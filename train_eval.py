import argparse
import logging, json
import torch
from tqdm import tqdm, trange
from argparse import Namespace
from typing import Dict, Tuple
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
     AdamW,
     AutoConfig,
     AutoTokenizer,
     AutoModel,
     PreTrainedModel,
     PreTrainedTokenizer,
     get_linear_schedule_with_warmup,
)

from model.gpt import run_batch_generation, LlamaLMHeadModel

import os
from utils.args import (
    set_default_params,
    update_additional_params
)
from torch.optim import lr_scheduler
import random
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

from transformers import LlamaForCausalLM, LlamaTokenizer

from model.gpt import decode_sample
from utils.metrics import (
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE, SPBLEU, SPUnigramMetric
)
from utils.data import write_generation_preds


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join(args.output_dir, args.exp_name, args.dataset) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    global_eval = 99999
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler=="linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=0.0001, verbose=True)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    global_step = 0
    model.zero_grad()
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    set_seed(args)  # for reproducibility

    for _ in train_iterator:
        tr_loss = 0.0
        local_steps = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss, _, _, _ = run_batch_fn_train(args, model, batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.scheduler=="linear":
                    scheduler.step()
                else:
                    pass
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / local_steps)

        results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
            if args.scheduler == "reducelr":
                scheduler.step(results["loss"])
                tb_writer.add_scalar("lr", args.learning_rate)
            else:
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)


            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            logger.info("Saving model checkpoint to %s", output_dir)
            global_eval=results["loss"]
            #model_to_save.save_pretrained(output_dir, safe_serialization=False)
            #tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
                logger.info("Saving model checkpoint to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step

def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            loss, lm_logits, mc_logits, mc_labels = run_batch_fn(args, model, batch)
            all_preds.append(mc_logits.detach().cpu().numpy())
            all_labels.append(mc_labels.detach().cpu().numpy())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    if args.task.lower() == "generation":
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {"perplexity": perplexity, "loss": eval_loss}

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, default="config/llama-base/params.json", help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--epochs", type=int, default=-1, help="number of epochs for training")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1, help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data", help="Path to dataset.")
    parser.add_argument("--dataset", type=str, default="lcquad2", choices=["lcquad2","qald9", "vquanda"],
                        help="dataset name.")
    parser.add_argument('--eval_partial', action='store_true')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--knowledge', action='store_true')

    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear","reducelr"])
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="all",
                        help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="sgpt",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--output_dir", type=str, default="runs", help="Output directory for checkpoints and predictions")

    parser.add_argument('--generate', action='store_true')
    parser.add_argument("--decode", type=str, default="basic", choices=["basic","beam"], help="decoding technique")
    parser.add_argument("--generation_params_file", type=str, default="config/llama-base/generation_params.json",
                        help="JSON configuration file for generation-related configurations.")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.dataset=="lcquad2":
        from scripts.dataset_lcquad2 import Dataset, SPECIAL_TOKENS
    if args.dataset=="vquanda":
        from scripts.dataset_vquanda import Dataset, SPECIAL_TOKENS
    if args.dataset=="qald9":
        from scripts.dataset_qald9 import Dataset, SPECIAL_TOKENS


    args = parser.parse_args()
    fromcommand = args
    params = json.load(open(args.params_file, "r"))
    params["num_train_epochs"]= fromcommand.epochs
    args = vars(args)
    update_additional_params(params, args)
    args.update(params)
    args = Namespace(**args)


    args.params = params  # used for saving checkpoints
    if fromcommand.epochs!=-1:
        args.num_train_epochs = fromcommand.epochs


    tokenizer = AutoTokenizer.from_pretrained(args.llama_output_path)
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.task = args.task
    dataset_args.knowledge = args.knowledge


    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    set_seed(args)
    args.train_batch_size = 2

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = Dataset, LlamaLMHeadModel, run_batch_generation, run_batch_generation

    if args.eval_only:
        pass
    else:
        config = AutoConfig.from_pretrained(args.llama_output_path)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        config.knowledge_max_tokens = dataset_args.knowledge_max_tokens
        tokenizer = AutoTokenizer.from_pretrained(args.llama_output_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model = model_class.from_pretrained(args.llama_output_path, config=config)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    #####   training    #######
    if not args.eval_only:

        train_dataset = dataset_class(dataset_args, tokenizer, name=args.dataset, split_type="train", masked=args.masked ,eval_partial=None)
        eval_dataset = dataset_class(dataset_args, tokenizer, name=args.dataset,split_type="val", masked=args.masked, eval_partial=None)

        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=train_dataset.collate_fn)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        global_step = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval)
        #logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            #model.save_pretrained(args.output_dir, safe_serialization=True)
            #tokenizer.save_pretrained(args.output_dir, safe_serialization=True)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)

            # Load a trained model and vocabulary that you have fine-tuned
            #config = AutoConfig.from_pretrained(args.output_dir)
            #config.output_past = False
            #config.knowledge_max_tokens = dataset_args.knowledge_max_tokens

            #tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
            #model = model_class.from_pretrained(args.output_dir, config=config)
            #model.resize_token_embeddings(128256)
            #model.to(args.device)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = dataset_class(dataset_args, tokenizer, name=args.dataset, split_type=args.eval_dataset, masked=args.masked, eval_partial=None)
        result = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or args.eval_dataset)
    return args, model, tokenizer



################


def is_num(text):
    """ Check if the string is a number"""
    text = text.replace("\'","")
    text = text.replace('\"', "")
    return text.replace('.','',1).isdigit()

def process_output(text, args):
    kewords = ["(","{",")","}",",",]
    if args.masked:
        kewords+=["ent1", "ent2", "ent3","ent4"]
    for kw in kewords:
        text = text.replace(kw, " " + kw + " ")
    text = text.replace("  ", " ")
    text = " ".join(w if ":" in w else w.lower() for w in text.split(" ")).replace("  "," ")
    text = text.replace("?", " ?").replace("  ", " ")
    out = " "
    for w in text.split(" "):
        if "." in w:
            if is_num(w):
                out += " " + w
            else:
                out += w.replace(".", " .")
        else:
            out+= " "+w
    kw = ['dbo:','dbp:','dbpedia2:','yago:','foaf:','onto:','res:','dbr:','dbc:','wd:','wdt:','ps:', 'pq:']
    for k in kw:
        text = " ".join( w.replace(k," "+k) if k in w else w for w in out.split(" ")).replace("  "," ")
    for k in kw:
        text = text.replace(k," "+k).replace("  "," ")

    return text.replace("  ", " ").strip()

def normalized_vars(query):
    ref_vars = list()
    cleaned_query = query
    for token in query.split():
        if token.startswith("?") and token not in ref_vars:
            ref_vars.append(token)

    for i,var in enumerate(ref_vars):
        cleaned_query = cleaned_query.replace(var,f"var{i+1}")
    return cleaned_query

def evaluate_ewo(args, eval_dataset, model, tokenizer, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,  # only support batch_size=1 for sampling right now
        collate_fn=eval_dataset.collate_fn
    )

    metrics = [
        UnigramMetric(metric="recall"),
        UnigramMetric(metric="precision"),
        SPUnigramMetric(metric="recall"),
        SPUnigramMetric(metric="precision"),
        NGramDiversity(n=1),
        NGramDiversity(n=2),
        NGramDiversity(n=3),
        NGramDiversity(n=4),
        CorpusNGramDiversity(n=1),
        CorpusNGramDiversity(n=2),
        CorpusNGramDiversity(n=3),
        CorpusNGramDiversity(n=4),
        BLEU(),
        SPBLEU(),
        METEOR(),
        ROUGE()
    ]

    args.tokenizer = tokenizer
    all_output_texts = []
    all_ground_truths = []
    dialog_ids = []
    tasks = []
    do_evaluate = False
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            sampled_output_ids, ground_truth, query_id, knowledge_text = decode_sample(args, model, batch, eval_dataset)
            sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)
            print("Truth: ", ground_truth)
            sampled_output_text = process_output(process_output(sampled_output_text, args), args)
            print("Predt: ", sampled_output_text)
            all_output_texts.append(sampled_output_text)
            all_ground_truths.append(ground_truth)
            dialog_ids.append(query_id)
            sampled_output_text_norm, ground_truth_norm = normalized_vars(sampled_output_text), normalized_vars(ground_truth)
        if ground_truth.strip() != "":
            do_evaluate = True
            for metric in metrics:
                metric.update((sampled_output_text, ground_truth))
                name = metric.name()
                if name.startswith("SP"):
                    metric.update((sampled_output_text_norm, ground_truth_norm))
                else:
                    metric.update((sampled_output_text, ground_truth))

    if args.output_file:
        write_generation_preds(args.output_file, dialog_ids, all_output_texts, all_ground_truths)

    result = dict()
    if do_evaluate and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        pre, rec, sp_pre, sp_rec = 0.0, 0.0, 0.0, 0.0
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for metric in metrics:
                name = metric.name()
                score = metric.compute()
                if name == "UnigramRecall":
                    rec = score
                elif name == "UnigramPrecision":
                    pre = score
                elif name == "SP-UnigramRecall":
                    sp_rec = score
                elif name == "SP-UnigramPrecision":
                    sp_pre = score
                print(name, str(score))
                result[name] = score
                logger.info("  %s = %s", name, str(score))
                writer.write("%s = %s\n" % (name, str(score)))
            f1 = (2 * pre * rec) / (pre + rec)
            sp_f1 = (2 * sp_pre * sp_rec) / (sp_pre + sp_rec)
            print("F1-score: ", round(f1, 4))
            print("SP-F1-score: ", round(sp_f1, 4))

    return result


def main_ewo(args, model, tokenizer):
    if args.dataset=="lcquad2":
        from scripts.dataset_lcquad2 import EvalDataset
    if args.dataset == "vquanda":
        from scripts.dataset_vquanda import EvalDataset
    if args.dataset=="qald9":
        from scripts.dataset_qald9 import EvalDataset

    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.output_dir, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        if len(args["generation_params_file"]) > 0:
            with open(args["generation_params_file"]) as fg:
                generation_params = json.load(fg)
            args.update(generation_params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.knowledge = args.knowledge

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    #args.output_dir = args.checkpoint
    #tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, ignore_mismatched_sizes=True)
    #model = LlamaLMHeadModel.from_pretrained(args.checkpoint, ignore_mismatched_sizes=True)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Generation parameters %s", args)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = EvalDataset(dataset_args, tokenizer, name=args.dataset, split_type=args.eval_dataset, masked=args.masked, eval_partial=args.eval_partial)
        result = evaluate_ewo(args, eval_dataset, model, tokenizer, desc=args.eval_desc or args.eval_dataset)
    return result


if __name__=="__main__":
    args, model, tokenizer = main()
    main_ewo(args, model, tokenizer)