import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import pad_ids, truncate_sequences
from itertools import chain
from tqdm import tqdm
import spacy
from os.path import join
import json

SPECIAL_TOKENS = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "[PAD]",
"additional_special_tokens": ["[Q]","[SUB]","[QUERY]"],
}

SPECIAL_TOKENS_VALUES = ["[BOS]", "[EOS]", "[PAD]", "[Q]","[SUB]","[QUERY]"]

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, name, split_type, masked=None, eval_partial=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type
        self.nlp = spacy.load("en_core_web_sm")
        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        # special tokens:   token to id
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.q_tok, self.ent_tok, self.query_tok = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["additional_special_tokens"])
        self.data = self._loaddata(dataset=name, split_type=split_type, masked=masked, load_partial=eval_partial) # loading data
        self.dep_mapping = json.load(open("data/qald9/dep_mapping.json"))

        self._create_examples()

    def build_input_from_segments(self, knowledge, question, ref_query, example, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and the question """
        instance = {}

        sequence = [[self.bos] + knowledge+ [self.q_tok]] + question+ [ref_query + ([self.eos] if with_eos else [])]
        #          [[bos]      +[SUB]+ENT1+..+[Q]]         + question + [[query] + sparql_query + [eos]]
        sequence_with_speaker = [ [self.query_tok] + s for i, s in enumerate(sequence[1:])]  # adding query token

        sequence = [sequence[0]]       +   sequence_with_speaker
        #       [bos]+ [SUB]+ENT1+...  +    [query] +q

        # Layer related tokens need to be processed here
        instance["input_ids"] = list(chain(*sequence))
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]

        if not isinstance(example["dep_ids"][0], int):
            example["dep_ids"] = [self.dep_mapping[dpid] if dpid is not None else 10 for dpid in example["dep_ids"]] # 10 means compound, setting compound
            example["dep_lvl"] = [dpid if dpid is not None else max(1,len(example["dep_lvl"])) for dpid in example["dep_lvl"]]  # if level is not found, setting the max level as the value

        example["dep_lvl"] = [dpid if dpid!=-1 else 2 for dpid in example["dep_lvl"]]

        inid ="input_ids"

        # aligning the shape of each layer by padding with default values or by truncating
        for labs in ["postag_ids", "dep_ids","dep_lvl"]:
            if len(example[labs]) == 0:
                instance[labs] = [2] * len(instance[inid])

            if len(example[labs]) < len(instance[inid]):
                instance[labs] = [2]+example[labs] + [2] * (len(instance[inid]) -len(example[labs])-1)

            if len(example[labs]) > len(instance[inid]):
                instance[labs] = example[labs][:len(instance[inid])]

            if len(example[labs]) != len(instance[inid]):
                instance[labs] = [2] * (len(instance[inid]) - len(instance[labs])) + instance[labs]

            if labs not in instance:
                instance[labs] = [2] * (len(instance[inid]) - len(example[labs])) + example[labs]
        instance["pos_ids"] = [j for j in range(1, len(instance[inid]) + 1)]

        try:
            assert len(instance["input_ids"])==len(instance["postag_ids"])==len(instance["dep_ids"])==len(instance["dep_lvl"])==len(instance["dep_lvl"])
        except:
            print("ALT: ",len(instance["input_ids"]),len(instance["postag_ids"]),len(instance["dep_ids"]),len(instance["dep_lvl"]))
        return instance

    def maskit(self, item):
        """mask the entity and relations in the question with generic tokens. i.e, ENT1,"""

        question = item["question"].replace("?", "").replace("<", "").replace(">", "").lower()
        query = item["sparql_dbpedia"]
        for i, ent in enumerate(item["entities"]):
            entlab = " ".join(ent.lower().split("_"))
            question = question.replace(entlab, f"ENT{i + 1}")
            query = " ".join(f"ENT{i + 1}" if ent in w else w for w in query.split()).replace("  ", " ").strip()
        return question.strip(), query

    def _loaddata(self, dataset="qald9", split_type="train", masked=None, load_partial=None):
        new_data = json.load(open(join("data", dataset, split_type+".json")))

        # process data
        formatted_data = []
        for d in new_data:
            temp = d.copy()
            temp["uid"] = temp["id"]
            temp["sparql_dbpedia"] = temp["fil_sparql"].replace("<http://www.w3.org/1999/02/22-rdf-syntax-ns#type","rdf_type")
            temp["sparql_original"] = temp["sparql"]
            temp["question"] = temp["en_ques"]
            formatted_data.append(temp)

        if load_partial:
            formatted_data=formatted_data[:150]  # only evaluate on 150 data
        if masked:     # mask the entity and relations in the question with generic tokens. i.e, ENT1, REL1
            masked_data = list()
            for data in formatted_data:
                temp_data = data.copy()
                question, sparql = self.maskit(data)
                temp_data["question"] = question
                temp_data["sparql_dbpedia"] = sparql
                masked_data.append(temp_data)
            return masked_data

        return formatted_data

    def clean_ref(self, text):
        """
        :param text: reference query
        :return:  format the query nicely, seperated by a space
        """
        kewords = ["(", "{", ")", "}", ","]
        for kw in kewords:
            text = text.replace(kw, " " + kw + " ")
        text = text.replace("  ", " ")
        text = " ".join(w if ":" in w else w.lower() for w in text.split(" ")).replace("  ", " ")
        return text

    def format_knowledge(self, item):
        """Linearize and seperated with special tokens
           Right now we only consider entity. in the next step we'll also use relation
        """
        seq = ""
        for e in item["entities"]:
            seq += " [SUB] " + e
        tokenized_knowledge = self.tokenizer.tokenize(seq.strip())
        return (tokenized_knowledge, len(tokenized_knowledge))


    def align_tokens(self,ques, tokenized_ques):
        """
        Align the huggingface-based tokenized text with the spacy based one.
        This is required because for the LM huggingface one is important where the dependency tree related stuffs are coming from spacy
        """
        special = "Ġ" # the special char appears before each token in the tokenized text from huggingface
        tok = [s.replace(special, "") for s in tokenized_ques]   # removing the special char from all the tokens to match with the spacy-based tokenized text
        doc = self.nlp(ques)

        t = list()  # text token
        tp = list() # token dependency relation/type
        ps = list() # token pos tags
        childlist = list()
        treelevel = {i: -1 for i in range(len(tok))}
        for token in doc:
            # print(token.text, '  -> ',token.dep_)
            t.append(token.text)
            tp.append(token.dep_)
            ps.append(token.pos_)
            childlist.append([child for child in token.children])

        allchilds = [child for child in childlist if child]
        listdone = [-1 for alist in allchilds]

        kk = 0
        while (kk < len(t)):
            for lvl, adep in enumerate(allchilds):
                if listdone[lvl] == -1:
                    for tlst, ac in enumerate(adep):
                        for k, v in treelevel.items():
                            if treelevel[k] == -1 and t[k] == ac.text:
                                treelevel[k]=lvl + 2
                                break
                    listdone[lvl] = 1
            kk += 1

        #root check
        temp = treelevel.copy()
        for m,n in temp.items():
            if n==-1:
                treelevel[m]=1
                break

        tlv = [-1 for i in range(len(tok))]

        mapping = {i: {"token": None, "pos": None, "dep-tok": None, "dep-lvl": None} for i in range(len(tok))}

        j = 0

        for i, a in enumerate(tok):
            for k, et in enumerate(t):
                if a in et and j >= k:
                    mapping[i]["token"] = a
                    mapping[i]["pos"] = ps[k]
                    mapping[i]["dep-tok"] = tp[k]
                    mapping[i]["dep-lvl"] = treelevel[k]
                    j += 1
        return [b["dep-tok"] for a,b in mapping.items()],[b["dep-lvl"] for a,b in mapping.items()]


    def format_knowledge_syn(self, pos_tokens, pos, dep, depl, l):
        idx = len(pos)-1  # setting the last item by default

        # looking for the index of "PROPN" or "NOUN" or "PRON" by default
        try:
            idx = pos_tokens.index("PROPN")
        except:
            try:
                idx = pos_tokens.index("NOUN")
            except:
                try:
                    idx = pos_tokens.index("PRON")
                except:
                    idx = len(pos)-1

        pos = [pos[idx]]*(l+1) + pos  # l+1 to align with knowledge sequence where self.bos was added
        dep = [dep[idx]]*(l+1) + dep
        depl = [depl[idx]] * (l + 1) + depl
        return pos, dep, depl

    def _create_examples(self):
        print("Creating examples")
        self.examples = []
        for item in tqdm(self.data):
            query_id = item["uid"]

            #prepare the knowledge sequence
            used_knowledge, klen = self.format_knowledge(item) if self.args.knowledge else ([],0)  # preparing entity sequence
            used_knowledge =  self.tokenizer.convert_tokens_to_ids(used_knowledge)                 # token to id
            used_knowledge = used_knowledge[:self.args.knowledge_max_tokens]                       # truncating extra sequence

            ques = item["question"].replace("?","")
            tok_ques = self.tokenizer.tokenize(ques)
            tokenized_question = self.tokenizer.convert_tokens_to_ids(tok_ques)

            query_text = self.clean_ref(item["sparql_dbpedia"])                    # making each token in the query single space seperated (for training)
            tokenized_query = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(query_text))

            pos_ids = item["question_pos_ids"]                    # already preprocessed using utils/dptree.py
            dep_ids, dep_lvls = item["question_dep_ids"], item["question_dep_lvl"]     # already preprocessed utils/dptree.py
            pos_ids, dep_ids, dep_lvls = self.format_knowledge_syn(item["question_pos_tokens"], pos_ids, dep_ids, dep_lvls, klen)

            # truncating extra lengths
            pos_ids = pos_ids[:self.args.input_max_tokens]              # postag ids
            dep_ids = dep_ids[:self.args.input_max_tokens]              # dependency relations
            dep_lvls = dep_lvls[:self.args.input_max_tokens]            # dependency levels

            # preparing a single data point for training and evaluation
            self.examples.append({
                "question": [tokenized_question[:self.args.input_max_tokens]],
                "knowledge": used_knowledge,
                "postag_ids": pos_ids,
                "dep_ids": dep_ids,
                "dep_lvl": dep_lvls,
                "sparql_tokenized": tokenized_query,
                "sparql_text": query_text,
                "id": query_id
            })

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class Dataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, masked, eval_partial):
        super(Dataset, self).__init__(args, tokenizer, name, split_type, masked=masked, eval_partial=eval_partial)

    def __getitem__(self, index):
        example = self.examples[index]
        instance= self.build_input_from_segments(example["knowledge"],example["question"],example["sparql_tokenized"], example)
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        pos_ids = [ins["pos_ids"] for ins in batch]
        postag_ids = [ins["postag_ids"] for ins in batch]
        dep_ids = [ins["dep_ids"] for ins in batch]
        dep_lvl = [ins["dep_lvl"] for ins in batch]

        # padding and converting into tensors
        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        pos_ids = torch.tensor(pad_ids(pos_ids, 0))
        postag_ids = torch.tensor(pad_ids(postag_ids, 0))
        dep_ids = torch.tensor(pad_ids(dep_ids, 0))
        dep_lvl = torch.tensor(pad_ids(dep_lvl, 0))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        return input_ids, pos_ids, postag_ids, dep_ids, dep_lvl, lm_labels


class EvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, masked, eval_partial):
        super(EvalDataset, self).__init__(args, tokenizer, name, split_type, masked, eval_partial)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch