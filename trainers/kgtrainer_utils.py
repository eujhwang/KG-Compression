import json
import linecache
import math
import os
import pickle
from logging import getLogger
from pathlib import Path
from random import sample
from typing import Callable, Dict, Iterable, List, Union

import networkx as nx
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, Sampler
from filelock import FileLock

from transformers import BartTokenizer, T5Tokenizer
from transformers.file_utils import cached_property
from transformers.modeling_bart import shift_tokens_right

try:
    import nltk

    NLTK_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    NLTK_AVAILABLE = False

if NLTK_AVAILABLE:
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def load_kg_vocab(path, tokenizer):
    concept2id = {'<pad>': 1, '<unk>': 3}
    with open(path, 'r') as f:
        for line in f.readlines():
            vocab, _ = line.strip().split()
            tokenized_vocab = tokenizer.encode(
                ' '+vocab, add_special_tokens=False)
            if len(tokenized_vocab) > 1:
                print('not covered vocab: ', vocab, tokenized_vocab)
            if concept2id.get(vocab):
                print('duplicated vocab: ', vocab, tokenized_vocab)
            concept2id[vocab] = tokenized_vocab[0]
    return concept2id


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()

        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        if not os.path.exists(self.src_file):
            self.src_file = Path(data_dir).joinpath(type_path + ".source.txt")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        if not os.path.exists(self.tgt_file):
            self.tgt_file = Path(data_dir).joinpath(type_path + ".target.txt")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")

        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

        ## load KG file
        self.kg_vocab = Path(data_dir).joinpath("kg_vocab.txt")
        self.concept2id = load_kg_vocab(self.kg_vocab, tokenizer)

        self.kg_path = Path(data_dir).joinpath(type_path + ".kg.json")

        self.concepts = []
        self.concepts_labels = []
        self.distances = []
        self.head_ids = []
        self.tail_ids = []
        self.relations = []
        self.triple_labels = []

        with open(self.kg_path, 'r') as f:
            for line in f.readlines():
                line = json.loads(line)
                assert(len(line['concepts']) == len(line['labels'])), (len(line['concepts']), len(line['labels']))
                self.concepts.append(line['concepts'])
                self.concepts_labels.append(line['labels'])
                self.distances.append(line['distances'])
                self.head_ids.append(line['head_ids'])
                self.tail_ids.append(line['tail_ids'])
                self.relations.append(line['relations'])
                self.triple_labels.append(line['triple_labels'])

        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

        self.train = True if type_path == "train" else False

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        src_line = self.prefix + linecache.getline(str(self.src_file), index+1).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index+1).rstrip("\n")

        assert src_line, f"empty source line for index {index+1}"
        assert tgt_line, f"empty tgt line for index {index+1}"

        source_inputs = self.encode_line(self.tokenizer, src_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()

        src_mask = source_inputs["attention_mask"].squeeze()
        tar_mask = target_inputs["attention_mask"].squeeze()

        concept = self.concepts[index]
        cpt_label = self.concepts_labels[index]
        dist = self.distances[index] 
        relations = self.relations[index]
        head_ids = self.head_ids[index]
        tail_ids = self.tail_ids[index]
        triple_labels = self.triple_labels[index]

        # FIXME: we should address all relations, not just taking random one relation out of all relations
        relations = [x[0] for x in relations] # taking only one relation out of many relations
        # concept: 202 cpt_label: 202 dist: 202 relations: 608 head_ids: 608 tail_ids: 608 triple_labels: 608 relations: 608
        # print("concept:", len(concept), "cpt_label:", len(cpt_label), "dist:", len(dist), "relations:", len(relations),
        #       "head_ids:", len(head_ids), "tail_ids:", len(tail_ids), "triple_labels:", len(triple_labels), "relations:", len(relations))
        # print()

        # create a graph
        # graph = nx.MultiDiGraph()
        # for head, tail, rels in zip(head_ids, tail_ids, relations):
        #     for rel in rels:
        #         graph.add_edge(head, tail, rel=rel, weight=1)
        # A = nx.adjacency_matrix(graph)

        # print("head_max:", max(head_ids), "tail_max", max(tail_ids))

        _concept = concept.copy()
        _cpt_label = cpt_label.copy()
        _head_ids = head_ids.copy()
        _tail_ids = tail_ids.copy()
        _relations = relations.copy()
        _triple_labels = triple_labels.copy()
        _dist = dist.copy()

        assert len(dist) == len(concept)

        max_concept_length = 300
        max_oracle_concept_length = 30
        max_triple_len = 600

        _concept_ids, _concept_labels, _concept_distances = self.encode_concept(
            self.concept2id, _concept, _cpt_label, _dist, max_concept_length)

        oracle_concept_ids, oracle_concept_mask = None, None
        if self.train:
            oracle_concept_ids, oracle_concept_mask, _ = self.encode_oracle_concept(
                self.concept2id, _concept, _cpt_label, _dist, max_oracle_concept_length)
            source_ids = torch.cat([source_ids, oracle_concept_ids], dim=0)
            src_mask = torch.cat([src_mask, oracle_concept_mask], dim=0)

        _head_ids, _tail_ids, _relation_ids, _triple_labels = self.encode_triples(
            _head_ids, _tail_ids, _relations, _triple_labels, max_triple_len)
        # print("_head_ids:", _head_ids)
        # print("_tail_ids:", _tail_ids)

        # new - eunjeong
        # graph = nx.DiGraph()
        adj = torch.zeros((max_concept_length, max_concept_length), device=_concept_ids.device)
        for head, tail, rel in zip(_head_ids, _tail_ids, _relation_ids):
            # print("head", head.item(), "tail", tail.item(), "rel", rel.item())
            src = head.item()
            dst = tail.item()
            if adj[src, dst] == 0:
                adj[src, dst] = 1
            else:
                adj[src, dst] += 1
            # graph.add_edge(head.item(), tail.item(), rel=rel.item(), weight=1)
            # if head.item() == 0:
                # print("head:", head.item(), "tail:", tail.item(), "rel:", rel.item())
            # break
        # print(graph.nodes())
        # assert False
        # A = nx.to_numpy_matrix(graph)
        # print("A:", adj.shape, adj)
        # print("A.nonzero():", adj.nonzero().tolist())
        # assert False
        # A = torch.tensor(A.todense(), device=_concept_ids.device, dtype=torch.float)
        # _A = torch.zeros((max_concept_length, max_concept_length), device=_concept_ids.device)
        # _A[:A.shape[0], :A.shape[1]] = A

        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
            "concept_ids": _concept_ids,
            "concept_distances": _concept_distances,
            "concept_labels": _concept_labels,
            "oracle_concept_ids": oracle_concept_ids,
            "oracle_concept_mask": oracle_concept_mask,
            "head_ids": _head_ids,
            "tail_ids": _tail_ids,
            "relation_ids": _relation_ids,
            "triple_labels": _triple_labels,
            "adj": adj,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer([line], max_length=max_length, padding="max_length" if pad_to_max_length else None,
            truncation=True, return_tensors=return_tensors, **self.dataset_kwargs,)

    def encode_concept(self, tokenizer, concepts, labels, distances, max_len, return_tensors="pt"):

        concept_ids = []
        for c in concepts:
            concept_ids.append(tokenizer[c])
        if len(concept_ids) >= max_len-1:
            concept_ids = [0] + concept_ids[:max_len-2] + [2]
            labels = [0] + labels[:max_len-2] + [0]
            distances = [0] + distances[:max_len-2] + [0]
        if len(concept_ids) < max_len-1:
            concept_ids = [0] + concept_ids + [2]
            labels = [0] + labels + [0]
            distances = [0] + distances + [0]
        while len(concept_ids) < max_len:
            concept_ids.append(1) # PAD_ID = 1
            labels.append(-1)
            distances.append(0)
        if return_tensors == 'pt':
            return torch.tensor(concept_ids), torch.tensor(labels), torch.tensor(distances)
        else:
            return concept_ids, labels, distances

    def encode_oracle_concept(self, tokenizer, concepts, labels, distances, max_len, return_tensors="pt"):

        # drop zero label
        _c, _l, _d = [], [], []
        for c, l, d in zip(concepts, labels, distances):
            if l == 1:
                _c.append(c)
                _l.append(1)
                _d.append(d)

        if len(concepts) > 20:
            sampled_concepts = sample(concepts, 20)
            _c += sampled_concepts
            _l += [1] * len(sampled_concepts)
            _d += [2] * len(sampled_concepts)

        concepts = _c
        labels = _l
        distances = _d

        concept_ids = []
        for c in concepts:
            concept_ids.append(tokenizer[c])
        if len(concept_ids) >= max_len-1:
            concept_ids = [0] + concept_ids[:max_len-2] + [2]
            labels = [1] + labels[:max_len-2] + [0]
            distances = [0] + distances[:max_len-2] + [0]
        if len(concept_ids) < max_len-1:
            concept_ids = [0] + concept_ids + [2]
            labels = [1] + labels + [0]
            distances = [0] + distances + [0]
        while len(concept_ids) < max_len:
            concept_ids.append(1) # PAD_ID = 1
            labels.append(0)
            distances.append(0)
        
        if return_tensors == 'pt':
            return torch.tensor(concept_ids), torch.tensor(labels), torch.tensor(distances)
        else:
            return concept_ids, labels, distances

    def encode_triples(self, head_ids, tail_ids, relation_ids, triple_labels, max_len, return_tensors='pt'):
        if len(head_ids) > max_len:
            head_ids = head_ids[:max_len]
            tail_ids = tail_ids[:max_len]
            relation_ids = relation_ids[:max_len]
            triple_labels = triple_labels[:max_len]
        while len(head_ids) < max_len:
            head_ids.append(0)
            tail_ids.append(0)
            relation_ids.append(0)
            triple_labels.append(-1)
        if return_tensors == 'pt':
            return torch.tensor(head_ids), torch.tensor(tail_ids), \
                torch.tensor(relation_ids), torch.tensor(triple_labels)
        else:
            return head_ids, tail_ids, relation_ids, triple_labels


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": isinstance(tokenizer, BartTokenizer)}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch) -> Dict[str, torch.Tensor]:

        # batch: list [{src: xxx, tag: xxx, id: xxx}, ...]
        input_ids = torch.stack([x["input_ids"] for x in batch])
        attention_mask = torch.stack([x["attention_mask"] for x in batch])
        labels = torch.stack([x["labels"] for x in batch])

        # batch concept information
        concept_ids = torch.stack([x["concept_ids"] for x in batch])
        concept_distances = torch.stack([x["concept_distances"] for x in batch])
        concept_labels = torch.stack([x["concept_labels"] for x in batch])

        # batch triple information 
        head_ids = torch.stack([x["head_ids"] for x in batch])
        tail_ids = torch.stack([x["tail_ids"] for x in batch])
        relation_ids = torch.stack([x["relation_ids"] for x in batch])
        triple_labels = torch.stack([x["triple_labels"] for x in batch])
        labels = trim_batch(labels, self.pad_token_id)
        adj = torch.stack([x["adj"] for x in batch])

        oracle_concept_ids = None
        if batch[0]["oracle_concept_ids"] is not None:
            oracle_concept_ids = torch.stack([x["oracle_concept_ids"] for x in batch])
            oracle_concept_mask = torch.stack([x["oracle_concept_mask"] for x in batch])
            oracle_concept_ids, oracle_concept_mask = trim_batch(oracle_concept_ids, self.pad_token_id, attention_mask=oracle_concept_mask)
            input_ids = torch.cat([input_ids, oracle_concept_ids], dim=1)
            attention_mask = torch.cat([attention_mask, oracle_concept_mask], dim=1)

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "concept_ids": concept_ids,
            "concept_distances": concept_distances,
            "concept_labels": concept_labels,
            "oracle_concept_ids": oracle_concept_ids,
            "head_ids": head_ids,
            "tail_ids": tail_ids,
            "relation_ids": relation_ids,
            "triple_labels": triple_labels,
            "adj": adj,
        }

        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

logger = getLogger(__name__)

def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def cost_matrix(x, y, p=2):
    "Returns the matrix of $|x_i-y_j|^p$."
    x_col = x.unsqueeze(1)
    y_lin = y.unsqueeze(0)
    #    x_col = x.unsqueeze(1)
    #    y_lin = y.unsqueeze(0)
    c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
    return c

def sinkhorn_loss(x, y, epsilon, mu, nu, n, m, p=2, niter=100, acc=1e-3, unbalanced=False, gpu=False):
    """
    Given two emprical measures with n points each with locations x and y
    outputs an approximation of the OT cost with regularization parameter epsilon
    niter is the max. number of steps in sinkhorn loop

    INPUTS:
        x : positions of diracs for the first distribution, torch.FloatTensor of size [n, d]
        y : positions of diracs for the second distribution, torch.FloatTensor of size [m, d]
        epsilon : importance of the entropic regularization
        mu : mass located at each dirac, torch.FloatTensor of size [n]
        nu : mass located at each dirac, torch.FloatTensor of size [m]
        n : total number of diracs of the first distribution
        m : total number of diracs of the second distribution
        niter : maximum number of Sinkhorn iterations
        acc : required accuracy to satisfy convergence
        unbalanced : specify if unbalanced OT needs to be solved
        gpu : specify usage of CUDA with pytorch

    OUTPUTs:
        cost : the cost of moving from distribution x to y
    """
    # The Sinkhorn algorithm takes as input three variables :
    # C = Variable(cost_matrix(x, y, p=p), requires_grad=True)  # Wasserstein cost function
    C= cost_matrix(x, y, p=p)

    # use GPU if asked to
    # if (gpu & torch.cuda.is_available()):
    #     C = C.cuda()
    #     mu = nu.cuda()
    #     nu = nu.cuda()

    # Parameters of the Sinkhorn algorithm.
    tau = -.8  # nesterov-like acceleration
    thresh = acc  # stopping criterion
    if (unbalanced):
        rho = 1(.5) ** 2  # unbalanced transport
        lam = rho / (rho + epsilon)  # Update exponent

    # Elementary operations .....................................................................
    def ave(u, u1):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

    def M(u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.repeat(m, 1).transpose(0, 1) + v.repeat(n, 1)) / epsilon

    def lse(A):
        "log-sum-exp"
        return torch.log(torch.exp(A).sum(1, keepdim=True) + 1e-6)  # add 10^-6 to prevent NaN

    # Actual Sinkhorn loop ......................................................................
    u, v, err = torch.zeros_like(mu).to(x.device), torch.zeros_like(nu).to(x.device), 0.
    u.requires_grad = True
    v.requires_grad = True
    actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

    for i in range(niter):
        u1 = u  # useful to check the update
        if (unbalanced):
            # accelerated unbalanced iterations
            u = ave(u, lam * (epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u))
            v = ave(v, lam * (epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v))
        else:
            u = epsilon * (torch.log(mu) - lse(M(u, v)).squeeze()) + u
            v = epsilon * (torch.log(nu) - lse(M(u, v).t()).squeeze()) + v
        err = (u - u1).abs().sum()

        actual_nits += 1
        if (err < thresh).data.numpy():
            break
    U, V = u, v
    pi = torch.exp(M(U, V))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * C)  # Sinkhorn cost

    return cost

def sinkhorn_loss_default(x, y, epsilon=0.01, p=2, niter=100, gpu=True):
    n = x.shape[0]
    m = y.shape[0]
    mu = torch.ones(n)/n
    nu = torch.ones(m)/m
    return sinkhorn_loss(x, y, epsilon, mu, nu, n, m, p, niter=niter, gpu=gpu)