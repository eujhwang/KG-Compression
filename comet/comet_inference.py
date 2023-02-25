import configparser
import json
import logging
import os
import pickle
import random
import sys
import argparse

import numpy as np
import torch
from collections import Counter

import tqdm

sys.path.append(os.getcwd())

import src.data.data as data
import src.data.config as comet_cfg
import src.interactive.functions as interactive

config = configparser.ConfigParser()
config.read("../preprocess/paths.cfg")

# cpnet = None
# cpnet_simple = None
concept2id = None
relation2id = None
id2concept = None
id2relation = None

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    """ Set all seeds to make results reproducible (deterministic mode).
         When seed is None, disables deterministic mode. """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_resources():
    global concept2id, relation2id, id2relation, id2concept
    concept2id = {}
    id2concept = {}
    with open(config["paths"]["concept_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            concept2id[w.strip()] = len(concept2id)
            id2concept[len(id2concept)] = w.strip()

    print("concept2id done")
    id2relation = {}
    relation2id = {}
    with open(config["paths"]["relation_vocab"], "r", encoding="utf8") as f:
        for w in f.readlines():
            id2relation[len(id2relation)] = w.strip()
            relation2id[w.strip()] = len(relation2id)

    print("relation2id done")


def load_kg(kg_path):
    print("loading train.kg.json...", end="")
    kgs = []
    with open(kg_path, 'r') as f:
        for line in f.readlines():
            kgs.append(json.loads(line))
    print("done!")

    return kgs


def load_comet(args):
    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)

    if args.device != "cpu":
        comet_cfg.device = int(args.device)
        comet_cfg.do_gpu = True
        torch.cuda.set_device(comet_cfg.device)
        model.cuda(comet_cfg.device)
    else:
        comet_cfg.device = "cpu"

    # input_event = "swimming"
    # relation = "relatedto"

    sampling_algorithm = args.sampling_algorithm
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    # if relation not in data.conceptnet_data.conceptnet_relations:
    #     relation = "all"
    return model, sampler, data_loader, text_encoder


def comet_inference(model, sampler, data_loader, text_encoder, input_event, relation):
    outputs = interactive.get_conceptnet_sequence(input_event, model, sampler, data_loader, text_encoder, relation)
    # output: {'CapableOf': {'e1': 'juice', 'relation': 'CapableOf', 'beams': ['taste good', 'come from apple', 'come from fruit']}}
    output_event = outputs[relation]['beams'][0]
    # print("input_event:", input_event, "relation:", relation, "output_event:", output_event)
    return output_event



def augment_kg_triples(args, kgs):
    print("start augmenting kg triples...")

    model, sampler, data_loader, text_encoder = load_comet(args)

    conceptnet_relations = data.conceptnet_data.conceptnet_relations
    _relation2id = dict()
    _id2relation = dict()
    for relation, relation_id in relation2id.items():
        for cr in conceptnet_relations:
            if relation == cr.lower():
                _relation2id[cr] = relation_id
                _id2relation[relation_id] = cr
                break

    # print("relation2id:", relation2id)
    # print("_relation2id:", _relation2id)
    # print("id2relation:", id2relation)
    # print("_id2relation:", _id2relation)

    max_new_rel_num = 3
    max_one_hop_concept_num = 30
    max_concept_num = 300
    max_triple_num = 750
    _data, _concepts = [], []
    for idx, kg in tqdm.tqdm(enumerate(kgs), total=len(kgs)):
        concepts = kg['concepts']
        labels = kg['labels']
        distances = kg['distances']
        relations = kg['relations']
        head_ids = kg['head_ids']
        tail_ids = kg['tail_ids']
        triple_labels = kg['triple_labels']
        relations = [rel[0] for rel in relations]

        # print("old concepts:", len(concepts), concepts)
        # print("old labels:", len(labels), labels)
        # print("old distances:", len(distances), distances)
        # print("old head_ids:", len(head_ids), head_ids)
        # print("old relations:", len(relations), relations)
        # print("old tail_ids:", len(tail_ids), tail_ids)
        # print("old triple_labels:", len(triple_labels), triple_labels)

        # extract concepts that are 0 or 1 hop away
        np_concepts = np.asarray(concepts, dtype=str)
        np_distances = np.asarray(distances, dtype=np.compat.long)
        np_head_ids = np.asarray(head_ids, dtype=np.compat.long)
        np_relations = np.asarray(relations, dtype=np.compat.long)
        np_tail_ids = np.asarray(tail_ids, dtype=np.compat.long)

        def _augment_kg(hop_concepts, hop_index, hop):
            # get relations related to zero-hop concept keywords
            hop_relation_dict = dict()
            for hop_concept, hop_id in zip(hop_concepts, hop_index):
                hop_relation_dict[(hop_concept, hop_id)] = set()
                hop_rels = np_relations[np_head_ids == hop_id].tolist()
                hop_relation_dict[(hop_concept, hop_id)] = set(hop_rels)

            # augmentation start...
            # find the relations that do not exist in the original, and then obtain new tail event using comet
            all_relation_ids = _id2relation.keys()
            for (hop_concept, hop_id), relation_ids in hop_relation_dict.items():
                # old_relation_ids = [rel_id for rel_id in list(relation_ids) if rel_id in all_relation_ids] # existing relations
                # new_relation_ids = all_relation_ids - old_relation_ids

                new_relation_ids = all_relation_ids - relation_ids
                new_relations = [_id2relation[rel] for rel in list(new_relation_ids)] # non-existing relations

                if hop != 0 and len(new_relations) > max_new_rel_num:
                    new_relations = random.sample(new_relations, max_new_rel_num)

                for rel in new_relations:
                    # obtain new tail event using comet inference
                    output_event = comet_inference(model, sampler, data_loader, text_encoder, hop_concept, rel)
                    if output_event not in concepts:
                        new_tail_id = len(concepts)
                        concepts.append(output_event)
                        labels.append(0)  # since new knowledge is not the target, label is 0
                        distances.append(hop+1)  # if 0-hop keyword is augmented, distance is 1. In the case of 1-hop, distance would be 2.
                        head_ids.append(hop_id)
                        relations.append(_relation2id[rel])
                        tail_ids.append(new_tail_id)
                        triple_labels.append(0)  # since we are adding new knowledge, triple label would be just 0

                        if len(concepts) >= max_concept_num and len(head_ids) >= max_triple_num:
                             break

        zero_hop_index = np.nonzero(np_distances == 0)[0].tolist()
        zero_hop_concepts = np_concepts[zero_hop_index]
        if len(zero_hop_concepts) > 0:
            _augment_kg(zero_hop_concepts, zero_hop_index, 0)

        one_hop_index = np.nonzero(np_distances == 1)[0].tolist()
        one_hop_concepts = np_concepts[one_hop_index]

        if len(one_hop_concepts) > max_one_hop_concept_num:
            one_hop_concepts = random.sample(one_hop_concepts.tolist(), max_one_hop_concept_num)
        if len(one_hop_concepts) > 0:
            _augment_kg(one_hop_concepts, one_hop_index, 1)

        # update kg values
        kg['concepts'] = concepts
        kg['labels'] = labels
        kg['distances'] = distances
        kg['relations'] = relations
        kg['head_ids'] = head_ids
        kg['tail_ids'] = tail_ids
        kg['triple_labels'] = triple_labels

        _concepts += concepts
        _data.append(kg)
        # print("")
        # print("new concepts:", len(concepts), concepts)
        # print("new labels:", len(labels), labels)
        # print("new distances:", len(distances), distances)
        # print("new head_ids:", len(head_ids), head_ids)
        # print("new relations:", len(relations), relations)
        # print("new tail_ids:", len(tail_ids), tail_ids)
        # print("new triple_labels:", len(triple_labels), triple_labels)
        # print("=================================================================")
    return _data, _concepts


def save_json(data, filename):
    with open(filename, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')


def main(args):
    dataset = args.data_dir
    DATA_PATH = config["paths"][dataset + "_dir"]
    kg_path = DATA_PATH + "/train.kg.json"

    load_resources()
    kgs = load_kg(kg_path)

    total_concepts = []
    _data, _concepts = augment_kg_triples(args, kgs)

    total_concepts += _concepts
    print("kg_path:", os.path.basename(kg_path))
    new_kg_file = ".".join(["augmented", os.path.basename(kg_path)])

    save_json(_data, DATA_PATH + f'/{new_kg_file}')

    words_by_frequency = sorted(Counter(total_concepts).items(), key=lambda x: x[1], reverse=True)
    print('total word counts: ', len(words_by_frequency))
    with open(DATA_PATH + '/augmented.kg_vocab.txt', 'w') as vocab_file:
        for word, frequency in words_by_frequency:
            vocab_file.write('{} {}\n'.format(word, frequency))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="eg")

    # comet-inference
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="pretrained_models/conceptnet_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="greedy", help="greedy, beam-#, top-#")

    args = parser.parse_args()
    set_seed(42)
    main(args)