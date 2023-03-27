import configparser
import json
import logging
import multiprocessing
import os
import pickle
import random
import sys
import argparse
import time
from multiprocessing import Process

import numpy
import numpy as np
import torch
from collections import Counter

import tqdm

sys.path.append(os.getcwd())

import src.data.data as data
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

    conceptnet_relations = data.conceptnet_data.conceptnet_relations
    _relation2id = dict()
    _id2relation = dict()
    for relation, relation_id in relation2id.items():
        for cr in conceptnet_relations:
            if relation == cr.lower():
                _relation2id[cr] = relation_id
                _id2relation[relation_id] = cr
                break

    relation2id = _relation2id
    id2relation = _id2relation
    print("relation2id done")


def load_kg(kg_path):
    print(f"loading {kg_path}...", end="")
    kgs = []
    with open(kg_path, 'r') as f:
        for line in f.readlines():
            kgs.append(json.loads(line))
    print("done!")

    return kgs

def load_concepts_nv(concepts_nv_path):
    print(f"loading {concepts_nv_path}...", end="")
    concepts_nv = []
    with open(concepts_nv_path, 'r') as f:
        for line in f.readlines():
            concepts_nv.append(json.loads(line))
    print("done!")

    return concepts_nv


def load_comet(args, device):
    opt, state_dict = interactive.load_model_file(args.model_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    model = model.to(device)
    # if args.device != "cpu":
    #     comet_cfg.device = int(args.device)
    #     comet_cfg.do_gpu = True
    #     torch.cuda.set_device(comet_cfg.device)
    #     model.cuda(comet_cfg.device)
    # else:
    #     comet_cfg.device = "cpu"

    sampling_algorithm = args.sampling_algorithm
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    # if relation not in data.conceptnet_data.conceptnet_relations:
    #     relation = "all"
    return model, sampler, data_loader, text_encoder


def load_comet(comet_file, sampling_algorithm, device):
    opt, state_dict = interactive.load_model_file(comet_file)

    data_loader, text_encoder = interactive.load_data("conceptnet", opt)

    n_ctx = data_loader.max_e1 + data_loader.max_e2 + data_loader.max_r
    n_vocab = len(text_encoder.encoder) + n_ctx

    model = interactive.make_model(opt, n_vocab, n_ctx, state_dict)
    model = model.to(device)
    # if args.device != "cpu":
    #     comet_cfg.device = int(args.device)
    #     comet_cfg.do_gpu = True
    #     torch.cuda.set_device(comet_cfg.device)
    #     model.cuda(comet_cfg.device)
    # else:
    #     comet_cfg.device = "cpu"

    # sampling_algorithm = args.sampling_algorithm
    sampler = interactive.set_sampler(opt, sampling_algorithm, data_loader)

    # if relation not in data.conceptnet_data.conceptnet_relations:
    #     relation = "all"
    return model, sampler, data_loader, text_encoder


def comet_inference(model, sampler, data_loader, text_encoder, input_event, relation, device):
    outputs = interactive.get_conceptnet_sequence(input_event, model, sampler, data_loader, text_encoder, relation, device)
    # output: {'CapableOf': {'e1': 'juice', 'relation': 'CapableOf', 'beams': ['taste good', 'come from apple', 'come from fruit']}}
    output_event = outputs[relation]['beams'][0]
    # print("input_event:", input_event, "relation:", relation, "output_event:", output_event)
    return output_event


def _augment_kg(kg, hop_concepts, hop_index, hop_distances, _relation2id, _id2relation, model, sampler, data_loader, text_encoder,
                max_new_rel_num, max_concept_num, max_triple_num, _concepts, _data, device):
    # c_proc = multiprocessing.current_process()
    # print("Running on Process",c_proc.name,"PID",c_proc.pid, "idx:", idx)
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


    # get relations related to zero-hop concept keywords
    hop_relation_dict = dict()
    for hop_concept, hop_id, hop_dist in zip(hop_concepts, hop_index, hop_distances):
        hop_relation_dict[(hop_concept, hop_id, hop_dist)] = set()
        hop_rels = np_relations[np_head_ids == hop_id].tolist()
        hop_relation_dict[(hop_concept, hop_id, hop_dist)] = set(hop_rels)

    # augmentation start...
    # find the relations that do not exist in the original, and then obtain new tail event using comet
    all_relation_ids = _id2relation.keys()
    for (hop_concept, hop_id, hop_dist), relation_ids in hop_relation_dict.items():
        # old_relation_ids = [rel_id for rel_id in list(relation_ids) if rel_id in all_relation_ids] # existing relations
        # new_relation_ids = all_relation_ids - old_relation_ids

        new_relation_ids = all_relation_ids - relation_ids
        new_relations = [_id2relation[rel] for rel in list(new_relation_ids)]  # non-existing relations

        if hop_dist != 1 and len(new_relations) > max_new_rel_num:
            new_relations = random.sample(new_relations, max_new_rel_num)

        for rel in new_relations:
            # obtain new tail event using comet inference
            output_event = comet_inference(model, sampler, data_loader, text_encoder, hop_concept, rel, device)
            if output_event not in concepts:
                new_tail_id = len(concepts)
                concepts.append(output_event)
                labels.append(0)  # since new knowledge is not the target, label is 0
                distances.append(
                    hop_dist)  # if 0-hop keyword is augmented, distance is 1. In the case of 1-hop, distance would be 2.
                head_ids.append(hop_id)
                relations.append(_relation2id[rel])
                tail_ids.append(new_tail_id)
                triple_labels.append(0)  # since we are adding new knowledge, triple label would be just 0

                if len(concepts) >= max_concept_num and len(head_ids) >= max_triple_num:
                    break
    # update kg values
    kg['concepts'] = concepts
    kg['labels'] = labels
    kg['distances'] = distances
    kg['relations'] = relations
    kg['head_ids'] = head_ids
    kg['tail_ids'] = tail_ids
    kg['triple_labels'] = triple_labels

    # print("")
    # print("new concepts:", len(concepts), concepts)
    # print("new labels:", len(labels), labels)
    # print("new distances:", len(distances), distances)
    # print("new head_ids:", len(head_ids), head_ids)
    # print("new relations:", len(relations), relations)
    # print("new tail_ids:", len(tail_ids), tail_ids)
    # print("new triple_labels:", len(triple_labels), triple_labels)
    # print("=================================================================")

    _concepts += concepts
    _data.append(kg)

def augment_kg_triples(args, kgs, device):
    print("start augmenting kg triples...")

    model, sampler, data_loader, text_encoder = load_comet(args.model_file, args.sampling_algorithm, device)

    # if torch.cuda.is_available():
    #     multiprocessing.set_start_method('spawn', force=True)
    #     pool = multiprocessing.get_context('spawn').Pool(args.num_proc)
    #     model.share_memory()
    # else:
    #     pool = multiprocessing.Pool(args.num_proc)
    # procs = []
    # start_time = time.perf_counter()



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

        print("old concepts:", len(concepts), "old labels:", len(labels), "old distances:", len(distances),
              "old head_ids:", len(head_ids), "old relations:", len(relations), "old tail_ids:", len(tail_ids),
              "old triple_labels:", len(triple_labels))
        # print("old concepts:", len(concepts)) #, concepts)
        # print("old labels:", len(labels)) #, labels)
        # print("old distances:", len(distances)) #, distances)
        # print("old head_ids:", len(head_ids)) #, head_ids)
        # print("old relations:", len(relations)) #, relations)
        # print("old tail_ids:", len(tail_ids)) #, tail_ids)
        # print("old triple_labels:", len(triple_labels)) #, triple_labels)

        # extract concepts that are 0 or 1 hop away
        np_concepts = np.asarray(concepts, dtype=str)
        np_distances = np.asarray(distances, dtype=np.compat.long)
        np_head_ids = np.asarray(head_ids, dtype=np.compat.long)
        np_relations = np.asarray(relations, dtype=np.compat.long)
        np_tail_ids = np.asarray(tail_ids, dtype=np.compat.long)

        new_concepts = []
        new_labels = []
        new_distances = []
        new_relations = []
        new_head_ids = []
        new_tail_ids = []
        new_triple_labels = []
        new_relations = []
        def _augment_kg(hop_concepts, hop_index, hop_distances):
            # get relations related to zero-hop concept keywords
            hop_relation_dict = dict()
            for hop_concept, hop_id, hop_dist in zip(hop_concepts, hop_index, hop_distances):
                hop_relation_dict[(hop_concept, hop_id, hop_dist)] = set()
                hop_rels = np_relations[np_head_ids == hop_id].tolist()
                hop_relation_dict[(hop_concept, hop_id, hop_dist)] = set(hop_rels)

            # augmentation start...
            # find the relations that do not exist in the original, and then obtain new tail event using comet
            all_relation_ids = _id2relation.keys()
            for (hop_concept, hop_id, hop_dist), relation_ids in hop_relation_dict.items():
                # old_relation_ids = [rel_id for rel_id in list(relation_ids) if rel_id in all_relation_ids] # existing relations
                # new_relation_ids = all_relation_ids - old_relation_ids

                new_rel_ids = all_relation_ids - relation_ids
                new_rels = [_id2relation[rel] for rel in list(new_rel_ids)] # non-existing relations

                if hop_dist != 1 and len(new_rels) > max_new_rel_num:
                    new_rels = random.sample(new_rels, max_new_rel_num)

                for rel in new_rels:
                    # obtain new tail event using comet inference
                    output_event = comet_inference(model, sampler, data_loader, text_encoder, hop_concept, rel, device)
                    if output_event not in concepts:
                        new_tail_id = len(concepts)
                        new_concepts.append(output_event)
                        new_labels.append(0)  # since new knowledge is not the target, label is 0
                        new_distances.append(hop_dist)  # if 0-hop keyword is augmented, distance is 1. In the case of 1-hop, distance would be 2.
                        new_head_ids.append(hop_id)
                        new_relations.append(_relation2id[rel])
                        new_tail_ids.append(new_tail_id)
                        new_triple_labels.append(0)  # since we are adding new knowledge, triple label would be just 0

                        # if len(concepts) >= max_concept_num and len(head_ids) >= max_triple_num:
                        #      break

        zero_hop_index = np.nonzero(np_distances == 0)[0].tolist()
        zero_hop_concepts = np_concepts[zero_hop_index].tolist()
        zero_hop_distances = [1] * len(zero_hop_concepts)

        one_hop_index = np.nonzero(np_distances == 1)[0].tolist()
        one_hop_concepts = np_concepts[one_hop_index].tolist()

        if len(one_hop_index) > max_one_hop_concept_num:
            one_hop_index = random.sample(one_hop_index, max_one_hop_concept_num)
            one_hop_concepts = np_concepts[one_hop_index].tolist()
            assert len(one_hop_index) == len(one_hop_concepts)
        one_hop_distances = [2] * len(one_hop_concepts)

        hop_index = zero_hop_index + one_hop_index
        hop_concepts = zero_hop_concepts + one_hop_concepts
        hop_distances = zero_hop_distances + one_hop_distances

        _augment_kg(hop_concepts, hop_index, hop_distances)

        # _concepts += concepts
        # _data.append(kg)
        print("")
        print("new concepts:", len(new_concepts), new_concepts)
        print("new labels:", len(new_labels), new_labels)
        print("new distances:", len(new_distances), new_distances)
        print("new head_ids:", len(new_head_ids), new_head_ids)
        print("new relations:", len(new_relations), new_relations)
        print("new tail_ids:", len(new_tail_ids), new_tail_ids)
        print("new triple_labels:", len(new_triple_labels), new_triple_labels)
        print("=================================================================")

        # proc =
        # proc.start()
        # procs.append(pool.apply_async(func=_augment_kg, args=(kg, hop_concepts, hop_index, hop_distances, _relation2id, _id2relation,
        #                                                 model, sampler, data_loader, text_encoder, max_new_rel_num,
        #                                                 max_concept_num, max_triple_num, _concepts, _data, device)))

    # Joins all the processes
    # for p in tqdm.tqdm(procs, total=len(procs)):
    #     p.get()
    # pool.close()
    # pool.join()
    # finish_time = time.perf_counter()
    # print("done")
    print("len(_data):", len(_data), _data[:10])
    return _data, _concepts


def save_json(data, filename):

    with open(filename, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')


def aggregate_concepts(args, kgs, concepts_nv, model, sampler, data_loader, text_encoder, device, DATA_PATH):
    concepts_nv_dict = dict()

    # [qc] = {'relations': kg['relations'],
    #         'ac': {ac},
    #         'index': [indices]}
    all_relation_ids = id2relation.keys()
    max_one_hop_concept_num = args.max_one_hop_concept_num

    if args.end > len(kgs):
        end = len(kgs)
    else:
        end = args.end

    kgs = kgs[args.start: end]
    concepts_nv = concepts_nv[args.start: args.end]

    for idx, (kg, nv) in tqdm.tqdm(enumerate(zip(kgs, concepts_nv)), total=len(kgs)):
        qc = tuple(nv['qc'])
        # ac = nv['ac']
        concepts = kg['concepts']
        head_ids = kg['head_ids']
        # tail_ids = kg['tail_ids']
        distances = kg['distances']
        relations = kg['relations']
        relations = [rel[0] for rel in relations]

        if len(concepts) == 0:
            continue

        # np_concepts = np.asarray(concepts, dtype=str)
        np_distances = np.asarray(distances, dtype=numpy.int32)
        np_head_ids = np.asarray(head_ids, dtype=numpy.int32)
        np_relations = np.asarray(relations, dtype=numpy.int32)

        if len(head_ids) == 0:
            topk_head_ids = np.nonzero(np_distances == 0)[0]
        else:
            # head_ids_by_freq = sorted(Counter(head_ids).items(), key=lambda x: x[1], reverse=True)
            # topk_head_ids = head_ids_by_freq[:10]
            # topk_head_ids = np.array(topk_head_ids, dtype=int)[:, 0]  # [:, 1] is frequency

            zero_hop_ids = np.nonzero(np_distances == 0)[0].flatten()
            one_hop_ids = np.nonzero(np_distances == 1)[0].flatten()
            one_hop_ids = set(head_ids).intersection(set(one_hop_ids))
            if len(one_hop_ids) > max_one_hop_concept_num:
                one_hop_ids = random.sample(one_hop_ids, max_one_hop_concept_num)

            topk_head_ids = zero_hop_ids.tolist() + list(one_hop_ids)
            # if len(one_hop_index) > max_one_hop_concept_num:
            #     one_hop_index = random.sample(one_hop_index, max_one_hop_concept_num)
            #     one_hop_concepts = np_concepts[one_hop_index].tolist()
            #     assert len(one_hop_index) == len(one_hop_concepts)

        if qc not in concepts_nv_dict.keys():
            new_concepts = []
            new_labels = []
            new_distances = []
            new_head_ids = []
            new_relations = []
            new_tail_ids = []
            new_triple_labels = []
            for head_id in topk_head_ids:
                hop = distances[head_id]
                distance = hop+1

                rels = np_relations[np_head_ids == head_id]
                rels = set(rels.flatten())
                new_rel_ids = all_relation_ids - rels

                for rid in new_rel_ids:
                    rname = id2relation[rid]

                    # obtain new tail event using comet inference
                    output_event = comet_inference(model, sampler, data_loader, text_encoder, concepts[head_id], rname, device)

                    event_split_dict = dict(Counter(output_event.split()))
                    if len(event_split_dict) == 1:
                        key = list(event_split_dict.keys())[0]
                        if event_split_dict[key] > 1:
                            continue

                    if output_event not in concepts and output_event not in new_concepts:
                        new_tail_id = len(concepts) + len(new_concepts)
                        new_concepts.append(output_event)
                        new_labels.append(0)  # since new knowledge is not the target, label is 0
                        new_distances.append(distance)  # if 0-hop keyword is augmented, distance is 1. In the case of 1-hop, distance would be 2.
                        new_head_ids.append(head_id)
                        new_relations.append(rid)
                        new_tail_ids.append(new_tail_id)
                        new_triple_labels.append(0)  # since we are adding new knowledge, triple label would be just 0

            concepts_nv_dict[qc] = {
                "concepts": new_concepts,
                "labels": new_labels,
                "distances": new_distances,
                "head_ids": new_head_ids,
                "relations": new_relations,
                "tail_ids": new_tail_ids,
                "triple_labels": new_triple_labels,
                "topk_head_ids": topk_head_ids,
            }
            # print("concepts_nv_dict:", concepts_nv_dict[qc])
        else:
            # TODO -- later enable!!
            # diff = set(topk_head_ids) - set(concepts_nv_dict[qc]["topk_head_ids"])
            # if len(diff) > 0:
            #     print("topk_head_ids:", topk_head_ids)
            #     print("concepts_nv_dict[qc].keys():", concepts_nv_dict[qc]["topk_head_ids"])
            #     print("diff:", diff)
            continue

    return concepts_nv_dict


def merge_concept_nv_dict(dir, kgs, concepts_nv, type):
    files = os.listdir(dir)
    all_concept_nv_dict = dict()
    for file in files:
        if file.startswith(type) and file.endswith(".pickle"):
            with open(os.path.join(dir, file), 'rb') as f:
                concept_nv_dict = pickle.load(f)
                all_concept_nv_dict.update(concept_nv_dict)

    print("len(all_concept_nv_dict.keys()):", len(all_concept_nv_dict.keys()))
    _data, _concepts = [], []
    max_num_tokens = 3
    for kg, nv in tqdm.tqdm(zip(kgs, concepts_nv), total=len(kgs)):
        qc = tuple(nv['qc'])

        if qc in all_concept_nv_dict.keys():

            concepts = all_concept_nv_dict[qc]['concepts']
            labels = all_concept_nv_dict[qc]['labels']
            distances = all_concept_nv_dict[qc]['distances']
            head_ids = all_concept_nv_dict[qc]['head_ids']
            tail_ids = all_concept_nv_dict[qc]['tail_ids']
            triple_labels = all_concept_nv_dict[qc]['triple_labels']
            relations = all_concept_nv_dict[qc]['relations']

            assert len(concepts) == len(labels) == len(distances)

            new_concepts = []
            new_labels = []
            new_distances = []
            new_head_ids = []
            new_tail_ids = []
            new_triple_labels = []
            new_relations = []
            invalid_concept_index = []
            for i in range(len(concepts)):
                concept = concepts[i]
                label = labels[i]
                dist = distances[i]
                # head_id = _head_ids[i]
                # tail_id = _tail_ids[i]
                # triple_label = _triple_labels[i]
                # relation = _relations[i]
                if len(concept.split(" ")) > max_num_tokens:
                    print("concept:", concept)
                    invalid_concept_index.append(i)
                else:
                    new_concepts.append(concept)
                    new_labels.append(label)
                    new_distances.append(dist)
                    # new_head_ids.append(head_id)
                    # new_tail_ids.append(tail_id)
                    # new_triple_labels.append(triple_label)
                    # new_relations.append(relation)
            print("invalid_concept_index:", invalid_concept_index)
            for head_id, tail_id, relation, triple_label in zip(head_ids, tail_ids, relations, triple_labels):
                if head_id in invalid_concept_index or tail_id in invalid_concept_index:
                    continue
                new_head_ids.append(head_id)
                new_tail_ids.append(tail_id)
                new_relations.append(relation)
                new_triple_labels.append(triple_label)

            assert len(new_head_ids) == len(new_tail_ids) == len(new_relations) == len(new_triple_labels)

            concepts = kg['concepts'] + new_concepts
            labels = kg['labels'] + new_labels
            distances = kg['distances'] + new_distances
            head_ids = kg['head_ids'] + [int(head_id) for head_id in new_head_ids]
            tail_ids = kg['tail_ids'] + new_tail_ids
            triple_labels = kg['triple_labels'] + new_triple_labels
            relations = kg['relations']
            relations = [rel[0] for rel in relations] + new_relations# + [rel.astype(numpy.int32) for rel in new_relations]

            assert len(concepts) == len(distances) == len(labels)
            # assert len(head_ids) == len(tail_ids) == len(relations) == len(triple_labels)

            _concepts += concepts
            _data.append({
                'concepts': concepts,
                'labels': labels,
                'distances': distances,
                'head_ids': head_ids,
                'tail_ids': tail_ids,
                'relations': relations,
                'triple_labels': triple_labels,
            })

            # print("new concepts:", len(concepts), type(concepts), concepts)
            # print("new labels:", len(labels), type(labels), labels)
            # print("new distances:", len(distances), type(distances), distances)
            # print("new head_ids:", len(head_ids), type(head_ids), head_ids)
            # print("new relations:", len(tail_ids), type(tail_ids), tail_ids)
            # print("new tail_ids:", len(relations), type(relations), relations)
            # print("new triple_labels:", len(triple_labels), type(triple_labels), triple_labels)
            # assert False
        else:
            print("qc:", qc)
            _concepts += concepts
            _data.append(kg)

    return _data, _concepts


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = args.data_dir
    DATA_PATH = config["paths"][dataset + "_dir"]
    # kg_path = DATA_PATH + "/train.kg.json"
    # concepts_nv_path = DATA_PATH + "/train.concepts_nv.json"

    load_resources()
    # kgs = load_kg(kg_path)
    # concepts_nv = load_concepts_nv(concepts_nv_path)

    if args.inference:
        type = args.type
        kg_path = DATA_PATH + f"/{type}.kg.json"
        concepts_nv_path = DATA_PATH + f"/{type}.concepts_nv.json"
        kgs = load_kg(kg_path)
        concepts_nv = load_concepts_nv(concepts_nv_path)
        assert len(kgs) == len(concepts_nv)

        model, sampler, data_loader, text_encoder = load_comet(args.model_file, args.sampling_algorithm, device)
        concepts_nv_dict = aggregate_concepts(args, kgs, concepts_nv, model, sampler, data_loader, text_encoder, device,
                                              DATA_PATH)
        pickle.dump(concepts_nv_dict,
                    open(DATA_PATH + f'/{type}.concepts_nv_dict_{args.start}_{args.end}_rel{args.max_one_hop_concept_num}.pickle',
                         'wb'))
    if args.merge:
        dir = DATA_PATH + "/concepts_nv_dict/"
        total_concepts = []
        for type in ["train", "val", "test"]:
            kg_path = DATA_PATH + f"/{type}.kg.json"
            concepts_nv_path = DATA_PATH + f"/{type}.concepts_nv.json"
            kgs = load_kg(kg_path)
            concepts_nv = load_concepts_nv(concepts_nv_path)
            assert len(kgs) == len(concepts_nv)

            _data, _concepts = merge_concept_nv_dict(dir, kgs, concepts_nv, type)
            print("data:", len(_data), "concepts:", len(_concepts))
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
    parser.add_argument("--type", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--merge", action="store_true", help="merge inference outputs")
    parser.add_argument("--inference", action="store_true", help="start comet inference")

    # comet-inference
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_file", type=str, default="pretrained_models/conceptnet_pretrained_model.pickle")
    parser.add_argument("--sampling_algorithm", type=str, default="greedy", help="greedy, beam-#, top-#")
    parser.add_argument("--num_proc", type=int, default=5, help="number of processors")
    parser.add_argument("--start", type=int, default=0, help="number of processors")
    parser.add_argument("--end", type=int, default=25597, help="number of processors")
    parser.add_argument("--max_one_hop_concept_num", type=int, default=30, help="number of processors")

    args = parser.parse_args()
    set_seed(42)
    main(args)