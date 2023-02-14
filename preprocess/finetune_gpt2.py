import argparse
import configparser
import datetime
import itertools
import json
import logging
import os
import pickle
import random
import time

import torch.cuda
import tqdm

import networkx as nx
from numpy import inf
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, set_seed, AdamW, \
    get_linear_schedule_with_warmup

from dataset import ConceptDataset

config = configparser.ConfigParser()
config.read("paths.cfg")

cpnet = None
cpnet_simple = None
concept2id = None
relation2id = None
id2concept = None
id2relation = None

logger = logging.getLogger(__name__)

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


def load_cpnet():
    global cpnet, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)


def find_neighbors(source_concepts, target_concepts):
    global concept2id
    # source = [concept2id[s_cpt] for s_cpt in source_concepts]
    # target = [concept2id[t_cpt] for t_cpt in target_concepts]
    print("source:", len(source_concepts), source_concepts)
    print("target:", len(target_concepts), target_concepts)

    # pairs = []
    common_concept_dict = {}
    # for pair in tqdm.tqdm(itertools.product(source_concepts, target_concepts),
    #                       total=len(list(itertools.product(source_concepts, target_concepts)))):
    for pair in tqdm.tqdm(zip(source_concepts, target_concepts), total=len(list(zip(source_concepts, target_concepts)))):
        s, t = pair[0], pair[1]
        if s == t:
            continue
        # pairs.append(pair)

        common_concepts = set()
        if not nx.has_path(cpnet_simple, source=s, target=t):
            continue

        all_shortest_paths = list(nx.all_shortest_paths(cpnet_simple, source=s, target=t))
        for path in all_shortest_paths:
            # print("path:", path)
            for node in path[1:-1]:
                # print(id2concept[node], end=", ")
                common_concepts.add(node)
        common_concept_dict[pair] = common_concepts
        # print()
        # break

    # print("pairs:", len(pairs), pairs)
    print("common_concept_dict:", common_concept_dict)

    return common_concept_dict


def preprocess(input_data_path, input_triple_path):
    ############################## train data ##############################
    data = []
    with open(input_data_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    triple = []
    with open(input_triple_path, 'r') as f:
        for line in f.readlines():
            triple.append(json.loads(line))
    # print("triple:", triple[:5])
    # assert False

    assert len(data) == len(triple)

    check = []
    questions = []
    answers = []
    zip_data = zip(data, triple)
    for d, t in tqdm.tqdm(zip_data, total=len(list(zip_data))):
        qc = d['qc']
        ac = d['ac']
        # concepts = list(set(t['concepts']))

        if len(qc) == 0 and len(ac) == 0:
            continue

        if qc+ac not in check:
            check.append(qc+ac)
        else:
            continue

        # print("qc:", qc, "ac:", ac, "concepts:", concepts)
        # question = "What are the related concepts among "+" and ".join(qc+ac) +"?"
        # answer = ",".join([c for c in concepts if c not in qc+ac])

        for pair in itertools.product(qc, ac):
            s_word, t_word = pair[0], pair[1]
            s_id, t_id = concept2id[pair[0]], concept2id[pair[1]]

            if not cpnet_simple.has_node(s_id) or not cpnet_simple.has_node(t_id):
                logging.info("not exist!! -- s_word: {:}, s_id: {:}, {:} and t_word: {:}, t_id: {:}, {:}".format(
                    s_word, s_id, cpnet_simple.has_node(s_id), t_word, t_id, cpnet_simple.has_node(t_id)))
                continue

            if not nx.has_path(cpnet_simple, source=s_id, target=t_id):
                continue

            question = f"What are the related concepts between {s_word} and {t_word}?"
            common_concepts = set()
            all_shortest_paths = list(nx.all_shortest_paths(cpnet_simple, source=s_id, target=t_id))[:10]
            for path in all_shortest_paths:
                for node in path[1:-1]:
                    common_concepts.add(id2concept[node])
            answer = ",".join([c for c in list(common_concepts) if c not in [s_word, t_word]])

            # print("question:", question)
            # print("answer:", answer)
            questions.append(question.replace("_", " "))
            answers.append(answer.replace("_", " "))

    return questions, answers

    ############################## train data ##############################

    ############################## random sampling ##############################
    # percent = 1
    # concept_nodes = list(id2concept.keys() & set(cpnet_simple.nodes()))
    # sample_num_nodes = int(len(concept_nodes) * (percent / 100))
    # random_nodes1 = sample(concept_nodes, 1000)
    # random_nodes2 = sample(concept_nodes, 1000)
    ############################## random sampling ##############################


    # common_concept_dict = find_neighbors(random_nodes1, random_nodes2)

    # return common_concept_dict


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, sample_every):
    total_train_loss = 0

    model.train()
    for step, batch in enumerate(train_dataloader):
        q_input_ids, q_attn_mask, a_input_ids = batch
        model.zero_grad()

        outputs = model(q_input_ids, labels=a_input_ids, attention_mask=q_attn_mask, token_type_ids=None)

        loss = outputs[0]

        batch_loss = loss.item()
        total_train_loss += batch_loss

        # Get sample every x batches.
        if step % sample_every == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. Loss: {:>5,}.'.format(step, len(train_dataloader), batch_loss))

            model.eval()
            input_seq = "<|startoftext|>"
            generated = torch.tensor(tokenizer.encode(input_seq)).unsqueeze(0)

            sample_outputs = model.generate(
                # generated,
                bos_token_id=random.randint(1,30000),
                do_sample=True,
                top_k=30,
                max_length=50,
                top_p=0.95,
                num_return_sequences=1
            )
            for i, sample_output in enumerate(sample_outputs):
                logging.info("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            model.train()

        loss.backward()
        optimizer.step()
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    return avg_train_loss


def eval_epoch(model, valid_dataloader):
    model.eval()
    total_eval_loss = 0

    # Evaluate data for one epoch
    for batch in valid_dataloader:
        q_input_ids, q_attn_mask, a_input_ids = batch

        with torch.no_grad():
            outputs = model(q_input_ids,
                            attention_mask=q_attn_mask,
                            labels=a_input_ids)

            loss = outputs[0]

        batch_loss = loss.item()
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(valid_dataloader)
    return avg_val_loss


def save_model(model, tokenizer, args):
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

    output_dir = './model_save/'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logging.info("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))


def finetune(args, questions, answers):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # setup tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')  # gpt2-medium
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    total_len = len(questions)
    # train_keys = list(common_concept_dict.keys())[:int(total_len * 0.8)]
    # valid_keys = list(common_concept_dict.keys())[int(total_len * 0.8):]
    train_qs = questions[:int(total_len * 0.8)]
    train_as = answers[:int(total_len * 0.8)]
    valid_qs = questions[int(total_len * 0.8):]
    valid_as = answers[int(total_len * 0.8):]

    train_dataset = ConceptDataset(train_qs, train_as, tokenizer, device)
    valid_dataset = ConceptDataset(valid_qs, valid_as, tokenizer, device)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # this produces sample output every 100 steps
    sample_every = 100
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.epsilon)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    total_t0 = time.time()
    training_stats = []
    best_val_loss = inf
    for epoch_i in range(0, args.epochs):
        # ========================================
        #               Training
        # ========================================

        print("")
        logging.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, args.epochs))
        logging.info('Training...')

        t0 = time.time()
        avg_train_loss = train_epoch(model, tokenizer, optimizer, scheduler, train_dataloader, sample_every)
        training_time = format_time(time.time() - t0) # Measure how long this epoch took.

        print("")
        logging.info("  Average training loss: {0:.2f}".format(avg_train_loss))
        logging.info("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================

        print("")
        logging.info("Running Validation...")

        t0 = time.time()
        avg_val_loss = eval_epoch(model, valid_dataloader)
        validation_time = format_time(time.time() - t0)

        logging.info("  Validation Loss: {0:.2f}".format(avg_val_loss))
        logging.info("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

        if best_val_loss > avg_val_loss:
            logging.info("Best model updated! at epoch {:}".format(epoch_i + 1))
            save_model(model, tokenizer, args)

    print("")
    logging.info("Training complete!")
    logging.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))


def predict(model, tokenizer, device):
    model.eval()

    prompt = "<|startoftext|>"

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    generated = generated.to(device)

    print(generated)

    sample_outputs = model.generate(
        generated,
        # bos_token_id=random.randint(1,30000),
        do_sample=True,
        top_k=50,
        max_length=300,
        top_p=0.95,
        num_return_sequences=3
    )

    for i, sample_output in enumerate(sample_outputs):
        logging.info("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="eg", type=str, help="data directory")
    parser.add_argument("--out_dir", default="output", type=str, help="log output directory")
    parser.add_argument("--finetune", action="store_true", help="whether to finetune or not")
    parser.add_argument("--predict", action="store_true", help="whether to predict or not")

    # training
    parser.add_argument("--epochs", default=30, type=int, help="number of epochs")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--warmup_steps", default=1e2, type=float, help="warmup steps")
    parser.add_argument("--epsilon", default=1e-8, type=float, help="epsilon for optimizer")

    args = parser.parse_args()


    dataset = args.data_dir
    DATA_PATH = config["paths"][dataset + "_dir"]
    sampled_concept_file = f'{DATA_PATH}/ckg_gpt2_{dataset}.pkl'

    set_seed(42)
    load_resources()

    logging.basicConfig(
        format=f'%(asctime)s %(message)s',
        datefmt='%H:%M:%S',
        # force=True,
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("log/out.log")
        ]
    )

    # logger.setLevel(log_level)

    if not os.path.exists(sampled_concept_file):
        load_cpnet()
        questions, answers = preprocess(DATA_PATH + "/{}.concepts_nv.json".format('train'), DATA_PATH + "/train.kg.json")
        assert len(questions) == len(answers)

        if questions is not None:
            with open(sampled_concept_file, 'wb') as handle:
                pickle.dump({"question": questions, "answer": answers}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(sampled_concept_file, 'rb') as handle:
            common_concept_dict = pickle.load(handle)
            questions = common_concept_dict["question"]
            answers = common_concept_dict["answer"]
            assert len(questions) == len(answers)

    if args.finetune:
        logging.info("finetune start...")
        model, tokenizer, device = finetune(args, questions, answers)


    if args.predict:
        logging.info("prediction start...")
        predict(model, tokenizer, device)
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=DataCollator(),
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    # )
    #
    # trainer.train()