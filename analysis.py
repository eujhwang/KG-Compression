import argparse
import json
import os
import tqdm
import torch.cuda
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet')


def parse_tokens(token_path):
    token_list = []
    with open(token_path) as f:
        lines = f.readlines()

        for line in lines:
            tokens = line.strip().replace("<s>", "").replace("</s>", "").replace("<pad>", "").split(" ")
            tokens = [token for token in tokens if token != ""]
            token_list.append(tokens)
    return token_list


def parse_preds(pred_path):
    pred_list = []
    with open(pred_path) as f:
        lines = f.readlines()

        for line in lines:
            pred_list.append(line.strip())
    return pred_list


def calculate_scores(token_list, sentences, model):
    total_len = len(token_list)
    step = 3
    all_tokens_len = []
    sent_sim_score, ans_sim_score = [], []
    sent_topk_score, sent_bottomk_score = [], []
    ans_topk_score, ans_bottomk_score = [], []
    num_diff = []
    for i in tqdm.tqdm(range(0, total_len, step), total=total_len // step):
        all_tokens = set()
        for j in range(i, i + step):
            # if j == i:
            #     index1 = j
            #     index2 = j + 1
            #     index3 = j + 2
            # elif j == i + 1:
            #     index1 = j
            #     index2 = j - 1
            #     index3 = j + 1
            # elif j == i + 2:
            #     index1 = j
            #     index2 = j - 2
            #     index3 = j - 1
            # diffs = [len(set(token_list[index1]) - (set(token_list[index2] + token_list[index3])))]
            # num_diff.append(diffs[0])

            sent = sentences[int(i / step)]['sent']
            ans = sentences[int(i / step)]['ans'][j - i]
            if len(token_list[j]) == 0:
                sent_topk_score.append(0)
                sent_bottomk_score.append(0)
                sent_sim_score.append(0)
                ans_sim_score.append(0)
            else:
                token_emb = model.encode(" ".join(token_list[j]))
                sent_emb = model.encode(sent)
                sent_cos_sim = util.cos_sim(token_emb, sent_emb).squeeze(-1)  # .tolist()
                # sent_cos_sim = torch.tensor([score / (i + 1) for i, score in enumerate(sent_cos_sim)])
                # sent_avg_cos_sim = torch.mean(sent_cos_sim)
                sent_avg_cos_sim = sent_cos_sim

                sent_topk = torch.max(sent_cos_sim)
                sent_bottomk = torch.min(sent_cos_sim)

                ans_emb = model.encode(ans)
                ans_cos_sim = util.cos_sim(token_emb, ans_emb).squeeze(-1)  # .tolist()
                # ans_cos_sim = torch.tensor([score / (i + 1) for i, score in enumerate(ans_cos_sim)])
                # ans_avg_cos_sim = torch.mean(ans_cos_sim)
                ans_avg_cos_sim = ans_cos_sim

                ans_topk = torch.max(ans_cos_sim)
                ans_bottomk = torch.min(ans_cos_sim)

                sent_topk_score.append(sent_topk.item())
                sent_bottomk_score.append(sent_bottomk.item())
                sent_sim_score.append(sent_avg_cos_sim.item())

                ans_topk_score.append(ans_topk.item())
                ans_bottomk_score.append(ans_bottomk.item())
                ans_sim_score.append(ans_avg_cos_sim.item())
            all_tokens.update(token_list[j])
        all_tokens_len.append(len(all_tokens))
    print("all_tokens_len", len(all_tokens_len), sum(all_tokens_len) / len(all_tokens_len))
    print("sent_avg_sim_score:", sum(sent_sim_score) / len(sent_sim_score))
    print("ans_sim_score:", sum(ans_sim_score) / len(ans_sim_score))

    # print("sent_avg_topk_score:", sum(sent_topk_score) / len(sent_topk_score))
    # print("sent_avg_bottomk_score:", sum(sent_bottomk_score) / len(sent_bottomk_score))
    # print("ans_avg_topk_score:", sum(ans_topk_score) / len(ans_topk_score))
    # print("ans_avg_bottomk_score:", sum(ans_bottomk_score) / len(ans_bottomk_score))
    # print("num_diff:", sum(num_diff) / len(num_diff))


def check_synonyms(word, token_list):
    word_synonyms = set()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemma_names():
            word_synonyms.add(lemma)

    similar_tokens = set()
    for token in token_list:
        if token == word:
            continue
        if token in word_synonyms:
            similar_tokens.add(token)
    print("word:", word, "similar_tokens:", similar_tokens)
    return similar_tokens


def count_synonyms(token_list):
    synonym_num_list = []
    similar_token_list = []
    for i, tokens in enumerate(token_list):
        num_synonyms = 0
        if len(tokens) == 0:
            continue
        for token in tokens:
            similar_tokens = check_synonyms(token, tokens)
            num_synonyms += len(similar_tokens)
        synonym_num_list.append(num_synonyms)
    print("synonym_num_list", len(synonym_num_list), synonym_num_list)
    return synonym_num_list


def count_all_distinct_token(token_list):
    all_tokens = set()
    for tokens in token_list:
        all_tokens.update(tokens)

    total_len = len(token_list)
    step = 3
    token_num_list = []
    synonym_num_list = []
    for i in range(0, total_len, step):
        tokens_per_sent = set()
        for j in range(i, i + step):
            tokens_per_sent.update(token_list[j])

        num_synonyms = 0
        for token in list(tokens_per_sent):
            similar_tokens = check_synonyms(token, tokens_per_sent)
            num_synonyms += len(similar_tokens)
        synonym_num_list.append(num_synonyms)

        token_num_list.append(len(tokens_per_sent))
    return len(all_tokens), sum(token_num_list) / len(token_num_list), sum(synonym_num_list) / len(synonym_num_list)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    concepts_nv_path = "data/eg/test.concepts_nv.json"

    data = []
    with open(concepts_nv_path, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))

    sentences = []
    target_pred_list = []
    for item in data:
        sent = item['sent']
        ans = item['ans']
        ans = ans.split('\t')
        # print("sent:", sent, "ans:", ans)
        sentences.append({
            "sent": sent,
            "ans": ans
        })
        target_pred_list.extend(ans)

    # sag_dir = "output-eg/KGMixtureOfExpertCho_Output_akzotqmi_83ju8zz9"
    # aug_dir = "output-eg/KGMixtureOfExpertCho_Output_7ts6r62l_l5vly7x7"
    # sag_aug_dir = "output-eg/KGMixtureOfExpertCho_Output_2uinyfdo_im13vzt7"
    # ori_dir = "../MoKGE-origin/output-eg/KGMixtureOfExpertCho_Output"

    sag_dir = "output-eg-infer/sag/KGMixtureOfExpertCho_Output_akzotqmi_83ju8zz9"
    aug_dir = "output-eg-infer/aug/KGMixtureOfExpertCho_Output_7ts6r62l_l5vly7x7"
    sag_aug_dir = "output-eg-infer/sag-aug/KGMixtureOfExpertCho_Output_2uinyfdo_im13vzt7"
    ori_dir = "output-eg-infer/origin/KGMixtureOfExpertCho_Output"

    token_file = "output_test_selected_tokens.txt"
    pred_file = "output_test_pred.txt"

    sag_token_path = os.path.join(sag_dir, token_file)
    aug_token_path = os.path.join(aug_dir, token_file)
    sag_aug_token_path = os.path.join(sag_aug_dir, token_file)
    ori_token_path = os.path.join(ori_dir, token_file)

    sag_pred_path = os.path.join(sag_dir, pred_file)
    aug_pred_path = os.path.join(aug_dir, pred_file)
    sag_aug_pred_path = os.path.join(sag_aug_dir, pred_file)
    ori_pred_path = os.path.join(ori_dir, pred_file)

    sag_token_list = parse_tokens(sag_token_path)
    aug_token_list = parse_tokens(aug_token_path)
    sag_aug_token_list = parse_tokens(sag_aug_token_path)
    ori_token_list = parse_tokens(ori_token_path)

    sag_pred_list = parse_preds(sag_pred_path)
    aug_pred_list = parse_preds(aug_pred_path)
    sag_aug_pred_list = parse_preds(sag_aug_pred_path)
    ori_pred_list = parse_preds(ori_pred_path)

    # assert len(sag_token_list) == len(aug_token_list) == len(sag_aug_token_list) == len(ori_token_list)
    assert len(target_pred_list) == len(sag_pred_list) == len(aug_pred_list) == len(sag_aug_pred_list) == len(
        ori_pred_list)

    # model = SentenceTransformer('all-distilroberta-v1')
    # model = model.to(device)
    # bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    bert_scorer = BERTScorer(model_type="facebook/bart-large")

    # print("################################## sag_token_list ##################################")
    # # calculate_scores(sag_token_list[:30], sentences, model)
    #
    # sag_synonym_num_list = count_synonyms(sag_token_list)
    # aug_synonym_num_list = count_synonyms(aug_token_list)
    # sag_aug_synonym_num_list = count_synonyms(sag_aug_token_list)
    ori_synonym_num_list = count_synonyms(ori_token_list[:150])

    # print("avg-sag-synonym-num:", sum(sag_synonym_num_list)/len(sag_synonym_num_list))
    # print("avg-aug-synonym-num:", sum(aug_synonym_num_list)/len(aug_synonym_num_list))
    # print("avg-sag_aug-synonym-num:", sum(sag_aug_synonym_num_list)/len(sag_aug_synonym_num_list))
    print("avg-ori-synonym-num:", sum(ori_synonym_num_list)/len(ori_synonym_num_list))
    #
    # sag_all_tokens, sag_tokens_per_sent, sag_synonym_num = count_all_distinct_token(sag_token_list)
    # aug_all_tokens, aug_tokens_per_sent, aug_synonym_num = count_all_distinct_token(aug_token_list)
    # sag_aug_all_tokens, sag_aug_tokens_per_sent, sag_aug_synonym_num = count_all_distinct_token(sag_aug_token_list)
    # ori_all_tokens, ori_tokens_per_sent, ori_synonym_num = count_all_distinct_token(ori_token_list)
    #
    # print("sag_all_tokens:", sag_all_tokens, "sag_tokens_per_sent", sag_tokens_per_sent, "sag_synonym_num", sag_synonym_num)
    # print("aug_all_tokens:", aug_all_tokens, "aug_tokens_per_sent", aug_tokens_per_sent, "aug_synonym_num", aug_synonym_num)
    # print("sag_aug_all_tokens:", sag_aug_all_tokens, "sag_aug_tokens_per_sent", sag_aug_tokens_per_sent, "sag_aug_synonym_num", sag_aug_synonym_num)
    # print("ori_all_tokens:", ori_all_tokens, "ori_tokens_per_sent", ori_tokens_per_sent, "ori_synonym_num", ori_synonym_num)

    # print("################################## aug_token_list ##################################")
    # calculate_scores(aug_token_list[:30], sentences, model)
    #
    # print("################################## sag_aug_token_list ##################################")
    # calculate_scores(sag_aug_token_list[:30], sentences, model)
    #
    # print("################################## ori_token_list ##################################")
    # calculate_scores(ori_token_list[:30], sentences, model)

    # P, R, F1 = bert_scorer.score(ori_pred_list, target_pred_list)
    # print("P:", P.shape, P, "R:", R.shape, R, "F1:", F1.shape, F1)
    # print("P:", torch.mean(P).item(), "R:", torch.mean(R).item(), "F1:", torch.mean(F1).item())
    #
    # P, R, F1 = bert_scorer.score(sag_pred_list, target_pred_list)
    # print("P:", P.shape, P, "R:", R.shape, R, "F1:", F1.shape, F1)
    # print("P:", torch.mean(P).item(), "R:", torch.mean(R).item(), "F1:", torch.mean(F1).item())
    #
    # P, R, F1 = bert_scorer.score(aug_pred_list, target_pred_list)
    # print("P:", P.shape, P, "R:", R.shape, R, "F1:", F1.shape, F1)
    # print("P:", torch.mean(P).item(), "R:", torch.mean(R).item(), "F1:", torch.mean(F1).item())
    #
    # P, R, F1 = bert_scorer.score(sag_aug_pred_list, target_pred_list)
    # print("P:", P.shape, P, "R:", R.shape, R, "F1:", F1.shape, F1)
    # print("P:", torch.mean(P).item(), "R:", torch.mean(R).item(), "F1:", torch.mean(F1).item())

    # P: tensor(0.3483) R: tensor(0.4114) F1: tensor(0.3800)
    # bert_score_p = torch.nn.functional.relu(P)
    # bert_score_r = torch.nn.functional.relu(R)
    # bert_score_f1 = torch.nn.functional.relu(F1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_dir", type=str, default='data/opinion-qa/', help="start collecting memes from reddit")
    args = parser.parse_args()

    main(args)