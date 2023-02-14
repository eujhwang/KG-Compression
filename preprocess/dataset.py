import torch
from torch.utils.data import Dataset


class ConceptDataset(Dataset):
    def __init__(self, questions, answers, tokenizer, device):
        self.tokenizer = tokenizer
        # self.common_concept_dict = common_concept_dict
        self.questions = questions
        self.answers = answers

        # self.id2concept = id2concept
        # self.keys = keys
        self.end_of_text_token = "<|endoftext|>"
        self.max_length = 32

        self.device = device

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # head, tail = self.keys[idx]
        # concepts = self.common_concept_dict[self.keys[idx]]
        #
        # question = f"What are the common concepts between {self.id2concept[head]} and {self.id2concept[tail]}?"
        # answer = ", ".join([self.id2concept[c] for c in concepts])
        # question = question.replace("_", " ")
        # answer = answer.replace("_", " ")
        # print("idx:", idx)
        # print("q:", len(self.questions), type(self.questions))
        # print("a:", self.answers[idx])

        question = self.questions[idx]
        answer = self.answers[idx]

        q_encode = self.tokenizer('<|startoftext|>' + question + '<|endoftext|>', truncation=True,
                                  max_length=self.max_length, padding="max_length")
        a_encode = self.tokenizer('<|startoftext|>' + answer + '<|endoftext|>', truncation=True,
                                  max_length=self.max_length, padding="max_length")

        q_input_ids = torch.tensor(q_encode['input_ids'], device=self.device)
        q_attn_mask = torch.tensor(q_encode['attention_mask'], device=self.device)

        a_input_ids = torch.tensor(a_encode['input_ids'], device=self.device)
        # a_attn_mask = a_encode['attention_mask']
        return q_input_ids, q_attn_mask, a_input_ids


class DataCollator:
    # def __init__(self, tokenizer):
    #     self.tokenizer = tokenizer

    def __call__(self, batch):
        q_input_ids, q_attn_mask, a_input_ids, a_attn_mask = batch
        return q_input_ids, q_attn_mask, a_input_ids, a_attn_mask

