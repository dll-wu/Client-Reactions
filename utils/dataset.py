import torch
from torch.utils.data import Dataset
from itertools import chain
from transformers import DataCollatorForLanguageModeling
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[unused1]", "[unused2]"]
THERA = "咨询师"
CLIENT = "来访者"

class CUSTdataset(Dataset):
    def __init__(self, data, tokenizer, batch_first=True):
        self.data = data
        # self.specific_speaker = specific_speaker
        # self.label_type = label_type
        #
        # if self.specific_speaker == "咨询师":
        #     assert self.label_type in ["coarse", "fine"]
        # elif self.specific_speaker == "来访者":
        #     assert self.label_type in ["coarse", "fine", "exp"]
        # else:
        #     raise ValueError("specific speaker must be one of counselor or client!")

        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        context = self.data[index]["context"]
        response = self.data[index]["response"]
        speaker = self.data[index]["speaker"]
        labels = self.data[index]["labels"]
        return self.process(context, response, speaker, labels)
    
    def process(self, context, response, speaker, labels):
        # [cls] his1 [sep] [his2] [sep] [client] [cls] sent1 [sep] [cls] sent2 [sep]

        bos, eos = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]"])
        context = [c + [eos] for c in context]
        response = [[bos] + r + [eos] for r in response]

        # post-process response, if reponse length is too long.
        # attention here: response and labels are tied
        if len(list(chain(*response))) > 510:
            if len(response) > 1:
                print("response num > 1")
                ## infer
                context = []
                response = [[bos] + sum([r[1:-1] for r in response], [])[-509:]]
                label = labels[0]["label"]
                labels = [{"start":0, "end": len(list(chain(*response))), "label": label}]
                assert len(labels) == len(response) == 1
            else:
                print("response num = 1")
                assert len(labels) == len(response) == 1
                context = []
                response = [[bos] + response[0][-509:]]
                labels[0]["end"] = 510
                assert len(labels) == len(response) == 1

        while True:
            context_length = len(list(chain(*context)))
            response_length = len(list(chain(*response)))

            if context_length + response_length <= 510:
                break
            if context_length == 0:
                break
            context = context[1:]


        input_ids = [bos] + sum(context, []) + [speaker] + sum(response, [])
        # print(input_ids)
        # print(len(input_ids))

        history_length = len(sum(context, [])) + 2  # [cls] + sum(context, []) + [speaker]
        token_type_ids = [0] * history_length

        assert len(response) == len(labels)
        for res, label in zip(response, labels):
            res_len = len(res)
            start, end = history_length, history_length + res_len
            label["start"] = start
            label["end"] = end
            history_length = end

        instance = {}
        instance["input_ids"] = input_ids
        instance["token_type_ids"] = token_type_ids + [1] * (history_length - len(token_type_ids))  # origin token_type

        # if self.specific_speaker == "咨询师":
        #     if self.label_type == "coarse":
        #         instance["labels"] = torch.LongTensor([label["coarse_strategy"] for label in labels])
        #     elif self.label_type == "fine":
        #         instance["labels"] = torch.LongTensor([label["fine_strategy"] for label in labels])
        #     else:
        #         raise ValueError("label type must be one of coarse and fine!")
        # else:
        #     if self.label_type == "coarse":
        #         instance["labels"] = torch.LongTensor([label["coarse_behav_label"] for label in labels])
        #     elif self.label_type == "fine":
        #         instance["labels"] = torch.LongTensor([label["fine_behav_label"] for label in labels])
        #     elif self.label_type == "exp":
        #         instance["labels"] = torch.LongTensor([label["experience_label"] for label in labels])
        #     else:
        #         raise ValueError("label type must be one of coarse, fine and exp!")
        # print(labels)
        instance["labels"] = torch.LongTensor([label["label"] for label in labels])


        instance["cls_pos"] = torch.LongTensor([label["start"] for label in labels])

        instance["attention_mask"] = [1] * len(input_ids)
        
        mlm_inputs, mlm_labels = self.mlm_collator.mask_tokens(torch.LongTensor([input_ids]))
        instance["mlm_input_ids"] = mlm_inputs[0]
        instance["mlm_labels"] = mlm_labels[0]
        
        # without MLM
        instance["input_ids"] = torch.LongTensor(instance["input_ids"])
        
        return instance
    
    
    def collate(self, batch):
        input_ids = pad_sequence(
            [
                instance["input_ids"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        token_type_ids = pad_sequence(
            [
                torch.tensor(instance["token_type_ids"], dtype=torch.long)
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        attention_mask = pad_sequence(
            [
                torch.tensor(instance["attention_mask"], dtype=torch.long)
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )
        mlm_input_ids = pad_sequence(
            [
                instance["mlm_input_ids"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        mlm_labels = pad_sequence(
            [
                instance["mlm_labels"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=-100,
        )
        cls_labels = pad_sequence(
            [
                instance["labels"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=-100,
        )
        cls_pos = pad_sequence(
            [
                instance["cls_pos"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=-100,
        )
       
        return input_ids, token_type_ids, attention_mask, cls_pos, cls_labels, mlm_input_ids, mlm_labels

