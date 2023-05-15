import torch
import torch.nn as nn
from transformers import (
        BertTokenizer,
        BertConfig,
        BertForPreTraining,
        AdamW,
        WEIGHTS_NAME,
        CONFIG_NAME
)



class BertClassifierBase(nn.Module):
    def __init__(self, config, args, num_class=None):
        super(BertClassifierBase, self).__init__()
        
        self.config = config
        self.bert = BertForPreTraining(self.config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        if args.__dict__.get("num_class") is not None:
            self.num_class = args.num_class
        else:
            self.num_class = num_class

        self.fc = nn.Linear(config.hidden_size, self.num_class)

        self.softmax = nn.Softmax(dim=-1)
        self.lossfn = nn.CrossEntropyLoss()

        torch.nn.init.xavier_uniform_(self.fc.weight)

        self.args = args
        self.mlm = True if self.args.mlm_prob > 0 else False

    def forward(self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            mlm_labels=None,
            cls_pos=None,
            cls_labels=None,
    ):
        batch_size, seq_len = input_ids.size()

        outputs = self.bert.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        sequence_output, pooled_output = outputs[:2]

        assert sequence_output.size(0) == cls_pos.size(0)

        MLMloss = None
        if self.mlm:
            # assert mlm_labels is not None
            mlm_prediction_scores, _ = self.bert.cls(sequence_output, pooled_output)
            MLMloss = self.lossfn(mlm_prediction_scores.view(-1, self.config.vocab_size), mlm_labels.view(-1))

        # 取每个sentence从cls到sep的平均向量表示
        # sequence_output [batch_size, seq_len, hidden_state]
        # cls_pos [batch_size, num_cls]

        flatten_cls_labels = []

        mean_reps = []
        for i in range(batch_size):
            current_sequence_output = sequence_output[i] #[seq_len, hidden_state]

            current_attention_mask = attention_mask[i]
            valid_length = (current_attention_mask == 1).sum().item()
            # print("valid_length", valid_length)

            current_cls_pos = cls_pos[i]  # [num_cls]
            valid_cls_pos = torch.LongTensor([int(pos) for pos in current_cls_pos if pos != -100])
            valid_bos_pos = torch.LongTensor(valid_cls_pos).repeat(2, 1).to(self.args.device)
            valid_bos_pos[1] = torch.roll(valid_bos_pos[1], -1)
            valid_bos_pos[-1, -1] = valid_length
            valid_bos_pos = torch.transpose(valid_bos_pos, 0, 1)
            # print(valid_bos_pos.size())

            valid_cls_labels = cls_labels[i][:len(valid_cls_pos)]

            flatten_cls_labels.append(valid_cls_labels)

            idx = torch.arange(seq_len).unsqueeze(0).expand(len(valid_cls_pos), -1).to(self.args.device)
            idx = (idx >= valid_bos_pos[:, 0].unsqueeze(1)) & (idx <= valid_bos_pos[:, 1].unsqueeze(1))
            mean_rep_per_sent = torch.matmul(idx.float(), current_sequence_output) / idx.sum(1).unsqueeze(1)
            mean_reps.append(mean_rep_per_sent)

        mean_reps = torch.cat(mean_reps, dim=0)


        flatten_cls_labels = torch.cat(flatten_cls_labels)

        logits = self.fc(mean_reps)

        CEloss = self.lossfn(logits, flatten_cls_labels)
        CEloss = CEloss.requires_grad_()

        return logits, flatten_cls_labels, CEloss, MLMloss

    def from_pretrained(self, args, model_checkpoint=None):
        if model_checkpoint:
            saved_model = torch.load(model_checkpoint)
            model_dict = self.state_dict()
            model_dict.update({k:v for k,v in saved_model.items() if k in model_dict.keys()})
            self.load_state_dict(model_dict)
        else:
            self.bert = BertForPreTraining.from_pretrained(args.model_name_or_path, config=self.config)
