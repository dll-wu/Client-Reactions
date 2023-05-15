from argparse import ArgumentParser
import torch
import json
import csv
import os
from tqdm import tqdm
import random
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix

from utils.dataloader import convert_data_to_input
from utils.dataset import CUSTdataset
from torch.utils.data import DataLoader
from train_dist import read_label_jsonfile
from bert_classifier import *
from transformers import (
        BertTokenizer,
        BertConfig)

import numpy as np

from pprint import pformat
import logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def prepare_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="all model are under this dir"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=None,
        help="Path or URL of the model",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/"
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="If False train from scratch"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use DataParallel or not",
    )
    # parser.add_argument(
    #     "--result_filename",
    #     required=True
    # )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=16
    )
    parser.add_argument(
        "--use_concate_session",
        type=bool,
        default=True
    )
    parser.add_argument(
        "--max_history_num",
        type=int,
        default=10,
        help="maximum number of context utterances"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mlm_prob",
        type=float,
        default=0
    )
    parser.add_argument(
        "--evaluate_invalidation_prob",
        type=float,
        default=0.2
    )
    parser.add_argument(
        "--specific_speaker",
        type=str,
        default="client",
        help="classification on which speaker's utterances."
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="fine_behav_label",
        help="the label type"
    )

    args = parser.parse_args()

    return args


# def load_ranker(model_path, args):
#     ranker = Ranker(model_path, args)
#     return ranker


def load_data(args, tokenizer):
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    test_data = data['test']


    all_samples = []
    for dialog in test_data:
        samples = convert_data_to_input(dialog, tokenizer, args)
        all_samples += samples

    test_dataset = CUSTdataset(all_samples, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        collate_fn=test_dataset.collate,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.test_batch_size,
        shuffle=False,
    )

    return len(all_samples), test_loader


def evaluate(args, model, data_loader, invalidation_prob=1):
    true = []
    predict = []
    for batch in tqdm(data_loader):
        input_ids, token_type_ids, attention_mask, cls_pos, cls_labels, _, _ = tuple(
            input_tensor.to(args.device) for input_tensor in batch
        )
        output, flatten_cls_labels, _, _ = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            cls_pos=cls_pos,
            cls_labels=cls_labels,
        )
        mean_logits = torch.nn.functional.softmax(output, dim=-1)
        # print(mean_logits)
        pred_labels = []
        if invalidation_prob < 1:
            for i in range(len(mean_logits)):
                logit = mean_logits[i]
                if logit[1] >= invalidation_prob:
                    pred_labels.append(1)
                else:
                    pred_labels.append(torch.argmax(logit).item())
        else:
            pred_labels = torch.argmax(mean_logits, dim=-1).cpu().numpy().tolist()


        # print("pred_labels", pred_labels)
        predict += pred_labels
        flatten_cls_labels = flatten_cls_labels.cpu().numpy().tolist()
        true += flatten_cls_labels
    return true, predict


def calculate(true_labels, pred_labels, label2id):
    accuracy = accuracy_score(y_true=true_labels, y_pred=pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    precision = precision_score(true_labels, pred_labels, average='macro')
    #     print('fine-grained macro avg precision:', precision)
    recall =  recall_score(true_labels, pred_labels, average='macro')
    #     print('fine-grained macro avg recall:', recall)
    # report = classification_report(y_true=true_labels, y_pred=pred_labels, target_names=list(label2id.keys()))
    confusion = confusion_matrix(y_true=true_labels, y_pred=pred_labels, labels=list(label2id.values()))
    # print("report", report)
    print("confusion matrix: ")
    print(confusion)
    print('output format:\n{0:.1f}\t{1:.1f}\t{2:.1f}\t{3:.1f}'.format(accuracy * 100,
                                                                          precision * 100,
                                                                          recall * 100,
                                                                          macro_f1 * 100))
    return accuracy, precision, recall, macro_f1


def calculate_mean_and_std(results):
    results = np.array(results)
    avg_score = np.mean(results, axis=0)
    std_score = np.std(results, axis=0)
    return avg_score.tolist(), std_score.tolist()


if __name__ == "__main__":
    args = prepare_args()

    if args.specific_speaker == "client":
        args.specific_speaker = "来访者"
    elif args.specific_speaker == "counselor":
        args.specific_speaker = "咨询师"
    else:
        raise ValueError

    labels = read_label_jsonfile(speaker=args.specific_speaker, label_type=args.label_type)

    num_class = len(labels)
    args.__dict__.update({"num_class": num_class})

    label2id = dict(zip(labels, range(len(labels))))
    args.__dict__.update({"label2id": label2id})

    logging.basicConfig(
        level=logging.INFO
    )
    logger.info("Arguments: %s", pformat(args))

    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large', do_lower_case=True)
    config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    model = BertClassifierBase(config, args)

    # load test data
    num_samples, test_loader = load_data(args, tokenizer)
    # print("test data num: ", num_samples)


    # acc_list, precision_list, recall_list, f1_list = [[]] * 4
    results = []
    invalidation_results = []
    model_paths = os.listdir(args.model_dir)
    for model_path in model_paths:
        print("model_path", model_path)
        model_files = os.listdir(os.path.join(args.model_dir, model_path)) 
        if 'pytorch_model.bin' in model_files:
            model_file = 'pytorch_model.bin'
        else:
            model_file = [f for f in model_files if 'checkpoint' in f]
            model_file = sorted(model_file)[-1]

        valid_f1 = float(model_file.rsplit('.', maxsplit=2)[0].split('_')[-1])

        model_path = os.path.join(os.path.join(args.model_dir, model_path), model_file)
        print('Choose model:', model_path)

        model.from_pretrained(args=args, model_checkpoint=model_path)
        model.to(args.device)
        model.eval()

        assert args.model_dir is not None or args.model_checkpoint is not None

        true, predict = evaluate(args, model, data_loader=test_loader, invalidation_prob=args.evaluate_invalidation_prob)
        accuracy, precision, recall, macro_f1 = calculate(true_labels=true, pred_labels=predict, label2id=label2id)
        confusion = confusion_matrix(true, predict)
        results.append([valid_f1, accuracy, precision, recall, macro_f1])
        classification_report(true, predict, output_dict=True)
        print(classification_report)

        print("confusion", confusion)

    print(invalidation_results)

    avg_score, std_score = calculate_mean_and_std(results)

    import csv

    with open(f"test_results/{'_'.join(args.model_dir.split('/'))}_inval_prob{args.evaluate_invalidation_prob}.csv", "w", encoding="utf8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["valid", "test"])
        csv_writer.writerow(["macro-f1", "accuracy", "precision", "recall", "macro-f1"])
        for result in results:
            csv_writer.writerow(result)
        csv_writer.writerow(avg_score)
        csv_writer.writerow(std_score)












    # csv_filename = args.result_filename
    # with open(csv_filename, 'w', encoding='utf8') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(["valid", "test"])
    #     csv_writer.writerow(["macro-f1", "accuracy", "macro-f1", "precision", "recall"])
    #
    #     if args.model_dir is not None:
    #         model_files = os.listdir(args.model_dir)
    #         for model in model_files:
    #             print("---------test on model: {} -------".format(model))
    #             ranker = load_ranker(os.path.join(args.model_dir, model), args)
    #
    #             valid_f1 = ranker.get_model_performance_on_valid()
    #             accuracy, macro_f1, precision, recall = predict(test_data, ranker)
    #             csv_writer.writerow([valid_f1, accuracy, macro_f1, precision, recall])
    #
    #     elif args.model_checkpoint is not None:
    #         ranker = load_ranker(args.model_checkpoint, args)
    #         predict(test_data, ranker)
    #
    #         valid_f1 = ranker.get_model_performance_on_valid()
    #         accuracy, macro_f1, precision, recall = predict(test_data, ranker)
    #         csv_writer.writerow([valid_f1, accuracy, macro_f1, precision, recall])



    # ranker = Ranker("runs/bert-base-MLM/bert-base-mlm-seed11", args)
    """
    export CUDA_VISIBLE_DEVICES=7 
    export model_name=roberta
    export model_size=large
    export mlm=noMLM
    
    
    nohup python test.py \
    --model_dir runs/$model_name/$model_size-$mlm/ \
    --data_path data/data_v8_split_seed1234.json \
    --result_filename results/$model_name-$model_size-$mlm-result.csv >data_v8_logs/$model_name-$model_size-$mlm-test.log 2>&1 &
    """



