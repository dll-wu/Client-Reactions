import os
import ujson

import torch
from transformers import cached_path

from utils.dataset import CUSTdataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

speaker_information = {"咨询师": "[unused1]", "来访者": "[unused2]"}


def convert_data_to_input(data, tokenizer, args):
    # split dialog to multiple short context-client_response pairs and convert text to token_id.
    # add speaker special token before each utterance. counselor--[unused1], client--[unused2].
    # each utterance contains several sentences. we convert it to [cls] sent1 [sep] [cls] sent1 [sep]

    if args.use_concate_session:
        session_type = "concate_session"
    else:
        session_type = "raw_session"


    process = lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x))
    # print(data)
    dialog = data[session_type]
    inputs = []
    context = []

    for utter in dialog:
        speaker = utter["speaker"]
        text = utter["text"]
        sents = []
        labels = []



        speaker_id = tokenizer.convert_tokens_to_ids(speaker_information[speaker])

        if speaker == args.specific_speaker:
            annotations = utter["annotation"]
            # print("text", text)
            # print("tokenized text", tokenizer.tokenize(text))

            for annotation in annotations:
                start, end = annotation["start"], annotation["end"]
                sent = process(text[start: end])
                # print("sent", text[start: end])
                # print("tokenized sent", tokenizer.tokenize(text[start: end]))
                sents.append(sent)
                label = args.label2id[annotation[args.label_type]]
                labels.append({"label": label})
            assert len(labels) == len(sents)

            res = {
                'context': context.copy(),
                'response': sents,
                'speaker': speaker_id,
                'labels': labels
            }
            inputs.append(res)
            # assert sum(sents, []) == process(text)

        text = process(text)
        text = [speaker_id] + text
        context = context + [text]

        if len(context) > args.max_history_num:
            context = context[-args.max_history_num:]

    return inputs


def get_context_response_pairs(tokenizer, args, logger):
    cache_dir = "cache/dataset_cache_" + type(tokenizer).__name__
    if args.dataset_cache:
        logger.info("Load tokenized dataset from cache at %s", cache_dir)
        dataset = torch.load(cache_dir)
    else:
        logger.info("Download dataset from %s", args.data_path)
        cache_file = cached_path(args.data_path)
        with open(cache_file, "r", encoding="utf-8") as f:
            corpus = ujson.load(f)

        # coarse_label2id = convert_label_to_id(speaker, label_type="coarse")
        # fine_label2id = convert_label_to_id(speaker, label_type="fine")
        # exp_label2id = convert_label_to_id(speaker, label_type="exp")

        dataset = {}
        for key in ['train', 'valid', 'test']:
            dialogs = corpus[key]
            all_samples = []
            for dialog in dialogs:
                samples = convert_data_to_input(data=dialog, tokenizer=tokenizer, args=args)
                all_samples += samples
            if key == "valid":
                args.valid_batch_size = len(all_samples)
            dataset[key] = all_samples

        # torch.save(dataset, cache_dir)

    return dataset


# def get_context_counselor_response_pairs(tokenizer, dataset_path, dataset_cache, max_history_num, use_concate_session, logger):
#     speaker = "counselor"
#     cache_dir = "cache/dataset_cache_" + type(tokenizer).__name__
#     if dataset_cache:
#         logger.info("Load tokenized dataset from cache at %s", cache_dir)
#         dataset = torch.load(cache_dir)
#     else:
#         logger.info("Download dataset from %s", dataset_path)
#         cache_file = cached_path(dataset_path)
#         with open(cache_file, "r", encoding="utf-8") as f:
#             corpus = ujson.load(f)
#
#         coarse_label2id = convert_label_to_id(speaker, label_type="coarse")
#         fine_label2id = convert_label_to_id(speaker, label_type="fine")
#
#         dataset = {}
#         for key in ['train', 'valid', 'test']:
#             dialogs = corpus[key]
#             all_samples = []
#             for dialog in dialogs:
#                 samples = convert_data_to_input(dialog, tokenizer, max_history_num, specific_speaker=speaker, use_concate_session=use_concate_session)
#                 for sample in samples:
#                     labels = sample["labels"]
#                     for label in labels:
#                         label['coarse_strategy'] = coarse_label2id[label['coarse_strategy']]
#                         label["fine_strategy"] = fine_label2id[label["fine_strategy"]]
#                 all_samples += samples
#
#             dataset[key] = all_samples
#
#         # torch.save(dataset, cache_dir)
#
#     return dataset



def build_dataloaders(args, tokenizer, logger): 
    logger.info("Build train and validation dataloaders")

    datasets = get_context_response_pairs(tokenizer=tokenizer, args=args, logger=logger)
    
    train_dataset, valid_dataset = (
        CUSTdataset(datasets["train"], tokenizer),
        CUSTdataset(datasets["valid"], tokenizer),
    )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.train_batch_size,
        shuffle=(not args.distributed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.valid_batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, train_sampler, valid_sampler

if __name__ == "__main__":
    import torch
    from transformers import BertTokenizer
    import logging
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    data_path = "../data/processed_for_train_after_TestStage4.json"
    datasets = get_context_response_pairs(tokenizer, args,
                                                 logger)

    print("write dataset to json")
    with open("dataset.json", "w", encoding="utf8") as f:
        ujson.dump(datasets, f, ensure_ascii=False)

