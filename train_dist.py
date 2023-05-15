import os
import ujson
import math
import torch
import numpy as np
import random
from argparse import ArgumentParser
from pprint import pformat
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim.lr_scheduler import LambdaLR
from bert_classifier import *
from utils.dataloader import build_dataloaders

from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.metrics import MetricsLambda, RunningAverage, Loss, Accuracy, Average
from ignite.handlers import (
        Timer,
        ModelCheckpoint,
        EarlyStopping,
        global_step_from_engine,
        )
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    OutputHandler,
    OptimizerParamsHandler,
    )


import logging
from sklearn.metrics import f1_score, confusion_matrix
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def average_distributed_scalar(scalar, args):
    if args.local_rank == -1:
        return scalar
    scalar_t = (
        torch.tensor(scalar, dtype=torch.float, device=args.device)
        / torch.distributed.get_world_size()
    )
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def read_label_jsonfile(speaker, label_type):
    if speaker == "咨询师":
        filename = "counselor_labels"
        assert label_type in ["coarse_strategy", "fine_strategy"]
        with open(os.path.join("./data", filename + ".json"), "r", encoding="utf8") as f:
            data = ujson.load(f)
            if label_type == "coarse_strategy":
                labels = list(data.keys())
            else:
                labels = sum(list(data.values()), [])
    elif speaker == "来访者":
        filename = "client_labels"
        assert label_type in ["coarse_behav_label", "fine_behav_label", "experience_label"]
        with open(os.path.join("./data", filename + ".json"), "r", encoding="utf8") as f:
            data = ujson.load(f)
            if label_type == "coarse_behav_label":
                labels = list(data.keys())
            elif label_type == "experience_label":
                labels = sum([list(value.keys()) for _, value in data.items()], [])
            else:
                labels = sum(sum([list(value.values()) for _, value in data.items()], []), [])
    else:
        raise ValueError("speaker must be one of counselor and client!")
    return labels


def train():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=int, default=768, help="Hs of the model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="hfl/chinese-roberta-wwm-ext-large",
        choices=['bert-base-chinese', 'hfl/chinese-roberta-wwm-ext', 'hfl/chinese-roberta-wwm-ext-large'],
        help="Name or Path of Model in HuggingFace Models Hub"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="models/",
        help="Path or URL of the model",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="If False train from scratch"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/data_v8_split_seed1234.json",
        help="Path or url of the dataset. "
    )
    parser.add_argument(
        "--dataset_cache",
        action="store_true",
        help="use dataset cache or not",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123
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
        help="Number of subprocesses for data loading",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=50, help="Patience for early stopping"
    )
    parser.add_argument("--n_saved", type=int, default=1, help="Save the best n models")
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=1000, help="Batch size for validation"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="noam",
        choices=["noam", "linear"],
        help="method of optim",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=3500, help="Warmup steps for noam"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1: not distributed)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Use DataParallel or not",
    )
    parser.add_argument(
        "--mlm_prob",
        type=float,
        default=0,
        help="masked probability"
    )
    parser.add_argument(
        "--label_type",
        type=str,
        default="fine_strategy",
        help="the label type"
    )
    parser.add_argument(
        "--use_concate_session",
        type=bool,
        default=True,
        help="use concate session or raw session. concate session: concate messages of each speaker and concate multiple sentences with same label"
    )
    parser.add_argument(
        "--specific_speaker",
        type=str,
        default="client",
        help="classification on which speaker's utterances."
    )
    args = parser.parse_args()
    print(args.specific_speaker)

    if args.specific_speaker == "client":
        args.specific_speaker = "来访者"
    elif args.specific_speaker == "counselor":
        args.specific_speaker = "咨询师"
    else:
        raise ValueError


    labels = read_label_jsonfile(speaker=args.specific_speaker, label_type=args.label_type)
    # print(labels)
    # exit(0)
    num_class = len(labels)
    args.__dict__.update({"num_class": num_class})

    label2id = dict(zip(labels, range(len(labels))))
    args.__dict__.update({"label2id": label2id})

    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = args.local_rank != -1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    logger.info("Prepare tokenizer, models and optimizer.")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=True)
    configuration = BertConfig.from_pretrained(args.model_name_or_path)
    # print('bert config', configuration)
    model = BertClassifierBase(configuration, args)
    if args.pretrained:
        model.from_pretrained(args, model_checkpoint=args.model_checkpoint)
        print('load checkpoint')
    else:
        model.from_pretrained(args)
    ##torch.save(model.state_dict(), 'models/pytorch_model.bin')
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(args.device)


    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = build_dataloaders(args, tokenizer, logger)

    # print("valid batch size", args.valid_batch_size)
    
    if args.scheduler == "noam":
        # make the peak lr to be the value we want
        args.warmup_steps = min(
            len(train_loader) // args.gradient_accumulation_steps + 1, args.warmup_steps
        )
        args.lr /= args.model_size ** (-0.5) * args.warmup_steps ** (-0.5)
    optimizer = AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

    # fp16
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    
    # better not use parallel. 
    # if you would like to use parallel, you shouldn't calculate loss in the model, just calculate it in the update and evaluate function
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # trainer
    def update(engine, batch):
        model.train()
        input_ids, token_type_ids, attention_mask, cls_pos, cls_labels, mlm_input_ids, mlm_labels = tuple(
            input_tensor.to(args.device) for input_tensor in batch
        )
        if args.mlm_prob > 0:
            output, flatten_cls_labels, CEloss, MLMloss = model(
                input_ids=mlm_input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
                cls_pos=cls_pos,
                cls_labels=cls_labels,
            )
        else:
            output, flatten_cls_labels, CEloss, MLMloss = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                cls_pos=cls_pos,
                cls_labels=cls_labels,
            )


        if MLMloss:
            loss = (MLMloss + CEloss)/ args.gradient_accumulation_steps
        else:
            loss = (CEloss) / args.gradient_accumulation_steps
        # print("loss", loss)
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if MLMloss:
            return loss.item()*args.gradient_accumulation_steps, MLMloss.item(), CEloss.item()
        else:
            return loss.item()*args.gradient_accumulation_steps, 0, CEloss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def evaluate(engine, batch):
        model.eval()
        with torch.no_grad():
            input_ids, token_type_ids, attention_mask, cls_pos, cls_labels, mlm_input_ids, mlm_labels = tuple(
                input_tensor.to(args.device) for input_tensor in batch
            )
            # 一条一条输入
            pred_labels = []
            flatten_cls_labels = []
            CEloss_ = 0
            for i in range(len(input_ids)):
                l = len(attention_mask[i][attention_mask[i] == 1])
                output, flatten_cls_label, CEloss, MLMloss = model(
                                    input_ids=input_ids[i:i+1,:l],
                                    token_type_ids=token_type_ids[i:i+1,:l],
                                    attention_mask=attention_mask[i:i+1,:l],
                                    cls_pos=cls_pos[i:i+1],
                                    cls_labels=cls_labels[i:i+1],
                                    mlm_labels=mlm_labels[i:i+1, :l]
                                )
                mean_logits = torch.nn.functional.softmax(output, dim=-1)
                pred_labels += torch.argmax(mean_logits, dim=-1).tolist()
                flatten_cls_labels += flatten_cls_label.tolist()
                CEloss_ += CEloss
                # if i == 372:
                #     print(output)
                #     print(mean_logits)
                #     print(input_ids[i:i+1,:l])
                #     print(token_type_ids[i:i+1,:l])
                #     print(attention_mask[i:i+1,:l])
            # print("pred_labels", pred_labels)
            f1 = f1_score(flatten_cls_labels, pred_labels, average='macro')# sum(flags).item()/cls_label.shape[0]
            confusion = confusion_matrix(flatten_cls_labels, pred_labels)
            print("eval")
            print("f1", f1)
            print("confusion", confusion)


        return f1, CEloss_

    evaluator = Engine(evaluate)


    # Attach evaluator to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader)
    )
    
    # Count running time
    timer = Timer(average=False)
    timer.attach(trainer, start=Events.STARTED)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: train_sampler.set_epoch(engine.state.epoch),
        )
        evaluator.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: valid_sampler.set_epoch(engine.state.epoch),
        )
    # learning rate schedule
    def noam_lambda(iteration):
        step = ((iteration - 1) // args.gradient_accumulation_steps) + 1
        # step cannot be zero
        step = max(step, 1)
        # calculate the noam learning rate
        lr = args.model_size ** (-0.5) * min(
            (step) ** (-0.5), (step) * args.warmup_steps ** (-1.5)
        )
        return lr

    if args.scheduler == "noam":
        noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)
        scheduler = LRScheduler(noam_scheduler)
    else:
        scheduler = PiecewiseLinear(
            optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)]
        )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    
    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "train_loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "MLM_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "CE_loss")
    # attach the metrics to the evaluator
    eval_metrics = {}
    eval_metrics["f1"] = Average(output_transform=lambda x: x[0])
    eval_metrics["eval_loss"] = Average(output_transform=lambda x: x[1])
    eval_metrics["avg_f1"] = MetricsLambda(
        average_distributed_scalar, eval_metrics["f1"], args
    )

    eval_metrics["avg_eval_loss"] = MetricsLambda(
        average_distributed_scalar, eval_metrics["eval_loss"], args
    )
  

    for name, metric in eval_metrics.items():
        metric.attach(evaluator, name)
    
    # log training info on the main process only
    if args.local_rank in [-1, 0]:
        # print log
        tb_logger = TensorboardLogger(log_dir=None)
        print("tb_logger.writer.logdir", tb_logger.writer.logdir)
        pbar_log_file = open(os.path.join(tb_logger.writer.logdir, "training.log"), "w")
        pbar = ProgressBar(persist=True, file=pbar_log_file, mininterval=4)
        pbar.attach(trainer, metric_names=["train_loss", 'MLM_loss', 'CE_loss'])
        trainer.add_event_handler(
            Events.ITERATION_STARTED,
            lambda _: pbar.log_message(
                "lr: %.5g @ iteration %d step %d\nTime elapsed: %f"
                % (
                    optimizer.param_groups[0]["lr"],
                    trainer.state.iteration,
                    ((trainer.state.iteration - 1) // args.gradient_accumulation_steps) + 1,
                    timer.value(),
                )
            ),
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            lambda _: pbar.log_message(
                "Validation metrics:\n%s\nTime elapsed: %f" % \
                (pformat(evaluator.state.metrics), timer.value())
            ),
        )
        
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(tag="training", metric_names=["train_loss", 'MLM_loss', 'CE_loss']),
            event_name=Events.ITERATION_COMPLETED,
        )
        tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(optimizer),
            event_name=Events.ITERATION_STARTED,
        )
        
        def global_step_transform(*args, **kwargs):
            return trainer.state.iteration

        tb_logger.attach(
            evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=list(eval_metrics.keys()),
                global_step_transform=global_step_transform
            ),
            event_name=Events.EPOCH_COMPLETED,
        )
        
        # we save the model with maximum acc
        def score_function(engine):
            score = engine.state.metrics["avg_f1"]
            return score

        checkpoint_handlers = [
            ModelCheckpoint(
                tb_logger.writer.logdir,
                "checkpoint",
                score_function=score_function,
                n_saved=args.n_saved,
            )
            for _ in range(1)
        ]
        
        # save n best models
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handlers[0],
            {"mymodel": getattr(model, "module", model)},
        )

        earlyStopping_handler = EarlyStopping(
            patience=args.patience, score_function=score_function, trainer=trainer
        )
        
        evaluator.add_event_handler(Events.COMPLETED, earlyStopping_handler)
        
        # save args
        torch.save(args, tb_logger.writer.logdir + "/model_training_args.bin")
    
        # save config
        getattr(model, "module", model).config.to_json_file(
            os.path.join(tb_logger.writer.logdir, CONFIG_NAME)
        )
        
        # save vocab
        tokenizer.save_vocabulary(tb_logger.writer.logdir)
    
    # start training
    trainer.run(train_loader, max_epochs=args.n_epochs)
    
    if args.local_rank in [-1, 0]:
        # rename the best model
        saved_name = checkpoint_handlers[0]._saved[-1][1]
        os.rename(
            os.path.join(tb_logger.writer.logdir, saved_name),
            os.path.join(tb_logger.writer.logdir, saved_name+"_"+WEIGHTS_NAME),
        )
        
        pbar_log_file.close()
        tb_logger.close()

    dir_name = f"runs/{args.model_name_or_path.split('/')[-1]}"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_name = f"{dir_name}/{args.specific_speaker}"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    dir_name = f"{dir_name}/labeltype{args.label_type}_mlm{args.mlm_prob}_seed{args.seed}_pretrain{args.pretrained}"
    os.rename(tb_logger.writer.logdir, dir_name)



if __name__ == "__main__":
    train()
    
