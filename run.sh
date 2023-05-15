#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export SEED=0
export SPEAKER=counselor
export LABELTYPE=fine_strategy
#export SPEAKER=counselor
#export LABELTYPE=coarse_strategy
export MLM=0.15
#export DATE=0831

nohup python train_dist.py \
       --model_name_or_path hfl/chinese-roberta-wwm-ext-large \
       --model_checkpoint pretrained_models/checkpoint_mymodel_-0.3839.pt \
       --pretrained \
       --data_path data/processed_for_train_concate_all_qualified_annotated_500_sessions_seed3.json \
       --train_batch_size 8 \
       --gradient_accumulation_steps 1 \
       --scheduler linear \
       --n_epochs 10 \
       --n_saved 1 \
       --label_type $LABELTYPE \
       --specific_speaker $SPEAKER \
       --mlm_prob $MLM \
       --seed $SEED \
       --lr 5e-5 > test_label_${LABELTYPE}.log 2>&1 & 
