#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#export LABELTYPE=fine_strategy
#export SPEAKER=counselor
export LABELTYPE=fine_strategy
export SPEAKER=counselor
export INVAL_PROB=1
# export DATE=0831
export zh_speaker=咨询师

nohup python test.py \
       --model_dir runs/chinese-roberta-wwm-ext-large/${zh_speaker}/${LABELTYPE} \
       --data_path data/processed_for_train_concate_all_qualified_annotated_500_sessions_seed3.json \
       --label_type $LABELTYPE \
       --specific_speaker $SPEAKER \
       --evaluate_invalidation_prob ${INVAL_PROB} \
       --test_batch_size 4 \
       --max_history_num 10 > 2023ACL_logs/test_${LABELTYPE}_${SPEAKER}_${INVAL_PROB}.log 2>&1 &
