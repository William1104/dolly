#!/bin/env bash

export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='dolly'
export checkpoint_dir_name="${model_name}__${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_training_root='./'
export local_output_dir="${local_training_root}/${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export DATASET_FILE_PATH=`pwd`/parquet-train.arrow
export MODEL_PATH=`pwd`/model/
deepspeed --num_gpus=8 \
    --module training.trainer \
    --deepspeed $deepspeed_config \
    --epochs 1 \
    --local-output-dir $local_output_dir \
    --dbfs-output-dir "" \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 8 \
    --lr 1e-5
