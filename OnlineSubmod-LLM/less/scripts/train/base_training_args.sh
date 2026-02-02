#!/bin/bash

ID=$RANDOM
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
export header="torchrun --nproc_per_node 8 --nnodes 1 --master_port=$MASTER_PORT -m less.train.train"

export base_training_args="--do_train True \
--max_seq_length 2048 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 1 \
--save_strategy no \
--num_train_epochs 1 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to none \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy epoch \
--fracinv 2.0 \
--n_test 500"
