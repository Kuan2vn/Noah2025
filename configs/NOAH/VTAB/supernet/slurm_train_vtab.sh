#!/usr/bin/env bash

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

JOB_NAME=VTAB-SUPERNET
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml
CKPT=$1
WEIGHT_DECAY=0.0001

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

for LR in 0.0005
do
    for DATASET in cifar100
    do
        # KHÔNG CẦN export MASTER_PORT (không distributed training trên Colab)
        # KHÔNG CẦN srun (chạy trực tiếp python)
        python supernet_train_prompt.py \
            --data-path=./data/vtab-1k/${DATASET} \
            --data-set=${DATASET} \
            --cfg=${CONFIG} \
            --resume=${CKPT} \
            --output_dir=./saves/${DATASET}_supernet_lr-${LR}_wd-${WEIGHT_DECAY} \
            --batch-size=64 \
            --lr=${LR} \
            --epochs=500 \
            --weight-decay=${WEIGHT_DECAY} \
            --no_aug \
            --direct_resize \
            --mixup=0 \
            --cutmix=0 \
            --smoothing=0 \
            --launcher="none" # hoặc bỏ --launcher nếu không cần

        echo -e "\033[32m[ Please check output for details. ]\033[0m" # Sửa thông báo log vì không có file log khi chạy trực tiếp
    done
done