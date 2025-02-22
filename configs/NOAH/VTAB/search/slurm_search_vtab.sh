#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

JOB_NAME=VTAB-SEARCH
CONFIG=./experiments/NOAH/supernet/supernet-B_prompt.yaml
LIMITS=$1

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

for DATASET in cifar100 caltech101 dtd oxford_flowers102 svhn eurosat dmlab dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
do
    # KHÔNG CẦN export MASTER_PORT TRÊN COLAB
    # export MASTER_PORT=$((12000 + $RANDOM % 20000))

    # CHẠY TRỰC TIẾP LỆNH PYTHON, KHÔNG CẦN srun
    python evolution.py \
        --data-path=./data/vtab-1k/${DATASET} \
        --data-set=${DATASET} \
        --cfg=${CONFIG} \
        --output_dir=saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/search_limit-${LIMITS} \
        --batch-size=64 \
        --resume=saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/checkpoint.pth \
        --param-limits=${LIMITS} \
        --max-epochs=15 \
        --no_aug \
        --inception \
        --direct_resize \
        --mixup=0 \
        --cutmix=0 \
        --smoothing=0 \
        --launcher="none" # hoặc có thể bỏ dòng này nếu không cần launcher

    # LƯU LOG VÀO FILE, VÀ IN RA MÀN HÌNH TRONG COLAB
    echo -e "\033[32m[ Please check output above and log: \"logs/${currenttime}-${DATASET}-vtab-search.log\" for details. ]\033[0m" 2>&1 | tee -a logs/${currenttime}-${DATASET}-vtab-search.log
done