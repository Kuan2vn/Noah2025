#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

JOB_NAME=VTAB-RT
CKPT=$1
WEIGHT_DECAY=0.0001

mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

for LR in 0.001
do
    for DATASET in cifar100 caltech101 dtd oxford_flowers102 svhn eurosat dmlab dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do
        # KHÔNG CẦN export MASTER_PORT TRÊN COLAB
        # export MASTER_PORT=$((12000 + $RANDOM % 20000))

        # CHẠY TRỰC TIẾP LỆNH PYTHON, KHÔNG CẦN srun
        python supernet_train_prompt.py \
            --data-path=./data/vtab-1k/${DATASET} \
            --data-set=${DATASET} \
            --cfg=experiments/NOAH/subnet/VTAB/ViT-B_prompt_${DATASET}.yaml \
            --resume=${CKPT} \
            --output_dir=saves/${DATASET}_supernet_lr-0.0005_wd-0.0001/retrain_${LR}_wd-${WEIGHT_DECAY} \
            --batch-size=64 \
            --mode=retrain \
            --epochs=100 \
            --lr=${LR} \
            --weight-decay=${WEIGHT_DECAY} \
            --no_aug \
            --direct_resize \
            --mixup=0 \
            --cutmix=0 \
            --smoothing=0 \
            --launcher="none" # hoặc có thể bỏ dòng này nếu không cần launcher

        # LƯU LOG VÀO FILE, VÀ IN RA MÀN HÌNH TRONG COLAB
        echo -e "\033[32m[ Please check output above and log: \"logs/${currenttime}-${DATASET}-${LR}-vtab-rt.log\" for details. ]\033[0m" 2>&1 | tee -a logs/${currenttime}-${DATASET}-${LR}-vtab-rt.log
    done
done