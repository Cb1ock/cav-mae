#!/bin/bash

set -x
# # comment this line if not running on sls cluster
# . /data/sls/scratch/share-201907/slstoolchainrc
# source /data/sls/scratch/yuangong/avbyol/venv-a5/bin/activate
export TORCH_HOME=../../pretrained_models

model=my-ft
ftmode=multimodal # or audioonly or videoonly

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
#wget -nc https://www.dropbox.com/s/l5t5geufdy3qvnv/audio_model.21.pth?dl=1 -O cav-mae-scale++.pth
#pretrain_path=${cur_dir}/cav-mae-scale++.pth

# pretrain_path=../celebv-text/exp/testmae01-audioset-cav-mae-balNone-lr5e-5-epoch25-bs16-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75-a5/models/best_audio_model.pth
# pretrain_model=my_pretrained

pretrain_path=../celebv-text/exp/testmae01-audioset-cav-mae-balNone-lr1e-4-epoch25-bs32-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75-a5/models/best_audio_model.pth
pretrain_model=my_pretrained_biger

# pretrain_path=/home/chenghao/Project/cav-mae/pretrained_model/audio_model_scale++.pth
# pretrain_model=cav-mae_pretrained_scale++

# pretrain_path=/home/chenghao/Project/cav-mae/pretrained_model/audio_model_base.pth
# pretrain_model=cav-mae_pretrained_base

pretrained_mae_path=../../pretrained_model/pretrained_maedfer.pth

freeze_base=False
head_lr=50 # newly initialized ft layers uses 50 times larger than the base lr

bal=None
lr=1e-4
epoch=10
lrscheduler_start=2
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=1
wa_end=10
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
batch_size=24
label_smooth=0.1

dataset=audioset
tr_data=train_data.json
te_data=test_data.json
label_csv=class_labels_indices_mafw.csv


exp_dir=./exp/testmae01-full-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-r3-${pretrain_model}
mkdir -p $exp_dir

CUDA_VISABLE_DEVICE=0,1 python -W ignore /home/chenghao/Project/cav-mae/src/run_my_ft.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--pretrained_mae_path ${pretrained_mae_path} \
--label-csv ${label_csv} --n_class 11 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss BCE --metrics emo --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 32 > $exp_dir/log.txt 2>&1