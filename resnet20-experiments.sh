#!/bin/bash

mkdir logs
mkdir logs/cifar

# Train a full-precision baseline
python main.py -a resnet20 -b 256 --img_size 32 --scale_size 32 -j 16 --weight-decay 1e-4 --lr 0.1 --decay_steps 100 150 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/cifar/resnet20-full

# Train a 1-bit weight quantized model
python main.py -a resnet20 -b 256 --img_size 32 --scale_size 32 -j 16 --weight-decay 1e-4  --lr 0.1 -T 5 --decay_steps 35 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/cifar/resnet20-w-1 --train_qw --wk 1 --pretrained logs/cifar/resnet20-full/model_best.pth.tar

# Train 1-bit to 4-bit activation quantized models
python main.py -a resnet20 -b 256 --img_size 32 --scale_size 32 -j 16 --weight-decay 1e-4 --lr 0.1 -T 5 --decay_steps 7 35 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/cifar/resnet20-w-1-a-1-7-35-0.01 --train_qa --ak 1 --qw --wk 1 --qa_sample_batch_size 100 --resume logs/cifar/resnet20-w-1/model_best.pth.tar
python main.py -a resnet20 -b 256 --img_size 32 --scale_size 32 -j 16 --weight-decay 1e-4 --lr 0.01 -T 5 --decay_steps 7 35 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/cifar/resnet20-w-1-a-2-7-35-0.01 --train_qa --ak 2 --qw --wk 1 --qa_sample_batch_size 100 --resume logs/cifar/resnet20-w-1/model_best.pth.tar
python main.py -a resnet20 -b 256 --img_size 32 --scale_size 32 -j 16 --weight-decay 1e-4 --lr 0.01 -T 5 --decay_steps 7 35 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/cifar/resnet20-w-1-a-3-7-35-0.01 --train_qa --ak 3 --qw --wk 1 --qa_sample_batch_size 100 --resume logs/cifar/resnet20-w-1/model_best.pth.tar
python main.py -a resnet20 -b 256 --img_size 32 --scale_size 32 -j 16 --weight-decay 1e-4 --lr 0.01 -T 5 --decay_steps 7 35 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/cifar/resnet20-w-1-a-4-7-35-0.01 --train_qa --ak 4 --qw --wk 1 --qa_sample_batch_size 100 --resume logs/cifar/resnet20-w-1/model_best.pth.tar
