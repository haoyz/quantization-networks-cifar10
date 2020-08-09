#!/bin/bash

mkdir logs
mkdir logs/simple

# Train a full-precision baseline
python main.py -a simplenet -b 256 --img_size 32 --scale_size 32 -j 0 --weight-decay 1e-4 --lr 0.1 --decay_steps 100 150 --epochs 200 --start_save 0 --print-info 1  --logs-dir logs/simple/simplenet-full

# Train a 1-bit weight quantized model
python main.py -a simplenet -b 256 --img_size 32 --scale_size 32 -j 0 --weight-decay 1e-4 --lr 0.1 --decay_steps 35 100 150 --epochs 200 --start_save 0 --print-info 1  --logs-dir logs/simple/simplenet-qw-1-35 --train_qw --wk 1 --pretrained logs/simple/simplenet-full/model_best.pth.tar

# Train a 1-bit and a 2-bit activation quantized model
python main.py -a simplenet -b 512 --img_size 32 --scale_size 32 -j 0 --weight-decay 1e-4 --lr 0.1 --decay_steps 35 --temperature 5 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/simple/simplenet-qa-1 --train_qa --qw --resume logs/simple/simplenet-qw-1-35/model_best.pth.tar --ak 1 --wk 1
python main.py -a simplenet -b 512 --img_size 32 --scale_size 32 -j 0 --weight-decay 1e-4 --lr 0.1 --decay_steps 20 --temperature 5 --epochs 200 --start_save 0 --print-info 1 --logs-dir logs/simple/simplenet-qa-2 --train_qa --qw --resume logs/simple/simplenet-qw-1-35/model_best.pth.tar --ak 2 --wk 1
