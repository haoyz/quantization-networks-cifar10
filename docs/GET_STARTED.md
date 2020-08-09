# Getting Started

## Our Scripts
We provide two bash scripts[resnet20-experiments.sh](../resnet20-experiments.sh) and [simplenet-experiments.sh](../simplenet-experiments.sh) which implements the full-precision / 1-bit weight quantization / 1-to-N-bit activation quantization of ResNet-20 and a simple CNN.
To replicate our experimental results, you can simply run:
```
bash resnet20-experiments.sh
bash simplenet-experiments.sh
```

## Training Configurations

`main.py` implements the training of full-precision, weight-quantized and activation-quantized model.
The arguments are：
```
-h, --help              Show help message.

# Data
-d, --dataset DATASET   Dataset to train on. Choose from cifar10/imagenet/mnist.
-b, --batch-size        Batch size for training.
-j, --workers WORKERS   Number of data loader workers.
--scale_size            Size of validation images. Default is 256 (for ImageNet). Set to 32 for CIFAR-10.
--img_size IMG_SIZE     Size of training images. Default is 256 (for ImageNet). Set to 32 for CIFAR-10.

# Model/Training
-a, --arch ARCH         CNN architecture name. Choose from alexnet/resnet110/resnet1202/resnet18/resnet20/resnet32/resnet34/resnet44/resnet50/resnet56/simplenet
--adam                  Set this flag to Use Adam optimizer or not (use SGD instead).
--lr LR                 Initial learning rate. Default is 0.001.
--momentum MOMENTUM     Momentum for SGD optimizer.. Default is 0.9.
--weight-decay WEIGHT_DECAY     Weight decay for SGD optimizer.. Default is 1e-4.
--decay_steps [DECAY_STEPS [DECAY_STEPS ...]]   Milestones(epoch numbers) for learning rate decay. The learning rate will be timed 0.1 at each milestone.
--pretrained PATH       Path to a pre-trained checkpoint. Use this argument only at the first quantization training.
--resume PATH           Path to a quantization-trained checkpoint. Use this argument for further quantization training.
--resume_epoch RESUME_EPOCH     Epoch number to resume from.
--evaluate              Set this flag to only evaluate and do not train.
--epochs EPOCHS         Total number of training epochs.
--start_save START_SAVE Number of training epochs to start saving checkpoint. Default is 0.
--seed SEED             Random seed. Default is 1.
--print-freq PRINT_FREQ Frequency of printing training log. Default is 1.
--print-info PRINT_INFO Frequency of printing model info and gradient info. Default is 10.
--data-dir PATH         Path to the datasets. Default is './data'.
--logs-dir PATH         Path to save the logs. Default is './logs'.

# Quantization
--qw                    Set this flag to quantize weights. If '--train_qw' is set, this argument is ignored.
--train_qw              Set this flag to train weight quantization.
--qa                    Set this flag to quantize activations. If '--train_qa' is set, this argument is ignored.
--train_qa              Set this flag to train activation quantization
--wk WK                 Number of bits for weight quantization. Expected a positive integer or a string in '3-pm-2'/'3-pm-4' (denoting 3(±2) and 3(±4)).
--ak AK                 Number of bits for activation quantization. Expected a positive integer.
-T, --temperature TEMPERATURE   Temperature for sigmoidT function. Default is 10.
--qa_gamma QA_GAMMA     The gamma value for definition of outliers in activation quantization. Expected a floating number in range [0, 0.5). Default is 0.0027 (3-sigma).
--qa_sample_batch_size  Batch size for initialization of activation quantization.
--freeze_bn             Set this flag to freeze BN layers while activation quantization training.
```

Note:
1. The higher bits, the lower initial learning rates.
2. ResNet-20/32/44/56/110/1202 are designed for CIFAR-10, while ResNet-18/34/50 are designed for ImageNet. 
They support different image sizes. Networks designed for ImageNet (e.g. ResNet-18) does not work well on CIFAR-10.
