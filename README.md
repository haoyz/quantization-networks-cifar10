# Quantization Networks on CIFAR-10

## Overview

This repository contains the training code of Quantization Networks introduced in the CVPR 2019 paper: [Quantization Networks](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Quantization_Networks_CVPR_2019_paper.pdf).

Our implementation is a modified version of the [original implementation](https://github.com/aliyun/alibabacloud-quantization-networks).
The main changes are:
1. Support of CIFAR-10 and MNIST dataset, besides ImageNet.
2. A unified training script for full-precision, weight quantization and activation quantization.
3. Training curve visualization using Tensorboard.
4. Clustering before quantization training, needless of manual clustering.
5. Freezing of BN layer during activation quantization training (idea borrowed from [FQN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf)).
6. Removal of outlier in clustering using 3-sigma rule (also borrowed from [FQN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf)).


## Usage

First, please create a compatible Python environment. 

### Environments

Here are our environment configurations during development.
+ Ubuntu 18.04
+ Python 3.7
+ opencv-python
+ numpy 1.17.4
+ pytorch 1.3.0
+ torchvision 0.4.2
+ Tensorboard 2.0.0
+ argparse 1.1
+ logging 0.5.1.2

### Preparing Data
Please refer to [PREPARE_DATA.md](docs/PREPARE_DATA.md).

### Getting Started
Please refer to [GET_STARTED.md](docs/GET_STARTED.md).

## Our Experimental Results
Please refer to [EXPERIMENTS.md](docs/EXPERIMENTS.md).

## License
This repository is forked from [aliyun/alibabacloud-quantization-networks](https://github.com/aliyun/alibabacloud-quantization-networks) and keep its Apache 2.0 license.

## Citation
Please cite the paper if it helps your research:
```
@inproceedings{yang2019quantization,
  title={Quantization Networks},
  author={Yang Jiwei, Shen Xu, Xing Jun, Tian Xinmei, Li Houqiang, Deng Bing, Huang Jianqiang and Hua Xian-sheng},
  booktitle={CVPR},
  year={2019}
}
```
