# Preparing Data

This repo supports 3 datasets: CIFAR-10, MNIST and ImageNet.

CIFAR-10 and MNIST can be automatically downloaded by torchvision and do not require manually preparation.

To use ImageNet, please download ImageNet (ILSVRC2012) to `data` folder.
+ To download ImageNet, one way is to participate in [Kaggle ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/data) or use the BT torrent available on Academic: [train](http://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) [val](http://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5).
+ Image label list `train.txt` and `val.txt` can be acquired from the `caffe_ilsvrc12.tar.gz` provided by Caffe：

    ```
    cd data
    mkdir imagenet
    cd imagenet
    wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz
    tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz
    ```

When ImageNet is prepared, the expected structure of this code should be like：

```
.
├── data
│   └── imagenet
│       ├── train
│       │   ├── n01440764
│       │   │   ├── n01440764_10026.JPEG
│       │   │   ├── ...
│       │   │   └── n01440764_9981.JPEG
│       │   ├── ...
│       │   └── n15075141
│       ├── train.txt
│       ├── val
│       │   ├── ILSVRC2012_val_00000001.JPEG
│       │   ├── ...
│       │   └── ILSVRC2012_val_00050000.JPEG
│       └── val.txt
├── datasets
├── models
├── utils
├── LICENSE
├── main.py
├── README.md
├── resnet20-experiments.sh
├── simplenet_experiments.sh
└── visualize.py
```
