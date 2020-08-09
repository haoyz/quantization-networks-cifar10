from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10

def get_cifar10(data_dir, img_size, scale_size, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # RGB imagenet
    # with data augmentation
    # train_transformer = T.Compose([
    #     T.Resize(scale_size),
    #     T.RandomCrop(img_size),
    #     # T.RandomResizedCrop(img_size),
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),  # [0, 255] to [0.0, 1.0]
    #     normalizer,  # normalize each channel of the input
    # ])
    train_transformer = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, 4),
        T.ToTensor(),
        normalizer,
    ])

    # test_transformer = T.Compose([
    #     T.Resize(scale_size),
    #     T.CenterCrop(img_size),
    #     T.ToTensor(),
    #     normalizer,
    # ])
    test_transformer = T.Compose([
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(CIFAR10(data_dir, train=True, transform=train_transformer, download=True),
                              batch_size=batch_size, shuffle=True, num_workers=workers)

    val_loader = DataLoader(CIFAR10(data_dir, train=False, transform=test_transformer, download=True),
                            batch_size=batch_size, shuffle=True, num_workers=workers)

    return train_loader, val_loader, 10
