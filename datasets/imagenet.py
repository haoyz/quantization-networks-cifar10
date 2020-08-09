import os.path as osp
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torch.utils.data.sampler import RandomSampler


class ImageNet(object):
    def __init__(self, dataset, root=None, transform=None):
        super(ImageNet, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, label = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        img = Image.open(fpath).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_imagenet(data_dir, img_size, scale_size, batch_size, workers):
    def readlist(fpath):
        lines = []
        with open(fpath, 'r') as f:
            data = f.readlines()

        for line in data:
            name, label = line.split()
            lines.append((name, int(label)))
        return lines

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # RGB imagenet
    # with data augmentation
    train_transformer = T.Compose([
        T.Resize(scale_size),
        T.RandomCrop(img_size),
        # T.RandomResizedCrop(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # [0, 255] to [0.0, 1.0]
        normalizer,  # normalize each channel of the input
    ])

    test_transformer = T.Compose([
        T.Resize(scale_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalizer,
    ])

    train_list = readlist(data_dir + '/train.txt')
    val_list = readlist(data_dir + '/val.txt')

    train_loader = DataLoader(
        ImageNet(train_list, root=data_dir + '/train/',
                 transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomSampler(train_list),
        pin_memory=True, drop_last=False)

    val_loader = DataLoader(
        ImageNet(val_list, root=data_dir + '/val/',
                 transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return train_loader, val_loader, 1000
