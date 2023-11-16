import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import gc
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from statistics import mean
import math


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img



def image_whiten(images):
    img_size = images.size()
    images = images.view(img_size[0], -1)
    mean = torch.mean(images, dim=1, keepdim=True)
    std = torch.std(images, dim=1, keepdim=True)

    new_image=  (images - mean) / std
    new_image = torch.reshape(new_image, img_size)


    return new_image


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def data_load(args):

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(root='../dataset/cifar10', train=True,
                                                download=True, transform=transform_train)
        train_loader= torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                           shuffle=True, num_workers=4, pin_memory=False)

        testset = torchvision.datasets.CIFAR10(root='../dataset/cifar10', train=False,
                                               download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                          shuffle=False, num_workers=4, pin_memory=False)
    elif  args.dataset == 'fmnist':

        transform_train_fmnist = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        transform_test_fmnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])


        trainset = torchvision.datasets.FashionMNIST(root='../dataset/fmnist', train=True,
                                                     download=True, transform=transform_train_fmnist)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                          shuffle=True, pin_memory=False, num_workers=4)
        valset = torchvision.datasets.FashionMNIST(root='../dataset/fmnist', train=False,
                                                   download=True, transform=transform_test_fmnist)
        # test_loader = torch.utils.data.DataLoader(valset,batch_size= args.batch_size,
        #                                                  shuffle=False, pin_memory=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(valset, batch_size=1000,
                                                  shuffle=True, pin_memory=False, num_workers=4)
    elif  args.dataset == 'mnist':

        transform_train_mnist = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        transform_test_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        trainset = torchvision.datasets.MNIST(root='../dataset/mnist', train=True,
                                                     download=True, transform=transform_train_mnist)
        valset = torchvision.datasets.MNIST(root='../dataset/mnist', train=False,
                                                   download=True, transform=transform_test_mnist)

        train_loader = torch.utils.data.DataLoader(trainset,
            batch_size=args.batch_size, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader
