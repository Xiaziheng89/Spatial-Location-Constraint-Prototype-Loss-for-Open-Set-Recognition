import os
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, KMNIST
from PIL import Image


class MNISTRGB(MNIST):
    """MNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class KMNISTRGB(KMNIST):
    """KMNIST Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')
        img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'mnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = MNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = MNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = 10


class KMNIST(object):
    def __init__(self, **options):
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'kmnist')

        pin_memory = True if options['use_gpu'] else False

        trainset = KMNISTRGB(root=data_root, train=True, download=True, transform=transform)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = KMNISTRGB(root=data_root, train=False, download=True, transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = 10


class CIFAR10(object):
    def __init__(self, **options):

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'cifar10')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.train_loader = train_loader
        self.test_loader = test_loader


class CIFAR100(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'cifar100')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 100
        self.train_loader = train_loader
        self.test_loader = test_loader


class SVHN(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'svhn')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train)
        
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=options['workers'], pin_memory=pin_memory,
        )
        
        testset = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=transform)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        self.num_classes = 10
        self.train_loader = train_loader
        self.test_loader = test_loader


class ImageNet(object):     # ImageNet2012
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'imagenet')
        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.ImageNet(root=data_root, split='train', download='True', transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                   num_workers=options['workers'], pin_memory=pin_memory)

        testset = torchvision.datasets.ImageNet(root=data_root, split='test', download='True', transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False,
                                                  num_workers=options['workers'], pin_memory=pin_memory)

        self.num_classes = 200
        self.train_loader = train_loader
        self.test_loader = test_loader


class Omniglot(object):
    def __init__(self, **options):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        batch_size = options['batch_size']
        data_root = os.path.join(options['data_root'], 'omniglot')

        pin_memory = True if options['use_gpu'] else False

        trainset = torchvision.datasets.Omniglot(root=data_root, download=True, transform=transform_train)
        # trainset = torchvision.datasets.

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['workers'], pin_memory=pin_memory)

        testset = torchvision.datasets.Omniglot(root=data_root, download=True, transform=transform)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=options['workers'], pin_memory=pin_memory,
        )

        # self.num_classes = 10
        self.train_loader = train_loader
        self.test_loader = test_loader


__factory = {
    'mnist': MNIST,
    'kmnist': KMNIST,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn': SVHN,
    'imagenet': ImageNet,
    'omniglot': Omniglot
}


def create(name, **options):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](**options)
