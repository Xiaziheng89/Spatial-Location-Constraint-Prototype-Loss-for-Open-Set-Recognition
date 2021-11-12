from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR


def load_osr_data(**options):
    if 'mnist' in options['dataset']:
        data = MNIST_OSR(known=options['known'], dataroot=options['data_root'], batch_size=options['batch_size'],
                         img_size=options['img_size'])
        train_loader, test_loader, out_loader = data.train_loader, data.test_loader, data.out_loader
    elif 'cifar10' == options['dataset']:
        data = CIFAR10_OSR(known=options['known'], dataroot=options['data_root'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        train_loader, test_loader, out_loader = data.train_loader, data.test_loader, data.out_loader
    elif 'svhn' in options['dataset']:
        data = SVHN_OSR(known=options['known'], dataroot=options['data_root'], batch_size=options['batch_size'],
                        img_size=options['img_size'])
        train_loader, test_loader, out_loader = data.train_loader, data.test_loader, data.out_loader
    elif 'cifar100' in options['dataset']:
        data = CIFAR10_OSR(known=options['known'], dataroot=options['data_root'], batch_size=options['batch_size'],
                           img_size=options['img_size'])
        train_loader, test_loader = data.train_loader, data.test_loader
        out_data = CIFAR100_OSR(known=options['unknown'], dataroot=options['data_root'],
                                batch_size=options['batch_size'], img_size=options['img_size'])
        out_loader = out_data.test_loader
    else:
        data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['data_root'],
                                 batch_size=options['batch_size'], img_size=options['img_size'])
        train_loader, test_loader, out_loader = data.train_loader, data.test_loader, data.out_loader

    return train_loader, test_loader, out_loader, data.num_classes
