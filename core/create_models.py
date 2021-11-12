from models import gan
from models.model import classifier32, classifier32ABN, MyCNN
import importlib
from models.resnet import *
from models.RAN import ResidualAttentionModel_92_32input_update
from models.resnest import *
from models.backbone_wide_resnet import wide_encoder


def init_network(**options):
    print("Creating networks:")
    if options['cs']:
        net_c = classifier32ABN(num_classes=options['num_classes'])
        nz, ns = options['nz'], 1
        if 'tiny_imagenet' in options['dataset']:
            net_g = gan.Generator(1, nz, 64, 3)
            net_d = gan.Discriminator(1, 3, 64)
        else:
            net_g = gan.Generator32(1, nz, 64, 3)
            net_d = gan.Discriminator32(1, 3, 64)
        if options['use_gpu']:
            net_c, net_g, net_d = nn.DataParallel(net_c).cuda(), nn.DataParallel(net_g).cuda(),\
                                  nn.DataParallel(net_d).cuda()
        return net_c, net_g, net_d, 0

    elif options['cs++']:
        net_c = classifier32ABN(num_classes=options['num_classes'])
        nz, ns = options['nz'], 1
        if 'tiny_imagenet' in options['dataset']:
            net_g1 = gan.Generator(1, nz, 64, 3)
            net_d1 = gan.Discriminator(1, 3, 64)
            net_g2 = gan.Generator(1, nz, 64, 3)
        else:
            net_g1 = gan.Generator32(1, nz, 64, 3)
            net_d1 = gan.Discriminator32(1, 3, 64)
            net_g2 = gan.Generator32(1, nz, 64, 3)
        if options['use_gpu']:
            net_c, net_g1, net_d1, net_g2 = nn.DataParallel(net_c).cuda(), nn.DataParallel(net_g1).cuda(),\
                                  nn.DataParallel(net_d1).cuda(), nn.DataParallel(net_g2).cuda()
        return net_c, net_g1, net_d1, net_g2

    elif options['loss'] != 'SLCPLoss':
        net_c = classifier32(num_classes=options['num_classes'])
        if options['model'] == 'wide_resnet':
            net_c = wide_encoder(options['feat_dim'], 40, 8, 0, num_classes=options['num_classes'])
        if options['use_gpu']:
            net_c = nn.DataParallel(net_c).cuda()
        return net_c, 0, 0, 0

    elif options['loss'] == 'SLCPLoss':
        net_c = classifier32ABN(num_classes=options['num_classes'])
        if options['model'] == 'resnet50':
            net_c = ResNet50(num_c=options['num_classes'])
        elif options['model'] == 'resnet18':
            net_c = ResNet18(num_c=options['num_classes'])
        elif options['model'] == 'resnet152':
            net_c = ResNet152(num_c=options['num_classes'])
        elif options['model'] == 'RAN':
            net_c = ResidualAttentionModel_92_32input_update(num_classes=options['num_classes'])
        elif options['model'] == 'resnest':
            net_c = resnest50(pretrained=False, num_classes=options['num_classes'])
        elif options['model'] == 'wide_resnet':
            net_c = wide_encoder(options['feat_dim'], 40, 4, 0, num_classes=options['num_classes'])
        elif options['model'] == 'MyCNN':
            net_c = MyCNN(num_classes=options['num_classes'])
        elif options['model'] == 'trans':
            net_c = VisionTransformer(CONFIGS['ViT-B_16'], options['img_size'], zero_head=True,
                                      num_classes=options['num_classes'])

        if 'tiny_imagenet' in options['dataset']:
            net_g = gan.Generator(1, options['feat_dim'], 64, 3)
            net_d = gan.Discriminator(1, 3, 64)
        else:
            net_g = gan.Generator32(1, options['feat_dim'], 64, 3)
            net_d = gan.Discriminator32(1, 3, 64)
        if options['use_gpu']:
            net_c, net_g, net_d = nn.DataParallel(net_c).cuda(), nn.DataParallel(net_g).cuda(),\
                                  nn.DataParallel(net_d).cuda()
        return net_c, net_g, net_d, 0


def init_optimizer(net_c, net_g, net_d, net_g2, criterion, **options):
    params_list = [{'params': net_c.parameters()}, {'params': criterion.parameters()}]
    if options['dataset'] == 'tiny_imagenet':
        opt_c = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        opt_c = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
        # opt_c = torch.optim.Adam(params_list, lr=options['lr'])

    opt_g, opt_d, opt_g2 = None, None, None
    if net_g != 0:
        opt_g = torch.optim.Adam(net_g.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
    if net_d != 0:
        opt_d = torch.optim.Adam(net_d.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
    if net_g2 != 0:
        opt_g2 = torch.optim.Adam(net_g2.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
    return opt_c, opt_g, opt_d, opt_g2


def init_loss(**options):
    loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(loss, options['loss'])(**options)
    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()

    if options['use_gpu']:
        criterion.cuda()
        criterion_bce.cuda()
        criterion_l1.cuda()
    return criterion, criterion_bce, criterion_l1
