import argparse
import os
from core.split import splits_2020 as splits
from core.create_dir_file import create_dir_path, create_record_file_name, create_model_file_name, create_model_dir_path
import pandas as pd
import torch
import time
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import datetime
from core import train, test
from pprint import pprint
from core.utils import save_networks
from core.load_osr_data import load_osr_data
from core.create_models import init_network, init_loss, init_optimizer
parser = argparse.ArgumentParser("Open Set Recognition")
# dataset
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet")
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--out_num', type=int, default=50, help='For CIFAR100, cifar+10-->10, cifar+50-->50')
parser.add_argument('--output', type=str, default='./results')
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for generator and discriminator")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--step_size', type=int, default=30)
# model
parser.add_argument('--feat_dim', type=int, default=128, help="classifier32:128,resnet50:2048,RAN:1024,resnest:2048")
parser.add_argument('--model', type=str, default='classifier32', help="classifier32,resnet50,RAN,resnest")
parser.add_argument('--lambda', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--loss', type=str, default='SLCPLoss', help="Softmax,GCPLoss,SLCPLoss")
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--cs', action='store_true', help="for ARPLoss and AMPFLoss", default=False)
parser.add_argument('--cs++', action='store_true', help="for AMPFLoss", default=False)


def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
        options.update({'use_gpu': use_gpu})
    else:
        print("Currently using CPU")

    train_loader, test_loader, out_loader, num_classes = load_osr_data(**options)   # load dataset
    options['num_classes'] = num_classes
    net_c, net_g, net_d, net_g2 = init_network(**options)                           # networks
    criterion, criterion_bce, criterion_l1 = init_loss(**options)                   # criterion
    opt_c, opt_g, opt_d, opt_g2 = init_optimizer(net_c, net_g, net_d, net_g2, criterion, **options)    # optimizers
    model_path, model_file_name = create_model_dir_path(**options), create_model_file_name(**options)

    scheduler = None
    if options['step_size'] > 0:
        scheduler = lr_scheduler.MultiStepLR(opt_c, milestones=[int(options['max_epoch'] * 0.3),
                                                                int(options['max_epoch'] * 0.6),
                                                                int(options['max_epoch'] * 0.9)], gamma=0.1)
    start_time = time.time()
    best_res = {'best_ACC': 0, 'best_AUROC': 0, 'best_OSCR': 0}
    res = {}
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))
        train(net_c, criterion, opt_c, train_loader, epoch, **options)

        if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print("==> Test", options['loss'])
            res = test(net_c, criterion, test_loader, out_loader, **options)

            print("Acc(%): {:.3f}\tAUROC(%): {:.3f}\tOSCR(%): {:.3f}\t".format(res['ACC'], res['AUROC'], res['OSCR']))

            save_networks(net_c, model_path, model_file_name, criterion=criterion, **options)

            if res['ACC'] > best_res['best_ACC']:
                best_res['best_ACC'] = res['ACC']
            if res['AUROC'] > best_res['best_AUROC']:
                best_res['best_AUROC'] = res['AUROC']
                save_networks(net_c, model_path, model_file_name + '_best_AUROC', criterion=criterion, **options)
            if res['OSCR'] > best_res['best_OSCR']:
                best_res['best_OSCR'] = res['OSCR']
            pprint(best_res)
        if options['step_size'] > 0:
            scheduler.step()
        print()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}\n".format(elapsed))
    return res, best_res


if __name__ == '__main__':
    args = parser.parse_args()
    my_option = vars(args)

    my_option['data_root'] = os.path.join(my_option['data_root'], my_option['dataset'])
    results = dict()
    dir_path, record_file_name = create_dir_path(**my_option), create_record_file_name(**my_option)

    for i in range(len(splits[my_option['dataset']])):
        known = splits[my_option['dataset']][len(splits[my_option['dataset']])-i-1]
        if my_option['dataset'] == 'cifar100':
            unknown = splits[my_option['dataset']+'-'+str(my_option['out_num'])][len(splits[my_option['dataset']])-i-1]
        elif my_option['dataset'] == 'tiny_imagenet':
            my_option['img_size'] = 64
            my_option['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))

        my_option.update({'item': i, 'known': known, 'unknown': unknown})
        if my_option['loss'] == 'AMPFLoss':
            my_option.update({'R_recording': [], 'kR_recording': []})

        result, best_result = main_worker(my_option)
        result['unknown'] = unknown
        result['known'] = known
        best_result['unknown'] = unknown
        best_result['known'] = known
        results[str(i)] = result
        results[str(i)].update(best_result)
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, record_file_name))        # dir_path/file_name
