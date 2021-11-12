import os


def create_dir_path(**options):
    if options['dataset'] == 'cifar100':
        dir_path = os.path.join(options['output'], 'cifar+{}'.format(str(options['out_num'])), 'records')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)           # ./results/cifar+10、cifar+50/records
    else:
        dir_path = os.path.join(options['output'], options['dataset'], 'records')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)           # ./results/mnist, svhn, cifar10, tiny_imagenet/records
    return dir_path


def create_record_file_name(**options):
    if options['dataset'] == 'cifar100':
        dataset_name = 'cifar+' + str(options['out_num'])
    else:
        dataset_name = options['dataset']

    if options['loss'] == 'Softmax':
        file_name = dataset_name + '_Softmax_' + options['model'] + '.csv'     # e.g.: mnist_Softmax.csv
    elif options['loss'] == 'GCPLoss':
        file_name = dataset_name + '_GCPL_' + options['model'] + '.csv'
    elif options['loss'] == 'RPLoss':
        file_name = dataset_name + '_RPL_' + options['model'] + '.csv'
    elif options['loss'] == 'ARPLoss':
        if options['cs']:
            file_name = dataset_name + '_ARPL+CS_' + options['model'] + '.csv'
        else:
            file_name = dataset_name + '_ARPL_' + options['model'] + '.csv'
    elif options['loss'] == 'AMPFLoss':
        if options['cs++']:
            file_name = dataset_name + '_AMPF++_' + options['model'] + '.csv'
        elif options['cs']:
            file_name = dataset_name + '_AMPF_' + options['model'] + '.csv'
        else:
            file_name = dataset_name + '_MPF_' + options['model'] + '.csv'
    else:
        # file_name = dataset_name + '_SLCPL_' + options['model'] + '.csv'
        file_name = dataset_name + '_' + options['loss'] + '_' + options['model'] + '.csv'

    return file_name


def create_model_file_name(**options):
    if options['dataset'] == 'cifar100':
        dataset_name = 'cifar+' + str(options['out_num'])
    else:
        dataset_name = options['dataset']

    if options['loss'] == 'Softmax':
        file_name = dataset_name + '_' + str(options['item']) + '_Softmax'  # e.g.: mnist_0_Softmax
    elif options['loss'] == 'GCPLoss':
        file_name = dataset_name + '_' + str(options['item']) + '_GCPL'
    elif options['loss'] == 'RPLoss':
        file_name = dataset_name + '_' + str(options['item']) + '_RPL'
    elif options['loss'] == 'ARPLoss':
        if options['cs']:
            file_name = dataset_name + '_' + str(options['item']) + '_ARPL+CS'
        else:
            file_name = dataset_name + '_ARPL'
    elif options['loss'] == 'AMPFLoss':
        if options['cs++']:
            file_name = dataset_name + '_' + str(options['item']) + '_AMPF++'
        elif options['cs']:
            file_name = dataset_name + '_' + str(options['item']) + '_AMPF'
        else:
            file_name = dataset_name + '_' + str(options['item']) + '_MPF'
    else:
        # file_name = dataset_name + '_' + str(options['item']) + '_SLCPL'
        file_name = dataset_name + '_' + str(options['item']) + '_' + options['loss']
    return file_name


def create_model_dir_path(**options):
    if options['dataset'] == 'cifar100':
        dir_path = os.path.join(options['output'], 'cifar+{}'.format(str(options['out_num'])))
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)           # ./results/cifar+10、cifar+50
    else:
        dir_path = os.path.join(options['output'], options['dataset'])
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)           # ./results/mnist, svhn, cifar10, tiny_imagenet
    return dir_path
