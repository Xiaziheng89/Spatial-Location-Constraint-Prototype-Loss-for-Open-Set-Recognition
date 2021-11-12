import numpy as np
import torch
from core import evaluation


def test(net, criterion, test_loader, out_loader, **options):
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _prediction_k, _prediction_u, _labels = [], [], []

    with torch.no_grad():
        net.eval()
        for data, labels in test_loader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logic, _ = criterion(x, y)

                predictions = logic.data.max(1)[1]           # logic 应该就是越大越好了，所以我之前的代码有问题
                total += labels.size(0)
                correct += predictions.eq(labels.data.view_as(predictions)).sum()

                _prediction_k.append(logic.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(out_loader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                x, y = net(data, True)
                logic, _ = criterion(x, y)
                _prediction_u.append(logic.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _prediction_k = np.concatenate(_prediction_k, 0)
    _prediction_u = np.concatenate(_prediction_u, 0)
    _labels = np.concatenate(_labels, 0)

    x1, x2 = np.max(_prediction_k, axis=1), np.max(_prediction_u, axis=1)
    results = evaluation.metric_auroc(x1, x2)['Bas']

    # OSCR
    _oscr_score = evaluation.compute_oscr(_prediction_k, _prediction_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_score * 100.

    return results
