import torch
import torch.nn as nn
import torch.nn.functional as f
from loss.Dist import Dist


class GCPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(GCPLoss, self).__init__()
        self.weight_pl = options['lambda']
        self.Dist = Dist(num_classes=options['num_classes'], feat_dim=options['feat_dim'])

    def forward(self, x, y, labels=None):
        dist = self.Dist(x)
        logits = f.softmax(-dist, dim=1)
        if labels is None:
            return logits, 0
        loss_main = f.cross_entropy(-dist, labels)

        center_batch = self.Dist.centers[labels, :]
        loss_r = f.mse_loss(x, center_batch) / 2

        loss = loss_main + self.weight_pl * loss_r
        return logits, loss
