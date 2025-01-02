r""" Helper functions """
import random

import torch
import numpy as np
from torch import nn


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


# def to_cuda(batch):
#     for key, value in batch.items():
#         if isinstance(value, torch.Tensor):
#             batch[key] = value.cuda()
#     return batch

def to_cuda(batch):
    bsz = len(batch['query_ins'])
    batch['query_img'] = batch['query_img'].cuda()
    batch['query_mask'] = batch['query_mask'].cuda()
    batch['support_imgs'] = batch['support_imgs'].cuda()
    batch['support_masks'] = batch['support_masks'].cuda()
    batch['class_id'] = batch['class_id'].cuda()
    for i in range(bsz):
        batch['query_ins'][i] = batch['query_ins'][i].cuda()
        # batch['query_point'][i] = batch['query_point'][i].cuda()

def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def f_score(pr, gt, beta=1, eps=1e-7, threshold=None):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1.,
                           eps=self.eps, threshold=None)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps)
        self.bce = nn.BCELoss(reduction='mean')

        self.lambda_dice=lambda_dice
        self.lambda_bce=lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice*dice) + (self.lambda_bce*bce)

