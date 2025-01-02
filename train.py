import utils
import numpy as np
from FSSdataset import DatasetISAID
import argparse
import torch.optim as optim
import torch
from HSNet import HypercorrSqueezeNetwork
from model import SEMPNet
from sota_models import PCFNet
from torch.utils.data import DataLoader
from evaluation import Logger, AverageMeter, Evaluator

import os


def train(epoch, model, dataloader, optimizer, training):
    r""" Train HSNet """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train_mode() if training else model.eval()

    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):
        # 1. forward pass
        utils.to_cuda(batch)
        # logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        logit_cls, logit_mask = model(batch['query_img'], batch['query_ins'], batch['support_imgs'], batch['support_masks'])
        # pred_mask = logit_mask.argmax(1)

        pred_mask = model.generate_predict_mask(batch['query_ins'], [logit_cls.unsqueeze(0)])

        # 2. Compute loss & update model parameters
        loss = model.compute_objective(logit_cls, logit_mask, batch['query_mask'], batch['query_ins'])
        # loss = criterion(logit_mask, batch['query_mask'].unsqueeze(1))
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.long(), batch['query_mask'].long())
        # average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=100)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='MaskNet Pytorch Implementation')
    parser.add_argument('--datapath', type=str, default="F:/datasets/iSAID_patches")
    # parser.add_argument('--datapath', type=str, default="F:/datasets/DLRSD")
    parser.add_argument('--benchmark', type=str, default='iSAID', choices=['pascal', 'coco', 'fss', 'iSAID', 'DLRSD'])
    parser.add_argument('--logpath', type=str, default='')
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'vgg16'])
    args = parser.parse_args(args=[])
    Logger.initialize(args, training=True)

    model = SEMPNet(args.backbone, args.shot)
    # model = HypercorrSqueezeNetwork(args.backbone, False)
    # model.load_state_dict(torch.load('logs/iSAID/HSNet-resnet-fold0/best_model.pt'))
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model.to(device)

    # Helper classes (for training) initialization
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr}])

    # Dataset initialization
    dataset_trn = DatasetISAID(datapath=args.datapath, fold=args.fold, split='trn', transform=True, shot=args.shot)
    dataset_val = DatasetISAID(datapath=args.datapath, fold=args.fold, split='val', transform=False, shot=args.shot)
    dataloader_trn = DataLoader(dataset_trn, args.bsz, shuffle=True, collate_fn=dataset_trn.collate_fn)
    dataloader_val = DataLoader(dataset_val, args.bsz, shuffle=False, collate_fn=dataset_val.collate_fn)

    # Train HSNet
    best_val_miou = float('-inf')
    best_val_loss = float('inf')
    for epoch in range(args.niter):
        trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)
        with torch.no_grad():
            val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

        # Save the best model
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            Logger.save_model_miou(model, epoch, val_miou)

        Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
        Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
        Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
        Logger.tbd_writer.flush()
    Logger.tbd_writer.close()
    Logger.info('==================== Finished Training ====================')