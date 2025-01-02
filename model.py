import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce
from operator import add
from torchvision.models import resnet
from torchvision.models import vgg
from FSSdataset import DatasetISAID
from torch.utils.data import DataLoader
from segment_anything.modeling.mask_decoder import MLP
import math


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone.conv1.forward(img)
    feat = backbone.bn1.forward(feat)
    feat = backbone.relu.forward(feat)
    feat = backbone.maxpool.forward(feat)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
        feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

        if bid == 0:
            res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

    return feats

def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat


class SEMPNet(nn.Module):
    def __init__(self, backbone, shot):
        super(SEMPNet, self).__init__()
        self.shot = shot
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            self.feat_channels = [512, 512, 512, 512, 512, 512]
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            self.feat_channels = [256, 512, 1024, 2048]
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            self.feat_channels = [256, 512, 1024, 2048]
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount()[-3:].cumsum(dim=0)
        self.MMMModule = MPM(self.feat_channels, self.stack_ids, nbottlenecks)
        self.criterion = nn.BCELoss(reduction='none')

        self.criterion_for_fewshot = WeightedDiceLoss()

    def forward(self, query_image, query_instances, support_images, support_masks):
        with torch.no_grad():
            support_feats_list = []
            query_feats = self.extract_feats(query_image, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            for i in range(self.shot):
                support_feats = self.extract_feats(support_images[:, i, :, :, :], self.backbone, self.feat_ids,
                                                   self.bottleneck_ids, self.lids)
                support_feats_list.append(support_feats)

        corr_list = []
        for i in range(self.shot):
            corr = self.MMMModule.correlation(query_feats, support_feats_list[i], support_masks[:, i, :, :].clone())
            # corr = self.correlation(query_feats, support_feats_list[i], [3, 9, 13], support_masks[:, i, :, :].clone())
            corr_list.append(corr)

        corr = corr_list[0]
        if self.shot > 1:
            for i in range(1, self.shot):
                corr[0] += corr_list[i][0]
                corr[1] += corr_list[i][1]
                corr[2] += corr_list[i][2]
            corr[0] /= self.shot
            corr[1] /= self.shot
            corr[2] /= self.shot


        output_categories, output_masks = self.MMMModule(corr, query_instances)

        return output_categories, output_masks

    def correlation(self, query_feats, support_feats, stack_ids, support_mask):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            c, h, w = support_feat.size()[1:]
            mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                 align_corners=True)
            prototype = Weighted_GAP(support_feat, mask)
            supp_feat_bin = prototype.expand(-1, -1, h, w)
            merge = torch.cat([query_feat, supp_feat_bin], dim=1)
            if c ==512:
                corr = self.sim_conv1(merge)
            elif c == 1024:
                corr = self.sim_conv2(merge)
            else:
                corr = self.sim_conv3(merge)
            corrs.append(corr.squeeze(1))


        corr_l4 = torch.stack(corrs[-stack_ids[0]:]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[-stack_ids[1]:-stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l2 = torch.stack(corrs[-stack_ids[2]:-stack_ids[1]]).transpose(0, 1).contiguous()

        return [corr_l4, corr_l3, corr_l2]

    def compute_objective(self, logit_category, logit_mask, gt_mask, query_instances):
        bsz = gt_mask.size(0)
        class_loss = 0.0
        for i in range(bsz):
            # gt_cate = torch.tensor([gt_mask[i][point[1], point[0]] for point in points[i]], dtype=torch.long, device=gt_mask.device)
            IoA = self.compute_intersection_over_area(query_instances[i], gt_mask[i])
            gt_cate = (IoA > 0.8).float()
            weight = torch.zeros_like(gt_cate, dtype=torch.float, device=gt_cate.device)
            weight = torch.fill_(weight, 1)
            weight[gt_cate > 0] = 9
            cls_loss = self.criterion(logit_category[i].squeeze(0), gt_cate)
            cls_loss = torch.mean(weight * cls_loss)
            # mse_loss = F.cross_entropy(logit_category[i].unsqueeze(0), gt_cate.unsqueeze(0), weight=torch.tensor([1.0, 10.0], device=gt_cate.device))
            class_loss += cls_loss
        class_loss /= bsz
        dice_loss = self.criterion_for_fewshot(logit_mask, gt_mask)
        # logit_mask = logit_mask.view(bsz, 2, -1)
        # gt_mask = gt_mask.contiguous().view(bsz, -1).long()
        # mask_loss = self.CELoss(logit_mask, gt_mask)
        return class_loss + dice_loss

    def compute_intersection_over_area(self, mask, label):
        k, h, w = mask.shape
        mask = mask.view(k, h*w)
        label = label.reshape(1, h*w)
        intersection = torch.sum(mask * label, dim=1)
        area = torch.sum(mask, dim=1)
        IoA = intersection / area
        return IoA

    def generate_predict_mask(self, masks, categories, threshold=0.5):
        output_masks = []
        for mask, category in zip(masks, categories):
            category = (category > threshold).float()
            out = torch.einsum('qk, khw->qhw', category, mask.float())
            out = out >= 1
            output_masks.append(out)
        output_masks = torch.cat(output_masks, dim=0)
        return output_masks

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging



class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = query.size()
        proj_query = query.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_key = key.view(m_batchsize, C, -1)
        energy = torch.bmm(proj_query, proj_key)/ math.sqrt(self.chanel_in)
        attention = self.softmax(energy)
        proj_value = value.view(m_batchsize, 1, -1)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, 1, width, height)

        return out


class MPM(nn.Module):
    def __init__(self, in_channels, stack_ids, nbottlenecks):
        super(MPM, self).__init__()

        def make_building_block(in_channel, out_channels, group=4):

            building_block_layers = []
            for idx, outch in enumerate(out_channels):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                building_block_layers.append(nn.Conv2d(inch, outch, 3, 1, 1))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        self.stack_ids = stack_ids[-3:]

        # MP blocks
        self.MP_blocks = nn.ModuleList()
        for inch in in_channels[-3:]:
            self.MP_blocks.append(CAM_Module(inch))

        outch1, outch2, outch3 = 16, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(nbottlenecks[-1], [outch1, outch2, outch3])
        self.encoder_layer3 = make_building_block(nbottlenecks[-2], [outch1, outch2, outch3])
        self.encoder_layer2 = make_building_block(nbottlenecks[-3], [outch1, outch2, outch3])

        # Mixing building blocks
        self.encoder_layer4to3 = nn.Sequential(
            nn.Linear(outch3, outch3 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(outch3 // 4, outch3),
            nn.ReLU(inplace=True)
        )
        self.encoder_layer3to2 = nn.Sequential(
            nn.Linear(outch3, outch3 // 4),
            nn.ReLU(inplace=True),
            nn.Linear(outch3 // 4, outch3),
            nn.ReLU(inplace=True)
        )
        self.decoder = MLP(outch3, 256, 1, 3)

    def forward(self, corr, query_instances):

        corr4 = self.encoder_layer4(corr[0])
        corr3 = self.encoder_layer3(corr[1])
        corr2 = self.encoder_layer2(corr[2])
        logit_categories, logit_masks = [], []
        for i, ins in enumerate(query_instances):
            hypercorr_ins4 = self.compute_instance_correlation(corr4[i], ins)
            hypercorr_ins3 = self.compute_instance_correlation(corr3[i], ins)
            hypercorr_ins2 = self.compute_instance_correlation(corr2[i], ins)
            hypercorr_mix43 = hypercorr_ins4 + hypercorr_ins3
            hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)
            hypercorr_mix432 = hypercorr_mix43 + hypercorr_ins2
            hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)
            logit_category = self.decoder(hypercorr_mix432)
            logit_category = logit_category.transpose(0, 1)
            logit_mask = torch.einsum("bq,qhw->bhw", logit_category, ins.float())
            mask_bg = 1 - logit_mask
            logit_mask = torch.cat([mask_bg, logit_mask], dim=0)
            logit_masks.append(logit_mask)
            logit_categories.append(logit_category.sigmoid())
        logit_masks = torch.stack(logit_masks).contiguous()

        return logit_categories, logit_masks

    def compute_instance_correlation(self, corr, ins):
        H, W = corr.shape[-2:]
        N = ins.size(0)
        correlation = corr.unsqueeze(0).repeat(N, 1, 1, 1)
        ins = ins.unsqueeze(1)
        ins = F.interpolate(ins.float(), (H, W), mode='bilinear', align_corners=True)
        correlation = correlation * ins
        area = torch.sum(ins, dim=(2, 3)) + 1e-7
        prototypes = torch.sum(correlation, dim=(2, 3)) / area
        return prototypes


    def compute_mask_prototypes(self, feature, masks):
        area = torch.sum(masks, dim=(2, 3)) + 1e-7                         # N
        feature = (feature.unsqueeze(0) * masks.unsqueeze(0)).squeeze(0)   # N, C, H, W
        prototypes = torch.sum(feature, dim=(2, 3)) / area
        return prototypes

    def compute_query_point_prototype(self, feature, points, ori_imsize):
        feat = F.interpolate(feature, size=ori_imsize, mode="bilinear", align_corners=True)
        prototype = torch.cat([feat[:, :, point[1], point[0]] for point in points], dim=0)
        return prototype

    def correlation(self, query_feats, support_feats, support_mask):
        corr4, corr3, corr2 = [], [], []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), support_feat.size()[2:], mode='bilinear',
                                 align_corners=True)
            if idx < self.stack_ids[0]:
                similarity = self.MP_blocks[0](query_feat, support_feat, mask)
                corr2.append(similarity)
            elif idx < self.stack_ids[1]:
                similarity = self.MP_blocks[1](query_feat, support_feat, mask)
                corr3.append(similarity)
            else:
                similarity = self.MP_blocks[2](query_feat, support_feat, mask)
                corr4.append(similarity)
        corr4 = torch.cat(corr4, dim=1)
        corr3 = torch.cat(corr3, dim=1)
        corr2 = torch.cat(corr2, dim=1)
        return [corr4, corr3, corr2]


def weighted_dice_loss(
        prediction,
        target_seg,
        weighted_val: float = 1.0,
        reduction: str = "sum",
        eps: float = 1e-8,
):
    """
    Weighted version of Dice Loss

    Args:
        prediction: prediction
        target_seg: segmentation target
        weighted_val: values of k positives,
        reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum' : The output will be summed.
        eps: the minimum eps,
    """
    target_seg_fg = target_seg == 1
    target_seg_bg = target_seg == 0
    target_seg = torch.stack([target_seg_bg, target_seg_fg], dim=1).float()

    n, _, h, w = target_seg.shape

    prediction = prediction.reshape(-1, h, w)
    target_seg = target_seg.reshape(-1, h, w)
    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, h * w)
    target_seg = target_seg.reshape(-1, h * w)

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum() / n
    elif reduction == "mean":
        loss = loss.mean()
    return loss


class WeightedDiceLoss(nn.Module):
    def __init__(
            self,
            weighted_val: float = 1.0,
            reduction: str = "sum",
    ):
        super(WeightedDiceLoss, self).__init__()
        self.weighted_val = weighted_val
        self.reduction = reduction

    def forward(self,
                prediction,
                target_seg, ):
        return weighted_dice_loss(
            prediction,
            target_seg,
            self.weighted_val,
            self.reduction,
        )


if __name__ == "__main__":
    shot = 1
    dataset = DatasetISAID(datapath="F:/datasets/iSAID_patches", fold=0, split='val', transform=True, shot=shot)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
    model = SEMPNet('resnet50', shot)
    for batch in dataloader:
        categories, masks = model(batch['query_img'], batch['query_ins'], batch['support_imgs'], batch['support_masks'])
        loss = model.compute_objective(categories, masks, batch['query_mask'], batch['query_ins'])
        print(loss)
        out = model.generate_predict_mask(batch['query_ins'], categories)
        print(out.shape)
        print(batch['query_img'].shape,  batch['query_ins'][0].shape, batch['support_imgs'].shape, batch['support_masks'].shape)

        break

