from torch.utils.data import Dataset
import torch
import numpy as np
import PIL.Image as Image
import os
import albumentations as albu
from torch.utils.data import DataLoader
import random
import json
from segment_anything.utils.amg import rle_to_mask
from torchvision import transforms
import cv2


def mask_small_object2(mask):
    """
    :param mask: 输入mask掩码
    :return: 权重图
    """

    retval, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, retval):
        mask[labels == i] = (1.0 - np.log(stats[i][4] / 65536))
    mask[labels == 0] = 1.0
    return mask

class DatasetISAID(Dataset):
    def __init__(self, datapath, fold, split, transform, shot=1, trn_all=False):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'iSAID'
        self.fold = fold
        self.trn_all = trn_all
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.378, 0.380, 0.356), std=(0.162, 0.156, 0.152)),
        ])
        self.transform = transform
        self.shot = shot

        self.img_path = os.path.join(os.path.join(datapath, split), 'images')
        self.ann_path = os.path.join(os.path.join(datapath, split), 'semantic_png')
        self.ins_path = os.path.join(os.path.join(datapath, split), 'masks')

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, query_instances, query_point_coords, support_imgs, support_masks = self.load_frame(query_name, support_names)
        query_mask = self.get_mask(query_mask, class_sample)
        support_masks = [self.get_mask(support_mask, class_sample) for support_mask in support_masks]
        if self.transform:
            query_img, query_mask, query_instances, query_point_coords = random_transform(query_img, query_mask, query_instances, query_point_coords)

            for i, support_img in enumerate(support_imgs):
                support_imgs[i], support_masks[i], _, _ = random_transform(support_imgs[i], support_masks[i], None, None)

        query_instances = torch.from_numpy(query_instances.copy())
        query_point_coords = torch.from_numpy(query_point_coords.copy()).int().squeeze()
        support_imgs = np.stack(support_imgs, axis=0)
        support_masks = np.stack(support_masks, axis=0)

        batch = {'query_img': query_img,                  # H, W, 3
                 'query_mask': query_mask,                # H, W
                 'query_ins': query_instances,            # M, H, W
                 'query_point': query_point_coords,       # M, 2
                 'query_name': query_name,                # 1
                 'support_imgs': support_imgs,            # shot, H, W, 3
                 'support_masks': support_masks,          # shot, H, W
                 # 'support_ins': support_instances,        # [[N_1, H, W], ...,[N_shot, H, W]]
                 # 'support_points': support_point_coords,  # [[N_1, 2], ...,[N_shot, 2]]
                 'support_names': support_names,          # shot
                 'class_id': torch.tensor(class_sample)}  # 1
        return batch

    def build_class_ids(self):
        nclass_val = self.nclass // self.nfolds
        # e.g. fold0 val: 1, 2, 3, 4, 5
        class_ids_val = [self.fold * nclass_val + i for i in range(1, nclass_val + 1)]
        if not self.trn_all:
            # e.g. fold0 trn: 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
            class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]
        else:
            # training on all classes
            class_ids_trn = [x for x in range(1, self.nclass + 1)]

        # assert len(set(class_ids_trn + class_ids_val)) == self.nclass
        assert 0 not in class_ids_val
        assert 0 not in class_ids_trn

        if self.split == 'trn':
            print("Training classes:", class_ids_trn)
            return class_ids_trn
        else:
            print("Testing classes:", class_ids_val)
            return class_ids_val


    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = ('F:/datasets/iSAID_patches/%s/list/split%d.txt' % (split, fold_id))
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data[:-3], int(data.split('_')[-1])] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if not self.trn_all and fold_id == self.fold: # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)
        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            img_metadata = read_metadata(self.split, self.fold)
        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(1, self.nclass + 1):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]

        # img_metadata_classwise.keys(): [1, 2, ..., 15]
        assert 0 not in img_metadata_classwise.keys()
        assert self.nclass in img_metadata_classwise.keys()

        return img_metadata_classwise

    def sample_episode(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        if self.split == 'trn':
            query_class = random.choice(self.class_ids)
            query_name = np.random.choice(self.img_metadata_classwise[query_class], 1, replace=False)[0]
        else:
            query_name, query_class = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[query_class], 1, replace=False)[0]
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name)
            if len(support_names) == self.shot:
                break

        return query_name, support_names, query_class

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        query_instances, query_point = self.read_instance_and_points(query_name)
        support_imgs = [self.read_img(support_name) for support_name in support_names]
        support_masks = [self.read_mask(support_name) for support_name in support_names]
        # support_instances, support_points = [], []
        # for support_name in support_names:
        #     support_instance, support_point = self.read_instance_and_points(support_name)
            # support_instances.append(support_instance)
            # support_points.append(support_point)
        return query_img, query_mask, query_instances, query_point, support_imgs, support_masks

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        return np.array(Image.open(os.path.join(self.ann_path, img_name)))

    def read_img(self, img_name):
        r"""Return binary image in PIL Image"""
        image = Image.open(os.path.join(self.img_path, img_name))
        image = self.normalize(image)
        return np.array(image.permute(1, 2, 0))

    def read_instance_and_points(self, img_name):
        r"""Return binary image in PIL Image"""
        name = os.path.join(self.ins_path, img_name).replace('.png', '.json')
        instances = []
        points = []
        with open(name) as f:
            data = json.load(f)
        for item in data:
            instances.append(np.array(rle_to_mask(item['segmentation'])))
            points.append(np.array(item['point_coords']))

        return np.array(instances), np.array(points)

    def get_mask(self, mask, cls):
        mask[mask != cls] = 0
        mask[mask == cls] = 1
        return mask

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        query_img = list()
        query_mask = list()
        query_ins = list()
        query_point = list()
        query_name = list()
        support_imgs = list()
        support_masks = list()
        # support_ins = list()
        # support_points = list()
        support_names = list()
        class_id = list()

        for b in batch:
            query_img.append(b['query_img'])
            query_mask.append(b['query_mask'])
            query_ins.append(b['query_ins'])
            query_point.append(b['query_point'])
            query_name.append(b['query_name'])
            support_imgs.append(b['support_imgs'])
            support_masks.append(b['support_masks'])
            # support_ins.append(b['support_ins'])
            # support_points.append(b['support_points'])
            support_names.append(b['support_names'])
            class_id.append(b['class_id'])

        query_img = torch.from_numpy(np.stack(query_img, axis=0)).permute(0, 3, 1, 2).contiguous()
        query_mask = torch.from_numpy(np.stack(query_mask, axis=0)).contiguous()
        support_imgs = torch.from_numpy(np.stack(support_imgs, axis=0)).permute(0, 1, 4, 2, 3).contiguous()
        support_masks = torch.from_numpy(np.stack(support_masks, axis=0)).contiguous()
        class_id = torch.tensor(class_id)

        new_batch = {'query_img': query_img,               # bs, 3, H, W
                     'query_mask': query_mask,             # bs, H, W
                     'query_name': query_name,             # bs
                     'query_ins': query_ins,               # [[M_1, H, W], ...,[M_bs, H, W]]
                     'query_point': query_point,           # [[M_1, 2], ...,[M_bs, H, W]]
                     'support_imgs': support_imgs,         # bs, shot, 3, H, W
                     'support_masks': support_masks,       # bs, shot, H, W
                     # 'support_ins': support_ins,           # [[N_1_1, H, W], ..., [N_bs_shot, H, W]]
                     # 'support_points': support_points,     # [[N_1_1, 2], ..., [N_bs_shot, 2]]
                     'support_names': support_names,       # bs, shot
                     'class_id': class_id}                 # bs

        return new_batch


class DatasetWHU(Dataset):
    def __init__(self, datapath, transform, shot=1):
        self.benchmark = 'WHU'
        self.class_ids = [1]
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.434, 0.433, 0.434), std=(0.209, 0.209, 0.209)),
        ])
        self.transform = transform
        self.shot = shot

        self.img_path = os.path.join(datapath, 'image')
        self.ann_path = os.path.join(datapath, 'label')
        self.ins_path = os.path.join(datapath, 'masks')

        self.imgs_data, self.targets_data = self.build_img_data()

    def __len__(self):
        return len(self.imgs_data)

    def __getitem__(self, idx):
        query_name, support_names = self.sample_episode(idx)
        query_img, query_mask, query_instances, query_point_coords, support_imgs, support_masks = self.load_frame(query_name, support_names)
        # query_mask = self.get_mask(query_mask, class_sample)
        # support_masks = [self.get_mask(support_mask, class_sample) for support_mask in support_masks]
        if self.transform:
            query_img, query_mask, query_instances, query_point_coords = random_transform(query_img, query_mask, query_instances, query_point_coords)

            for i, support_img in enumerate(support_imgs):
                support_imgs[i], support_masks[i], _, _ = random_transform(support_imgs[i], support_masks[i], None, None)

        query_instances = torch.from_numpy(query_instances.copy())
        query_point_coords = torch.from_numpy(query_point_coords.copy()).int().squeeze()
        support_imgs = np.stack(support_imgs, axis=0)
        support_masks = np.stack(support_masks, axis=0)

        batch = {'query_img': query_img,                  # H, W, 3
                 'query_mask': query_mask,                # H, W
                 'query_ins': query_instances,            # M, H, W
                 'query_point': query_point_coords,       # M, 2
                 'query_name': query_name,                # 1
                 'support_imgs': support_imgs,            # shot, H, W, 3
                 'support_masks': support_masks,          # shot, H, W
                 # 'support_ins': support_instances,        # [[N_1, H, W], ...,[N_shot, H, W]]
                 # 'support_points': support_point_coords,  # [[N_1, 2], ...,[N_shot, 2]]
                 'support_names': support_names,
                 'class_id': 1}                  # shot
        return batch


    def build_img_data(self):
        image_names = os.listdir(self.ann_path)
        targets = []
        for image_name in image_names:
            img = cv2.imread(os.path.join(self.ann_path, image_name), 0)
            # the target area in support images should larger than 64*64
            if np.sum(img == 255) >= 4096:
                targets.append(image_name)
        return image_names, targets


    def sample_episode(self, idx):
        idx %= len(self.imgs_data)  # for testing, as n_images < 1000
        query_name = self.imgs_data[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.targets_data, 1, replace=False)
            if query_name != support_name and support_name not in support_names:
                support_names.append(support_name[0])
            if len(support_names) == self.shot:
                break

        return query_name, support_names

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        query_instances, query_point = self.read_instance_and_points(query_name)
        support_imgs = [self.read_img(support_name) for support_name in support_names]
        support_masks = [self.read_mask(support_name) for support_name in support_names]
        # support_instances, support_points = [], []
        # for support_name in support_names:
        #     support_instance, support_point = self.read_instance_and_points(support_name)
            # support_instances.append(support_instance)
            # support_points.append(support_point)
        return query_img, query_mask, query_instances, query_point, support_imgs, support_masks

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        return np.array(Image.open(os.path.join(self.ann_path, img_name)), dtype=np.uint8)

    def read_img(self, img_name):
        r"""Return binary image in PIL Image"""
        image = Image.open(os.path.join(self.img_path, img_name))
        image = self.normalize(image)
        return np.array(image.permute(1, 2, 0))

    def read_instance_and_points(self, img_name):
        r"""Return binary image in PIL Image"""
        name = os.path.join(self.ins_path, img_name).replace('.tif', '.json')
        instances = []
        points = []
        with open(name) as f:
            data = json.load(f)
        for item in data:
            instances.append(np.array(rle_to_mask(item['segmentation'])))
            points.append(np.array(item['point_coords']))

        return np.array(instances), np.array(points)

    def get_mask(self, mask, cls):
        mask[mask != cls] = 0
        mask[mask == cls] = 1
        return mask

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        query_img = list()
        query_mask = list()
        query_ins = list()
        query_point = list()
        query_name = list()
        support_imgs = list()
        support_masks = list()
        # support_ins = list()
        # support_points = list()
        support_names = list()
        class_ids = list()


        for b in batch:
            query_img.append(b['query_img'])
            query_mask.append(b['query_mask'])
            query_ins.append(b['query_ins'])
            query_point.append(b['query_point'])
            query_name.append(b['query_name'])
            support_imgs.append(b['support_imgs'])
            support_masks.append(b['support_masks'])
            # support_ins.append(b['support_ins'])
            # support_points.append(b['support_points'])
            support_names.append(b['support_names'])
            class_ids.append(b['class_id'])

        query_img = torch.from_numpy(np.stack(query_img, axis=0)).permute(0, 3, 1, 2).contiguous()
        query_mask = torch.from_numpy(np.stack(query_mask, axis=0)).contiguous()
        support_imgs = torch.from_numpy(np.stack(support_imgs, axis=0)).permute(0, 1, 4, 2, 3).contiguous()
        support_masks = torch.from_numpy(np.stack(support_masks, axis=0)).contiguous()
        class_id = torch.tensor(class_ids)

        new_batch = {'query_img': query_img,               # bs, 3, H, W
                     'query_mask': query_mask,             # bs, H, W
                     'query_name': query_name,             # bs
                     'query_ins': query_ins,               # [[M_1, H, W], ...,[M_bs, H, W]]
                     'query_point': query_point,           # [[M_1, 2], ...,[M_bs, H, W]]
                     'support_imgs': support_imgs,         # bs, shot, 3, H, W
                     'support_masks': support_masks,       # bs, shot, H, W
                     # 'support_ins': support_ins,           # [[N_1_1, H, W], ..., [N_bs_shot, H, W]]
                     # 'support_points': support_points,     # [[N_1_1, 2], ..., [N_bs_shot, 2]]
                     'support_names': support_names,       # bs, shot
                     'class_id': class_id}                 # bs


        return new_batch



def random_transform(image, mask, instances, points):
    flip = 0
    rotate = 1
    if flip > 0.5:
        image, mask, instances, points = random_flip(image, mask, instances, points)
    if rotate > 0.5:
        image, mask, instances, points = random_rotate(image, mask, instances, points)
    return image, mask, instances, points

def random_flip(image, mask, instances, points):
    height, width = mask.shape
    mode = random.choice([0, 1])
    new_image = np.flip(image, mode)
    new_mask = np.flip(mask, mode)
    if instances is not None:
        new_instances = np.flip(instances, mode + 1)
        new_points = points
        if mode == 0:
            # new_points = np.stack([[width - 1 - point[0, 0], point[0, 1]] for point in points])
            new_points[:, :, 0] = width - 1 - points[:, :, 0]
        else:
            new_points[:, :, 1] = height - 1 - points[:, :, 1]
            # new_points = np.stack([[point[0, 0], height - 1 - point[0, 1]] for point in points])
        return new_image, new_mask, new_instances, new_points
    else:
        return new_image, new_mask

def random_rotate(image, mask, instances, points):
    height, width = mask.shape
    mode = random.choice([1, 2, 3])
    new_image = np.rot90(image, mode)
    new_mask = np.rot90(mask, mode)
    if instances is not None:
        new_instances = np.rot90(instances, mode, axes=(1, 2))
        new_points = points.copy()
        if mode == 1:
            new_points[:, :, 0] = points[:, :, 1]
            new_points[:, :, 1] = width - 1 - points[:, :, 0]
        elif mode == 2:
            new_points[:, :, 0] = width - 1 - points[:, :, 0]
            new_points[:, :, 1] = width - 1 - points[:, :, 1]
        else:
            new_points[:, :, 0] = height - 1 - points[:, :, 1]
            new_points[:, :, 1] = points[:, :, 0]
        return new_image, new_mask, new_instances, new_points
    else:
        return new_image, new_mask, None, None



if __name__ == "__main__":
    transform_trn = albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        # albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=0.5, border_mode=0),
        # albu.GridDistortion(p=0.5),
        # albu.Resize(256, 256),
        albu.Normalize(mean=(0.378, 0.380, 0.356), std=(0.162, 0.156, 0.152)),  # iSAID
        # albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2(p=1),
    ])
    dataset = DatasetWHU(datapath="G:/deep learning/deeplearning.ai-master/buildings/whu_buildings/3. The cropped image tiles and raster labels/test", transform=False, shot=5)
    # dataset = DatasetISAID(datapath="F:/datasets/iSAID_patches", fold=0, split='trn', transform=True, shot=1, trn_all=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate_fn)
    for batch in dataloader:
        print(batch['query_name'][0])
        print(batch['query_img'].shape)
        print(batch['support_imgs'].shape)
        print(batch['query_point'][0].shape)
        print(batch['query_ins'][0].dtype)
        print(batch['query_mask'].dtype)
        print(batch['class_id'][0])
        break




