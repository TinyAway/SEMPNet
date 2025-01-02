import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import tqdm
import os
import json
import torch.nn.functional as F
import sys
sys.path.append("..")
from segment_anything.utils.amg import rle_to_mask
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

sam_checkpoint = "F:/SAM_checkpoint/sam_vit_h_4b8939.pth"
device = "cuda"
# model_type = "default"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

mask_generator_2 = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    points_per_batch=128,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.9,
    box_nms_thresh=0.7,
    output_mode = "uncompressed_rle",
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
    min_mask_region_area=64,  # Requires open-cv to run post-processing
)

image_path = 'F:/datasets/iSAID_patches/val/images'
image_names = os.listdir(image_path)
mask_path = 'F:/datasets/iSAID_patches/val/masks'
for name in tqdm.tqdm(image_names):
    img_name = name.split('.')[0]
    masks_name = os.path.join(mask_path, img_name) + '.json'
    if not os.path.exists(masks_name):
        image = cv2.imread(os.path.join(image_path, name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks2 = mask_generator_2.generate(image)
        with open(masks_name,'w') as file_obj:
            json.dump(masks2, file_obj)