# SEMPNet
Official implementation of SAM enhanced mask parsing network.

## Data preparateion
1. Downloading datatset at [iSAID](https://captain-whu.github.io/iSAID/dataset.html)
2. Spliting images to 515*512 following [iSAID_Devkit](https://github.com/CAPTAIN-WHU/iSAID_Devkit)

## Mask generation
1. Downloading SAM and its checkpoint from [Segmetn anything](https://github.com/facebookresearch/segment-anything)
2. Generating masks using mask_generating.py.

## Train the model
`python train.py`
