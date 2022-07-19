import os
import glob
import cv2
import numpy as np

import torch
from torch import nn
from torchvision import transforms

import argparse
from segmentation.models import ModelBuilder, SegmentationModule
from segmentation.dataset import round2nearest_multiple
from segmentation.mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from segmentation.mit_semseg.lib.utils import as_numpy, mark_volatile
from scipy.io import loadmat

# parameter
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--dir', default = False)
args = parser.parse_args()

colors = loadmat('segmentation/data/color150.mat')['colors']

SEG_NET_PATH = 'segmentation'
MODEL_PATH = 'ckpt/ade20k-hrnetv2-c1'
SUFFIX = '_epoch_30.pth'
WEIGHTS_ENCODER = os.path.join(SEG_NET_PATH, MODEL_PATH, 'encoder' + SUFFIX)
WEIGHTS_DECODER = os.path.join(SEG_NET_PATH, MODEL_PATH, 'decoder' + SUFFIX)
ARCH_ENCODER = 'hrnetv2'
ARCH_DECODER = 'c1'
FC_DIM = 720
IMG_SIZE = [300, 375, 450, 525, 600]
IMG_MAX_SIZE = 1000
NUM_CLASS=150
PADDING_CONSTANT = 32
SEQM_DOWNSAMPLING_RATE = 4
GPU_ID = 0
LABEL_PROP_MAX = 0.1
NUMBER_OF_RES_IMAGE = 5 

builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=ARCH_ENCODER, fc_dim=FC_DIM, weights=WEIGHTS_ENCODER)
net_decoder = builder.build_decoder(arch=ARCH_DECODER, fc_dim=FC_DIM, num_class=NUM_CLASS, weights=WEIGHTS_DECODER, use_softmax = True)

crit =  nn.NLLLoss(ignore_index =- 1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()

transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

def visualize_result(label_map):
    label_map = label_map.astype('int')
    label_map_rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    for label in np.unique(label_map):
        label_map_rgb += (label_map == label)[:, :, np.newaxis] * \
            np.tile(colors[label],(label_map.shape[0], label_map.shape[1], 1))
    return label_map_rgb

def segment_this_img(f, flag=False):
    img = cv2.imread(f)    
    img = img[:, :, ::-1]  # BGR to RGB!!!
    if flag:
        cv2.imwrite('./abc.png',img)
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    for this_short_size in IMG_SIZE:
        scale = min(this_short_size / float(min(ori_height, ori_width)), IMG_MAX_SIZE / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, PADDING_CONSTANT)
        target_width = round2nearest_multiple(target_width, PADDING_CONSTANT)
        
        img_resized = cv2.resize(img.copy(), (target_width, target_height))
        img_resized = img_resized.astype(np.float32)
        img_resized = np.float32(np.array(img)) / 255.
        img_resized = img_resized.transpose((2, 0, 1))
        img_resized = transform(torch.from_numpy(img_resized))

        img_resized = torch.unsqueeze(img_resized, 0)
        img_resized_list.append(img_resized)
    input_ = dict()
    input_['img_ori'] = img.copy()
    input_['img_data'] = [x.contiguous() for x in img_resized_list]
    segSize = (img.shape[0],img.shape[1])
    with torch.no_grad():
        pred = torch.zeros(1, NUM_CLASS, segSize[0], segSize[1])
        pred = async_copy_to(pred,GPU_ID)
        for timg in img_resized_list:
            feed_dict = input_.copy()
            #feed_dict = img.copy()
            feed_dict['img_data'] = timg.cuda()
            del feed_dict['img_ori']
            feed_dict = async_copy_to(feed_dict, GPU_ID)
            
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp / len(IMG_SIZE)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0).cpu())
    return preds

if __name__ == "__main__":
    pat = glob.glob(args.dir+"/*")
    for i in pat:
        seg_img = segment_this_img(i)
        cv2.imwrite(i[:-4] +"_seg.png", seg_img)
        seg_img = visualize_result(seg_img)
        cv2.imwrite(i[:-4] +"_seg_visualize.png", seg_img)
