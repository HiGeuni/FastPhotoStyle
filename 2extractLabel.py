"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import os
import torch
from torch import nn
from segmentation.dataset import round2nearest_multiple 
from segmentation.models import ModelBuilder, SegmentationModule
from segmentation.mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from segmentation.mit_semseg.lib.utils import as_numpy, mark_volatile
from skimage.metrics import structural_similarity as ssim
from imageio import imread
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
from segmentation.mit_semseg.utils import colorEncode
from scipy.io import loadmat
import csv
import glob
torch.cuda.set_device(0)

CFG_PATH='./segmentation/config/ade20k_hrnetv2.yaml'
parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--img')
parser.add_argument('--csv')
parser.add_argument('--dir', help="directory to csv file")
parser.add_argument('--num', default = 0)
args = parser.parse_args()

# Absolute paths of segmentation model weights
SEG_NET_PATH = 'segmentation'
SUFFIX='_epoch_30.pth'
MODEL_PATH='ckpt/ade20k-hrnetv2-c1'
weights_encoder = os.path.join(SEG_NET_PATH, MODEL_PATH, 'encoder' + SUFFIX)
weights_decoder = os.path.join(SEG_NET_PATH, MODEL_PATH, 'decoder' + SUFFIX)
arch_encoder = 'hrnetv2'
arch_decoder = 'c1'
fc_dim = 720
imgSize=[300, 375, 450, 525, 600]
imgMaxSize = 1000
num_class = 150
gpu_id = 0
padding_constant = 32


# tmp
# Load semantic segmentation network module
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=arch_encoder, fc_dim=fc_dim, weights=weights_encoder)
net_decoder = builder.build_decoder(arch=arch_decoder, fc_dim=fc_dim, num_class=num_class, weights=weights_decoder, use_softmax=True)
crit = nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
#transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#dictionary define
colors = loadmat('./segmentation/data/color150.mat')['colors']
names = {}
with open('./object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def segment_this_img(f, flag=False):
    img = imread(f)
    
    img = img[:, :, ::-1]  # BGR to RGB!!!
    if flag:
        cv2.imwrite('./abc.png',img)
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    for this_short_size in imgSize:
        scale = min(this_short_size / float(min(ori_height, ori_width)), imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, padding_constant)
        target_width = round2nearest_multiple(target_width, padding_constant)
        
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
        pred = torch.zeros(1, num_class, segSize[0], segSize[1])
        pred = async_copy_to(pred,gpu_id)
        for timg in img_resized_list:
            feed_dict = input_.copy()
            #feed_dict = img.copy()
            feed_dict['img_data'] = timg.cuda()
            del feed_dict['img_ori']
            feed_dict = async_copy_to(feed_dict, gpu_id)
            
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp / len(imgSize)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0).cpu())
    return preds

def writeCSV(lists):
    print(lists)
    CSV_FILE_PATH = args.csv
    csvFileRead = open(CSV_FILE_PATH, "r")
    
    rdr = csv.reader(csvFileRead)
    csvFileWrite = open(CSV_FILE_PATH, "a", newline = '')
    wr = csv.writer(csvFileWrite)
    wr.writerow(lists)

def getDc(PATH):
    segImage = segment_this_img(PATH)
    predImage = np.int32(segImage)
    pixs = predImage.size
    dc = {}
    for idx in range(150):
        dc[idx] = 0.0
    uniquesImage, countsImage = np.unique(predImage, return_counts = True)
    for idx in np.argsort(countsImage)[::-1]:
        dc[idx] = round(countsImage[idx] / pixs, 5)
    return dc

def extractImage(FILE_PATH):
    imageDc = getDc(FILE_PATH)
    tmp = [FILE_PATH]
    for idx in range(150):
        tmp.append(imageDc[idx])
    return tmp

def extractDir(DIR_PATH):
    DIR_PATH = DIR_PATH + "/*"
    file_list = glob.glob(DIR_PATH)
    num = int(args.num)
    while num <= len(file_list):
    #    if os.path.getsize(FILE_PATH) > 400000:
    #        continue
        print(num, file_list[num])
        tmp = extractImage(file_list[num])
        num += 1
        writeCSV(tmp)


if args.img:
    print(extractImage(args.img))
else:
    extractDir(args.dir)
