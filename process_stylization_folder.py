"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
from mimetypes import read_mime_types
import os
from ossaudiodev import control_names
import torch
from torch import nn
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from segmentation.models import ModelBuilder, SegmentationModule
from segmentation.dataset import round2nearest_multiple
from segmentation.mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from segmentation.mit_semseg.lib.utils import as_numpy, mark_volatile
from photo_wct import PhotoWCT
import process_stylization_ade20k_ssn
from torchvision import transforms
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imageio import imread
import cv2
from PIL import Image
from segmentation.mit_semseg.utils import colorEncode
from scipy.io import loadmat
import csv
import styleTransfer
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from heapq import heappush, heappop
import json

parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth')
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA.')
parser.add_argument('--save_intermediate', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False)
parser.add_argument('--no_post', action='store_true', default=True)
parser.add_argument('--folder', type=str, default='images/examples')
parser.add_argument('--beta', type=float, default=0.9999)
parser.add_argument('--content_image', help="content folder Path")
parser.add_argument('--style_image', help="style folder Path")
parser.add_argument('--cont_img_ext', type=str, default='.png')
parser.add_argument('--styl_img_ext', type=str, default='.png')
parser.add_argument('--theme', help = "clear, autumn, cloudy, snowy, sunset, summer, night", default = False)
args = parser.parse_args()

folder = args.folder

CSV_PATH = './theme.csv'
JSON_PATH = './data.json'
f = open(JSON_PATH, encoding='UTF-8')
json_data = json.loads(f.read())

cont_img_folder = args.content_image
cont_img_list = [f for f in os.listdir(cont_img_folder) if os.path.isfile(os.path.join(cont_img_folder, f))]
cont_img_list.sort()
if args.style_image:
    styl_img_folder = args.style_image
    styl_img_list = [f for f in os.listdir(styl_img_folder) if os.path.isfile(os.path.join(styl_img_folder, f))]
    styl_img_list.sort()
print(cont_img_list)
#define segmentation module
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

model = VGG19(weights='imagenet')
model = Model(inputs = model.input, outputs=model.get_layer('block5_pool').output)
        
segReMapping = process_stylization_ade20k_ssn.SegReMapping('ade20k_semantic_rel.npy')

WEIGHTS_ENCODER = os.path.join(SEG_NET_PATH, MODEL_PATH, 'encoder' + SUFFIX)
WEIGHTS_DECODER = os.path.join(SEG_NET_PATH, MODEL_PATH, 'decoder' + SUFFIX)
builder = ModelBuilder()
net_encoder = builder.build_encoder(arch=ARCH_ENCODER, fc_dim=FC_DIM, weights=WEIGHTS_ENCODER)
net_decoder = builder.build_decoder(arch=ARCH_DECODER, fc_dim=FC_DIM, num_class=NUM_CLASS, weights=WEIGHTS_DECODER, use_softmax = True)
crit =  nn.NLLLoss(ignore_index =- 1)

segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.cuda()
segmentation_module.eval()
transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# Load model
p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load(args.model))

# Load Propagator
if args.fast:
    from photo_gif import GIFSmoothing
    p_pro = GIFSmoothing(r=35, eps=0.01)
else:
    from photo_smooth import Propagator
    p_pro = Propagator(args.beta)

def segment_this_img(f, flag=False):
    img = imread(f)    
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

def getVggFeatures(file_path):
    img = load_img(file_path, target_size = (224, 224))
    img = img_to_array(img)
    tmp = np.expand_dims(img, axis=0)
    tmp = preprocess_input(tmp)
    features = model.predict(tmp)
    return features

def getDc(segImage):
    predImage = np.int32(segImage)
    pixs = predImage.size
    dc = {}
    for idx in range(150):
        dc[idx] = 0.0
    uniquesImage, countsImage = np.unique(predImage, return_counts = True)
    for idx in np.argsort(countsImage)[::-1]:
        dc[idx] = countsImage[idx] / pixs
    return dc

#target : csv file
# equation1 : before segremapping, calculate similarity Using label propotion
def getEquation(content_image, target):
    alpha = 1
    beta = 0
    # h = [] 
    eq1_list = []
    eq2_list = []
    # content = cv2.imread(content_image)
    cont_seg = segment_this_img(content_image)
    contentDc = getDc(cont_seg)
    # contentVgg = getVggFeatures(content_image)
    for t in range(target.shape[0]):
        filename = target.loc[t, 'file_name']
        # eq1 = 0.0
        # eq2 = 0.0
        # deno = 0
        # mole = 0
        ##########################
        # eq 1-1 : 기존 equation #
        ##########################
        #for idx in contentDc.keys():
        #    d = target.loc[t, str(idx)]
        #    if float(contentDc[idx]) == 0.0:
        #        deno += float(d)
        #    elif float(d) == 0.0:
        #        deno += float(contentDc[idx])
        #    else:
        #        deno += max(float(contentDc[idx]), float(d))
        #        mole += min(float(contentDc[idx]), float(d))
        #eq1 = float(mole/deno)
         
        #####################################
        # eq 1-2 : 교수님께서 말씀하신 내용 #
        #####################################
        #for idx in contentDc.keys():
        #    contentLabel = float(contentDc[idx])
        #    styleLabel = float(target.loc[t, str(idx)])
        #    if contentLabel == 0.0:
        #        continue
        #    else:
        #        deno += max(contentLabel, styleLabel)
        #        mole += min(contentLabel, styleLabel)
        #eq1 = float(mole/deno)

        #####################################################
        # eq 1-3 : 교수님께서 말씀하신 내용 +  maximum prop #
        #####################################################
        #for idx in contentDc.keys():
        #   contentLabel = min(float(contentDc[idx]), LABEL_PROP_MAX)
        #   styleLabel = min(float(target.loc[t, str(idx)]), LABEL_PROP_MAX)
        #   if contentLabel == 0.0:
        #        continue
        #   else:
        #       deno += max(contentLabel, styleLabel)
        #       mole += min(contentLabel, styleLabel)
        #eq1 = float(mole/deno)
         
        #####################################################
        # eq 1-4 : 생각을 해보니까...                       #
        #####################################################
        tmp = 0
        for idx in contentDc.keys():
            contentLabel = float(contentDc[idx])
            styleLabel = float(target.loc[t, str(idx)])
            if contentLabel < 0.01:
                continue
            else:
                tmp += (min((styleLabel/contentLabel), 1.0) * contentLabel)
        # print("filename : ",filename, "equation : ",tmp)
        eq1_list.append(tmp) 
        # eq to list
        #eq1_list.append(eq1)
        
        # content = cv2.imread(content_image)
        # style = cv2.imread(filename)
        # if content.shape[0]*content.shape[1] > style.shape[0] * style.shape[1]:
        #     style = cv2.resize(style, dsize=(content.shape[1], content.shape[0]), interpolation=cv2.INTER_CUBIC)
        # else:
        #     content = cv2.resize(content, dsize = (style.shape[1], style.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        ########################
        # eq 2 : inner product #
        ########################
        # inner = 0
        # for i in range(7):
        #     for j in range(7):
        #         for k in range(512):
        #             inner += (contentVgg[0][i][j][k] * json_data[target.loc[t, 'file_name']][0][i][j][k])
        # eq2_list.append(inner)

    # eq1_list = MinMaxScaler().fit_transform(np.array(eq1_list).reshape(-1,1))   
    # eq2_list = MinMaxScaler().fit_transform(np.array(eq2_list).reshape(-1,1))
    
    total_eq = []
    for t in range(target.shape[0]):
        filename = target.loc[t, 'file_name']
        # eq = eq1_list[t] + beta * eq2_list[t]
        total_eq.append(eq1_list[t])
        # heappush(h, [eq, filename])
        # if len(h) > NUMBER_OF_RES_IMAGE:
            # heappop(h)
    # for i in h:
        # print(i)
    return total_eq

# content_image : numpy
# target : meta file
def getEq2(content_image, target):
    # beta = 0
    eq1_list = []
    # eq2_list = []
    # content = cv2.imread(content_image)
    cont_seg = segment_this_img(content_image)
    self_cont_seg = segReMapping.self_remapping(cont_seg)
    contentDc = getDc(cont_seg)
    # contentVgg = getVggFeatures(content_image)

    for t in range(target.shape[0]):
        filename = target.loc[t, 'file_name']
        # style = cv2.imread(filename)
        styl_seg = segment_this_img(filename)
        self_styl_seg = segReMapping.self_remapping(styl_seg)
        cross_cont_seg, cross_styl_seg = segReMapping.cross_remapping(self_cont_seg, self_styl_seg)
        # get propotion
        cont_new_dc = getDc(cross_cont_seg)
        styl_new_dc = getDc(cross_styl_seg)
        tmp = 0
        for idx in cont_new_dc.keys():
            contentLabel = float(cont_new_dc[idx])
            styleLabel = float(styl_new_dc[idx])
            if contentLabel < 0.01:
                continue
            if idx in contentDc:
                tmp += (min((styleLabel/contentLabel), 1.0) * contentLabel)
            else:
                tmp += ((min((styleLabel/contentLabel), 1.0) * contentLabel) * 0.05)
        # print("filename : ",filename, "equation : ",tmp)
        eq1_list.append(tmp) 

        ########################
        # eq 2 : inner product #
        ########################
        # inner = 0
        # for i in range(7):
        #     for j in range(7):
        #         for k in range(512):
        #             inner += (contentVgg[0][i][j][k] * json_data[target.loc[t, 'file_name']][0][i][j][k])
        # eq2_list.append(inner)

    # eq1_list = MinMaxScaler().fit_transform(np.array(eq1_list).reshape(-1,1))   
    # eq2_list = MinMaxScaler().fit_transform(np.array(eq2_list).reshape(-1,1))
    total_eq = []
    filename_list = []
    for t in range(target.shape[0]):
        filename = target.loc[t, 'file_name']
        # eq = eq1_list[t] + beta * eq2_list[t]
        filename_list.append(filename)
        # total_eq.append(eq)
        total_eq.append(eq1_list[t])
        # heappush(h, [eq, filename])
        # if len(h) > NUMBER_OF_RES_IMAGE:
            # heappop(h)
    # for i in h:
        # print(i)
    return total_eq, filename_list


colors = loadmat('./segmentation/data/color150.mat')['colors']

names = {}
with open('./object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

if args.theme:
    meta = pd.read_csv(CSV_PATH)
    meta = meta[meta['label'] == args.theme]
    meta = meta.reset_index()
    for f in cont_img_list:
        outp_img_path = './results/'+f[:-4]+'_'+args.theme
        if not os.path.exists(outp_img_path):
            os.makedirs(outp_img_path)
        if not os.path.exists(os.path.join(outp_img_path, 'segmentation')):
            os.makedirs(os.path.join(outp_img_path, 'segmentation'))

        if not os.path.exists(os.path.join(outp_img_path, 'content')):
            os.makedirs(os.path.join(outp_img_path, 'content'))
        if not os.path.exists(os.path.join(outp_img_path, 'style')):
            os.makedirs(os.path.join(outp_img_path, 'style'))

        seg_path = os.path.join(outp_img_path, 'segmentation')
        outp_cont_path = os.path.join(outp_img_path, 'content')
        outp_style_path = os.path.join(outp_img_path, 'style')

        content_image_path = os.path.join(cont_img_folder, f)
        content_seg_path = os.path.join(seg_path, f[:-4]+'_seg.png')
        cont_seg = segment_this_img(content_image_path, True)
        cv2.imwrite(content_seg_path, cont_seg)
        equation_time = time.time()
        eq1 = getEquation(content_image_path, meta)
        eq2, file_list = getEq2(content_image_path, meta)
        print("Equation Time : ", time.time() - equation_time)
        eq = []
        for i in range(len(eq2)):
            print("filename : ", file_list[i], " eq1 : ", eq1[i], " eq2 : ", eq2[i])
            heappush(eq, [eq1[i] + eq2[i], file_list[i]])
            if len(eq) > 5:
                heappop(eq)
        top = NUMBER_OF_RES_IMAGE
        while eq:
            cur = heappop(eq)
            print(top, cur)
            style_image_path = cur[1]
            style_seg_path = os.path.join(seg_path,style_image_path.split('/')[-1].split('.')[-2]+'_seg.png')
            style_seg = segment_this_img(style_image_path, True)
            cv2.imwrite(style_seg_path, style_seg)
            print("Content Image : " + content_image_path)
            print("Style Image : " + style_image_path)
            output_image_path = os.path.join(outp_img_path, "Top"+str(top)+content_image_path.split('/')[-1][:-4]+'_'+style_image_path.split('/')[-1][:-4]+'.png')
            top -= 1
            print(output_image_path)
            process_stylization_ade20k_ssn.stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_image_path,
                style_image_path=style_image_path,
                content_seg_path=content_seg_path,
                style_seg_path=style_seg_path,
                output_image_path=output_image_path,
                cuda=args.cuda,
                save_intermediate=args.save_intermediate,
                no_post=args.no_post,
                label_remapping=segReMapping,
                output_visualization=False
            )
            pred_color = colorEncode(np.int32(style_seg), colors).astype(np.uint8)
            Image.fromarray(pred_color).save(style_seg_path)
            style_img = cv2.imread(style_image_path)
            cv2.imwrite(os.path.join(outp_style_path, style_image_path.split('/')[-1]), style_img)
        pred_color = colorEncode(np.int32(cont_seg), colors).astype(np.uint8)
        Image.fromarray(pred_color).save(content_seg_path)
        cont_img = cv2.imread(content_image_path)
        cv2.imwrite(os.path.join(outp_cont_path, content_image_path.split('/')[-1]), cont_img)
else:
    outp_img_path = './results/'+args.content_image.split('/')[-2]+'_'+args.style_image.split('/')[-2]
    print("###################",outp_img_path)
    if not os.path.exists(outp_img_path):
        os.makedirs(outp_img_path)
    if not os.path.exists(os.path.join(outp_img_path, 'segmentation')):
        os.makedirs(os.path.join(outp_img_path, 'segmentation'))
    
    if not os.path.exists(os.path.join(outp_img_path, 'content')):
        os.makedirs(os.path.join(outp_img_path, 'content'))
    if not os.path.exists(os.path.join(outp_img_path, 'style')):
        os.makedirs(os.path.join(outp_img_path, 'style'))

    seg_path = os.path.join(outp_img_path, 'segmentation')
    outp_cont_path = os.path.join(outp_img_path, 'content')
    outp_style_path = os.path.join(outp_img_path, 'style')
    print("###################", outp_cont_path, outp_style_path)
    for f in cont_img_list:
        content_image_path = os.path.join(cont_img_folder, f)
        content_seg_path = os.path.join(seg_path, f[:-4]+'_seg.png')
        cont_seg = segment_this_img(content_image_path, True)
        cv2.imwrite(content_seg_path, cont_seg)
    
        for sf in styl_img_list:
            style_image_path = os.path.join(styl_img_folder, sf)
            style_seg_path = os.path.join(seg_path, sf[:-4] + '_seg.png')
            style_seg = segment_this_img(style_image_path, True)
            cv2.imwrite(style_seg_path, style_seg)
            output_image_path = os.path.join(outp_img_path, f[:-4]+'_'+sf[:-4]+'.png')
            print("output_image_path :"+output_image_path)
            print("Content image: " + content_image_path )

            print("Style image: " + style_image_path )

            process_stylization_ade20k_ssn.stylization(
                stylization_module=p_wct,
                smoothing_module=p_pro,
                content_image_path=content_image_path,
                style_image_path=style_image_path,
                content_seg_path=content_seg_path,
                style_seg_path=style_seg_path,
                output_image_path=output_image_path,
                cuda=args.cuda,
                save_intermediate=args.save_intermediate,
                no_post=args.no_post,
                label_remapping=segReMapping,
                output_visualization=False
            )
            pred_color = colorEncode(np.int32(style_seg), colors).astype(np.uint8)
            Image.fromarray(pred_color).save(style_seg_path)
            style_img = cv2.imread(style_image_path)
            cv2.imwrite(os.path.join(outp_style_path, sf[:-4]+'.jpg'), style_img)
        pred_color = colorEncode(np.int32(cont_seg), colors).astype(np.uint8)
        Image.fromarray(pred_color).save(content_seg_path)
        cont_img = cv2.imread(content_image_path)
        cv2.imwrite(os.path.join(outp_cont_path, f[:-4]+'.jpg'), cont_img)
