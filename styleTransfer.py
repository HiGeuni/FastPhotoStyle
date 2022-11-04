"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from __future__ import print_function
import argparse
import os
import torch
import process_stylization_ade20k_ssn
from torch import nn
from photo_wct import PhotoWCT
from segmentation.dataset import round2nearest_multiple 
from segmentation.models import ModelBuilder, SegmentationModule
from segmentation.mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from segmentation.mit_semseg.lib.utils import as_numpy, mark_volatile
from skimage.metrics import structural_similarity as ssim
# from scipy.misc import imread, imresize
from imageio import imread
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from segmentation.mit_semseg.utils import colorEncode
from scipy.io import loadmat
import csv
import time

start = time.time()
torch.cuda.set_device(0)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def segment_this_img(f, flag=False):
    img = imread(f)
    
    img = img[:, :, ::-1]  # BGR to RGB!!!
    if flag:
        cv2.imwrite('./abc.png',img)
    ori_height, ori_width, _ = img.shape
    img_resized_list = []
    for this_short_size in args.imgSize:
        scale = min(this_short_size / float(min(ori_height, ori_width)), args.imgMaxSize / float(max(ori_height, ori_width)))
        target_height, target_width = int(ori_height * scale), int(ori_width * scale)
        target_height = round2nearest_multiple(target_height, args.padding_constant)
        target_width = round2nearest_multiple(target_width, args.padding_constant)
        
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
        pred = torch.zeros(1, args.num_class, segSize[0], segSize[1])
        pred = async_copy_to(pred,args.gpu_id)
        for timg in img_resized_list:
            feed_dict = input_.copy()
            #feed_dict = img.copy()
            feed_dict['img_data'] = timg.cuda()
            del feed_dict['img_ori']
            feed_dict = async_copy_to(feed_dict, args.gpu_id)
            
            # forward pass
            pred_tmp = segmentation_module(feed_dict, segSize=segSize)
            pred = pred + pred_tmp / len(args.imgSize)
        _, preds = torch.max(pred, dim=1)
        preds = as_numpy(preds.squeeze(0).cpu())
    return preds

def getDc(PATH):
    segImage = segment_this_img(PATH)
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
def getEquation(content_image, target):
    alpha = 1
    beta = 0.5
    
    comp = 0
    resv = ''
    contentDc = getDc(content_image)
    equation = [0.0, 0.0]
    for t in range(target.shape[0]):
        eq1 = 0.0
        eq2 = 0.0
        deno = 0
        mole = 0
        for idx in contentDc.keys():
            d = target.loc[t, str(idx)]
            if float(contentDc[idx]) == 0.0:
                deno += float(d)
            elif float(d) == 0.0:
                deno += float(contentDc[idx])
            else:
                deno += max(float(contentDc[idx]), float(d))
                mole += min(float(contentDc[idx]), float(d))
        eq1 = float(mole/deno)
        content = cv2.imread(content_image)
        style = cv2.imread(target.loc[t, 'file_name'])
        if content.shape[0]*content.shape[1] > style.shape[0] * style.shape[1]:
            style = cv2.resize(style, dsize=(content.shape[1], content.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            content = cv2.resize(content, dsize = (style.shape[1], style.shape[0]), interpolation=cv2.INTER_CUBIC)

        grayContent = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        grayStyle = cv2.cvtColor(style, cv2.COLOR_BGR2GRAY)
        (score, diff) = ssim(grayContent, grayStyle, full=True)
        eq2 = score
        eq = alpha*eq1 + beta*eq2
        if eq > comp:
            equation[0] = eq1
            equation[1] = eq2
            comp = eq
            resv = target.loc[t, 'file_name']
    return resv, equation


#Complete
def getEquationTwo(content_image, style_image):
    contentDc = getDc(content_image)
    styleDc = getDc(style_image) 
    deno = 0
    mole = 0
    for idx in range(150):
        if contentDc[idx] == 0.0:
            deno += styleDc[idx]
        elif styleDc[idx] == 0.0:
            deno += contentDc[idx]
        else:
            deno += max(contentDc[idx], styleDc[idx]) 
            mole += min(contentDc[idx], styleDc[idx])

    eq1 = float(mole/deno)
    content = cv2.imread(content_image)
    style = cv2.imread(style_image)

    if content.shape[0] * content.shape[1] > style.shape[0] * style.shape[1]:
        style = cv2.resize(style, dsize = (content.shape[1], content.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        content = cv2.resize(content, dsize = (style.shape[1], style.shape[0]), interpolation=cv2.INTER_CUBIC)

    grayContent = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    grayStyle = cv2.cvtColor(style, cv2.COLOR_BGR2GRAY)
    (score, diff) = ssim(grayContent, grayStyle, full = True)
    eq2 = score
    print(eq1, eq2)

if __name__ == "__main__":
    CFG_PATH='./segmentation/config/ade20k_hrnetv2.yaml'
    parser = argparse.ArgumentParser(description='Photorealistic Image Stylization')
    parser.add_argument('--model_path', help='folder to model path', default='ckpt/ade20k-hrnetv2-c1')
    parser.add_argument('--suffix', default='_epoch_30.pth', help="which snapshot to load")
    parser.add_argument('--arch_encoder', default='hrnetv2', help="architecture of net_encoder")
    parser.add_argument('--arch_decoder', default='c1', help="architecture of net_decoder")
    parser.add_argument('--fc_dim', default=2048, type=int, help='number of features between encoder and decoder')


    parser.add_argument('--num_val', default=-1, type=int, help='number of images to evalutate')
    parser.add_argument('--num_class', default=150, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='batchsize. current only supports 1')

    parser.add_argument('--imgSize', default=[300, 400, 500, 600], nargs='+', type=int, help='list of input image sizes.' 'for multiscale testing, e.g. 300 400 500')

    parser.add_argument('--imgMaxSize', default=1000, type=int, help='maximum input image size of long edge')

    parser.add_argument('--padding_constant', default=32, type=int, help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=4, type=int, help='downsampling rate of the segmentation label')

    parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id for evaluation')

    parser.add_argument('--model', default='./PhotoWCTModels/photo_wct.pth', help='Path to the PhotoWCT model. These are provided by the PhotoWCT submodule, please use `git submodule update --init --recursive` to pull.')
    parser.add_argument('--content_image_path', default="./images/content3.png")
    parser.add_argument('--content_seg_path', default='./results/content3_seg.png')
    parser.add_argument('--style_image_path')
    parser.add_argument('--style_seg_path', default='./results/style3_seg.png')
    parser.add_argument('--output_image_path', default='./results/example3.png')
    parser.add_argument('--save_intermediate', action='store_true', default=False)
    parser.add_argument('--fast', action='store_true', default=False)
    parser.add_argument('--no_post', action='store_true', default=False)
    parser.add_argument('--output_visualization', action='store_true', default=False)
    parser.add_argument('--cuda', type=int, default=1, help='Enable CUDA.')
    parser.add_argument('--label_mapping', type=str, default='ade20k_semantic_rel.npy')
    parser.add_argument('--csv')
    args = parser.parse_args()

    segReMapping = process_stylization_ade20k_ssn.SegReMapping(args.label_mapping)

    # Absolute paths of segmentation model weights
    SEG_NET_PATH = 'segmentation'
    args.weights_encoder = os.path.join(SEG_NET_PATH,args.model_path, 'encoder' + args.suffix)
    args.weights_decoder = os.path.join(SEG_NET_PATH,args.model_path, 'decoder' + args.suffix)
    args.arch_encoder = 'hrnetv2'
    args.arch_decoder = 'c1'
    args.fc_dim = 720
    args.imgSize=[300, 375, 450, 525, 600]
    args.imgMaxSize = 1000

    # tmp
    # Load semantic segmentation network module
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(arch=args.arch_encoder, fc_dim=args.fc_dim, weights=args.weights_encoder)
    net_decoder = builder.build_decoder(arch=args.arch_decoder, fc_dim=args.fc_dim, num_class=args.num_class, weights=args.weights_decoder, use_softmax=True)
    crit = nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
    segmentation_module.cuda()
    segmentation_module.eval()
    #transform = transforms.Compose([transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])])
    transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # Load FastPhotoStyle model
    p_wct = PhotoWCT()
    p_wct.load_state_dict(torch.load(args.model))
    if args.fast:
        from photo_gif import GIFSmoothing
        p_pro = GIFSmoothing(r=35, eps=0.001)
    else:
        from photo_smooth import Propagator
        p_pro = Propagator()
    if args.cuda:
        p_wct.cuda(0)

    # SELECT STYLE IMAGE(BY EQUATION)
    # ExtractLabel -> extractImage : Dictionary 불러오기
    # up-style_transfer2-getEquation -> style이미지 찾기
    # 그 이미지로 style transfer 돌리기
    if args.style_image_path:
        getEquationTwo(args.content_image_path, args.style_image_path)
        styleImage = args.style_image_path
    else:
        meta = pd.read_csv(args.csv)
        resv, eq = getEquation(args.content_image_path, meta)
        styleImage = resv
        print(styleImage, eq)

    colors = loadmat('./segmentation/data/color150.mat')['colors']
    names = {}
    with open('./object150_info.csv') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            names[int(row[0])] = row[5].split(";")[0]


    args.output_image_path = './results/'+args.content_image_path.split('/')[-1][:-4]+'-'+styleImage.split('/')[-1][:-4]+'.png'

    cont_seg = segment_this_img(args.content_image_path, True)
    pred = np.int32(cont_seg)

    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions")
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print(" {}: {:.2f}%".format(name, ratio))

    pred_color = colorEncode(pred, colors).astype(np.uint8)
    Image.fromarray(pred_color).save(os.path.join('./', 'tmp_seg.png'))

    cv2.imwrite(args.content_seg_path, cont_seg)
    #style_seg = segment_this_img(args.style_image_path)
    style_seg = segment_this_img(styleImage)
    cv2.imwrite(args.style_seg_path, style_seg)

    process_stylization_ade20k_ssn.stylization(
        stylization_module=p_wct,
        smoothing_module=p_pro,
        content_image_path=args.content_image_path,
        #style_image_path=args.style_image_path,
        style_image_path=styleImage,
        content_seg_path=args.content_seg_path,
        style_seg_path=args.style_seg_path,
        output_image_path=args.output_image_path,
        cuda=True,
        save_intermediate=args.save_intermediate,
        no_post=args.no_post,
        label_remapping=segReMapping,
        output_visualization=args.output_visualization
    )
    end = time.time()
    print(f"{end-start:.5f} sec")
