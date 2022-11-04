import os
import cv2
import csv
import numpy as np
import pandas as pd
import torch
from torch import nn
from torchvision import transforms
from scipy.io import loadmat
from heapq import heappush, heappop

from segmentation.models import ModelBuilder, SegmentationModule
from segmentation.dataset import round2nearest_multiple
from segmentation.mit_semseg.lib.nn import async_copy_to
from segmentation.mit_semseg.lib.utils import as_numpy
import process_stylization_ade20k_ssn
from photo_gif import GIFSmoothing
from photo_wct import PhotoWCT

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
# model = VGG19(weights='imagenet')
# model = Model(inputs = model.input, outputs=model.get_layer('block5_pool').output)

p_wct = PhotoWCT()
p_wct.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))

p_pro = GIFSmoothing(r=35, eps=0.01)

CSV_PATH = './theme.csv'

colors = loadmat('./segmentation/data/color150.mat')['colors']

names = {}
with open('./object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def segment_this_img(f):
    # img = imread(f)    
    # img = img[:, :, ::-1]  # BGR to RGB!!!
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

    total_eq = []
    filename_list = []
    for t in range(target.shape[0]):
        filename = target.loc[t, 'file_name']
        total_eq.append(eq1_list[t])
        filename_list.append(filename)
  
    return total_eq, filename_list

def styleTransfer(content_image_path, theme, NUMBER_OF_RES_IMAGE):
    # Load Database Information
    meta = pd.read_csv(CSV_PATH)
    meta = meta[meta['label'] == theme]
    meta = meta.reset_index()

    # get Content Image list
    print("Contentn Image List : ", content_image_path)
    outp_img_path = './results/'+content_image_path.split("/")[-1][:-4]+'_'+theme

    if not os.path.exists(outp_img_path):
        os.makedirs(outp_img_path)
    if not os.path.exists(os.path.join(outp_img_path, 'segmentation')):
        os.makedirs(os.path.join(outp_img_path, 'segmentation'))

    # if not os.path.exists(os.path.join(outp_img_path, 'content')):
    #     os.makedirs(os.path.join(outp_img_path, 'content'))
    # if not os.path.exists(os.path.join(outp_img_path, 'style')):
    #     os.makedirs(os.path.join(outp_img_path, 'style'))

    seg_path = os.path.join(outp_img_path, 'segmentation')

    content_seg_path = os.path.join(seg_path, content_image_path.split("/")[-1][:-4]+'_seg.png')
    cont_seg = segment_this_img(content_image_path)
    cv2.imwrite(content_seg_path, cont_seg)
    eq1, file_list = getEquation(content_image_path, meta)

    # Select Top NUMBER_OF_RES_IMAGE style image
    eq = []
    for i in range(len(eq1)):
        print("filename : ", file_list[i], "eq1 : ", eq1[i])
        heappush(eq, [eq1[i], file_list[i]])
        if len(eq) > NUMBER_OF_RES_IMAGE:
            heappop(eq)
    
    top = NUMBER_OF_RES_IMAGE

    output = []

    while eq:
        cur = heappop(eq)
        style_image_path = cur[1]
        style_seg_path = os.path.join(seg_path,style_image_path.split('/')[-1].split('.')[-2]+'_seg.png')
        style_seg = segment_this_img(style_image_path)
        cv2.imwrite(style_seg_path, style_seg)
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
            cuda=True,
            save_intermediate=False,
            no_post=True,
            label_remapping=segReMapping,
            output_visualization=False
        )
        output.append(output_image_path)
    return output
