# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import math
import os
import scipy.io
import yaml
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from model import ft_net, ft_net_angle, ft_net_dense, ft_net_NAS, PCB, PCB_test, CPB
from evaluate_gpu import calculate_result
from evaluate_rerank import calculate_result_rerank
from re_ranking import re_ranking, re_ranking_one
from utils import load_network
from losses import L2Normalization
from shutil import copyfile
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--which_epoch',default='59', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test_data',type=str, help='./test_data')
parser.add_argument('--crop_dir',default='./data/cropped_aicity',type=str, help='./test_data')
parser.add_argument('--names', default='ft_ResNet50,xxxx,xxxxx', type=str, help='save model path')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')
parser.add_argument('--inputsize', default=320, type=int, help='batchsize')
parser.add_argument('--h', default=384, type=int, help='batchsize')
parser.add_argument('--w', default=384, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--CPB', action='store_true', help='use CPB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--pool',default='avg', type=str, help='last pool')
parser.add_argument('--k1', default=50, type=int, help='batchsize')
parser.add_argument('--k2', default=15, type=int, help='batchsize')
parser.add_argument('--lam', default=0.1, type=float, help='batchsize')
parser.add_argument('--dba', default=10, type=int, help='batchsize')

opt = parser.parse_args()
str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
if opt.h == opt.w:
    data_transforms = transforms.Compose([
        transforms.Resize( ( round(opt.inputsize*1.1), round(opt.inputsize*1.1)), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else:
    data_transforms = transforms.Compose([
        transforms.Resize( (round(opt.h*1.1), round(opt.w*1.1)), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384,192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])


data_dir = test_dir

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}
cropped_image_datasets = {x: datasets.ImageFolder( os.path.join(opt.crop_dir,x) ,data_transforms) for x in ['gallery','query']}
cropped_dataloaders = {x: torch.utils.data.DataLoader(cropped_image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        #print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                outputs = model(input_img)
                ff += outputs

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        #print(ff.shape)
        features = torch.cat((features,ff.data.cpu().float()), 0)
    return features

def extract_cam(model, dataloaders):
    cams = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        input_img = Variable(img.cuda())
        ff = torch.FloatTensor(n,512).zero_().cuda()
        for scale in ms:
            if scale != 1:
                 input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
            ff += model(input_img)

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        #outputs = nn.functional.softmax(outputs, dim=1) 
        cams = torch.cat((cams, ff.data.cpu().float()), 0)          
    return cams

def predict_cam(model, dataloaders):
    cams = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        input_img = Variable(img.cuda())
        #for scale in ms:
        #    if scale != 1:
        #         input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
        outputs = model(input_img)
        outputs = nn.functional.softmax(outputs)
        cams = torch.cat((cams, outputs.data.cpu().float()), 0)
    return cams


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs


######################################################################

names = opt.names.split(',')
models = nn.ModuleList()

for name in names:
    model_tmp, _, epoch = load_network(name, opt)
    model_tmp.classifier.classifier = nn.Sequential()
    model_tmp = torch.nn.DataParallel(model_tmp)
    models.append(model_tmp.cuda().eval())

# Extract feature\
snapshot_feature_mat = './feature/submit_result_%s.mat'%opt.names
print('Feature Output Path: %s'%snapshot_feature_mat)
if not os.path.isfile(snapshot_feature_mat):
    with torch.no_grad():
        gallery_feature, query_feature = torch.FloatTensor(), torch.FloatTensor()
        for model in models:
            q_f = extract_feature(model,dataloaders['query']) 
            q_f_crop = extract_feature(model,cropped_dataloaders['query']) 
            q_f = q_f + q_f_crop
            qnorm = torch.norm(q_f, p=2, dim=1, keepdim=True)
            q_f = q_f.div(qnorm.expand_as(q_f)) / np.sqrt(len(names))

            g_f = extract_feature(model,dataloaders['gallery']) 
            g_f_crop = extract_feature(model,cropped_dataloaders['gallery']) 
            g_f = g_f + g_f_crop
            gnorm = torch.norm(g_f, p=2, dim=1, keepdim=True)
            g_f = g_f.div(gnorm.expand_as(g_f)) / np.sqrt(len(names))            

            gallery_feature = torch.cat((gallery_feature,g_f), 1)
            query_feature = torch.cat((query_feature,q_f), 1)

    result = {'gallery_f':gallery_feature.numpy(),'query_f':query_feature.numpy()}
    scipy.io.savemat(snapshot_feature_mat,result)
else:
    result = scipy.io.loadmat(snapshot_feature_mat)
    query_feature = torch.FloatTensor(result['query_f']).cuda()
    gallery_feature = torch.FloatTensor(result['gallery_f']).cuda()
    print(np.where(np.isnan(gallery_feature.cpu().numpy())))

