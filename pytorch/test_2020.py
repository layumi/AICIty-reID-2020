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
import math
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
from utils import load_network
from tqdm import tqdm
from model import ft_net, ft_net_angle, ft_net_EF4, ft_net_EF5, ft_net_EF6,ft_net_arc, ft_net_IR, ft_net_SE, ft_net_DSE, ft_net_dense, ft_net_NAS, PCB, PCB_test, CPB
from evaluate_gpu import calculate_result
from evaluate_rerank import calculate_result_rerank
#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/pytorch2020',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='save model path')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')
parser.add_argument('--inputsize', default=320, type=int, help='batchsize')
parser.add_argument('--h', default=320, type=int, help='batchsize')
parser.add_argument('--w', default=320, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use densenet121' )
parser.add_argument('--use_SE', action='store_true', help='use densenet121' )
parser.add_argument('--use_EF4', action='store_true', help='use densenet121' )
parser.add_argument('--use_EF5', action='store_true', help='use densenet121' )
parser.add_argument('--use_EF6', action='store_true', help='use densenet121' )
parser.add_argument('--use_DSE', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./data/outputs',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
opt.fp16 = config['fp16'] 
opt.PCB = config['PCB']
opt.CPB = config['CPB']
opt.inputsize = config['inputsize']
opt.stride = config['stride']
opt.angle = config['angle']
opt.use_EF4 = config['use_EF4']
opt.use_EF5 = config['use_EF5']
opt.use_EF6 = config['use_EF6']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'pool' in config:
    opt.pool = config['pool']

opt.use_dense = config['use_dense']
if 'use_NAS' in config: # compatible with early config
    opt.use_NAS = config['use_NAS']
else:
    opt.use_NAS = False

if 'use_SE' in config: # compatible with early config
    opt.use_SE = config['use_SE']
else:
    opt.use_SE = False

if 'use_DSE' in config: # compatible with early config
    opt.use_DSE = config['use_DSE']
else:
    opt.use_DSE = False

if 'use_IR' in config: # compatible with early config
    opt.use_IR = config['use_IR']
else:
    opt.use_IR = False


if 'arc' in config:
    opt.arc = config['arc']
else:
    opt.arc = False

opt.nclasses = config['nclasses']

#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

str_ids = opt.gpu_ids.split(',')
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
        transforms.Resize( (round(opt.inputsize*1.1), round(opt.inputsize*1.1)), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
else: 
    data_transforms = transforms.Compose([
        transforms.Resize( (round(opt.h*1.1), round(opt.w*1.1)), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print(opt.h)


data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=20) for x in ['gallery','query','multi-query']}
else:
    image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=20) for x in ['gallery','query']}
class_names = image_datasets['query'].classes
use_gpu = torch.cuda.is_available()

######################################################################
# Load model
#---------------------------
'''
def load_network(network):
    which_epoch = opt.which_epoch
    if which_epoch.isdigit():
        save_path = os.path.join('./data/outputs',name,'net_%03d.pth'%int(which_epoch))
    else:
        save_path = os.path.join('./data/outputs',name,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network
'''
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
        if opt.use_dense:
            ff = torch.FloatTensor(n,512).zero_().cuda()
        elif opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts
        elif opt.CPB:
            ff = torch.FloatTensor(n,512,4).zero_().cuda() # we have three parts
        else:
            ff = torch.FloatTensor(n,512).zero_().cuda()
        ff = torch.FloatTensor(n,512).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                outputs = model(input_img)
                if opt.CPB:
                    #outputs = torch.cat( (outputs[0], outputs[1], outputs[2], outputs[3]) ,dim=-1) 
                    #outputs = outputs.view(n, 512, 4)
                    outputs = outputs[1]
                
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        elif opt.CPB:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(4)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
        #print(ff.shape)
        features = torch.cat((features,ff.data.cpu().float()), 0)
    return features

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        filename = os.path.basename(path)
        # Test Gallery Image
        if not 'c' in filename: 
            labels.append(9999999)
            camera_id.append(9999999)
        else:
            label = filename[0:6]
            camera = filename.split('c')[1]
            if label[0:2]=='-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0:3]))
        #print(camera[0:3])
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)

if opt.multi:
    mquery_path = image_datasets['multi-query'].imgs
    mquery_cam,mquery_label = get_id(mquery_path)

######################################################################
# Load Collected data Trained model
print('-------test-----------')
model, _, epoch = load_network(opt.name, opt)
'''
if opt.use_dense:
    model_structure = ft_net_dense(opt.nclasses, stride=opt.stride, pool = opt.pool)
elif opt.use_NAS:
    model_structure = ft_net_NAS(opt.nclasses, stride=opt.stride)
elif opt.use_SE:
    model_structure = ft_net_SE(opt.nclasses, stride=opt.stride, pool = opt.pool)
elif opt.use_DSE:
    model_structure = ft_net_DSE(opt.nclasses, stride=opt.stride, pool = opt.pool)
elif opt.use_IR:
    model_structure = ft_net_IR(opt.nclasses, stride=opt.stride)
elif opt.use_EF4:
    model_structure = ft_net_EF4(opt.nclasses)
elif opt.use_EF5:
    model_structure = ft_net_EF5(opt.nclasses)
elif opt.use_EF6:
    model_structure = ft_net_EF6(opt.nclasses)
elif opt.PCB:
    model_structure = PCB(opt.nclasses)
elif opt.CPB:
    model_structure = CPB(opt.nclasses)
elif opt.angle:
    model_structure = ft_net_angle(opt.nclasses, stride=opt.stride)
elif opt.arc:
    model_structure = ft_net_arc(opt.nclasses, stride=opt.stride)
else:
    model_structure = ft_net(opt.nclasses, stride=opt.stride, pool = opt.pool)

#if opt.fp16:
#    model_structure = network_to_half(model_structure)

model = load_network(model_structure)
print(model)
'''
# Remove the final fc layer and classifier layer
if opt.PCB:
    model = PCB_test(model)
elif opt.CPB:
    #model.classifier0 = nn.Sequential()
    #model.classifier1 = nn.Sequential()
    #model.classifier2 = nn.Sequential()
    #model.classifier3 = nn.Sequential()
    model.classifier0.classifier = nn.Sequential()
    model.classifier1.classifier = nn.Sequential()
    model.classifier2.classifier = nn.Sequential()
    model.classifier3.classifier = nn.Sequential()
    #model[1].model.fc = nn.Sequential()
    #model[1].classifier.classifier = nn.Sequential()
else:
    model.classifier.classifier = nn.Sequential()

# Change to test mode
model = model.eval()
if use_gpu:
    model = model.cuda()
    model = torch.nn.DataParallel(model)

gallery_label = np.asarray(gallery_label)
query_label = np.asarray(query_label)
gallery_cam = np.asarray(gallery_cam)
query_cam = np.asarray(query_cam)
print('Gallery Size: %d'%len(gallery_label))
print('Query Size: %d'%len(query_label))
# Extract feature
with torch.no_grad():
    gallery_feature = extract_feature(model,dataloaders['gallery'])
    query_feature = extract_feature(model,dataloaders['query'])

# Save to Matlab for check
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

result_file = './data/outputs/%s/result.txt'%opt.name
calculate_result( gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, result_file)
#calculate_result_rerank( gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, result_file)
