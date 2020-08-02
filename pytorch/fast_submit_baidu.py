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
import pickle
import PIL
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
parser.add_argument('--which_epoch',default='59', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='./data/test_data',type=str, help='./test_data')
parser.add_argument('--crop_dir',default='./data/cropped_aicity',type=str, help='./test_data')
parser.add_argument('--names', default='ft_ResNet50,xxxx,xxxxx', type=str, help='save model path')
parser.add_argument('--batchsize', default=100, type=int, help='batchsize')
parser.add_argument('--inputsize', default=384, type=int, help='batchsize')
parser.add_argument('--h', default=384, type=int, help='batchsize')
parser.add_argument('--w', default=384, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use densenet121' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--CPB', action='store_true', help='use CPB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--pool',default='avg', type=str, help='last pool')
parser.add_argument('--k1', default=70, type=int, help='batchsize')
parser.add_argument('--k2', default=10, type=int, help='batchsize')
parser.add_argument('--lam', default=0.2, type=float, help='batchsize')
parser.add_argument('--dba', default=0, type=int, help='batchsize')

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

        cams = torch.cat((cams, outputs.data.cpu().float()), 0)
    return cams

def load_pickle(filename):
    fr=open(filename,'rb')
    try:
        data = pickle.load(fr, encoding='latin1')
    except:
        data = pickle.load(fr)
    index = 0
    for name, f in data.items():
        if index == 0:
            feature = torch.zeros( len(data), len(f))
        feature[int(name[:-4])-1,:] = torch.FloatTensor(f)
        index +=1
    feature = L2Normalization(feature, dim=1)
    return feature

def load_attribute(filename):
    fr=open(filename,'rb')
    data = pickle.load(fr, encoding='latin1')
    index = 0
    direction_total = np.ndarray( len(data))
    color_total = np.ndarray( len(data))
    vtype_total = np.ndarray( len(data))
    for name, value in data.items():
        direction_total[int(name[:-4])-1] = value[0]
        color_total[int(name[:-4])-1] = value[1]
        vtype_total[int(name[:-4])-1] = value[2]
    return vtype_total

def get_shape(path):
    shape_total = np.zeros(len(path))
    count = 0
    for name, label in path:
        img = np.asarray(PIL.Image.open(name))
        shape_total[count] = img.shape[0] * img.shape[1]
        count += 1
    return shape_total


gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

query_shape = get_shape(query_path)
gallery_shape = get_shape(gallery_path)

#with open('q_g_direct_sim.pkl','rb') as fid:
#    q_g_direction_sim = pickle.load(fid)

with open('pkl_feas/q_g_direct_sim_track.pkl','rb') as fid:
    q_g_direction_sim = pickle.load(fid)

with open('pkl_feas/q_q_direct_sim.pkl','rb') as fid:
    q_q_direction_sim = pickle.load(fid)

#with open('pkl_feas/g_g_direct_sim.pkl','rb') as fid:
#    g_g_direction_sim = pickle.load(fid)
with open('pkl_feas/g_g_direct_sim_track.pkl','rb') as fid:
    g_g_direction_sim = pickle.load(fid)

######################################################################
# Extract feature
result = scipy.io.loadmat('feature/submit_result_ft_SE_imbalance_s1_384_p0.5_lr2_mt_d0_b24+v+aug.mat')
query_feature0 = torch.FloatTensor(result['query_f']).cuda()
gallery_feature0 = torch.FloatTensor(result['gallery_f']).cuda()

query_path = 'pkl_feas/query_fea_ResNeXt101_vd_64x4d_cos_alldata_final.pkl'
query_feature1 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature1 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/query_fea_ResNeXt101_vd_64x4d_twosource_alldata_final.pkl'
query_feature2 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature2 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/real_query_fea_ResNeXt101_32x8d_wsl_416_416_final.pkl'
query_feature3 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature3 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/query_fea_ResNeXt101_vd_64x4d_twosource_cos_autoaug_final2.pkl'
query_feature4 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature4 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/real_query_fea_ResNeXt101_32x16d_wsl_384_384_final.pkl'
query_feature5 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature5 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/real_query_fea_ResNeXt101_32x8d_wsl_384_384_final.pkl'
query_feature6 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature6 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/bzc_res50ibn_ensemble_query_4307.pkl'
query_feature7 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature7 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/real_query_fea_ResNeXt101_32x8d_wsl_400_400_final.pkl'
query_feature8 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature8 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/real_query_fea_ResNeXt101_32x8d_wsl_rect_final.pkl'
query_feature9 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature9 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/0403/query_fea_ResNeXt101_vd_64x4d_twosource_cos_trans_merge.pkl'
query_feature10 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature10 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/query_fea_Res2Net101_vd_final2.pkl'
query_feature11 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature11 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_path = 'pkl_feas/res50ibn_ensemble_query_bzc.pkl'
query_feature12 = torch.FloatTensor(load_pickle(query_path)).cuda()
gallery_feature12 = torch.FloatTensor(load_pickle(query_path.replace('query', 'gallery'))).cuda()

query_feature = torch.cat( (query_feature0, query_feature1, query_feature2, query_feature3, query_feature4, query_feature5, query_feature6, query_feature7, query_feature8, query_feature9, query_feature10, query_feature11,query_feature12), dim =1)
gallery_feature = torch.cat( (gallery_feature0, gallery_feature1, gallery_feature2, gallery_feature3, gallery_feature4, gallery_feature5, gallery_feature6, gallery_feature7, gallery_feature8, gallery_feature9, gallery_feature10, gallery_feature11, gallery_feature12), dim=1)

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

query_feature = L2Normalization(query_feature, dim=1)
gallery_feature = L2Normalization(gallery_feature, dim=1)

print(query_feature.shape)
threshold = 0.5
#query cluster
nq = query_feature.shape[0]
nf = query_feature.shape[1]
q_q_dist = torch.mm(query_feature, torch.transpose(query_feature, 0, 1))
q_q_dist = q_q_dist.cpu().numpy() 
q_q_dist[q_q_dist>1] = 1  #due to the epsilon
q_q_dist = 2-2*q_q_dist
eps = threshold
# first cluster
min_samples= 2
cluster1 = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', algorithm='auto', n_jobs=-1)
cluster1 = cluster1.fit(q_q_dist)
qlabels = cluster1.labels_
nlabel_q = len(np.unique(cluster1.labels_))
# gallery cluster
ng = gallery_feature.shape[0]
### Using tracking ID
g_g_dist = torch.ones(ng,ng).numpy()
nlabel_g = 0
glabels = torch.zeros(ng).numpy() - 1
with open('data/test_track_id.txt','r') as f:
    for line in f:
        line = line.replace('\n','')
        g_name = line.split(' ')
        g_name.remove('')
        g_name = list(map(int, g_name))
        for i in g_name:
            glabels[i-1] = nlabel_g
            for j in g_name:
                g_g_dist[i-1,j-1] = 0 
        nlabel_g +=1 
nimg_g = len(np.argwhere(glabels!=-1))
print('Gallery Cluster Class Number:  %d'%nlabel_g)
print('Gallery Cluster Image per Class:  %.2f'%(nimg_g/nlabel_g))

query_feature = L2Normalization(query_feature, dim=1)
gallery_feature = L2Normalization(gallery_feature, dim=1)

# Gallery Video fusion
gallery_feature_clone = gallery_feature.clone()
g_g_direction_sim_clone = g_g_direction_sim.copy()
junk_index_g = np.argwhere(gallery_shape< 15000).flatten() # 150x150
junk_index_q = np.argwhere(query_shape< 15000).flatten() # 150x150
print('Low Qualtiy Image in Query: %d'% len(junk_index_q))
print('Low Qualtiy Image in Gallery: %d'% len(junk_index_g))
for i in range(nlabel_g):
    index = np.argwhere(glabels==i).flatten()  #from small to large, start from 0
    high_quality_index = np.setdiff1d(index, junk_index_g)
    if len(high_quality_index) == 0:
        high_quality_index = index
    gf_mean = torch.mean(gallery_feature_clone[high_quality_index,:], dim=0)
    gd_mean = np.mean(g_g_direction_sim_clone[high_quality_index,:], axis=0)
    for j in range(len(index)):
        gallery_feature[index[j],:] +=  0.5*gf_mean
        #g_g_direction_sim[index[j],:] = (g_g_direction_sim[index[j],:] + gd_mean)/2

# Query Feature fusion
query_feature_clone = query_feature.clone()
for i in range(nlabel_q-1):
    index = np.argwhere(qlabels==i).flatten()  #from small to large, start from 0
    high_quality_index = np.setdiff1d(index, junk_index_q)
    if len(high_quality_index) == 0:
        high_quality_index = index
    qf_mean = torch.mean(query_feature_clone[high_quality_index,:], dim=0)
    for j in range(len(index)):
        query_feature[index[j],:] = qf_mean


query_feature = L2Normalization(query_feature, dim=1)
gallery_feature = L2Normalization(gallery_feature, dim=1)

######################################################################
# Predict Camera
q_cam = []
g_cam = []
with open('query_cam_preds_baidu.txt','r') as f:
    for line in f:
        line = line.replace('\n','')
        ID = line.split(' ')
        q_cam.append(int(ID[1]))

with open('gallery_cam_preds_baidu.txt','r') as f:
    for line in f:
        line = line.replace('\n','')
        ID = line.split(' ')
        g_cam.append(int(ID[1]))

q_cam = np.asarray(q_cam)
g_cam = np.asarray(g_cam)

cam_names = 'Cam-p0.5-s1-dense_veri_b8_lr1,Cam-p0.5-s1-dense_veri_b8_lr1_d0.2,Cam-p0.5-s1-dense,Cam-p0.5-s1-dense-d0.75'
cam_names = cam_names.split(',')

cam_models = nn.ModuleList()
# Extractr Cam Feature
if not os.path.isfile('submit_cam.mat'):
    for name in cam_names:
        model_tmp, _, epoch = load_network(name, opt)
        model_tmp.classifier.classifier = nn.Sequential()
        cam_models.append(model_tmp.cuda().eval())
    with torch.no_grad():
        gallery_cam, query_cam = torch.FloatTensor(), torch.FloatTensor()
        for cam_model in cam_models:
            q_c = extract_cam(cam_model,dataloaders['query'])
            #q_c_crop = extract_cam(cam_model,cropped_dataloaders['query'])
            #q_c = q_c + q_c_crop
            qnorm = torch.norm(q_c, p=2, dim=1, keepdim=True)
            q_c = q_c.div(qnorm.expand_as(q_c)) / np.sqrt(len(cam_names))

            g_c = extract_cam(cam_model,dataloaders['gallery'])
            #g_c_crop = extract_cam(cam_model,cropped_dataloaders['gallery'])
            #g_c = g_c + g_c_crop
            gnorm = torch.norm(g_c, p=2, dim=1, keepdim=True)
            g_c = g_c.div(gnorm.expand_as(g_c)) / np.sqrt(len(cam_names))

            query_cam = torch.cat((query_cam,q_c), 1)
            gallery_cam = torch.cat((gallery_cam,g_c), 1)

    result = {'gallery_cam':gallery_cam.numpy(),'query_cam':query_cam.numpy()}
    scipy.io.savemat('submit_cam.mat',result)
else:
    result = scipy.io.loadmat('submit_cam.mat')
    query_cam = torch.FloatTensor(result['query_cam']).cuda()
    gallery_cam = torch.FloatTensor(result['gallery_cam']).cuda()

# cam_expand
print(query_cam.shape)
cam_total = torch.cat((query_cam,gallery_cam), dim=0)
cam_dist = torch.mm(cam_total, torch.transpose(cam_total, 0, 1))
cam_dist[cam_dist>1] = 1  #due to the epsilon
cam_dist = 2 - 2*cam_dist
cam_dist = cam_dist.cpu().numpy()
min_samples= 50
eps = 0.3
cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed',n_jobs=8)
cluster = cluster.fit(cam_dist)
print('Cam Cluster Class Number:  %d'%len(np.unique(cluster.labels_)))
cam_label = cluster.labels_
q_cam_cluster = cam_label[0:nq]
g_cam_cluster = cam_label[nq:]

print(len(q_cam_cluster))
print(q_cam_cluster[747])
print(len(g_cam_cluster))
print(g_cam_cluster[16238])

print('Unsure_Query_Cam:%d'%len(np.argwhere(q_cam_cluster==-1)))
print('Unsure_Gallery_Cam:%d'%len(np.argwhere(g_cam_cluster==-1)))
# Camera Complete Plan

for i in range(nlabel_q-1):  # one class is -1, we ignore it
    index = np.argwhere(qlabels==i)
    index = index.flatten()
    flag = np.unique(q_cam_cluster[index])
    if len(flag)==1: 
        continue 
    if len(flag)>2:
        continue 
    for j in range(len(index)):
        if q_cam_cluster[index[j]] == -1:
            q_cam_cluster[index[j]] = flag[1]

for i in range(nlabel_g):  # one class is -1, we ignore it
    index = np.argwhere(glabels==i)
    index = index.flatten()
    flag = np.unique(g_cam_cluster[index])
    if len(flag)==1:
        continue
    if len(flag)>2:
        continue
    for j in range(len(index)):
        if g_cam_cluster[index[j]] == -1:
            g_cam_cluster[index[j]] = flag[1]

print('After complete, Unsure_Query_Cam:%d'%len(np.argwhere(q_cam_cluster==-1)))
print('After complete, Unsure_Gallery_Cam:%d'%len(np.argwhere(g_cam_cluster==-1)))

# generate the rank result
print('-------generate the rank result-----------')

nq = query_feature.shape[0]
result_file = 'track2-m.txt'
if os.path.isfile(result_file):
    os.system('rm %s'%result_file)
score_total = torch.mm(query_feature, torch.transpose(gallery_feature, 0, 1))
print(score_total.shape)
for i in range(nq):
    if q_cam[i] !=-1:
        ignore_index = np.argwhere(g_cam==q_cam[i])
        score_total[i,ignore_index] = score_total[i,ignore_index] - 0.3
    else:
    #    # same direction in 6,7,8,9
        ignore_index = np.argwhere(q_g_direction_sim[i,:] >= 0.25)
        #ignore_index2 = np.argwhere(g_cam == -1)
        #ignore_index = np.intersect1d(ignore_index1, ignore_index2)
        score_total[i,ignore_index] = score_total[i,ignore_index] - 0.1

    if q_cam_cluster[i] != -1:
        ignore_index = np.argwhere(g_cam_cluster==q_cam_cluster[i])
        score_total[i,ignore_index] = score_total[i,ignore_index] - 0.2

score_total = score_total.cpu().numpy()
ntop = 99 # delete the different tracklet with the same camera of top 5
for j in range(ntop):
    for i in range(nq):
        topk_index = np.argsort(score_total[i,:])
        topk_index = topk_index[-1-j]
        good_index = np.argwhere(glabels==glabels[topk_index] )
        if g_cam[topk_index] !=-1:
            bad_index = np.argwhere(g_cam==g_cam[topk_index])
            ignore_index = np.setdiff1d(bad_index, good_index)
            score_total[i,ignore_index] = score_total[i,ignore_index] - 0.3/(1+j)
        else:
        #    # same direction in 6,7,8,9
            bad_index = np.argwhere(g_g_direction_sim[topk_index,:] >= 0.25)
            #bad_index2 = np.argwhere(g_cam == -1)
            #bad_index = np.intersect1d(bad_index1, bad_index2)
            ignore_index = np.setdiff1d(bad_index, good_index)
            score_total[i,ignore_index] = score_total[i,ignore_index] - 0.1/(1+j)
        if g_cam_cluster[topk_index] != -1:
            bad_index = np.argwhere(g_cam_cluster==g_cam_cluster[topk_index])
            ignore_index = np.setdiff1d(bad_index, good_index)
            score_total[i,ignore_index] = score_total[i,ignore_index] - 0.1/(1+j)

score_total_copy = score_total

# remove the same cam
for i in range(nq):
    score = score_total[i,:]
    index = np.argsort(score)  #from small to large, start from 0
    index = index[::-1]
    index = index[0:100] + 1
    str_index = np.array2string(index, separator=' ', suppress_small=True)
    str_index = str_index[1:-1].replace('\n','')
    str_index = str_index.replace('    ',' ')
    str_index = str_index.replace('   ',' ')
    str_index = str_index.replace('  ',' ')
    if str_index[0] == ' ':
        str_index = str_index[1:]
    with open(result_file, 'a') as text_file:
        text_file.write(str_index+'\n')


print('-------generate the re-ranking result-----------')
# Re-ranking
result_file = 'track2-rerank-m.txt'
if os.path.isfile(result_file):
    os.system('rm %s'%result_file)
q_q_dist = torch.mm(query_feature, torch.transpose(query_feature, 0, 1))
g_g_dist = torch.mm(gallery_feature, torch.transpose(gallery_feature, 0, 1))
# to cpu
q_q_dist, g_g_dist = q_q_dist.cpu().numpy(), g_g_dist.cpu().numpy()
q_g_dist = score_total_copy
min_value = np.amin(q_g_dist)
print(min_value)

# Different query
for i in range(nlabel_q-1):
    index = np.argwhere(qlabels==i)
    bad_index = np.argwhere(qlabels!=i)
    valid_index = np.argwhere(qlabels!=-1)
    bad_index = np.intersect1d(bad_index, valid_index)
    for j in range(len(index)):
        q_q_dist[index[j], bad_index] =  q_q_dist[index[j], bad_index] - 0.1

# Same cluster query
for i in range(nlabel_q-1):
    index = np.argwhere(qlabels==i)
    for j in range(len(index)):
        q_q_dist[index[j], index] =  q_q_dist[index[j], index] + 0.1

# Same cluster gallery
for i in range(nlabel_g):
    index = np.argwhere(glabels==i)
    for j in range(len(index)):
        g_g_dist[index[j], index] = 1.0

# Different Gallery
# Only vehicle from same trackID  has same camera
# Same camera different trackID is different car
for i in range(ng):
    good_index = np.argwhere(glabels==glabels[i])
    if g_cam[i] != -1:
        junk_index = np.argwhere(g_cam == g_cam[i])
        index = np.setdiff1d(junk_index, good_index)
        g_g_dist[i, index] = g_g_dist[i, index] - 0.3
    else:
        # same direction in 6,7,8,9
        junk_index = np.argwhere(g_g_direction_sim[i,:] >= 0.25)
        index = np.setdiff1d(junk_index, good_index)
        g_g_dist[i,index] = g_g_dist[i,index] - 0.1
    if g_cam_cluster[i] != -1:
        junk_index = np.argwhere(g_cam_cluster == g_cam_cluster[i])
        index = np.setdiff1d(junk_index, good_index)
        g_g_dist[i, index] = g_g_dist[i, index] - 0.2

for i in range(nq):
    good_index = np.argwhere(qlabels==qlabels[i])
    if q_cam[i] != -1:
        junk_index = np.argwhere(q_cam == q_cam[i])
        index = np.setdiff1d(junk_index, good_index)
        q_q_dist[i, index] = q_q_dist[i, index] - 0.3
    else:
        # same direction in 6,7,8,9
        junk_index = np.argwhere(q_q_direction_sim[i,:] >= 0.25)
        index = np.setdiff1d(junk_index, good_index)
        q_q_dist[i,index] = q_q_dist[i,index] - 0.1
    if q_cam_cluster[i] != -1:
        junk_index = np.argwhere(q_cam_cluster == q_cam_cluster[i])
        index = np.setdiff1d(junk_index, good_index)
        q_q_dist[i, index] = q_q_dist[i, index] - 0.2

if not os.path.isfile('rerank_score.mat'):
    score_total = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1 = opt.k1, k2 = opt.k2, lambda_value=opt.lam)
    score = {'score_total':score_total}
    scipy.io.savemat('rerank_score.mat', score)
else: 
    score = scipy.io.loadmat('rerank_score.mat')
    score_total = score['score_total']
for i in range(nq):
    if q_cam[i] !=-1:
        ignore_index = np.argwhere(g_cam==q_cam[i])
        score_total[i,ignore_index] = score_total[i,ignore_index] + 0.3
    else:
        # same direction in 6,7,8,9
        ignore_index = np.argwhere(q_g_direction_sim[i,:] >= 0.25)
        #ignore_index2 = np.argwhere(g_cam == -1)
        #ignore_index = np.intersect1d(ignore_index1, ignore_index2)
        score_total[i,ignore_index] = score_total[i,ignore_index] + 0.1

    if q_cam_cluster[i] != -1:
        ignore_index = np.argwhere(g_cam_cluster==q_cam_cluster[i])
        score_total[i,ignore_index] = score_total[i,ignore_index] + 0.2

for i in range(nq):
    score = score_total[i,:]
    index = np.argsort(score)
    #index = index[::-1]
    index = index[0:100] + 1
    str_index = np.array2string(index, separator=' ', suppress_small=True)
    str_index = str_index[1:-1].replace('\n','')
    str_index = str_index.replace('    ',' ')
    str_index = str_index.replace('   ',' ')
    str_index = str_index.replace('  ',' ')
    if str_index[0] == ' ':
        str_index = str_index[1:]
    with open(result_file, 'a') as text_file:
        text_file.write(str_index+'\n')

