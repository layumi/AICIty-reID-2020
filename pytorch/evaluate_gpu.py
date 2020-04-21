import scipy.io
import torch
import numpy as np
#import time
import os
import pickle
#######################################################################
# Evaluate
def evaluate(score,ql,qc,gl,gc):
    #print(score.shape)
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    #index = index[0:100]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    if qc == -1: #VeID has no camera ID
        camera_index = []
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())

    #print(good_index)    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
def calculate_result(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, result_file, pre_compute_score=None):
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    ap = 0.0
    if pre_compute_score is None:
        query_feature = torch.FloatTensor(query_feature).cuda()
        gallery_feature = torch.FloatTensor(gallery_feature).cuda()
        score = torch.mm(query_feature, torch.transpose( gallery_feature, 0, 1))
        score = score.cpu().numpy()
    else: 
        score = pre_compute_score

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(score[i],query_label[i],query_cam[i],gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
       # if CMC_tmp[0]==0:
       #     print(i)
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    str_result = 'Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f\n'%(CMC[0],CMC[4],CMC[9],ap/len(query_label))
    print(str_result)
    text_file = open(result_file, "a")
    text_file.write(str_result)
    text_file.close()
    return score

if __name__ == '__main__':
    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = torch.FloatTensor(result['query_f'])[:, 0:512]
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = torch.FloatTensor(result['gallery_f'])[:, 0:512]
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]
    print(query_feature.shape)
    print(np.unique(query_cam))
    '''
    query_feature = torch.zeros( query_feature.shape[0], 2048)
    gallery_feature = torch.zeros( gallery_feature.shape[0], 2048)
    fr=open('val_query_fea_SEResNeXt50.pkl','rb')
    query_data = pickle.load(fr)
    index = 0
    for name, feature in query_data.items():
        query_feature[index,:] = torch.FloatTensor(feature)
        index +=1
    fr=open('val_gallery_fea_SEResNeXt50.pkl', 'rb')  
    gallery_data = pickle.load(fr)
    index = 0
    for name, feature in gallery_data.items():
        gallery_feature[index,:] = torch.FloatTensor(feature)
        index +=1
    '''
    calculate_result(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, 'tmp.txt')
