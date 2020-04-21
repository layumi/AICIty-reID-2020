import scipy.io
import torch
import numpy as np
import time
from  re_ranking import re_ranking
#######################################################################
# Evaluate
def evaluate(score,ql,qc,gl,gc):
    index = np.argsort(score)  #from small to large
    #index = index[::-1]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
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
def calculate_result_rerank(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, result_file, k1=100, k2=15, lambda_value=0):
    query_feature = torch.FloatTensor(query_feature).cuda()
    gallery_feature = torch.FloatTensor(gallery_feature).cuda()
    CMC = torch.IntTensor(len(gallery_label)).zero_()
    
    ap = 0.0
    print('calculate initial distance')
    since = time.time()
    q_g_dist = torch.mm(query_feature, torch.transpose(gallery_feature, 0, 1))
    q_q_dist = torch.mm(query_feature, torch.transpose(query_feature, 0, 1))
    g_g_dist = torch.mm(gallery_feature, torch.transpose(gallery_feature, 0, 1))

    # to cpu
    q_g_dist, q_q_dist, g_g_dist = q_g_dist.cpu().numpy(), q_q_dist.cpu().numpy(), g_g_dist.cpu().numpy()
    re_rank = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
    time_elapsed = time.time() - since
    print('Reranking complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(re_rank[i,:],query_label[i],query_cam[i],gallery_label,gallery_cam)
        if CMC_tmp[0]==-1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
    CMC = CMC.float()
    CMC = CMC/len(query_label) #average CMC
    str_result = 're-ranking Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f\n'%(CMC[0],CMC[4],CMC[9],ap/len(query_label))
    print(k1, k2, lambda_value, str_result)
    text_file = open(result_file, "a")
    text_file.write(str_result)
    text_file.close()


if __name__ == '__main__':
    result = scipy.io.loadmat('pytorch_result.mat')
    query_feature = result['query_f']
    query_cam = result['query_cam'][0]
    query_label = result['query_label'][0]
    gallery_feature = result['gallery_f']
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    k1_list = [80, 100, 120]
    k2_list = [15, 20]
    lambda_list = [0, 0.2]
    for k1 in k1_list:
        for k2 in k2_list:
            if k2>k1:
                continue
            for lambda_value in lambda_list:
                calculate_result_rerank(gallery_feature, gallery_label, gallery_cam, query_feature, query_label, query_cam, 'tmp.txt', k1, k2, lambda_value)
