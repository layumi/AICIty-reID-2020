import os
from shutil import copyfile
import scipy.io
download_path = './data/'
save_path = download_path + 'pytorch2020-cam/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#train_all
train_path = download_path + '/2020AICITY/'
virtual_path = download_path + '/pytorch2020-cam/virtual'
train_real_save_path = download_path + '/pytorch2020-cam/train_real'
train_all_save_path = download_path + '/pytorch2020-cam/train'
if not os.path.isdir(virtual_path):
    os.mkdir(virtual_path)
    os.mkdir(train_real_save_path)
    os.mkdir(train_all_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            cID = name.split('c')
            cID = cID[1][0:3]
            print(cID)
            src_path = train_path + '/' + name
            dst_path = virtual_path + '/' + cID
            dst_real_path = train_real_save_path + '/' + cID
            dst_all_path = train_all_save_path + '/' + cID
            if int(cID)<=666:
                if not os.path.isdir(dst_real_path):
                    os.mkdir(dst_real_path)
                copyfile(src_path, dst_real_path + '/' + name)
            else:
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)
            if not os.path.isdir(dst_all_path):
                    os.mkdir(dst_all_path)
            copyfile(src_path, dst_all_path + '/' + name)

result = scipy.io.loadmat('label.mat')
query_c = result['query_c'].flatten()
gallery_c = result['gallery_c'].flatten()
query_l = result['query_l'].flatten()
gallery_l = result['gallery_l'].flatten()

query_path = download_path + '/image_query/'
query_save_path = download_path + '/pytorch2020-cam/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)
    for i in range(40):
        dst_path = query_save_path + '/' + '%03d'%(i+1)
        if not os.path.isdir(dst_path):
                os.mkdir(dst_path)

    for root, dirs, files in os.walk(query_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            src_path = query_path + '/' + name
            index = int(name[0:-4])
            dst_path = query_save_path + '/' + '%03d'%(query_c[index-1])
            if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
            print(dst_path + '/' + name)

