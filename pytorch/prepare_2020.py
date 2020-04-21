import os
from shutil import copyfile
import numpy as np

download_path = './data/'
save_path = download_path + 'pytorch2020/'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

#train_all
train_path = download_path + '/2020AICITY/'
virtual_path = download_path + '/pytorch2020/virtual'
train_real_save_path = download_path + '/pytorch2020/train_real_all'
if not os.path.isdir(virtual_path):
    os.mkdir(virtual_path)
    os.mkdir(train_real_save_path)

    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            src_path = train_path + '/' + name
            dst_path = virtual_path + '/' + ID[0]
            dst_real_path = train_real_save_path + '/' + ID[0]
            if int(ID[0])<=666:
                if not os.path.isdir(dst_real_path):
                    os.mkdir(dst_real_path)
                copyfile(src_path, dst_real_path + '/' + name)
            else:
                if not os.path.isdir(dst_path):
                    os.mkdir(dst_path)
                copyfile(src_path, dst_path + '/' + name)

val_ID_list = []

# train + val
train_save_path = download_path + '/pytorch2020/train'
val_save_path = download_path + '/pytorch2020/query'
gallery_save_path = download_path + '/pytorch2020/gallery'
m = np.zeros((500,100))
if not os.path.isdir(val_save_path):
    os.mkdir(val_save_path)
    os.mkdir(train_save_path)
    os.mkdir(gallery_save_path)
    '''
    with open('./data/val_2020.txt', 'r') as f:
        for name in f:
            name = name.replace('\n','')
            val_ID = name.split('_')
            src_path = train_path + '/' + name
            dst_path = val_save_path + '/' + val_ID[0]
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/' + name)
            val_ID = int(val_ID[0])
            if not val_ID in val_ID_list:
                val_ID_list.append(val_ID)
    print(len(val_ID_list) )
    '''
    for root, dirs, files in os.walk(train_path, topdown=True):
        for name in files:
            if not name[-3:]=='jpg':
                continue
            ID  = name.split('_')
            camID = name.split('c')
            src_path = train_path + '/' + name
            dst_path = train_save_path + '/' + ID[0]
            dst_path_g = gallery_save_path + '/' + ID[0]
            dst_path_q = val_save_path + '/' + ID[0]
            if int(ID[0])<=666:
                if int(ID[0])<=400 : #train
                    if not os.path.isdir(dst_path):
                        os.mkdir(dst_path)
                    copyfile(src_path, dst_path + '/' + name)
                else:
                    cam_id = int(camID[1][0:3])
                    vid = ID[0]
                    print(int(vid), int(cam_id))
                    if m[int(vid)-400][int(cam_id)] == 0:
                        m[int(vid)-400][int(cam_id)] = 1
                        if not os.path.isdir(dst_path_q):
                             os.mkdir(dst_path_q)
                        copyfile(src_path, dst_path_q + '/'+ name)

#os.system('ln -s %s  ./data/pytorch2020/gallery_large'%os.path.abspath(train_real_save_path))
#os.system('rsync -r %s  ./data/pytorch2020/train+virtual'%os.path.abspath(train_save_path))
#os.system('rsync -r %s  ./data/pytorch2020/train+virtual'%os.path.abspath(virtual_path))
