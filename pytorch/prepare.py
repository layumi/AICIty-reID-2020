import os
from shutil import copyfile
from xml.dom import minidom
import numpy as np
# convert gb2312 to utf8
os.system('python convert.py')

train_path = 'data/image_train/'
query_path = 'data/image_query/'
gallery_path = 'data/image_test/'

save_path = 'data/pytorch'
save_query_path = 'data/pytorch/query/'
save_train_path = 'data/pytorch/train/' # 233 classes
save_train_all_path = 'data/pytorch/train_all/' # include train and val
save_test_path = 'data/pytorch/gallery'

save_test_data_path = 'data/test_data'
save_test_query_path = 'data/test_data/query'
save_test_gallery_path = 'data/test_data/gallery'

if not os.path.isdir(save_path):
    os.mkdir(save_path)
    os.mkdir(save_query_path)
    os.mkdir(save_train_path)
    os.mkdir(save_train_all_path)
    os.mkdir(save_test_data_path)
    os.mkdir(save_test_query_path)
    os.mkdir(save_test_gallery_path)

#---------prepare training all set-------
xmldoc = minidom.parse('data/train_label_utf8.xml')
itemlist = xmldoc.getElementsByTagName('Items')[0]
itemlist = itemlist.getElementsByTagName('Item')

m = np.zeros((500,100))
print(len(itemlist))
for s in itemlist:
    # ---------first we read the camera and ID info and rename the images.
    name = s.attributes['imageName'].value
    vid = s.attributes['vehicleID'].value
    cam = s.attributes['cameraID'].value
    src_path = train_path + name
    dst_path = save_train_all_path + vid  #train_all
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/'+vid + '_' + cam + '_'+ name)

    # ---------then we split the file in train_all to train/query

    if int(vid) <= 400:  # ID smaller than 400 will used as train
        dst_path = save_train_path + vid
        if not os.path.isdir(dst_path):
            os.mkdir(dst_path)
        copyfile(src_path, dst_path + '/'+vid + '_' + cam + '_'+ name)
    elif int(vid) > 400:  # ID larger than 400 will used as query/gallery
        cam_id = cam[1:]
        print(int(vid), int(cam_id))
        if m[int(vid)-400][int(cam_id)] == 0:
            m[int(vid)-400][int(cam_id)] = 1
            dst_path = save_query_path + vid
            if not os.path.isdir(dst_path):
                os.mkdir(dst_path)
            copyfile(src_path, dst_path + '/'+vid + '_' + cam + '_'+ name)

# We use the train_all as the gallery. 
# We view the images in the trainging set as distractors
os.system('ln -s  %s %s'%(os.path.abspath(save_train_all_path) , os.path.abspath(save_test_path) ) )
os.system('ln -s  %s %s'%(os.path.abspath(query_path) , os.path.abspath(save_test_query_path) ) )
os.system('ln -s  %s %s'%(os.path.abspath(gallery_path) , os.path.abspath(save_test_gallery_path) ) )
