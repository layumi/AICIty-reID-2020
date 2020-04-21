# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from PIL import Image
import time
import os
from losses import AngleLoss, ArcLoss
from model import ft_net, ft_net_dense, ft_net_EF4, ft_net_EF5, ft_net_EF6, ft_net_IR, ft_net_NAS, ft_net_SE, ft_net_DSE, PCB, CPB, ft_net_angle, ft_net_arc
from random_erasing import RandomErasing
import yaml
from AugFolder import AugFolder
from shutil import copyfile
import random
from autoaugment import ImageNetPolicy
from utils import get_model_list, load_network, save_network, make_weights_for_balanced_classes

version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')

# make the output
if not os.path.isdir('./data/outputs'):
    os.mkdir('./data/outputs')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--adam', action='store_true', help='use all training data' )
parser.add_argument('--name',default='ft_ResNet50', type=str, help='output model name')
parser.add_argument('--init_name',default='imagenet', type=str, help='initial with ImageNet')
parser.add_argument('--data_dir',default='./data/pytorch2020',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--train_veri', action='store_true', help='use part training data + veri' )
parser.add_argument('--train_virtual', action='store_true', help='use part training data + virtual' )
parser.add_argument('--train_comp', action='store_true', help='use part training data + comp' )
parser.add_argument('--train_pku', action='store_true', help='use part training data + pku' )
parser.add_argument('--train_comp_veri', action='store_true', help='use part training data + comp +veri' )
parser.add_argument('--train_milktea', action='store_true', help='use part training data + com + veri+pku' )
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--inputsize', default=299, type=int, help='batchsize')
parser.add_argument('--h', default=299, type=int, help='height')
parser.add_argument('--w', default=299, type=int, help='width')
parser.add_argument('--stride', default=2, type=int, help='stride')
parser.add_argument('--pool',default='avg', type=str, help='last pool')
parser.add_argument('--autoaug', action='store_true', help='use Color Data Augmentation' )
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_NAS', action='store_true', help='use nasnetalarge' )
parser.add_argument('--use_SE', action='store_true', help='use se_resnext101_32x4d' )
parser.add_argument('--use_DSE', action='store_true', help='use senet154' )
parser.add_argument('--use_IR', action='store_true', help='use InceptionResNetv2' )
parser.add_argument('--use_EF4', action='store_true', help='use EF4' )
parser.add_argument('--use_EF5', action='store_true', help='use EF5' )
parser.add_argument('--use_EF6', action='store_true', help='use EF6' )
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
parser.add_argument('--PCB', action='store_true', help='use PCB+ResNet50' )
parser.add_argument('--CPB', action='store_true', help='use Center+ResNet50' )
parser.add_argument('--fp16', action='store_true', help='use float16 instead of float32, which will save about 50% memory' )
parser.add_argument('--balance', action='store_true', help='balance sample' )
parser.add_argument('--angle', action='store_true', help='use angle loss' )
parser.add_argument('--arc', action='store_true', help='use arc loss' )
parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--resume', action='store_true', help='use arc loss' )
opt = parser.parse_args()

if opt.resume:
    model, opt, start_epoch = load_network(opt.name, opt)
else:
    start_epoch = 0

print(start_epoch)


fp16 = opt.fp16
data_dir = opt.data_dir
name = opt.name

if not opt.resume:
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >=0:
            gpu_ids.append(gid)
    opt.gpu_ids = gpu_ids

# set gpu ids
if len(opt.gpu_ids)>0:
    cudnn.enabled = True
    cudnn.benchmark = True
######################################################################
# Load Data
# ---------
#

if opt.h == opt.w:
    transform_train_list = [
        #transforms.RandomRotation(30), 
        transforms.Resize((opt.inputsize, opt.inputsize), interpolation=3),
        transforms.Pad(15),
        #transforms.RandomCrop((256,256)),
        transforms.RandomResizedCrop(size=opt.inputsize, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    transform_val_list = [
        transforms.Resize(size=opt.inputsize,interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
else:
    transform_train_list = [
        #transforms.RandomRotation(30),
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.Pad(15),
        #transforms.RandomCrop((256,256)),
        transforms.RandomResizedCrop(size=(opt.h, opt.w), scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

    transform_val_list = [
        transforms.Resize((opt.h, opt.w),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]


if opt.PCB:
    transform_train_list = [
        transforms.Resize((384,192), interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    transform_val_list = [
        transforms.Resize(size=(384,192),interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]

if opt.erasing_p>0:
    transform_train_list = transform_train_list +  [RandomErasing(probability = opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0)] + transform_train_list


transform_train_list_aug = [ImageNetPolicy()] + transform_train_list

print(transform_train_list)
data_transforms = {
    'train': transforms.Compose( transform_train_list ),
    'train_aug': transforms.Compose( transform_train_list_aug ), 
    'val': transforms.Compose(transform_val_list),
}


train_all = ''
if opt.train_all:
     train_all = '_all'

if opt.train_veri:
     train_all = '+veri'

if opt.train_comp:
     train_all = '+comp'

if opt.train_virtual:
     train_all = '+virtual'

if opt.train_pku:
     train_all = '+pku'

if opt.train_comp_veri:
     train_all = '+comp+veri'

if opt.train_milktea:
     train_all = '+comp+veri+pku'

image_datasets = {}

if not opt.autoaug:
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'])
else:
    image_datasets['train'] = AugFolder(os.path.join(data_dir, 'train' + train_all),
                                          data_transforms['train'], data_transforms['train_aug'])

if opt.balance:
    dataset_train = image_datasets['train']
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))                                                                
    weights = torch.DoubleTensor(weights)   
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize, sampler=sampler, num_workers=8, pin_memory=True) # 8 workers may work faster
else: 
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=True, num_workers=8, pin_memory=True) # 8 workers may work faster
                                             for x in ['train']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

#since = time.time()
#inputs, classes = next(iter(dataloaders['train']))
#print(time.time()-since)

######################################################################
# Training the model
# ------------------
#
# Now, let's write a general function to train a model. Here, we will
# illustrate:
#
# -  Scheduling the learning rate
# -  Saving the best model
#
# In the following, parameter ``scheduler`` is an LR scheduler object from
# ``torch.optim.lr_scheduler``.

y_loss = {} # loss history
y_loss['train'] = []
y_loss['val'] = []
y_err = {}
y_err['train'] = []
y_err['val'] = []

def train_model(model, criterion, optimizer, scheduler, start_epoch=0, num_epochs=25):
    since = time.time()

    warm_up = 0.1 # We start from the 0.1*lrRate
    gamma = 0.0 #auto_aug
    warm_iteration = round(dataset_sizes['train']/opt.batchsize)*opt.warm_epoch # first 5 epoch
    total_iteration = round(dataset_sizes['train']/opt.batchsize)*num_epochs

    best_model_wts = model.state_dict()
    best_loss = 9999
    best_epoch = 0

    for epoch in range(num_epochs-start_epoch):
        epoch = epoch + start_epoch
        print('gamma: %.4f'%gamma)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0
            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                if opt.autoaug:
                    inputs, inputs2, labels = data
                    if random.uniform(0, 1)> gamma:
                        inputs = inputs2
                    gamma = min(1.0, gamma + 1.0 / total_iteration)
                else:
                    inputs, labels = data

                now_batch_size,c,h,w = inputs.shape
                if now_batch_size<opt.batchsize: # skip the last batch
                    continue
                #print(inputs.shape)
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                # if we use low precision, input also need to be fp16
                #if fp16:
                #    inputs = inputs.half()
 
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)


                if opt.PCB:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 6
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) +sm(part[4]) +sm(part[5])
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                elif opt.CPB:
                    part = {}
                    sm = nn.Softmax(dim=1)
                    num_part = 4
                    for i in range(num_part):
                        part[i] = outputs[i]

                    score = sm(part[0]) + sm(part[1]) +sm(part[2]) + sm(part[3]) 
                    _, preds = torch.max(score.data, 1)

                    loss = criterion(part[0], labels)
                    for i in range(num_part-1):
                        loss += criterion(part[i+1], labels)
                else:
                    loss = criterion(outputs, labels)
                    if opt.angle or opt.arc:
                        outputs = outputs[0]
                    _, preds = torch.max(outputs.data, 1)

                # backward + optimize only if in training phase
                if epoch<opt.warm_epoch and phase == 'train': 
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                # backward + optimize only if in training phase
                if phase == 'train':
                    if fp16: # we use optimier to backward loss
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()

                #print('Iteration: loss:%.2f accuracy:%.2f'%(loss.item(), float(torch.sum(preds == labels.data))/now_batch_size ) )
                # statistics
                if int(version[0])>0 or int(version[2]) > 3: # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                else :  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                running_corrects += float(torch.sum(preds == labels.data))
                
                del(loss, outputs, inputs, preds)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            y_loss[phase].append(epoch_loss)
            y_err[phase].append(1.0-epoch_acc)            
            # deep copy the model
            if len(opt.gpu_ids)>1:
                save_network(model.module, opt.name, epoch+1)
            else:
                save_network(model, opt.name, epoch+1)
            draw_curve(epoch)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            last_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch: {:d} Best Train Loss: {:4f}'.format(best_epoch, best_loss))

    # load best model weights
    model.load_state_dict(last_model_wts)
    save_network(model, opt.name, 'last')
    return model


######################################################################
# Draw Curve
#---------------------------
x_epoch = []
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")
def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    ax0.plot(x_epoch, y_loss['train'], 'bo-', label='train')
    #ax0.plot(x_epoch, y_loss['val'], 'ro-', label='val')
    ax1.plot(x_epoch, y_err['train'], 'bo-', label='train')
    #ax1.plot(x_epoch, y_err['val'], 'ro-', label='val')
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig( os.path.join('./data/outputs',name,'train.png'))


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrainied model and reset final fully connected layer.
#

if not opt.resume:
    opt.nclasses = len(class_names)
    if opt.use_dense:
        model = ft_net_dense(len(class_names), opt.droprate, opt.stride, None, opt.pool)
    elif opt.use_NAS:
        model = ft_net_NAS(len(class_names), opt.droprate, opt.stride) 
    elif opt.use_SE:
        model = ft_net_SE(len(class_names), opt.droprate, opt.stride, opt.pool)
    elif opt.use_DSE:
        model = ft_net_DSE(len(class_names), opt.droprate, opt.stride, opt.pool)
    elif opt.use_IR:
        model = ft_net_IR(len(class_names), opt.droprate, opt.stride)
    elif opt.use_EF4:
        model = ft_net_EF4(len(class_names), opt.droprate)
    elif opt.use_EF5:
        model = ft_net_EF5(len(class_names), opt.droprate)
    elif opt.use_EF6:
        model = ft_net_EF6(len(class_names), opt.droprate)
    else:
        model = ft_net(len(class_names), opt.droprate, opt.stride, None, opt.pool)

    if opt.PCB:
        model = PCB(len(class_names))

    if opt.CPB:
        model = CPB(len(class_names))

    if opt.angle:
        model = ft_net_angle(len(class_names), opt.droprate, opt.stride)
    elif opt.arc:
        model = ft_net_arc(len(class_names), opt.droprate, opt.stride)

if opt.init_name != 'imagenet':
    old_opt = parser.parse_args()
    init_model, old_opt, _ = load_network(opt.init_name, old_opt)
    print(old_opt)
    opt.stride = old_opt.stride
    opt.pool = old_opt.pool
    opt.use_dense = old_opt.use_dense
    if opt.use_dense:
        model = ft_net_dense(opt.nclasses, droprate=opt.droprate, stride=opt.stride, init_model=init_model, pool = opt.pool)
    else:
        model = ft_net(opt.nclasses, droprate=opt.droprate, stride=opt.stride, init_model=init_model, pool = opt.pool)


##########################
#Put model parameter in front of the optimizer!!!

# For resume:
if start_epoch>=60:
    opt.lr = opt.lr*0.1
if start_epoch>=75:
    opt.lr = opt.lr*0.1

if len(opt.gpu_ids)>1:
    model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids).cuda()
    if not opt.CPB:
        ignored_params = list(map(id, model.module.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.module.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = (list(map(id, model.module.classifier0.parameters() ))
                     +list(map(id, model.module.classifier1.parameters() ))
                     +list(map(id, model.module.classifier2.parameters() ))
                     +list(map(id, model.module.classifier3.parameters() ))
                      )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.module.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.module.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.module.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.module.classifier3.parameters(), 'lr': opt.lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
else:
    model = model.cuda()
    if not opt.CPB:
        ignored_params = list(map(id, model.classifier.parameters() ))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier.parameters(), 'lr': opt.lr}
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = (list(map(id, model.classifier0.parameters() )) 
                     +list(map(id, model.classifier1.parameters() ))
                     +list(map(id, model.classifier2.parameters() ))
                     +list(map(id, model.classifier3.parameters() ))
                      )
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer_ft = optim.SGD([
             {'params': base_params, 'lr': 0.1*opt.lr},
             {'params': model.classifier0.parameters(), 'lr': opt.lr},
             {'params': model.classifier1.parameters(), 'lr': opt.lr},
             {'params': model.classifier2.parameters(), 'lr': opt.lr},
             {'params': model.classifier3.parameters(), 'lr': opt.lr},
         ], weight_decay=5e-4, momentum=0.9, nesterov=True)

if opt.adam:
    optimizer_ft = optim.Adam(model.parameters(), opt.lr, weight_decay=5e-4)

# Decay LR by a factor of 0.1 every 40 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=40, gamma=0.1)
exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[60-start_epoch, 75-start_epoch], gamma=0.1)

######################################################################
# Train and evaluate
# ^^^^^^^^^^^^^^^^^^
#
# It should take around 1-2 hours on GPU. 
#
dir_name = os.path.join('./data/outputs',name)

if not opt.resume:
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
#record every run
    copyfile('./train.py', dir_name+'/train.py')
    copyfile('./model.py', dir_name+'/model.py')
# save opts
    with open('%s/opts.yaml'%dir_name,'w') as fp:
        yaml.dump(vars(opt), fp, default_flow_style=False)

# model to gpu
if fp16:
    #model = network_to_half(model)
    #optimizer_ft = FP16_Optimizer(optimizer_ft, dynamic_loss_scale=True)
    model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level = "O1")


if opt.angle:
    criterion = AngleLoss()
elif opt.arc:
    criterion = ArcLoss()
else:
    criterion = nn.CrossEntropyLoss()

print(model)
model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                    start_epoch=start_epoch, num_epochs=80)

