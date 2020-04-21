import argparse
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from losses import AngleLinear, ArcLinear
from efficientnet_pytorch import EfficientNet
import pretrainedmodels
from torch.nn import functional as F

######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

def fix_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True
# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x


# Define the ResNet50-based Model
class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate)

        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))
        # avg pooling to global pooling

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        if self.pool == 'avg+max':
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
            x = x.view(x.size(0), x.size(1))
        elif self.pool == 'avg':
            x = self.model.avgpool(x)
            x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x

# Define the ResNet50  Model with angle loss
# The code is borrowed from https://github.com/clcarwin/sphereface_pytorch
class ft_net_angle(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_angle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        #self.classifier.classifier=nn.Sequential()
        self.classifier.classifier = AngleLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        #x = self.fc(x)
        return x

class ft_net_arc(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_arc, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num, droprate)
        #self.classifier.classifier=nn.Sequential()
        self.classifier.classifier = ArcLinear(512, class_num)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        #x = self.fc(x)
        return x

# Define the DenseNet121-based Model
class ft_net_dense(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, init_model=None, pool='avg'):
        super().__init__()
        model_ft = models.densenet121(pretrained=True)
        if stride == 1:
            model_ft.features.transition3.pool.stride = 1
        model_ft.fc = nn.Sequential()

        self.pool = pool
        if pool =='avg+max':
            model_ft.features.avgpool = nn.Sequential()
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='avg':
            model_ft.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.model = model_ft
            self.classifier = ClassBlock(1024, class_num, droprate)

        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.pool = init_model.pool
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))

    def forward(self, x):
        if self.pool == 'avg':
            x = self.model.features(x)
        elif self.pool == 'avg+max':
            x = self.model.features(x)
            x1 = self.model.avgpool2(x)
            x2 = self.model.maxpool2(x)
            x = torch.cat((x1,x2), dim = 1)
        x = x.view(x.size(0), x.size(1))
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x

class ft_net_EF4(nn.Module):
    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(1792, class_num, droprate)

    def forward(self, x):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x

class ft_net_EF5(nn.Module):
    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x

class ft_net_EF6(nn.Module):
    def __init__(self, class_num, droprate=0.2):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b6')
        self.model._fc = nn.Sequential()
        self.classifier = ClassBlock(2304, class_num, droprate)

    def forward(self, x):
        # Convolution layers
        x = self.model.extract_features(x)
        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x
# Define the NAS-based Model
class ft_net_NAS(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        model_name = 'nasnetalarge' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        #if stride == 1:
        #    model_ft.layer4[0].downsample[0].stride = (1,1)
        #    model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        # relu -> inplace
        model_ft.cell_17.apply(fix_relu)
        self.model = model_ft
        # For DenseNet, the feature dim is 4032
        self.classifier = ClassBlock(4032, class_num, droprate)

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the SE-based Model
class ft_net_SE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg', init_model=None):
        super().__init__()
        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.layer4[0].conv2.stride = (1,1)
            model_ft.layer4[0].downsample[0].stride = (1,1)
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1,1))
        elif pool == 'avg+max':
            model_ft.avg_pool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.max_pool2 = nn.AdaptiveMaxPool2d((1,1))
        else:
           print('UNKNOW POOLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        self.pool  = pool
        # For DenseNet, the feature dim is 2048
        if pool == 'avg+max':
            self.classifier = ClassBlock(4096, class_num, droprate)
        else:
            self.classifier = ClassBlock(2048, class_num, droprate)
        self.flag = False
        if init_model!=None:
            self.flag = True
            self.model = init_model.model
            self.classifier.add_block = init_model.classifier.add_block
            self.new_dropout = nn.Sequential(nn.Dropout(p = droprate))

    def forward(self, x):
        x = self.model.features(x)
        if self.pool == 'avg+max':
            x1 = self.model.avg_pool2(x)
            x2 = self.model.max_pool2(x)
            x = torch.cat((x1,x2), dim = 1)
        else:
            x = self.model.avg_pool(x)
        x = x.view(x.size(0), x.size(1))
        # Convolution layers
        # Pooling and final linear layer
        if self.flag:
            x = self.classifier.add_block(x)
            x = self.new_dropout(x)
            x = self.classifier.classifier(x)
        else:
            x = self.classifier(x)
        return x

# Define the SE-based Model
class ft_net_DSE(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2, pool='avg'):
        super().__init__()
        model_name = 'senet154' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.layer4[0].conv2.stride = (1,1)
            model_ft.layer4[0].downsample[0].stride = (1,1)
        if pool == 'avg':
            model_ft.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        elif pool == 'max':
            model_ft.avg_pool = nn.AdaptiveMaxPool2d((1,1))
        else:
           print('UNKNOW POOLING!!!!!!!!!!!!!!!!!!!!!!!!!!')
        #model_ft.dropout = nn.Sequential()
        model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 2048
        self.classifier = ClassBlock(2048, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Define the inceptionresnetv2-based Model
class ft_net_IR(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super().__init__()
        model_name = 'inceptionresnetv2' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if stride == 1:
            model_ft.mixed_7a.branch0[1].conv.stride = (1,1)
            model_ft.mixed_7a.branch1[1].conv.stride = (1,1)
            model_ft.mixed_7a.branch2[2].conv.stride = (1,1)
            model_ft.mixed_7a.branch3.stride = 1
        model_ft.avgpool_1a = nn.AdaptiveAvgPool2d((1,1))
        #model_ft.dropout = nn.Sequential()
        model_ft.last_linear = nn.Sequential()
        self.model = model_ft
        # For DenseNet, the feature dim is 2048
        self.classifier = ClassBlock(1536, class_num, droprate)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    
# Define the ResNet50-based Model (Middle-Concat)
# In the spirit of "The Devil is in the Middle: Exploiting Mid-level Representations for Cross-Domain Instance Matching." Yu, Qian, et al. arXiv:1711.08106 (2017).
class ft_net_middle(nn.Module):

    def __init__(self, class_num, droprate=0.5):
        super(ft_net_middle, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048+1024, class_num, droprate)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x0  n*1024*1*1
        x0 = self.model.avgpool(x)
        x = self.model.layer4(x)
        # x1  n*2048*1*1
        x1 = self.model.avgpool(x)
        x = torch.cat((x0,x1),1)
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x

# Part Model proposed in Yifan Sun etal. (2018)
class PCB(nn.Module):
    def __init__(self, class_num ):
        super(PCB, self).__init__()

        self.part = 6 # We cut the pool5 to 6 parts
        model_ft = models.resnet50(pretrained=True)
        self.model = model_ft
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=256))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        part = {}
        predict = {}
        # get six part feature batchsize*2048*6
        for i in range(self.part):
            part[i] = torch.squeeze(x[:,:,i])
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

class PCB_test(nn.Module):
    def __init__(self,model):
        super(PCB_test,self).__init__()
        self.part = 6
        self.model = model.model
        self.avgpool = nn.AdaptiveAvgPool2d((self.part,1))
        # remove the final downsample
        self.model.layer4[0].downsample[0].stride = (1,1)
        self.model.layer4[0].conv2.stride = (1,1)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.avgpool(x)
        y = x.view(x.size(0),x.size(1),x.size(2))
        return y

# Center Part Model proposed in Yifan Sun etal. (2018)
class CPB(nn.Module):
    def __init__(self, class_num ):
        super(CPB, self).__init__()

        self.part = 4 # We cut the pool5 to 4 parts
        #model_ft = models.resnet50(pretrained=True)
        #self.model = EfficientNet.from_pretrained('efficientnet-b5')
        #self.model._fc = nn.Sequential()
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
        model_ft = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.layer4[0].downsample[0].stride = (1,1)
        self.model = model_ft
       #self.dropout = nn.Dropout(p=0.5)
        # remove the final downsample
        #self.model.layer4[0].downsample[0].stride = (1,1)
        #self.model.layer4[0].conv2.stride = (1,1)
        # define 6 classifiers
        for i in range(self.part):
            name = 'classifier'+str(i)
            setattr(self, name, ClassBlock(2048, class_num, droprate=0.2, relu=False, bnorm=True, num_bottleneck=512))

    def forward(self, x):
        x =  self.model.features(x)
        #x = self.dropout(x)
        #print(x.shape)
        part = {}
        predict = {}
        d = 2+2+2
        for i in range(self.part):
            N,C,H,W = x.shape
            p = 2 #max(2-i,1)
            if i==0:
                part_input = x[:,:,d:W-d,d:H-d]
                part[i] = torch.squeeze(self.avgpool(part_input))
                last_input = torch.nn.functional.pad(part_input, (p,p,p,p), mode='constant', value=0)
                #print(part_input.shape)
            else:
                part_input = x[:,:,d:W-d,d:H-d] - last_input
                #print(part_input.shape)
                part[i] = torch.squeeze(self.avgpool(part_input))
                last_input = torch.nn.functional.pad(part_input, (p,p,p,p), mode='constant', value=0)
            name = 'classifier'+str(i)
            c = getattr(self,name)
            predict[i] = c(part[i])
            d = d - p

        y = []
        for i in range(self.part):
            y.append(predict[i])
        return y

'''
# debug model structure
# Run this code with:
python model.py
'''
if __name__ == '__main__':
# Here I left a simple forward function.
# Test the model, before you train it. 
    net = CPB(751)
    #net = ft_net_SE(751)
    print(net)
    input = Variable(torch.FloatTensor(4, 3, 320, 320))
    output = net(input)
    print('net output size:')
    #print(output.shape)
