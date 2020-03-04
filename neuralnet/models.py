import torch
import torch.nn as nn
import torch.nn.functional as TF
from pycocotools.coco import COCO
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pack_padded_sequence
import logging
from abc import ABC,abstractmethod
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms

class ContentLayer(nn.Module):
    '''Custom Content Layer'''
    def __init__(self, saved_feature):
        super(ContentLayer, self).__init__()
        self.saved_feature = saved_feature.detach()

    def forward(self, input):
        self.loss = TF.mse_loss(input, self.saved_feature)
        return input

class StyleLayer(nn.Module):
    '''Custom Style Layer'''
    def __init__(self, saved_feature):
        super(StyleLayer, self).__init__()
        self.saved_feature = self.convariance_matrix(saved_feature).detach()

    def forward(self, input):
        c_matrix = self.convariance_matrix(input)
        self.loss = TF.mse_loss(c_matrix, self.saved_feature)
        return input

    def convariance_matrix(self,input):
        '''Cacluate covariance matrix'''
        b, w, h, c = input.size()  # batch, width, height, channels (RBG)
        features = input.view(b * w, h * c)  
        c_matrix = torch.mm(features, features.t())  
        return c_matrix.div(b * w * h * c)

class vgg19(nn.Module):
    def __init__(self):
        super(vgg19, self).__init__()

        # Initialize model
        self.model = models.vgg19(pretrained=True)

        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.c_loss, self.c_layers = [],[]
        self.s_loss, self.s_layers = [],[]

        self.conv1 = nn.Sequential(
            self.model.features[0], # conv2d
            self.model.features[1], # relu
            self.model.features[2], # conv2d
            self.model.features[3], # relu
            self.model.features[4], # maxpool
        )
        self.conv2 = nn.Sequential(
            self.model.features[5], # conv2d
            self.model.features[6], # relu
            self.model.features[7], # conv2d
            self.model.features[8], # relu
            self.model.features[9], # maxpool
        )
        self.conv3 = nn.Sequential(
            self.model.features[10], # conv2d  
            self.model.features[11], # relu 
            self.model.features[12], # conv2d  
            self.model.features[13], # relu 
            self.model.features[14], # conv2d  
            self.model.features[15], # relu 
            self.model.features[16], # conv2d  
            self.model.features[17], # relu 
            self.model.features[18], # maxpool
        )
        self.conv4 = nn.Sequential(
            self.model.features[19], # conv2d 
            self.model.features[20], # relu
            self.model.features[21], # conv2d 
            self.model.features[22], # relu
            self.model.features[23], # conv2d 
            self.model.features[24], # relu
            self.model.features[25], # conv2d 
            self.model.features[26], # relu
            self.model.features[27], # maxpool
        )
        self.conv5 = nn.Sequential(
            self.model.features[28], # conv2d 
            self.model.features[29], # relu
            self.model.features[30], # conv2d 
            self.model.features[31], # relu
            self.model.features[32], # conv2d 
            self.model.features[33], # relu
            self.model.features[34], # conv2d 
            self.model.features[35], # relu
            self.model.features[36], # maxpool
        )

    def __call__(self, x, img_type):
        ''' Make Model callable with default set to forward()'''
        return self.forward(x, img_type)

    def encoder(self, x, img_type='generated'):
        # Reset loss for each fwd pass:
        self.s_loss = []
        self.c_loss = []

        layer = 0
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv1(x)

        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv2(x)
        
        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv3(x)

        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'content': self.c_layers.append(ContentLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
            x = self.c_layers[0].forward(x)
            self.c_loss.append(self.c_layers[0].loss)
        x = self.conv4(x)

        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer](x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv5(x)

        return sum(self.s_loss), sum(self.c_loss)

    def forward(self, x,img_type):
        s_loss, c_loss = self.encoder(x,img_type)
        return s_loss,c_loss
    
    
class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()

    def forward(self, x):
        height = x.size()[2]
        width = x.size()[3]
        
        n_width_next = x[:, :, :, 1:]
        n_width = x[:, :, :, :width-1]

        n_height_next = x[:, :, 1:, :]
        n_height = x[:, :, :height-1, :]

        tv_height = torch.sum(torch.abs(n_height_next - n_height), axis=[1, 2, 3])
        tv_width = torch.sum(torch.abs(n_width_next - n_width), axis=[1, 2, 3])

        tv_loss = torch.mean(tv_height + tv_width)
        return tv_loss

class vgg19_new(nn.Module):
    def __init__(self):
        super(vgg19_new, self).__init__()

        # Initialize model
        self.model = models.vgg19(pretrained=True)

        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.c_loss, self.c_layers = [],[]
        self.s_loss, self.s_layers = [],[]

        self.conv0 = nn.Sequential(
            self.model.features[0], # conv2d
            self.model.features[1], # relu
        )

        self.conv1 = nn.Sequential(
            self.model.features[2], # conv2d
            self.model.features[3], # relu
            self.model.features[4], # maxpool
            self.model.features[5], # conv2d
            self.model.features[6], # relu
        )
        self.conv2 = nn.Sequential(
            self.model.features[7], # conv2d
            self.model.features[8], # relu
            self.model.features[9], # maxpool
            self.model.features[10], # conv2d  
            self.model.features[11], # relu 
        )
        self.conv3 = nn.Sequential(
            self.model.features[12], # conv2d  
            self.model.features[13], # relu 
            self.model.features[14], # conv2d  
            self.model.features[15], # relu 
            self.model.features[16], # conv2d  
            self.model.features[17], # relu 
            self.model.features[18], # maxpool
            self.model.features[19], # conv2d 
            self.model.features[20], # relu
        )
        self.conv4a = nn.Sequential(
            self.model.features[21],  # conv2d 
            self.model.features[22], # relu
        )
        self.conv4b = nn.Sequential(
            self.model.features[23], # conv2d 
            self.model.features[24], # relu
            self.model.features[25], # conv2d 
            self.model.features[26], # relu
            self.model.features[27], # maxpool
            self.model.features[28], # conv2d 
        )
        self.conv5 = nn.Sequential(
            self.model.features[29], # relu
            self.model.features[30], # conv2d 
            self.model.features[31], # relu
            self.model.features[32], # conv2d 
            self.model.features[33], # relu
            self.model.features[34], # conv2d 
            self.model.features[35], # relu
            self.model.features[36], # maxpool
        )
        self.final = nn.Sequential(
            self.model.avgpool,
            self.model.classifier
        )

    def __call__(self, x, img_type):
        ''' Make Model callable with default set to forward()'''
        return self.forward(x, img_type)

    def encoder(self, x, img_type='generated'):
        # Reset loss for each fwd pass:
        self.s_loss = []
        self.c_loss = []

        # first conv layer
        x = self.conv0(x)

        # insert style layer @ conv_1
        layer = 0 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv1(x)

        # insert style layer @ conv_2
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv2(x)
        
        # insert style layer @ conv_3
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv3(x)

        # insert style layer @ conv_4
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv4a(x)

        # insert content layer @ conv_4_2
        if img_type == 'content': self.c_layers.append(ContentLayer(x))
        elif img_type == 'generated': 
            x = self.c_layers[0].forward(x)
            self.c_loss.append(self.c_layers[0].loss)
        x = self.conv4b(x)

        # insert style layer @ conv_5
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv5(x)

        return sum(self.s_loss), sum(self.c_loss)

    def forward(self, x,img_type):
        s_loss, c_loss = self.encoder(x,img_type)
        return s_loss,c_loss

class vgg19_new_fc(nn.Module):
    def __init__(self):
        super(vgg19_new_fc, self).__init__()

        # Initialize model
        self.model = models.vgg19(pretrained=True)

        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.c_loss, self.c_layers = [],[]
        self.s_loss, self.s_layers = [],[]

        self.conv0 = nn.Sequential(
            self.model.features[0], # conv2d
            self.model.features[1], # relu
        )

        self.conv1 = nn.Sequential(
            self.model.features[2], # conv2d
            self.model.features[3], # relu
            self.model.features[4], # maxpool
            self.model.features[5], # conv2d
            self.model.features[6], # relu
        )
        self.conv2 = nn.Sequential(
            self.model.features[7], # conv2d
            self.model.features[8], # relu
            self.model.features[9], # maxpool
            self.model.features[10], # conv2d  
            self.model.features[11], # relu 
        )
        self.conv3 = nn.Sequential(
            self.model.features[12], # conv2d  
            self.model.features[13], # relu 
            self.model.features[14], # conv2d  
            self.model.features[15], # relu 
            self.model.features[16], # conv2d  
            self.model.features[17], # relu 
            self.model.features[18], # maxpool
            self.model.features[19], # conv2d 
            self.model.features[20], # relu
        )
        self.conv4a = nn.Sequential(
            self.model.features[21],  # conv2d 
            self.model.features[22], # relu
        )
        self.conv4b = nn.Sequential(
            self.model.features[23], # conv2d 
            self.model.features[24], # relu
            self.model.features[25], # conv2d 
            self.model.features[26], # relu
            self.model.features[27], # maxpool
            self.model.features[28], # conv2d 
        )
        self.conv5 = nn.Sequential(
            self.model.features[29], # relu
            self.model.features[30], # conv2d 
            self.model.features[31], # relu
            self.model.features[32], # conv2d 
            self.model.features[33], # relu
            self.model.features[34], # conv2d 
            self.model.features[35], # relu
            self.model.features[36], # maxpool
        )
        self.final = nn.Sequential(
            self.model.avgpool,
            self.model.classifier
        )

    def __call__(self, x, img_type):
        ''' Make Model callable with default set to forward()'''
        return self.forward(x, img_type)

    def encoder(self, x, img_type='generated'):
        # Reset loss for each fwd pass:
        self.s_loss = []
        self.c_loss = []

        # first conv layer
        x = self.conv0(x)

        # insert style layer @ conv_1
        layer = 0 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv1(x)

        # insert style layer @ conv_2
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv2(x)
        
        # insert style layer @ conv_3
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv3(x)

        # insert style layer @ conv_4
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv4a(x)

        # insert content layer @ conv_4_2
        if img_type == 'content': self.c_layers.append(ContentLayer(x))
        elif img_type == 'generated': 
            x = self.c_layers[0].forward(x)
            self.c_loss.append(self.c_layers[0].loss)
        x = self.conv4b(x)

        # insert style layer @ conv_5
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv5(x)

        return sum(self.s_loss), sum(self.c_loss)

    def forward(self, x,img_type):
        s_loss, c_loss = self.encoder(x,img_type)
        return s_loss,c_loss

class resnet152(nn.Module):
    def __init__(self):
        super(resnet152, self).__init__()

        # Initialize model
        self.model = models.resnet152(pretrained=True)

        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False

        self.c_loss, self.c_layers = [],[]
        self.s_loss, self.s_layers = [],[]

        self.conv0 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu
        )
        self.conv1 = nn.Sequential(
            self.model.maxpool,
            self.model.layer1
        )
        self.conv2 = self.model.layer2
        self.conv3 = self.model.layer3
        self.conv4 = self.model.layer4


    def __call__(self, x, img_type):
        ''' Make Model callable with default set to forward()'''
        return self.forward(x, img_type)

    def encoder(self, x, img_type='generated'):
        # Reset loss for each fwd pass:
        self.s_loss = []
        self.c_loss = []

        # first conv layer
        x = self.conv0(x)

        # insert style layer @ conv_1
        layer = 0 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv1(x)

        # insert style layer @ conv_2
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv2(x)
        
        # insert style layer @ conv_3
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv3(x)

        # insert style layer @ conv_4
        layer += 1 
        if img_type == 'style': self.s_layers.append(StyleLayer(x))
        elif img_type == 'content': self.c_layers.append(ContentLayer(x))
        elif img_type == 'generated': 
            x = self.c_layers[0].forward(x)
            self.c_loss.append(self.c_layers[0].loss)
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv4(x)

        return sum(self.s_loss), sum(self.c_loss)

    def forward(self, x,img_type):
        s_loss, c_loss = self.encoder(x,img_type)
        return s_loss,c_loss