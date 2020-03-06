import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools.coco import COCO
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pack_padded_sequence
import logging
from abc import ABC,abstractmethod
from matplotlib import pyplot as plt
from torchvision import datasets, models, transforms

from neuralnet.layers import *
from neuralnet.loss import *

class vgg19(nn.Module):
    def __init__(self, content_mask=None, style_mask=None):
        super(vgg19, self).__init__()

        # Initialize model
        self.model = models.vgg19(pretrained=True)
        
        self.content_mask = content_mask
        self.style_mask = content_mask
        
        if self.content_mask is not None:
            self.content_mask = torch.Tensor(self.content_mask).cuda()
            self.style_mask = torch.Tensor(self.style_mask).cuda()

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
        if img_type == 'style': self.s_layers.append(StyleLayer(x, self.content_mask, self.style_mask))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv1(x)

        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x, self.content_mask, self.style_mask))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv2(x)
        
        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x, self.content_mask, self.style_mask))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv3(x)

        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x, self.content_mask, self.style_mask))
        elif img_type == 'content': self.c_layers.append(ContentLayer(x))
        elif img_type == 'generated': 
            x = self.s_layers[layer].forward(x)
            self.s_loss.append(self.s_layers[layer].loss)
            x = self.c_layers[0].forward(x)
            self.c_loss.append(self.c_layers[0].loss)
        x = self.conv4(x)

        layer += 1
        if img_type == 'style': self.s_layers.append(StyleLayer(x, self.content_mask, self.style_mask))
        elif img_type == 'generated': 
            x = self.s_layers[layer](x)
            self.s_loss.append(self.s_layers[layer].loss)
        x = self.conv5(x)

        return sum(self.s_loss), sum(self.c_loss)

    def forward(self, x,img_type):
        s_loss, c_loss = self.encoder(x,img_type)
        return s_loss,c_loss
