import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.nn.functional as F
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
    
    def __init__(self, saved_feature, content_mask, style_mask):
        super(StyleLayer, self).__init__()
        
        self.saved_feature = self.gram_tensor(saved_feature).detach()
        target_feature = saved_feature

        self.style_mask = style_mask.detach()
        self.content_mask = content_mask.detach()

        _, channel_f, height, width = target_feature.size()
        channel = self.style_mask.size()[1]
        
        
        # downsample mask
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2) 
        grid = grid.unsqueeze_(0).cuda()
        mask_ = F.grid_sample(self.style_mask, grid).squeeze(0).cuda()
           
        
        target_feature_3d = target_feature.squeeze(0).clone().cuda()

        size_of_mask = (channel, channel_f, height, width)             
        target_feature_masked = torch.einsum('acd,bcd->abcd', mask_, target_feature_3d)                              
        self.targets = self.gram_tensor(target_feature_masked.detach())
        means = torch.mean(mask_, axis=[1, 2], keepdims=True)
        means[means == 0] = 1
        self.targets /= means
             

    def forward(self, input_feature):
        self.loss = 0
        _, channel_f, height, width = input_feature.size()
        channel = len(self.targets)
        xc = torch.linspace(-1, 1, width).repeat(height, 1).cuda()
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width).cuda()
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2).cuda()
        grid = grid.unsqueeze_(0).to("cuda")
        mask = F.grid_sample(self.content_mask, grid).squeeze(0).cuda()
        input_feature_3d = input_feature.squeeze(0).clone().cuda()
        size_of_mask = (channel, channel_f, height, width)
        
        input_feature_masked = torch.einsum('acd,bcd->abcd', mask, input_feature_3d)        
        self.inputs_G = self.gram_tensor(input_feature_masked)
        means = torch.mean(mask, axis=[1, 2], keepdims=True)
        means[means == 0] = 1

        self.inputs_G /= means
                      
        # Mean across width, height
        # Sum across class channels
        self.loss = torch.sum(torch.mean(((self.inputs_G - self.targets) ** 2), axis=[1, 2]) * torch.squeeze(means))                       
        return input_feature
        

    def gram_tensor(self,input):
        '''Cacluate covariance matrix'''
        class_channels, feat_channels, feat_height, feat_width = input.size()
        features = input.view(class_channels, feat_channels, feat_height * feat_width)
        c_tensor = torch.einsum('mnr,mrk->mnk', features, features.permute(0, 2, 1))
        return c_tensor.div(feat_channels * feat_height * feat_width)
    
    def gram_matrix(self,input):
        '''Cacluate covariance matrix'''
        b, w, h, c = input.size()  # batch, width, height, channels (RBG)
        features = input.view(b * w, h * c).cuda()
        c_matrix = torch.mm(features, features.t())
        return c_matrix.div(b * w * h * c)
    

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
        s_loss, c_loss = self.encoder(x, img_type)
        return s_loss,c_loss
    
    
class TVLoss(nn.Module):
    def __init__(self):
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
        return 2 * tv_loss