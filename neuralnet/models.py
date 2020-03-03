import torch
import torch.nn as nn
import logging

from torchvision import models
from neuralnet.loss import *
from neuralnet.layers import *
from neuralnet.base_model import BaseModel


class vgg19(BaseModel):
    def __init__(self, config):
        super(vgg19, self).__init__(config)
        
        self.alpha = config['alpha']
        self.beta = config['beta']
        # Initialize model
        self.model = models.vgg19(pretrained=True)

        # Freeze the weights
        for param in self.model.parameters():
            param.requires_grad = False
        self.loss_names = ['S','C','TV','final']
        self.model_names = ['S']
        self._setup_cnn()
        self.set_style_content(self.style, self.content)
        if self.isTrain:
            self.optimizer_CS = torch.optim.Adam(self.net_S.parameters(), lr=config['lr'])
            self.optimizer_GEN = torch.optim.Adam([self.generated.requires_grad_(True)], lr=config['lr'])
            self.optimizers.append(self.optimizer_CS)
            self.optimizers.append(self.optimizer_GEN)

    def _setup_cnn(self):
        self.conv1 = nn.Sequential(
            self.model.features[0], # conv2d
            self.model.features[1], # relu
            self.model.features[2], # conv2d
            self.model.features[3], # relu
            self.model.features[4], # maxpool
        ).to(self.device)
        self.conv2 = nn.Sequential(
            self.model.features[5], # conv2d
            self.model.features[6], # relu
            self.model.features[7], # conv2d
            self.model.features[8], # relu
            self.model.features[9], # maxpool
        ).to(self.device)
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
        ).to(self.device)
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
        ).to(self.device)
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
        ).to(self.device)

    def set_style_content(self, style, content):
        modules = []
        modules.append(StyleLayer(style))
        modules.append(self.conv1)

        style   = self.conv1(style)
        content = self.conv1(content)
        modules.append(StyleLayer(style))
        modules.append(self.conv2)

        style   = self.conv2(style)
        content = self.conv2(content)
        modules.append(StyleLayer(style))
        modules.append(self.conv3)

        style   = self.conv3(style)
        content = self.conv3(content)
        modules.append(StyleLayer(style))
        modules.append(ContentLayer(content))
        modules.append(self.conv4)

        style   = self.conv4(style)
        content = self.conv4(content)
        modules.append(StyleLayer(style))
        modules.append(self.conv5)

        self.net_S = nn.Sequential(*modules).to(self.device)

    def set_input(self, input):
        self.data_batch = input.to(self.device)

    def __call__(self, x):
        ''' Make Model callable with default set to forward()'''
        self.set_input(x)
        return self.forward()

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        return self.net_S.forward(self.data_batch)
        
    def backward(self):
        self.loss_C = sum(layer.loss for layer in self.net_S if type(layer) is ContentLayer)
        self.loss_S = sum(layer.loss for layer in self.net_S if type(layer) is StyleLayer)
        self.loss_TV = TVLoss()(self.generated)
        self.loss_final = (self.alpha * self.loss_C) + (self.beta * self.loss_S) + (0.0001 * self.loss_TV)
        self.loss_final.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.optimizer_CS.zero_grad()
        self.optimizer_GEN.zero_grad()
        self.forward()
        self.backward()
        self.optimizer_CS.step()
        self.optimizer_GEN.step()