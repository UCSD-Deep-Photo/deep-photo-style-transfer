import torch
import torch.nn as nn
import torch.nn.functional as TF


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
