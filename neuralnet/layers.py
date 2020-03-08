import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLayer(nn.Module):
    '''Custom Content Layer'''
    def __init__(self, saved_feature):
        super(ContentLayer, self).__init__()
        self.saved_feature = saved_feature.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.saved_feature)
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
    