import torch
import torch.nn as nn
import torch.nn.functional as TF


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