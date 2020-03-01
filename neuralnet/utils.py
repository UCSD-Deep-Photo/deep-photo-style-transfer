import os
import torch
import logging

from datetime import datetime
from torchvision import transforms
from matplotlib import pyplot as plt

def checkGPU():
    logging.info("GPU Usage - cached:{}/{} GB, allocated:{}/{} GB".format(round(torch.cuda.memory_cached()/1024/1024/1024,5), 
                round(torch.cuda.max_memory_cached()/1024/1024/1024,5), 
                round(torch.cuda.memory_allocated()/1024/1024/1024, 5), 
                round(torch.cuda.max_memory_allocated()/1024/1024/1024, 5)))

def showImage(x, title='Image', save_file=False):
    '''Function to visualize images'''
    img = x[0].cpu().clone().detach() # get first image in batch and make a copy
    img = unNormalize(img)
    plt.title(title)
    plt.imshow(transforms.ToPILImage()(img))
    if save_file:
        if not os.path.exists('out'):
	        os.makedirs('out')
        plt.savefig(('out/' + save_file + '.png'))
    else:
        plt.show() 
    del img 

def unNormalize(x):
    '''Undo normalization so image is displayed with original colors'''
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    unnorm = transforms.Normalize((-mean / std),(1.0 / std)) # ImageNet mean and std
    return unnorm(x)