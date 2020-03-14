import os
import torch
import logging

from datetime import datetime
from torchvision import transforms
from matplotlib import pyplot as plt

from PIL import Image
from skimage.color import rgb2yuv, yuv2rgb

import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML

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
    #plt.imshow(transforms.ToPILImage()(img))
    plt.imshow(np.transpose(img,(1,2,0)))
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

def animate_progress(img_progress, save_file): 
    
    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    
    fig = plt.figure(figsize=(10,10))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_progress]
    ani = animation.ArtistAnimation(fig, ims, blit=True)

    ani.save('out/' + save_file + '.mp4', writer=writer)
    
    HTML(ani.to_jshtml())
    
def plot_losses(losses, save_file, title='Loss over Epochs'):
    
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set(xlabel='epoch', ylabel='loss', title=title)
    fig.savefig('out/' + save_file + '.png')

def original_colors(generated_img,content_img,use_gpu):
    '''Replaces generate image colors with original content image colors'''
    # From RGB to YUV color space
    content_img = unNormalize(content_img[0].cpu().clone().detach()).permute(1,2,0)
    generated_img = unNormalize(generated_img[0].cpu().clone().detach()).permute(1,2,0)

    c_yuv = rgb2yuv(content_img)
    g_yuv = rgb2yuv(generated_img)

    # Swap colors, keep greyscale
    c_yuv[:,:,0] = g_yuv[:,:,0] 

    # Back to RGB color space
    g_rgb = yuv2rgb(c_yuv)
    orig_color_img = torch.from_numpy(g_rgb).permute(2,0,1).unsqueeze(0)

    return orig_color_img

def convert_tensor_to_image(t):
    tmp = t.cpu().detach().numpy() # don't mess with the original, convert tensor to numpy array
    tmp = np.transpose(tmp, (0, 2, 3, 1)) # change structure
    tmp = np.squeeze(tmp) # get rid of last dimension added for Tensor
    return tmp

# def convert_image_to_tensor(img):
#     tmp = np.transpose(img, (2, 0, 1))
#     return torch.Tensor(tmp).unsqueeze(0)

def get_laplacian_grad_loss(img_as_tensor, laplacian):
    original_shape = img_as_tensor.shape # save for later
    
    tmp = convert_tensor_to_image(img_as_tensor)
    tmp = np.clip(tmp, 0, 1) # laplacian requires values to be [0, 1]
    
    gradient = laplacian @ tmp.reshape(-1, 3)
    loss = (gradient * tmp.reshape(-1, 3)).sum() # matrix sum
    gradient = 2 * gradient.reshape(original_shape) 
    
    # gradient_as_tensor = convert_image_to_tensor(gradient)
    return loss, torch.Tensor(gradient)
