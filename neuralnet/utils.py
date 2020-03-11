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
from PIL import Image

def checkGPU():
    logging.info("GPU Usage - cached:{}/{} GB, allocated:{}/{} GB".format(round(torch.cuda.memory_cached()/1024/1024/1024,5), 
                round(torch.cuda.max_memory_cached()/1024/1024/1024,5), 
                round(torch.cuda.memory_allocated()/1024/1024/1024, 5), 
                round(torch.cuda.max_memory_allocated()/1024/1024/1024, 5)))

def showImage(x, title='Image', save_file=False, orig_color=False):
    '''Function to visualize images'''
    img = x[0].cpu().clone().detach() # get first image in batch and make a copy
    if not orig_color:
        img = unNormalize(img)
    plt.title(title)
    transpose_im = np.transpose(img,(1,2,0))
    
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0.,0.,1.,1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(transpose_im, aspect='auto')
    fig.savefig('out/fig_'+save_file+'.jpeg')

    plt.imshow(transpose_im)
    if save_file:
        if not os.path.exists('out'):
	        os.makedirs('out')
        plt.savefig(('out/' + save_file + '.png'))
    else:
        plt.show() 
        
#     fig = plt.figure()
#     fig.canvas.draw()
#     w,h = fig.canvas.get_width_height()
#     buf = np.fromstring(fig.canvas.tostring_argb(),dtype=np.uint8)
#     buf.shape = (w,h,4)
#     pil_im = Image.frombytes("RGBA", (w,h), buf.tostring())
#     pil_im.save('out/pil_'+save_file+'.png')    

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
    
    fig = plt.figure(figsize=(10,10),frameon=False)
    fig.patch.set_visible(False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
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
    
# Print iterations progress
def progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
