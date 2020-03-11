import os
import time
import torch
import logging
import torch.optim as optim
from neuralnet.utils import *
import numpy as np
import torch.nn as nn
from neuralnet.models import TVLoss

def train(model, content_img, style_img, generated_img, save_file, alpha=5, beta=0.01,  gamma=0, lr=0.05, epochs=100,early_stop=5,timestamp='',orig_colors=False, LBFGS=False):
    use_gpu      = next(model.parameters()).is_cuda
    result       = []
    train_loss   = 0.0
    counter      = 0
    model.train()    
    img_progress = []
    losses = []

    if use_gpu:
        style_img     = style_img.cuda()
        content_img   = content_img.cuda()
        generated_img = generated_img.cuda()

    """
    Init content features
    """
    logging.info('Initializing Content Features.')
    _, _ = model(content_img,img_type='content')
    showImage(content_img,'Content Image',(timestamp + '_content' + '_' + save_file))

    """
    Init style features 
    """
    logging.info('Initializing Style Features.')
    _, _ = model(style_img,img_type='style')
    showImage(style_img,'Style Image',(timestamp + '_style' + '_' + save_file))              
    
    """
    Generate Image
    """
    logging.info('Generating Image.')
    tv = TVLoss()

    # Toggle LBFGS and Adam optimizers
    if LBFGS == True:
        optimizer = optim.LBFGS([generated_img.requires_grad_(True)])
    else :
        optimizer = optim.Adam([generated_img.requires_grad_(True)], lr=lr)
    

    print("Training...", end="")
    progress_bar(0, epochs, prefix='Progress:', suffix='Complete', length=50)
    
    for epoch in range(1,epochs+1):
        ts = time.time()

        # For testing: save image every X epochs
        # if (epoch % 50) == 0 or (epoch == 1):
        #     padded_epoch = '{0:04}'.format(epoch)
        #     if orig_colors:
        #         g_img_with_orig_color = original_colors(generated_img, content_img, use_gpu)
        #         showImage(g_img_with_orig_color,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch)),orig_colors) 
        #     else: 
        #         showImage(generated_img,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch)))

        if epoch <= 50:
            save_int = 1
        elif epoch <= 100:
            save_int = 5
        elif epoch <= 500:
            save_int = 10
        else:
            save_int = 50
            
        if (epoch % save_int == 0) or (epoch == 1):
            if orig_colors:
                g_img_with_orig_color = original_colors(generated_img, content_img, use_gpu)
                img_progress.append(g_img_with_orig_color[0])
            else: 
                img_progress.append(unNormalize(generated_img[0].detach().cpu()))

        optimizer.zero_grad()    

        tv_loss = tv(generated_img)
        s_loss, c_loss = model(generated_img, img_type='generated')
        loss = (alpha * c_loss) + (beta * s_loss) + (gamma * tv_loss)
        loss.backward()
        optimizer.step(closure=(loss.item))
        train_loss += loss.item()
        counter += 1

        if (epoch % 10) == 0:
            logging.info('Epoch: {}, Loss: {}, Time: {}'.format(epoch,loss.item(), ts))
            print('Epoch: {}, Loss: {}, Time: {}'.format(epoch,loss.item(), ts))
            losses.append(loss.item())
            
        if (epoch == 3000):
            padded_epoch = '{0:04}'.format(epoch)
            
            if orig_colors:
                g_img_with_orig_color = original_colors(generated_img, content_img, use_gpu)
                showImage(g_img_with_orig_color,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch))) 
            else: 
                showImage(generated_img,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch)))  

    train_loss /= counter
    result.append((train_loss))
    #checkGPU()
    logging.info('Final Loss: {}'.format(loss.item()))
    
    # save final output image
    padded_epoch = '{0:04}'.format(epoch)
    if orig_colors:
        g_img_with_orig_color = original_colors(generated_img, content_img, use_gpu)
        showImage(g_img_with_orig_color,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch))) 
    else: 
        showImage(generated_img,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch))) 
    
    animate_progress(img_progress, timestamp+'_'+save_file+'_animated')
    print("done.")
    plot_losses(losses, timestamp+'_'+save_file+'_losses')
    
    
    return result
