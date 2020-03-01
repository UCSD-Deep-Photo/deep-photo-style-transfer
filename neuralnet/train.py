import os
import time
import torch
import logging
import torch.optim as optim
from neuralnet.utils import showImage, checkGPU

def train(model, content_img, style_img, generated_img, save_file, alpha=5, beta=0.01,  lr=0.05, epochs=1000,early_stop=5,timestamp=''):
    use_gpu      = next(model.parameters()).is_cuda
    result       = []
    ts           = time.time()
    train_loss   = 0.0
    counter      = 0
    model.train()
    # TODO: ADD EARLY STOPPING AND SAVE BEST IMAGE
    # TODO: ADD LOSS PLOTS
    
    if use_gpu:
        style_img     = style_img.cuda()
        content_img   = content_img.cuda()
        generated_img = generated_img.cuda()

    """
    Init content features
    """
    logging.info('Initializing Content Features.')
    showImage(content_img, 'Content Image', (timestamp + '__img_content'))
    _, _ = model(content_img,img_type='content')

    """
    Init style features 
    """
    logging.info('Initializing Style Features.')
    showImage(style_img, 'Style Image', (timestamp + '__img_style'))
    _, _ = model(style_img,img_type='style')
    
    """
    Generate Image
    """
    logging.info('Generating Image.')
    optimizer = optim.Adam([generated_img.requires_grad_(True)], lr=lr)
    
    for epoch in range(1,epochs+1):
        if (epoch % 100) == 0 or (epoch == 1):
            padded_epoch = '{0:04}'.format(epoch)
            showImage(generated_img,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch))) #save img for first and every 100 epochs

        optimizer.zero_grad()    
        s_loss, c_loss = model(generated_img,img_type='generated')
        loss = (alpha * c_loss) + (beta * s_loss)
        loss.backward()
        optimizer.step(closure=(loss.item))
        train_loss += loss.item()
        counter += 1

        if (epoch % 10) == 0:
            logging.info('Epoch: {}, Loss: {}, Time: {}'.format(epoch,loss.item(), ts))
    train_loss /= counter
    result.append((train_loss))
    checkGPU()
    logging.info('Final Loss: {}'.format(loss.item()))
    return result

