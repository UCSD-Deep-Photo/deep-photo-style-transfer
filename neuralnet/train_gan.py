import os
import time
import torch
import logging
import torch.optim as optim
from neuralnet.utils import showImage, checkGPU, animate_progress, unNormalize, plot_losses
import numpy as np
import torch.nn as nn
from neuralnet.models import TVLoss


# TODO: ADD EARLY STOPPING AND SAVE BEST IMAGE
# TODO: ADD LOSS PLOTS
# TODO: save train losses
# TODO: models should handle normalization
def train(model, config):
    result       = []
    ts           = time.time()
    train_loss   = 0.0
    counter      = 0
    model.train()
    img_progress = []

    t_data_time = time.time()
    for epoch in range(config['epoch_count'], config['n_epochs'] + config['n_epochs_decay'] + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        iter_data_time = time.time()    # timer for data loading per iteration
        if (epoch % 100) == 0 or (epoch == 1):
            padded_epoch = '{0:04}'.format(epoch)
            #showImage(generated_img,'Generated Image',(timestamp + '_' + save_file + '_e' + str(padded_epoch))) #save img for first and every 100 epochs

        if (epoch % 10 == 0) or (epoch == 1):
            img_progress.append(unNormalize(model.generated[0].detach().cpu()))

        iter_start_time = time.time()  # timer for computation per iteration

        model.set_input(model.generated)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        if epoch % config['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
            logging.info('saving the model at the end of epoch {}'.format(epoch))
            model.save_networks('latest')
            model.save_networks(epoch)
        if epoch % config['log_freq'] == 0: 
            logging.info('Epoch: {}/{}, Loss: {}, Time: {}'.format(
                            epoch,
                            config['n_epochs'] + config['n_epochs_decay'], 
                            getattr(model, 'loss_' + model.loss_names[-1]).item(), 
                            time.time() - t_data_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    
    
    animate_progress(img_progress, config['save_file']+'_animated')
    # plot_losses(losses, config['save_file']+'_losses')
    
    return result