import os
import time
import torch
import logging
import torch.optim as optim
from neuralnet.utils import showImage, checkGPU, animate_progress, unNormalize, plot_losses
import numpy as np
import torch.nn as nn
from neuralnet.models import TVLoss


def train(model, dataset, config):
    result       = []
    ts           = time.time()
    train_loss   = 0.0
    counter      = 0
    model.train()
    total_iters = 0

    t_data_time = time.time()
    for epoch in range(config['epoch_count'], config['n_epochs'] + config['n_epochs_decay'] + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += config['batch_size']
            epoch_iter += config['batch_size']
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if total_iters % config['print_iter_freq'] == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                logging.info('Total iterations {}, Loss:{}'.format(total_iters, getattr(model, 'loss_' + model.loss_names[-1]).item(), time.time() - t_data_time))

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
    
    return None