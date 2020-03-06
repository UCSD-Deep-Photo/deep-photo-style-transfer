"""
This file is copied and customized from CycleGAN GitHub repo:
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/train.py
"""
import os
from gans.util import html
import time
from gans.dataset import create_dataset
from gans.models import create_model
from gans.util.visualizer import Visualizer, save_images
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path
import logging

def train(config):
    dataset = create_dataset(config)  # create a dataset given config
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logging.info('The number of training images = %d' % dataset_size)

    model = create_model(config)      # create a model given config
    model.setup(config)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(config)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(config['epoch_count'], config['n_epochs'] + config['n_epochs_decay'] + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % config['print_freq'] == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += config['batch_size']
            epoch_iter += config['batch_size']
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % config['display_freq'] == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % config['update_html_freq'] == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % config['print_freq'] == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / config['batch_size']
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if config['display_id'] > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % config['save_latest_freq'] == 0:   # cache our latest model every <save_latest_freq> iterations
                logging.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if config.get('save_by_iter') else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % config['save_epoch_freq'] == 0:              # cache our model every <save_epoch_freq> epochs
            logging.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, config['n_epochs'] + config['n_epochs_decay'], time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

def test(config):
    config['isTrain'] = False
    config['num_threads'] = 0   # test code only supports num_threads = 1
    config['batch_size'] = 1    # test code only supports batch_size = 1
    config['serial_batches'] = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    config['no_flip'] = True    # no flip; comment this line if results on flipped images are needed.
    config['display_id'] = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(config)  # create a dataset given config['dataset_mode'] and other options
    model = create_model(config)      # create a model given config['model and other options
    model.setup(config)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(config['results_dir'], config['model'], '{}_{}'.format(config['test_phase'], config['epoch']))  # define the website directory
    if config['load_iter'] > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, config['load_iter'])
    logging.info('creating web directory {}'.format(web_dir))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (config['model'], config['test_phase'], config['epoch']))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if config['eval']:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= config['num_test']:  # only apply our model to config['num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            logging.info('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=config['aspect_ratio'], width=config['display_winsize'])
    webpage.save()  # save the HTML

def load_config(path):
    """
    Load the configuration from config.yaml.
    """
    config = yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)

    # Setup logging
    config['timestamp'] = datetime.now().strftime('%m%d_%H%M%S')
    logging.basicConfig(filename='out/{}__worker.log'.format(config['timestamp']), 
                        level=logging.INFO, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Configuration file: {}".format(args.config))
    logging.info("Using model {}".format(config['model']))

    return config

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Train our GAN')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    parser.add_argument('--test', dest='test', help='test model', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)

    if args.test:
        test(config)
    else:
        train(config)