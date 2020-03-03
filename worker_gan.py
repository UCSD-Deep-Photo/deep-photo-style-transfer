import os
import yaml
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path
import gans
from gans.train_gan import train
from gans.dataset import create_dataset

def load_config():
    """
    Load the configuration from config.yaml.
    """
    parser = argparse.ArgumentParser(description='Worker to train our models')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)

    # Setup logging
    config['timestamp'] = datetime.now().strftime('%m%d_%H%M%S')
    logging.basicConfig(filename='out/{}__worker.log'.format(config['timestamp']), 
                        level=logging.INFO, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Configuration file: {}".format(args.config))
    logging.info("Using model {}".format(config['model']))
    return config

def model_loader(config):
    """
    Loads new model
    """
    model = getattr(gans, config['model'])(config)
    model.setup(config)
    return model

def data_loader(config):
    dataset = create_dataset(config)  # create a dataset given config['dataset_mode'] and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logging.info('The number of training images = %d' % dataset_size)
    return dataset, dataset_size

def main():
    """
    Main
    """
    config = load_config()

    # Load Model
    model = model_loader(config)

    # Load Dataset
    dataset, dataset_size = data_loader(config)
    # Train
    train(model, dataset, config)

    logging.info("Worker completed!")

if __name__ == '__main__':
    main()
