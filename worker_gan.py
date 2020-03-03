import os
import yaml
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path
import neuralnet.gans
from neuralnet.train_gan import train

def load_config():
    """
    Load the configuration from config.yaml.
    """
    parser = argparse.ArgumentParser(description='Worker to train our models')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)

    # Set generated image file name
    filename_c_img = Path(config['content_image']).stem
    filename_s_img = Path(config['style_image']).stem
    config['save_file'] = filename_c_img + '_' + filename_s_img
    
    # Setup logging
    config['timestamp'] = datetime.now().strftime('%m%d_%H%M%S')
    logging.basicConfig(filename='out/{}__worker.log'.format(config['timestamp']), 
                        level=logging.INFO, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Configuration file: {}".format(args.config))
    logging.info("Using model {}".format(config['model']))
    logging.info("Epochs: {}, Learning rate: {}, Content Image: {}, Style Image: {}".format(config['n_epochs']+config['n_epochs_decay'],
                                                                                            config['lr'], 
                                                                                            config['content_image'], 
                                                                                            config['style_image']))

    return config

def model_loader(config):
    """
    Loads new model
    """
    tt = getattr(neuralnet.gans, config['model'])
    model = getattr(neuralnet.gans, config['model'])(config)
    model.setup(config)
    return model

def main():
    """
    Main
    """
    config = load_config()

    # Load Model
    model = model_loader(config)

    # Train
    train(model, config)

    logging.info("Worker completed!")

if __name__ == '__main__':
    main()