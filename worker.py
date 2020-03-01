import os
import yaml
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path
import neuralnet.models
from neuralnet.image_loader import image_loader
from neuralnet.train import train

def load_config():
    """
    Load the configuration from config.yaml.
    """
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description='Worker to train our models')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    args = parser.parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.SafeLoader)

    # Set generated image file name
    filename_c_img = Path(config['content_image']).stem
    filename_s_img = Path(config['style_image']).stem
    config['save_file'] =  filename_c_img + '_' + filename_s_img + '_' + timestamp 

    # Setup logging
    logging.basicConfig(filename='out/worker_{}.log'.format(timestamp), 
                        level=logging.INFO, 
                        format='[%(asctime)s] %(message)s', 
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info("Configuration file: {}".format(args.config))
    logging.info("Using model {}".format(config['model']))
    logging.info("Epochs: {}, Learning rate: {}, Content Image: {}, Style Image: {}".format(config['train_epoch'], config['learning_rate'], config['content_image'], config['style_image']))
    return config

def model_loader(config):
    """
    Loads new model
    """
    return getattr(neuralnet.models, config['model'])()

def main():
    """
    Main
    """
    config = load_config()

    # Load Model
    model = model_loader(config)

    # Load images
    content_image = image_loader(config['content_image'])
    style_image = image_loader(config['style_image'])

    # Use GPU, if available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info("Using GPU for training.")
        model = model.cuda()
    else:
        logging.info("Using only CPU for training.")

    # Train    
    train(model, content_image, style_image, config['save_file'], alpha=config['alpha'], beta=config['beta'], lr=config['learning_rate'], epochs=config['train_epoch'])
    logging.info("Worker completed!")

if __name__ == '__main__':
    main()