import os
import yaml
import torch
import logging
import argparse
from datetime import datetime
from pathlib import Path
import neuralnet.models
import numpy as np
from neuralnet.image_loader import image_loader, generate_image
from neuralnet.train import train

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
    logging.info("Epochs: {}, Learning rate: {}, Content Image: {}, Style Image: {}".format(config['train_epoch'],
                                                                                            config['learning_rate'], 
                                                                                            config['content_image'], 
                                                                                            config['style_image']))

    return config

def model_loader(config, content_mask, style_mask):
    """
    Loads new model
    """
    return getattr(neuralnet.models, config['model'])(content_mask, style_mask)

def main():
    """
    Main
    """
    config = load_config()
    use_mask = config['use_mask']
    
    # Load images, masks
    content_img, content_mask   = image_loader(config['content_image'], use_mask)
    style_img, style_mask       = image_loader(config['style_image'], use_mask)
    generated_img = generate_image(content_img, config['generate_image'])
    

    # Load Model
    model = model_loader(config, content_mask, style_mask)


    # Use GPU, if available
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logging.info("Using GPU for training.")
        model = model.cuda()
    else:
        logging.info("Using only CPU for training.")
        
    # Train    
    train(
        model, 
        content_img, 
        style_img, 
        generated_img, 
        config['save_file'], 
        alpha=config['alpha'], 
        beta=config['beta'], 
        gamma=config['gamma'],
        lr=config['learning_rate'],
        epochs=config['train_epoch'],
        early_stop=config['early_stop'],
        timestamp=config['timestamp'],
        orig_colors=config['original_colors']
    )
    
    logging.info("Worker completed!")

if __name__ == '__main__':
    main()