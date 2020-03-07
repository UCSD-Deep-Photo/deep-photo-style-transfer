import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

def image_transforms(image):
    img_transforms = transforms.Compose([
        transforms.Resize(512), 
        #transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    return img_transforms(image).float()

def mask_and_image_transforms(image, mask):
    
    
    mask = transforms.ToTensor()(mask)
    mask = mask.unsqueeze(0)
    min_length = (min(mask.squeeze().size()))
    scale = 512/min_length
    mask = F.interpolate(mask.float(), scale_factor=scale)
    
    
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0)
    print(image.shape)
    image = F.interpolate(image, scale_factor=scale)
    print(image.shape)
    

    img_transforms = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    
    print(mask.shape)
    print(image.shape)
    print(torch.max(mask).item())
    
    
    return img_transforms(image.squeeze(0)).float(), mask.int().squeeze()


def load_image(image_name, use_mask=False):
    if use_mask:
        mask_name = image_name[:len(image_name) - 4] + ".npy"
        mask = np.load(mask_name)
        image = Image.open(image_name)
        image, mask = mask_and_image_transforms(image, mask)
        image = image.cpu().clone().detach().requires_grad_(False)
        mask = mask.cpu().clone().detach().requires_grad_(False)
        image = image.unsqueeze(0)
        
        classes = [21, 1, 2, 6, 5, 16, 53, 19, 20]
        
        mask = torch.stack([mask == c for c in classes], axis=0).float()
        mask = mask.unsqueeze(0)                       
    else:
        image = Image.open(image_name)
        image = image_transforms(image)
        image = image.cpu().clone().detach().requires_grad_(False)
        image = image.unsqueeze(0)
        mask = torch.ones([1, 1, image.size(0), image.size(1)])
        
    return image, mask

def generate_image(content_img, generate=False):
    if generate:
        return torch.zeros(content_img.data.size()).data.normal_(0.449, 0.226)
    else: 
        return content_img.cpu().clone()
        
