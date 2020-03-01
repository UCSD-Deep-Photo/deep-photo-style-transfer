import torch
import torchvision.transforms as transforms
from PIL import Image

def image_transforms(image):
    img_transforms = transforms.Compose([
        transforms.Resize(512), # Only this small for training
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    return img_transforms(image).float()


def image_loader(image_name, generate=False):
    image = Image.open(image_name)
    image = image_transforms(image)
    image = torch.tensor(image.cpu().clone().detach().requires_grad_(False))
    image = image.unsqueeze(0)
    if generate:
        return torch.randn(image.data.size()) ## TODO: CHANGE THIS TO PINK NOISE
    else: 
        return image
