import torch
import torchvision.transforms as transforms
from PIL import Image

def image_transforms(image):
    img_transforms = transforms.Compose([
        transforms.Resize(128), # Only this small for training
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    return img_transforms(image).float()


def image_loader(image_name):
    image = Image.open(image_name)
    image = image_transforms(image)
    image = torch.tensor(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image
