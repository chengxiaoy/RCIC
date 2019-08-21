from torchvision import transforms
from PIL import Image
import torch
import random

img = Image.open('test.jpg')

trans = transforms.Compose([
    transforms.FiveCrop(256),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    # transforms.Lambda(lambd=lambda crops: crops[random.randint(0, 4)])

])
tt = trans(img)
tt.save('test1.jpg')
