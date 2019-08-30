from torchvision import transforms
from PIL import Image
import torch
import random
import cv2
import numpy as np


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, Flip, OneOf, Compose
)
import torch

def strong_aug(p=.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            # CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        # HueSaturationValue(p=1),
    ], p=p)

aug = strong_aug(p=1)
#
#
#
#
#
# image = cv2.imread('test.jpg')
# cv2.imshow('bgr',image)
# cv2.waitKey(1000)
# image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
# cv2.imshow('rgb',image)





# pytorchvision
img = Image.open('B02_s1_w1.png')
tensor = transforms.ToTensor()(img)
img1 = Image.open('B02_s1_w1.png')
img = np.array(img)
img1 = np.array(img1)
img = np.array([img,img1,img,img1,img,img1]).transpose([1,2,0])
img3 = cv2.flip(img,0)
hehe = aug(image = img)

trans = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))
    # transforms.Lambda(lambd=lambda crops: crops[random.randint(0, 4)])

])
tt = trans(img)
tt.save('test1.jpg')
