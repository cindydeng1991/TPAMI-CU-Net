import os
import cv2
import numpy as np
import random
from PIL import Image, ImageFilter
path='E:/dataset/multi-focus/trainset/'
dir=os.listdir(path + 'label')
dir.sort()
length=len(dir)
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, w=None,h=None,width=None,height=None):
        self.radius = radius
        self.bounds = (w-70, h-70,w+70,h+70)

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)
class MyGaussianBlur2(ImageFilter.Filter):
    name = "GaussianBlur2"

    def __init__(self, radius=2, w=None,h=None,width=None,height=None):
        self.radius = radius
        self.bounds1 = (0, 0, w - 70, height)
        self.bounds2 = (w-70, 0, w+70, h-70)
        self.bounds3 = (w-70, h+70, w+70, height)
        self.bounds4 = (w+70, 0,width, height)

    def filter(self, image):
        if self.bounds1:
            clips1 = image.crop(self.bounds1).gaussian_blur(self.radius)
            clips2 = image.crop(self.bounds2).gaussian_blur(self.radius)
            clips3 = image.crop(self.bounds3).gaussian_blur(self.radius)
            clips4 = image.crop(self.bounds4).gaussian_blur(self.radius)
            image.paste(clips1, self.bounds1)
            image.paste(clips2, self.bounds2)
            image.paste(clips3, self.bounds3)
            image.paste(clips4, self.bounds4)
            return image
        else:
            return image.gaussian_blur(self.radius)

def creatlabel():
    for i in range(length):
        img=cv2.imread(path+'DIV2K_train_HR/'+dir[i])
        img = cv2.pyrDown(cv2.pyrDown(img))
        cv2.imwrite(path+'trainset/label/'+dir[i],img)

for i in range(length):
    image = Image.open(path + 'label/'+dir[i])
    (width,height)=image.size
    w=random.randrange(70,width-70)
    h=random.randrange(70,height-70)

    image_left = image.filter(MyGaussianBlur(radius=3, w=w,h=h,width=width,height=height))
    image_right = image.filter(MyGaussianBlur2(radius=3, w=w,h=h,width=width,height=height))
    image_left.save(path+'SourceA/'+dir[i])
    image_right.save(path+'SourceB/'+dir[i])

