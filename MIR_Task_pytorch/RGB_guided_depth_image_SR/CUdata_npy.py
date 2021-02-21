import torch.utils.data as data
from glob import glob
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from PIL import Image
import random
import os
import numpy as np
import torch

class cudataset(data.Dataset):
    def __init__(self):
        super(cudataset, self).__init__()
        self.depth = np.load('trainset/data_x.npy')#(batch,height,width,c)
        #print(self.depth.shape)
        self.depth = np.transpose(self.depth, (0, 3, 1, 2))
        self.depth_t = torch.from_numpy(self.depth)

        self.gt = np.load('trainset/label.npy')  # (batch,height,width,c)
        self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        self.gt_t = torch.from_numpy(self.gt)

        self.rgb = np.load('trainset/data_y.npy')  # (batch,height,width,c)
        self.rgb = np.transpose(self.rgb, (0, 3, 1, 2))
        self.rgb_t = torch.from_numpy(self.rgb)

    def __getitem__(self, item):
        img_depth = self.depth_t[item]
        img_gt = self.gt_t[item]
        img_rgb = self.rgb_t[item]

        return (img_depth, img_gt,img_rgb)

    def __len__(self):
        return len(self.depth)

if __name__ =='__main__':
    dataset=cudataset()
    dataloader=data.DataLoader(dataset,batch_size=1)
    for b1,(img_L,img_H,img_RGB) in enumerate(dataloader):
        print(b1)
        print(img_L.shape,img_H.shape,img_RGB.shape)
