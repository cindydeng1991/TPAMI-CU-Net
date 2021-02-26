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
        self.over = np.load('trainset/data_over.npy')#(batch,height,width,c)
        self.over = np.transpose(self.over, (0, 3, 1, 2))
        self.over_t = torch.from_numpy(self.over)

        self.gt = np.load('trainset/label.npy')  # (batch,height,width,c)
        self.gt = np.transpose(self.gt, (0, 3, 1, 2))
        self.gt_t = torch.from_numpy(self.gt)

        self.under = np.load('trainset/data_under.npy')  # (batch,height,width,c)
        self.under = np.transpose(self.under, (0, 3, 1, 2))
        self.under_t = torch.from_numpy(self.under)

    def __getitem__(self, item):
        img_over = self.over_t[item]
        img_gt = self.gt_t[item]
        img_under = self.under_t[item]

        return (img_over, img_gt,img_under)

    def __len__(self):
        return len(self.over)

if __name__ =='__main__':
    dataset=cudataset()
    dataloader=data.DataLoader(dataset,batch_size=1)
    for b1,(img_L,img_H,img_RGB) in enumerate(dataloader):
        print(b1)
        print(img_L.shape,img_H.shape,img_RGB.shape)
