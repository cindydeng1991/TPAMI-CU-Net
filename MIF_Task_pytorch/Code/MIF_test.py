import os
import math
import time
import torch
import random
import matplotlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms
from CUtest_npy import cudatatest
from MIF_CUNet import CUNet
from torch.utils.data import DataLoader
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def test():
    print("===> Loading testset")
    test_set = cudatatest()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    psnr_list = []
    model = CUNet()
    model = model.cuda()
    lr_img = ['one','two','three','four','five','six','Pendant','Plant','Pot','Tampax','Towel','Typewriter']
    for i in range(100,150):
        print('multi_exposure_model/' + str(i+1) + '.pth')
        state = torch.load('multi_exposure_model/' + str(i+1) + '.pth')
        model.load_state_dict(state['model'])
        model.eval()

        dic = {}
        dig = {}

        with torch.no_grad():

            for batch, (lr, hr, rgb) in enumerate(test_loader):
                hr = hr.float()
                lr = lr.float()
                rgb = rgb.float()

                hr = hr.cuda()
                lr = lr.cuda()
                rgb = rgb.cuda()

                sr = model(lr, rgb)
                sr = sr.cpu().numpy()
                hr = hr.cpu().numpy()
                sr=np.transpose(sr,(0,2,3,1))
                hr=np.transpose(hr,(0,2,3,1))
                #print(sr.shape)
                dic[lr_img[batch] + '_dn'] = sr
                dig[lr_img[batch] + '_gt'] = hr

            sio.savemat('multi_exposure_result/result' + str(i + 1)  + '.mat', dic)
            sio.savemat('multi_exposure_gt/gt' + str(i + 1)  + '.mat', dig)
    return psnr_list


val_psnr = test()
print(val_psnr)
