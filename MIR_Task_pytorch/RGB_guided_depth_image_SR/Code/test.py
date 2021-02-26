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
from testdata import cudataset
from testCUData import cudatatest1
from testCUData import cudatatest2
from CUNet import CUNet
from torch.utils.data import DataLoader
import cv2
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def test():
    print("===> Loading testset")
    test_set1 = cudatatest1()
    test_loader1 = DataLoader(test_set1, batch_size=1, shuffle=False, num_workers=0)
    test_set2 = cudatatest2()
    test_loader2 = DataLoader(test_set2, batch_size=1, shuffle=False, num_workers=0)
    psnr_list = []
    model = CUNet()
    model = model.cuda()
    lr_img1 = ['Art', 'Books', 'Moebius', 'Reindeer']
    lr_img2 = ['Ambush', 'Cave', 'Market']
    mean_d1 = [0.5187, 0.5024, 0.4298, 0.4785]
    mean_d2 = [0.6757, 0.5308, 0.5865]
    mean_c = [0.3068, 0.3956, 0.5767, 0.1755, 0.3952, 0.4296, 0.3157]
    save_path = 'test_result/'
    for i in range(69):
        
        print('model/' + str((i+1)*5) + '.pth')
        state = torch.load('model/' + str((i+1)*5) + '.pth')
        model.load_state_dict(state['model'])
        model.eval()

        dic = {}
        dig = {}

        with torch.no_grad():

            for batch, (lr, hr, rgb) in enumerate(test_loader1):
                hr = hr.float()
                lr = lr.float()
                rgb = rgb.float()
                
                
                
                hr = hr.cuda()
                lr = lr.cuda()
                rgb = rgb.cuda()

                sr = model(lr, rgb)
                sr = sr.cpu().numpy()
                hr = hr.cpu().numpy()
                #            sr=np.transpose(sr,(0,2,3,1))
                #            hr=np.transpose(hr,(0,2,3,1))
                #                print(sr.shape)
                dic[lr_img1[batch] + '_sr'] = sr
                dig[lr_img1[batch] + '_hr'] = hr

            for batch, (lr, hr, rgb) in enumerate(test_loader2):
                hr = hr.float()
                lr = lr.float()
                rgb = rgb.float()

                hr = hr.cuda()
                lr = lr.cuda()
                rgb = rgb.cuda()

                sr = model(lr, rgb)
                sr = sr.cpu().numpy()
                hr = hr.cpu().numpy()
                #            sr=np.transpose(sr,(0,2,3,1))
                #            hr=np.transpose(hr,(0,2,3,1))
                #                print(sr.shape)
                dic[lr_img2[batch] + '_sr'] = sr
                dig[lr_img2[batch] + '_hr'] = hr
            sio.savemat('test_result/result' + str((i+1)*5)  + '.mat', dic)
            sio.savemat('gt/gt' + str((i+1)*5)  + '.mat', dig)
    return psnr_list


val_psnr = test()
print(val_psnr)
