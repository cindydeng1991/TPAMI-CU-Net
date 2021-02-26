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
from traindata import cudataset
from CUNet import CUNet
from torch.utils.data import DataLoader
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Trainer:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 64
        self.lr = 0.00001

        print("===> Loading datasets")
        self.train_set = cudataset()
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

        print("===> Building model")
        self.model = CUNet()
        self.model = self.model.cuda()
        self.criterion = nn.MSELoss(reduction='mean')

        print("===> Setting Optimizer")
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer,gamma=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)
        self.train_loss = []
        self.val_psnr = []

        if os.path.exists('multi_exposure_model/latest.pth'):
            print('===> Loading pre-trained model...')
            state = torch.load('multi_exposure_model/latest.pth')
            self.train_loss = state['train_loss']
            self.model.load_state_dict(state['model'])

    def train(self):
        seed = random.randint(1, 1000)
        print("===> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)

        for ep in range(1, self.epoch + 1):
            print(self.lr)
            epoch_loss = []
            for batch, (lr, hr, rgb) in enumerate(self.train_loader):
                hr = hr.float()
                lr = lr.float()
                rgb = rgb.float()

                hr = hr.cuda()
                lr = lr.cuda()
                rgb = rgb.cuda()

                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                z = self.model(lr, rgb)
                loss = self.criterion(z, hr)
                loss = loss * 1000
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

                torch.cuda.synchronize()
                end_time = time.time()

                if batch % 100 == 0:
                    print('Epoch:{}\tcur/all:{}/{}\tAvg Loss:{:.4f}\tTime:{:.2f}'.format(ep, batch,len(self.train_loader),loss.item(),end_time - start_time))

            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss))
            print(np.mean(epoch_loss))

            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            if not os.path.exists("multi_exposure_model/"):
                os.makedirs("multi_exposure_model/")
            torch.save(state, os.path.join('multi_exposure_model/', 'latest.pth'))
            torch.save(state, os.path.join('multi_exposure_model/', str(ep) + '.pth'))
            matplotlib.use('Agg')
            plot_loss_list = self.train_loss
            fig1 = plt.figure()
            plt.plot(plot_loss_list)
            plt.savefig('train_loss_curve_multi_exposure.png')
            plt.close('all')
        print('===> Finished Training!')

if __name__ == '__main__':
    hhh = Trainer()
    hhh.train()
