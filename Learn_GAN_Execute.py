# -*- coding: utf-8 -*-
'''
brief  : 使用Pytorch搭建GAN网络模型，生成二次元头像
Author : 起风了
Date   : 2021.08.04
'''
import time

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# 定义生成器
class Generator(nn.Module):
    def __init__(self, nc, ngf, nz, feature_size):
        super(Generator, self).__init__()
        self.prj = nn.Linear(feature_size, nz * 6 * 6)
        # nn.Sequential：一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
        self.layer1 = nn.Sequential(nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 4), nn.ReLU())
        self.layer2 = nn.Sequential(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf * 2), nn.ReLU())
        self.layer3 = nn.Sequential(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ngf), nn.ReLU())
        self.layer4 = nn.Sequential(nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                                    nn.Tanh())

    def forward(self, x):
        out = self.prj(x).view(-1, 1024, 6, 6)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


# 图片显示
def img_show(inputs, picname):
    plt.ion()
    inputs = inputs.cpu()
    inputs = inputs / 2 + 0.5
    torchvision.utils.save_image(inputs, 'Pic/' + picname + '.jpg')
    # inputs = inputs.numpy().transpose((1, 2, 0))
    # plt.imshow(inputs)
    # plt.savefig('Pic/' + picname + '.png')
    # plt.pause(0.01)
    plt.close()


# 主程序
if __name__ == '__main__':
    # 串联多个变换操作
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 依概率p水平翻转，默认p=0.5
        transforms.ToTensor(),  # 转为tensor，并归一化至[0-1]
        # 标准化，把[0-1]变换到[-1,1]，其中mean和std分别通过(0.5,0.5,0.5)和(0.5,0.5,0.5)进行指定。
        # 原来的[0-1]最小值0变成(0-0.5)/0.5=-1，最大值1变成(1-0.5)/0.5=1
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 初始化鉴别器和生成器
    g = Generator(3, 128, 1024, 100)
    g.cuda()

    g.load_state_dict(torch.load("SavedModel/GAN/Train/g_280.pth"))

    print(g)



    for i in range(100):
        fake_inputs = g(torch.randn(64, 100).cuda())
        picname = "Test{}".format(i)
        img_show(torchvision.utils.make_grid(fake_inputs.data), picname)

