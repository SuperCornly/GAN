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


# 定义鉴别器
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        # 使用深度卷积网络作为鉴别器
        self.layer1 = nn.Sequential(nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf), nn.LeakyReLU(0.2, inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True))
        self.layer3 = nn.Sequential(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
                                    nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True))
        self.fc = nn.Sequential(nn.Linear(256 * 6 * 6, 1), nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc(out.view(-1, 256 * 6 * 6))
        return out


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
    torchvision.utils.save_image(inputs, 'Pic/OldPic_600_epochs/' + picname + '.jpg')
    # inputs = inputs.numpy().transpose((1, 2, 0))
    # plt.imshow(inputs)
    # plt.savefig('Pic/' + picname + '.png')
    # plt.pause(0.01)
    plt.close()


# 训练过程
def train(d, g, criterion, d_optimizer, g_optimizer, epochs=1, show_every=1000, print_every=10):
    iter_count = 0
    writer = SummaryWriter("GAN_logs")
    start_time = time.time()
    for epoch in range(300):
        print("-----------第 {} 轮训练开始-----------".format(epoch))

        D_total_Loss = 0
        G_total_Loss = 0

        batch_count = 0

        for inputs, _ in train_loader:
            real_inputs = inputs  # 真实样本
            fake_inputs = g(torch.randn(64, 100).cuda())  # 伪造样本

            real_labels = torch.ones((real_inputs.size(0), 1))  # 真实标签
            fake_labels = torch.zeros((64, 1))  # 伪造标签

            real_outputs = d(real_inputs.cuda())
            d_loss_real = criterion(real_outputs.cuda(), real_labels.cuda())

            fake_outputs = d(fake_inputs)
            d_loss_fake = criterion(fake_outputs.cuda(), fake_labels.cuda())

            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            fake_inputs = g(torch.randn(64, 100).cuda())
            outputs = d(fake_inputs)
            real_labels = torch.ones((outputs.size(0), 1))
            g_loss = criterion(outputs.cuda(), real_labels.cuda())

            G_total_Loss = G_total_Loss +g_loss.item()
            D_total_Loss = D_total_Loss +d_loss.item()

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (batch_count % 32 == 0):
                print('Epoch:{}, Iter:{}, D:{}, G:{}'.format(epoch,
                                                                 batch_count,
                                                                 d_loss.item(),
                                                                 g_loss.item()))
                picname = "Epoch_" + str(epoch) + "Iter_" + str(batch_count)
                img_show(torchvision.utils.make_grid(fake_inputs.data), picname)

            # if (iter_count % print_every == 0):
            #     print('Epoch:{}, Iter:{}, D:{}, G:{}'.format(epoch,
            #                                                      iter_count,
            #                                                      d_loss.item(),
            #                                                      g_loss.item()))
            #     iter_count += 1

            if batch_count == 64:
                break
            else:
                batch_count = batch_count + 1

        end_time = time.time()
        print("训练时间: {}".format(end_time - start_time))
        print("训练次数: {}, D_Loss: {}".format(epoch, D_total_Loss))
        print("训练次数: {}, G_Loss: {}".format(epoch, G_total_Loss))
        writer.add_scalar("D_LOSS", D_total_Loss, global_step=epoch)
        writer.add_scalar("G_LOSS", G_total_Loss, global_step=epoch)

        if epoch % 20 == 0:
            torch.save(d.state_dict(), "SavedModel/GAN/Train_600_epochs/d_{}.pth".format(epoch))
            torch.save(g.state_dict(), "SavedModel/GAN/Train_600_epochs/g_{}.pth".format(epoch))
            # print('Finished Training！')


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

    # 参数data_transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    train_set = datasets.ImageFolder('Dataset', data_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
                                               shuffle=True, num_workers=4)  # 数据加载

    inputs, _ = next(iter(train_loader))
    # make_grid的作用是将若干幅图像拼成一幅图像
    img_show(torchvision.utils.make_grid(inputs), "RealDataSample")

    # 初始化鉴别器和生成器
    d = Discriminator(3, 32)
    g = Generator(3, 128, 1024, 100)
    # 继续训练（可选）
    d.load_state_dict(torch.load("SavedModel/GAN/Train_300_epochs/d_280.pth"))
    g.load_state_dict(torch.load("SavedModel/GAN/Train_300_epochs/g_280.pth"))
    d.cuda()
    g.cuda()

    criterion = nn.BCELoss()  # 损失函数
    criterion.cuda()
    lr = 0.0003  # 学习率
    d_optimizer = torch.optim.Adam(d.parameters(), lr=lr)  # 定义鉴别器的优化器
    g_optimizer = torch.optim.Adam(g.parameters(), lr=lr)  # 定义生成器的优化器

    # 训练
    train(d, g, criterion, d_optimizer, g_optimizer, epochs=300)

    torch.save(d.state_dict(), "SavedModel/GAN/Final/d.pth")
    torch.save(g.state_dict(), "SavedModel/GAN/Final/g.pth")