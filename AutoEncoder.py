import time
import torch
import torch.nn as nn
import torchvision.utils

import DataLoader
import torch.utils.data.dataloader as dataloader
from torch.utils.tensorboard import SummaryWriter


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        # 输入并进行压缩
        self.encoder = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 96 * 96, 64*64),
            nn.BatchNorm1d(64*64)
            # nn.Linear(3*96*96, 96*96),
            # nn.Tanh(),
            # nn.Linear(96*96, 48*48),
            # nn.Tanh(),
            # nn.Linear(48*48, 24*24),
            # nn.Tanh(),
            # nn.Linear(24*24, 100)
        )

        # 解压
        self.decoder = torch.nn.Sequential(
            nn.Linear(64*64, 3*96*96),
            nn.BatchNorm1d(3*96*96),
            # nn.Linear(100, 24*24),
            # nn.Tanh(),
            # nn.Linear(24*24, 48*48),
            # nn.Tanh(),
            # nn.Linear(48*48, 96*96),
            # nn.Tanh(),
            # nn.Linear(96*96, 3*96*96),
            nn.Sigmoid(),
            nn.Unflatten(1, torch.Size([3, 96, 96]))
        )

    def forward(self, input):
        middle = self.encoder(input)
        output = self.decoder(middle)
        return middle, output


# torch.Size([128, 1, 94, 94])


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # 加载writer
    writer = SummaryWriter("AE_logs")
    start_time = time.time()

    # 使用GPU训练
    # 因此可以多加一层判定，但是我这里没写，因为我的电脑有一块GPU

    # 加载数据集
    data_set = DataLoader.GetData()
    # 数据集太大了，RTX3060的GPU很难跑,可以考虑丢弃一部分
    data_set.__int__("D:\Desktop\GAN\DataSet\\1")

    print("The number of Original Data is 50000")
    # data_set = data_set[0:64*128]
    print("The number of After Dropped Data is {}".format(len(data_set)))

    # 加载数据到dataloader
    batch_size = 4
    dataloader = dataloader.DataLoader(data_set, batch_size=batch_size)

    # 加载模型并加载到GPU中
    AutoEncoder = AutoEncoder()
    print(AutoEncoder)
    AutoEncoder.cuda()

    # 加载损失函数
    mse_loss = torch.nn.MSELoss()
    mse_loss.cuda()

    # 加载优化器
    optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=0.01)

    torch.cuda.memory_summary(device=None, abbreviated=False)

    print("---------------训练开始-------------")
    # 开始训练
    for epoch in range(500):
        print("-----------第 {} 轮训练开始-----------".format(epoch))
        #
        total_loss = 0
        batch_count = 0

        for data in dataloader:

            # 加载到GPU
            input = data.cuda()

            # 获取输出
            middle, output = AutoEncoder(input)
            # writer.add_images("PIC", output, global_step=epoch)

            # 设定零梯度
            optimizer.zero_grad()

            # 计算损失
            current_loss = mse_loss(output, input)
            current_loss.backward()
            total_loss = total_loss + current_loss

            # 反向传播
            optimizer.step()



            if batch_count == 128*64/batch_size:
                break
            else:
                batch_count = batch_count+1

            if batch_count % 256 == 0:
                torchvision.utils.save_image(input, "Pic/AE/{}_{}_Original.jpg".format(epoch, batch_count))
                torchvision.utils.save_image(output, "Pic/AE/{}_{}_Compressed.jpg".format(epoch, batch_count))

        # 保存每一轮的模型
        # 50了
        torch.save(AutoEncoder.state_dict(), 'SavedModel/AutoEncoder/Train/AE_{}.pth'.format(epoch))

        end_time = time.time()
        print("训练时间: {}".format(end_time - start_time))
        print("训练次数: {}, Loss: {}".format(epoch, total_loss))
        # writer.add_scalar("LOSS", total_loss, global_step=epoch)

    # 保存最终模型
    # torch.save(AutoEncoder.state_dict(), 'SavedModel/AutoEncoder/Final/final.pth')
    writer.close()


    # x = 0
    # for data in dl:
    #     x = data
    #     break
    # print(x.shape)
    # print(x[0])
    # output = AE(x)
    #
    # print(output.shape)
    # print(output[0])
