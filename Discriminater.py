import torch
import torch.nn as nn

# 判别器的模型结构、
# 模型结构为卷积的拓展，但是我想把其变成一个二位的线性层


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 三通道
        # 涉及到反卷积的计算，因此需要自己算一下，保证最后的输出为3*96*96（原图片尺寸）
        self.Dis = nn.Sequential(
            nn.ConvTranspose2d(3, 32, kernel_size=4, stride=)
        )



    def forward(self):
        return 0
