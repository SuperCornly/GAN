import torch
import Discriminater as Dis
import Generator as Gen
import torch.optim as op

# 获取训练集，要么选ACG要么就是猫猫

# 生成模型
# 生成器
D = Dis.Discriminator()
# 判别器
G = Gen.Generator()

# 损失函数(待定义)
loss = 0

# 优化器
optimizer = op.Adam

# 训练
print("__________________________TRAIN___________________________")
for epoch in range(200):
    print("____________________TRAIN_in_{}_times_____________________".format(epoch))



    # print("Train in {} times.".format(epoch))


# 输出
print("__________________________OUTPUT__________________________")
with torch.no_grad():
    print("Output in {} times.".format(1))


