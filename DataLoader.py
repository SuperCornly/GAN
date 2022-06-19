import os
import torchvision
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image


class GetData(dataset.Dataset):
    def __int__(self, file_root_path):
        # super(GetData, self).__int__()
        """
        :param file_root_path: 要读取的数据的路径
        :param total_size: 想要的数据长度，比如原先有50000个数据，而我只要200个，那total_size = 200即可
        :return:
        """
        self.imgs = self.read_file(file_root_path)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = Image.open(img)

        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imgs)

    def read_file(self, path):
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, img) for img in files_list]
        return file_path_list

    def transform(self, img):
        transform = torchvision.transforms.ToTensor()
        img = transform(img)
        return img


if __name__ == '__main__':
    data = GetData()
    data.__int__("D:\Desktop\GAN\DataSet")

    # print(data[64*128-1].shape)


#     print(data.__len__())
#
#     writer = SummaryWriter(log_dir="TensorBoard_Imgs")
#
#     img_loader = dataloader.DataLoader(data, batch_size=256)
#
#     print("The length of Dataloader is {}".format(len(img_loader)))
#
#     step = 0
#     for data in img_loader:
#         writer.add_images("Batch", data, global_step=step)
#         step = step+1
#         print("It's in {} times for tensorboard to show images".format(step))
#
#     writer.close()