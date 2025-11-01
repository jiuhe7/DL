import os
import pandas as pd
import torch
import torchvision
from d2l import torch as d2l

d2l.DATA_HUB['banana-detection'] = (d2l.DATA_URL + 'banana-detection.zip','5de25c8fce5ccdea9f91267273465dc968d20d72')
# 读取香蕉检测数据集
def read_data_bananas(is_train=True):
    """读取香蕉检测数据集中的图像和标签"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir,
                            'bananas_train' if is_train else 'bananas_val',
                            'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    # 把图片、标号全部读到内存里面
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(os.path.join(data_dir,'bananas_train' if is_train else 'bananas_val',
                                                            'images',f'{img_name}')))
        targets.append(list(target))
    print("len(targets)：",len(targets))
    print("len(targets[0])：",len(targets[0]))
    print("targets[0][0]....targets[0][4]：",targets[0][0], targets[0][1], targets[0][2], targets[0][3], targets[0][4])
    print("type(targets)：",type(targets))
    print("torch.tensor(targets).unsqueeze(1).shape：",torch.tensor(targets).unsqueeze(1).shape) # unsqueeze函数在指定位置加上维数为一的维度
    print("len(torch.tensor(targets).unsqueeze(1) / 256)：", len(torch.tensor(targets).unsqueeze(1) / 256))
    print("type(torch.tensor(targets).unsqueeze(1) / 256)：", type(torch.tensor(targets).unsqueeze(1) / 256))
    return images, torch.tensor(targets).unsqueeze(1) / 256 # 归一化使得收敛更快


# 创建一个自定义Dataset实例
class BananasDataset(torch.utils.data.Dataset):
    """一个用于加载香蕉检测数据集的自定义数据集"""

    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if is_train else f'validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)


# 为训练集和测试集返回两个数据加载器实例
def load_data_bananas(batch_size):
    """加载香蕉检测数据集"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                            batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                          batch_size)
    return train_iter, val_iter

# 读取一个小批量，并打印其中的图像和标签的形状
batch_size, edge_size = 32, 256
train_iter, _ = load_data_bananas(batch_size)
batch = next(iter(train_iter))
# ([32,1,5]) 中的1是每张图片中有几种类别，这里只有一种香蕉要识别的类别
# 5是类别标号、框的四个参数
batch[0].shape, batch[1].shape


# 示例
# pytorch里permute是改变参数维度的函数，
# Dataset里读的img维度是[batch_size, RGB, h, w]，
# 但是plt画图的时候要求是[h, w, RGB]，所以要调整一下

# 做图片的时候，一般是会用一个ToTensor()将图片归一化到【0, 1】，这样收敛更快
print("原始图片:\n", batch[0][0])
print("原始图片:\n", (batch[0][0:10].permute(0,2,3,1)))
print("归一化后图片:\n", (batch[0][0:10].permute(0,2,3,1)) / 255 )
imgs = (batch[0][0:10].permute(0,2,3,1)) / 255
#imgs = (batch[0][0:10].permute(0,2,3,1))
# d2l.show_images输入的imgs图片参数是归一化后的图片
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch[1][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
