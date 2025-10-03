import random # 用于随机操作（如打乱数据顺序）
import torch
from d2l import torch as d2l# 深度学习工具库（来自《动手学深度学习》），用于绘图等
from pyexpat import features

from numpy.ma.core import indices


def synthetic_data(w,b,num_examples):
    '''生成y=Xw+b+噪声'''
    X=torch.normal(0,1,(num_examples,len(w)))  # 生成特征X：均值0、方差1的随机数，形状为（样本数，特征数）
    y=torch.matmul(X,w)+b
    y+=torch.normal(0,0.01,y.shape)  # 加入噪声（真实数据总会有误差）：均值0、方差0.01的小噪声
    return X,y.reshape((-1,1))

true_w=torch.tensor([2.0,-3])
true_b=4.2
features,labels=synthetic_data(true_w,true_b,1000)# 生成1000个样本
# print('features:',features[0],'nlabel:',labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:,1].detach().numpy(),labels.detach().numpy(),1)
# d2l.plt.show()


def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))# 生成样本索引（0到999）
    # 这些样本是随机读取的，没有特定顺序
    random.shuffle(indices)# 打乱索引（随机读取样本，避免顺序影响训练）
    for i in range(0,num_examples,batch_size):
        batch_indices=torch.tensor(indices[i:min(i+batch_size,num_examples)]) # 取当前批量的索引（避免越界）
        yield features[batch_indices],labels[batch_indices]# 返回当前批量的特征和标签

batch_size=10
for X,y in data_iter(batch_size,features,labels):
    print(X,'\n',y)
    break

# 定义初始化模型
w=torch.normal(0,0.01,size=(2,1),requires_grad=True)
b=torch.zeros(1,requires_grad=True)

# 定义模型
def linreg(X,w,b):
    '''线性回归模型'''
    return torch.matmul(X,w)+b # 矩阵乘法Xw + 偏置b，得到预测值y_hat

# 定义损失函数
def squared_loss(y_hat,y):
    '''均方损失'''
    return (y_hat-y.reshape(y_hat.shape))**2/2# 确保y和y_hat形状一致后计算损失

# 定义优化算法
def sgd(params,lr,batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():
        for param in params:
            # 参数更新公式：param = param - 学习率 * 梯度 / 批量大小
            param-=lr*param.grad/batch_size # 梯度是批量的总和，除以批量大小取平均
            param.grad.zero_()# 清空梯度（避免下一次计算时累积）


#训练过程
lr=0.03
num_epochs=30
net=linreg
loss=squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):
        l=loss(net(X,w,b),y)#X,y的小批量损失
        l.sum().backward()
        sgd([w,b],lr,batch_size)#使用参数的梯度更新参数
    with torch.no_grad():
        train_l=loss(net(features,w,b),labels)
        print(f'epoch{epoch+1},loss{float(train_l.mean())}')

# 真实参数与训练参数

print(true_w)
print(true_b)


print(w)
print(b)
# print(f'w的估计误差：{true_w-w.reshape(true_w.shape)}')
# print(f'b的估计误差：{true_b-b}')