import matplotlib.pyplot as np
from d2l import torch as d2l
import torch
from torch import nn
import matplotlib.pyplot as plt
def dropout_layer(X,dropout):
    assert 0<=dropout<=1
    if dropout==1:
        return torch.zeros_like(X)
    if dropout==0:
        return X
    mask=(torch.rand(X.shape)>dropout).float()
    #(torch.rand(X.shape)>dropout).float() 这行代码的作用是生成一个随机掩码（mask）
    '''这个 mask 中，1.0 的位置对应输入 X 中会被保留的元素，
    0.0 的位置对应会被丢弃的元素。由于随机值服从均匀分布，
    mask 中 1.0 的比例约为 1 - dropout（即保留比例），
    0.0 的比例约为 dropout（即丢弃比例），
    从而实现 “随机丢弃部分元素” 的 dropout 功能。'''
    return mask*X/(1.0-dropout)
#实例
# X=torch.arange(16,dtype=torch.float32).reshape((2,8))
# print(X)
# print(dropout_layer(X,0.1))
#
# print(dropout_layer(X,0.9))

num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256

dropout1,dropout2=0.2,0.5

class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net,self).__init__()
        self.num_inputs=num_inputs
        self.training=is_training
        self.lin1=nn.Linear(num_inputs,num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu=nn.ReLU()

    def forward(self,X):
            H1=self.relu(self.lin1(X.reshape((-1,self.num_inputs))))
            if self.training==True:
                H1=dropout_layer(H1,dropout1)
            H2=self.relu(self.lin2(H1))
            if self.training==True:
                H2=dropout_layer(H2,dropout2)
            out=self.lin3(H2)
            return out
net=Net(num_inputs,num_outputs,num_hiddens1,num_hiddens2)


num_epochs,lr,batch_size=10,0.5,256
loss=nn.CrossEntropyLoss(reduction='none')
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
trainer=torch.optim.SGD(net.parameters(),lr=lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)

plt.show()