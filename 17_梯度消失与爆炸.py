from d2l import  torch as d2l
import torch
import matplotlib.pyplot as plt

x=torch.arange(-8.0,8.0,0.1,requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

'''detach().numpy()：由于x和y是需要计算梯度的张量（requires_grad=True），
需先用detach()将其从计算图中 “分离”，才能转换为 numpy 数组用于绘图。'''
d2l.plot(x.detach().numpy(),[y.detach().numpy(),x.grad.numpy()],
         legend=['sigmoid','gradient'],figsize=(4.5,2.5))



M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)



plt.show()