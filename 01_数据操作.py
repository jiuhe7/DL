import torch
# x=torch.arange(12).reshape(3,4)
# print(x)

# x=torch.zeros((2,3,4))
# print(x)

# x=torch.ones((2,3,4))
# print(x)

# x=torch.tensor([[2,3,4,6],[3,57,9,2],[32,78,0,1]])
# print(x)

# x=torch.arange(12,dtype=torch.float32).reshape(3,4)
# y=torch.tensor([[2,3,4,6],[3,57,9,2],[32,78,0,1]])
# z=torch.cat((x,y),dim=0)
# print(z)
#
# z=torch.cat((x,y),dim=1)
# print(z)
#
# print(x==y)
#
# print(x.sum())

# a=torch.arange(3).reshape((3,1))
# b=torch.arange(2).reshape((1,2))
# print(a)
# print(b)
# print(a+b)


x=torch.arange(12,dtype=torch.float32).reshape(3,4)
y=torch.tensor([[2,3,4,6],[3,57,9,2],[32,78,0,1]])
before=id(y)
y=y+x
print(id(x)==before)

# 原地操作
z=torch.zeros_like(y)
print('id(z)',id(z))
z[:]=x+y
print('id(z)',id(z))

# 如果后续没有重复使用x,可以下[:]=x+y

