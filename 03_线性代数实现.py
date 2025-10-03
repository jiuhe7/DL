import torch
# x=torch.arange(20).reshape(5,4)
# print(x)
# print(x.T)

# Y=torch.arange(24).reshape(2,3,4)
# print(Y)

# A=torch.arange(20,dtype=torch.float32).reshape(5,4)
# B=A.clone()#通过内存新分配，将A的一个副本分配给B
# print(A)
# print(A+B)

# A=torch.arange(20*2).reshape(2,5,4)
# A=A.float()
# print(A)
# print(A.shape)
# print(A.sum())

# A_sum_axis0=A.sum(axis=0)
# print(A_sum_axis0)
# print(A_sum_axis0.shape)

# A_sum_axis1=A.sum(axis=1)
# print(A_sum_axis1)
# print(A_sum_axis1.shape)

# A_sum_axis2=A.sum(axis=2)
# print(A_sum_axis2)
# print(A_sum_axis2.shape)

# print(A.float().mean())
# print(A.sum())
# print(A.numel())
# print(A.numel()*(A.float().mean()))

# print(A.float().mean(axis=0))
# print(A.float().mean(axis=2))

# 计算总和或均值时保持轴数不变

# sum_A=A.sum(axis=1,keepdims=True)
# print(sum_A)
# print(A/sum_A)

# 某个轴计算A元素的累计和
# print(A.cumsum(axis=0))



# A=torch.arange(20).reshape(5,4)
# A=A.float()
# print(A)
# B=torch.ones(4,3)
# print(B)
# C=torch.mm(A,B)
# print(C)



# L1范数 向量绝对值之和
# u=torch.tensor([3.0,-4])
# print(torch.abs(u).sum())


# L2范数  向量平方和开根
# x=torch.norm(u)
# print(x)

# F范数  矩阵元素平方和开根
# x=torch.norm(torch.ones((4,9)))
# print(x)


