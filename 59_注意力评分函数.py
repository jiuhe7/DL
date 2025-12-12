import math
import torch
from torch import nn
from d2l import torch as d2l


def masked_softmax(X,valid_lens):
    if valid_lens is None:
        '''dim=-1：最后一维。'''
        return nn.functional.softmax(X,dim=-1)

    else:
        shape=X.shape
        if valid_lens.dim()==1:
            # shape[1] 是 Num_queries
            valid_lens=torch.repeat_interleave(valid_lens,shape[1])
        else:
            # [[1, 2], [1, 2]] -> [1, 2, 1, 2]
            valid_lens=valid_lens.reshape(-1)
            #X.reshape(-1,shape[-1])把 3D 的 X (Batch, Query, Key) 拍扁成 2D (Batch * Query, Key)
        X=d2l.sequence_mask(X.reshape(-1,shape[-1]),valid_lens,value=-1e6)
        return nn.functional.softmax(X.reshape(shape),dim=-1)

# a=masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# print(a)
'''$$Score(q, k) = w_v^T \cdot \tanh(W_q q + W_k k)$$'''
class AdditiveAttention(nn.Module):
    '''加性注意力'''
    def __init__(self,key_size,query_size,num_hiddens,dropout,**kwargs):
        super(AdditiveAttention,self).__init__(**kwargs)
        self.W_k=nn.Linear(key_size,num_hiddens,bias=False)
        self.W_q=nn.Linear(query_size,num_hiddens,bias=False)
        self.w_v=nn.Linear(num_hiddens,1,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,queies,keys,values,valid_lens):
        queries,keys=self.W_q(queies),self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features=queries.unsqueeze(2)+keys.unsqueeze(1)
        features=torch.tanh(features)
        scores=self.w_v(features).squeeze(-1)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values的小批量，两个值矩阵是相同的
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(
    2, 1, 1)
valid_lens = torch.tensor([2, 6])

attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
a=attention(queries, keys, values, valid_lens)
print(a)

'''$$\text{Attention}(Q, K, V) 
= \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$'''
class DotProductAttention(nn.Module):
    '''缩放点积注意力'''
    def __init__(self,dropout,**kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout=nn.Dropout(dropout)

    def forward(self,queies,keys,values,valid_lens=None):
        d=queries.shape[-1]
        scores=torch.bmm(queies,keys.transpose(1,2))/math.sqrt(d)
        self.attention_weights=masked_softmax(scores,valid_lens)
        return torch.bmm(self.dropout(self.attention_weights),values)

queries = torch.normal(0, 1, (2, 1, 2))

attention = DotProductAttention(dropout=0.5)
attention.eval()
a=attention(queries, keys, values, valid_lens)
print(a)


















































