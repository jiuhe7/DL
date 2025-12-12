import torch
from jieba.lac_small.predict import batch_size
from torch import nn
from d2l import torch as d2l

def get_token_and_segments(tokens_a,tokens_b=None):
    tokens=['<cls>']+tokens_a+['<sep>']
    segments=[0]*(len(tokens_a)+2)
    if tokens_b is not None:
        tokens+=tokens_b+['<sep>']
        segments+=[1]*(len(tokens_b)+1)
'''['[CLS]', 'I', 'like', 'cat', '[SEP]', 'Cat', 'is', 'cute', '[SEP]']'''
'''[0, 0, 0, 0, 0, 1, 1, 1, 1]'''


class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)

        # 1. Token Embedding: 将单词索引转为向量
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)

        # 2. Segment Embedding: 区分第一句和第二句 (维度固定为 2)
        self.segment_embedding = nn.Embedding(2, num_hiddens)

        # 3. Transformer Blocks
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True
            ))

        # 4. 可学习的位置嵌入 (Position Embedding)
        # 形状为 (1, max_len, num_hiddens)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 核心逻辑：三种 Embedding 直接相加
        # 这里的相加触发了广播机制
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]

        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    def __init__(self,vocab_size,num_hiddens,num_inputs=768,**kwargs):
        super(MaskLM,self).__init__(**kwargs)
        self.mlp=nn.Sequential(nn.Linear(num_inputs,num_hiddens),
                               nn.ReLU(),
                               nn.LayerNorm(num_hiddens),
                               nn.Linear(num_hiddens,vocab_size))

    def forward(self,X,pred_positions):
        """
        X: BERTEncoder 的输出，形状 (batch_size, seq_len, num_inputs)
        pred_positions: 需要预测的掩码位置，形状 (batch_size, num_pred_positions)

                """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    def __init__(self,num_inputs,**kwargs):
        super(NextSentencePred,self).__init__(**kwargs)
        self.output=nn.Linear(num_inputs,2)
    def forward(self,X):
        return self.output(X)

class BERTModel(nn.Module):
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,
                 ffn_num_hiddens,num_heads,num_layers,dropout,
                 max_len=1000,key_size=768,query_size=768,value_size=768,
                 hid_in_features=768,mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel,self).__init__()
        self.encoder=BERTEncoder(vocab_size,num_hiddens,norm_shape,
                                 ffn_num_input,ffn_num_hiddens,num_heads,num_layers,
                                 dropout,max_len=max_len,key_size=key_size,
                                 query_size=query_size,value_size=value_size)
        self.hidden=nn.Sequential(nn.Linear(hid_in_features,num_hiddens),
                                  nn.Tanh())
        self.mlm=MaskLM(vocab_size,num_hiddens,mlm_in_features)
        self.nsp=NextSentencePred(nsp_in_features)

    def forward(self,tokens,segments,valid_lens=None,
                pred_positions=None):
        encoded_X=self.encoder(tokens,segments,valid_lens)
        if pred_positions is not None:
            mlm_Y_hat=self.mlm(encoded_X,pred_positions)
        else:
            mlm_Y_hat=None
        nsp_Y_hat=self.nsp(self.hidden(encoded_X[:,0,:]))
        return encoded_X,mlm_Y_hat,nsp_Y_hat



















