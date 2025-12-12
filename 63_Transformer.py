import math
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import os
import io

# ========== 核心修复：重写读取数据的函数，指定 UTF-8 编码 ==========
def read_data_nmt():
    """载入英语-法语数据集"""
    data_dir = d2l.download_extract('fra-eng')
    with io.open(os.path.join(data_dir, 'fra.txt'), encoding='utf-8') as f:
        return f.read()

# 覆盖 d2l 原有的函数
d2l.read_data_nmt = read_data_nmt
# ========== 修复结束 ==========

# 修复1：类继承名拼写错误 PositionWiseFNN -> PositionWiseFFN
# 修复2：变量名拼写错误 dens1->dense1, dense2参数ffn_um_hiddens->ffn_num_hiddens
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,** kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)  # 修正类名
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)  # 修正变量名
        self.relu = nn.ReLU()
        # 修正变量名 ffn_um_hiddens -> ffn_num_hiddens
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
                X: 原始输入 (残差路径)
                Y: 经过子层（如 Attention 或 FFN）处理后的输出
                """
        # 核心逻辑：LayerNorm( X + Dropout(Sublayer(X)) )
        # 这里体现的是 Post-Norm 结构
        return self.ln(self.dropout(Y) + X)

# 修复3：EncoderBlock参数名错误 ffn_num_inputs -> ffn_num_input
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,  # 修正参数名
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens
        )
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False):
        super(TransformerEncoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)

        # 位置编码：赋予模型识别词序的能力
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)

        # 使用 nn.Sequential 堆叠多个 EncoderBlock
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"block_{i}",
                                 EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

        self.attention_weights = None

    def forward(self, X, valid_lens):
        # 1. Embedding + Scaling + Positional Encoding
        # 乘以 sqrt(d_model) 是为了保持数据分布的方差，防止位置信息主导语义信息
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        self.attention_weights = [None] * len(self.blks)

        # 2. 逐层通过 EncoderBlock
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            # 记录每一层的注意力权重（用于可视化分析）
            self.attention_weights[i] = blk.attention.attention.attention_weights

        return X

# 修复4：PositionWiseFFN初始化多传了dropout参数（原代码中该类没有dropout参数）
class DecoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 第一层：掩码自注意力（Masked Self-Attention）
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)

        # 第二层：交叉注意力（Encoder-Decoder Attention）
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)

        # 修复：移除多余的dropout参数（PositionWiseFFN没有这个参数）
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        # state[0]: enc_outputs, state[1]: enc_valid_lens, state[2]: key_values_cache
        enc_outputs, enc_valid_lens = state[0], state[1]

        # 维护 KV Cache（重点：把当前 X 的特征存入该层的缓存中）
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values  # 更新缓存

        # 训练模式和推理模式的掩码处理
        if self.training:
            batch_size, num_steps, _ = X.shape
            # 生成因果掩码：确保每个位置只看过去
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # 1. Masked Self-Attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)

        # 2. Encoder-Decoder Attention (Query来自Y, Key/Value来自Encoder)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)

        # 3. FFN
        return self.addnorm3(Z, self.ffn(Z)), state
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    # 修复5：添加缺失的attention_weights属性（d2l库的AttentionDecoder需要）
    @property
    def attention_weights(self):
        return self._attention_weights

# 超参数设置
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()  # 减少训练轮数，加快测试
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

# 加载翻译数据集
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

# 初始化编码器和解码器
encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)

# 构建完整的Encoder-Decoder模型
net = d2l.EncoderDecoder(encoder, decoder)

# 训练模型
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
# 显示训练损失曲线
d2l.plt.show()