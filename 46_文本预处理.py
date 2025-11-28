# 导入依赖库：collections用于统计词频，re用于正则清洗，d2l提供数据集和工具函数
import collections
import re
from d2l import torch as d2l

# -----------------------------------------------------------------------------
# 1. 配置数据集（《时间机器》文本）
# -----------------------------------------------------------------------------
# d2l.DATA_HUB：d2l的数据集注册中心，存储下载地址和校验码
# @save：d2l装饰器，将该配置保存到d2l库，方便后续复用
d2l.DATA_HUB['time_machine'] = (
    d2l.DATA_URL + 'timemachine.txt',  # 数据集下载链接
    '090b5e7e70c295757f55df93cb0a180b9691891a'  # 数据校验码（确保下载完整无损坏）
)


# -----------------------------------------------------------------------------
# 2. 加载并清洗文本：将原始文本转为干净的文本行列表
# -----------------------------------------------------------------------------
def read_time_machine():  # @save
    """
    加载《时间机器》数据集并进行文本清洗
    返回：清洗后的文本行列表（每行仅含小写字母和空格）
    """
    # 下载并打开文件：d2l.download自动处理下载/缓存，'r'表示只读模式
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()  # 按行读取文本，lines是列表，每个元素是一行原始文本

    # 文本清洗三步法（正则表达式+字符串处理）
    cleaned_lines = []
    for line in lines:
        # 1. re.sub('[^A-Za-z]+', ' ', line)：将所有非字母字符（数字、标点、换行等）替换为单个空格
        # 2. strip()：去除行首/行尾的空白字符（避免首尾多余空格）
        # 3. lower()：所有字母转为小写（统一大小写，减少词表冗余）
        cleaned_line = re.sub('[^A-Za-z]+', ' ', line).strip().lower()
        cleaned_lines.append(cleaned_line)

    return cleaned_lines


# 测试：加载清洗后的文本
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')  # 打印总行数（约3221行）
print(f'第0行文本: {lines[0]}')  # 输出：the time machine by h g wells
print(f'第10行文本: {lines[10]}')  # 输出：the time traveller for so it will be convenient to speak of him


# -----------------------------------------------------------------------------
# 3. 词元化：将文本行拆分为最小处理单位（单词/字符）
# -----------------------------------------------------------------------------
def tokenize(lines, token='word'):  # @save
    """
    将文本行列表拆分为词元（token）列表
    参数：
        lines: 清洗后的文本行列表
        token: 词元类型（'word'=单词级，'char'=字符级）
    返回：
        词元列表（2D列表：每行对应一个词元子列表）
    """
    if token == 'word':
        # 单词级词元化：按空格分割（如"hello world" → ["hello", "world"]）
        return [line.split() for line in lines]
    elif token == 'char':
        # 字符级词元化：按字符分割（如"hello" → ["h", "e", "l", "l", "o"]）
        return [list(line) for line in lines]
    else:
        # 非法词元类型提示
        raise ValueError(f'错误：未知词元类型「{token}」，仅支持「word」或「char」')


# 测试：单词级词元化
tokens = tokenize(lines, token='word')
print('\n前11行的单词词元：')
for i in range(11):
    print(f'第{i}行：{tokens[i]}')


# -----------------------------------------------------------------------------
# 4. 词频统计：为构建词表提供依据
# -----------------------------------------------------------------------------
def count_corpus(tokens):  # @save
    """
    统计词元的出现频率
    参数：
        tokens: 2D词元列表（每行对应一个词元子列表）
    返回：
        词元-频率字典（collections.Counter类型）
    """
    # 先将2D词元列表展平为1D列表（如[[a,b],[c,d]] → [a,b,c,d]）
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    # 统计每个词元的出现次数（如{"the": 1000, "time": 500}）
    return collections.Counter(tokens)


# -----------------------------------------------------------------------------
# 5. 构建词表：实现词元↔索引的双向映射（模型仅能处理数字）
# -----------------------------------------------------------------------------
class Vocab:  # @save
    """
    文本词表类：管理词元与索引的映射，支持低频词过滤和未知词处理
    """

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化词表
        参数：
            tokens: 词元列表（1D或2D）
            min_freq: 最小出现频率（低于该值的词元将被舍弃）
            reserved_tokens: 预留词元（如<unk>未知词、<pad>填充符等）
        """
        # 初始化默认值
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 步骤1：统计词频并按频率降序排序（高频词在前，提升编码效率）
        self._token_freqs = sorted(
            count_corpus(tokens).items(),  # 词元-频率对
            key=lambda x: x[1],  # 按频率排序
            reverse=True  # 降序（高频词优先）
        )

        # 步骤2：初始化索引→词元、词元→索引映射（默认包含<unk>未知词元）
        self.idx_to_token = ['<unk>'] + reserved_tokens  # 索引到词元的列表（索引0=未知词）
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)
        }  # 词元到索引的字典

        # 步骤3：将词元加入词表（过滤低频词）
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break  # 频率低于阈值，后续词元更低频，直接终止
            if token not in self.token_to_idx:  # 避免重复（如预留词元）
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 新词元的索引=当前列表长度-1

    def __len__(self):
        """返回词表大小（即词元总数）"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        重载[]运算符：将词元（单个/列表）转换为索引
        支持：单个词元（str）→ 单个索引（int）；词元列表 → 索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            # 单个词元：存在则返回索引，否则返回<unk>的索引（0）
            return self.token_to_idx.get(tokens, self.unk)
        # 多个词元：递归转换每个词元
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        将索引（单个/列表）转换为词元
        支持：单个索引（int）→ 单个词元（str）；索引列表 → 词元列表
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]  # 单个索引直接查找
        # 多个索引：逐个查找并返回词元
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):
        """未知词元的索引（固定为0）"""
        return 0

    @property
    def token_freqs(self):
        """返回词元-频率对（用于分析词频分布）"""
        return self._token_freqs


# 测试：构建单词级词表
vocab = Vocab(tokens, min_freq=1)  # min_freq=1：保留所有出现过的词元
print('\n词表前10个词元→索引映射：')
print(list(vocab.token_to_idx.items())[:10])  # 输出：[('<unk>', 0), ('the', 1), ('and', 2), ...]

# 测试：词元↔索引双向转换
print('\n词元→索引转换示例：')
for i in [0, 10]:
    print(f'文本词元：{tokens[i]}')
    print(f'对应索引：{vocab[tokens[i]]}')
    print(f'索引→词元：{vocab.to_tokens(vocab[tokens[i]])}')
    print('-' * 50)


# -----------------------------------------------------------------------------
# 6. 整合全流程：加载数据集并返回索引序列和词表
# -----------------------------------------------------------------------------
def load_corpus_time_machine(max_tokens=-1):  # @save
    """
    整合文本加载→清洗→词元化→词表→索引序列的全流程
    参数：
        max_tokens: 最大索引序列长度（-1表示不限制，取全部）
    返回：
        corpus: 1D索引序列（模型输入数据）
        vocab: 字符级词表（映射关系）
    """
    # 步骤1：加载并清洗文本
    lines = read_time_machine()
    # 步骤2：字符级词元化（适合字符级语言模型训练）
    tokens = tokenize(lines, token='char')
    # 步骤3：构建字符级词表
    vocab = Vocab(tokens)
    # 步骤4：将所有词元展平为1D索引序列（模型仅接受数字输入）
    corpus = [vocab[token] for line in tokens for token in line]
    # 步骤5：限制最大长度（可选，用于快速测试）
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


# 测试：加载最终的数据集（字符级）
corpus, vocab = load_corpus_time_machine()
print(f'\n最终索引序列长度（总字符数）：{len(corpus)}')  # 输出约10万+
print(f'字符级词表大小：{len(vocab)}')  # 输出28（26字母+空格+<unk>）
print(f'词表包含的所有字符：{vocab.idx_to_token}')  # 查看完整词表