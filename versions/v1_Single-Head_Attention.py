import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# 参数设置
batch_size = 32    # 一次并行处理多少条独立的序列
block_size = 8     # 最大上下文窗口，每个训练样本包含8个Token
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选取计算设备
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

# 数据链接
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
DATA_PATH = Path(__file__).resolve().parent.parent / "input.txt"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 创建词表
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]            # encode编码(str转换为对应index)
decode = lambda l: ''.join([itos[i] for i in l])   # decode编码(index转换为对应str)

# 训练集和验证集切分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) 
train_data = data[:n]
val_data = data[n:]

# 从原始数据中，提取出这样一种用于训练的数据模式：input x / target y
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)  # 将张量 x 和 y 移动到指定的计算设备上
    return x, y

# 模型评估
@torch.no_grad()  # 执行这个函数时，不需要记录梯度，也不需要构建用于反向传播的计算图，仅用于模型推理、文本生成和模型评估
def estimate_loss():
    out = {}
    model.eval()                           # .eval()：切换到评估模式（函数内部没有定义model，Python会到函数外部的全局作用域中寻找它）
    for split in ['train', 'val']:         # 依次评估两个数据集
        losses = torch.zeros(eval_iters)   # 创建一个包含 eval_iters 个零的一维张量
        for k in range(eval_iters):    
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()         # out[split] = losses.mean()：计算平均损失
    model.train()
    return out

# 实现一个注意力头
class Head(nn.Module):
    def __init__(self, head_size):  # 传入qkv向量维度参数head_size
        super().__init__()
        # 同一个 Head 中的所有 Token 共用 W_q W_k W_v 权重矩阵，qkv向量的维度由 head_size 决定
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # register_buffer()方法，将一个张量注册到神经网络模块中，使它成为模型状态的一部分，但不会作为可训练参数参与梯度更新。

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,C) Query（Q）：当前 Token 想寻找什么信息
        q = self.query(x)  # (B,T,C) Key（K）：每个 Token 提供什么样的信息匹配线索

        # 计算注意力权重矩阵
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5             # (B, T, C) @ (B, C, T) -> (B, T, T)  计算历史token的注意力分数
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)  Mask：屏蔽未来 Token，确保当前 Token 只能关注自己和之前的 Token
        wei = F.softmax(wei, dim=-1)                                  # (B, T, T)  Softmax：将匹配分数转换为非负且每行总和为1的注意力权重

        v = self.value(x)  # (B,T,C) Value（V）：每个 Token 实际提供什么信息供其他 Token 聚合
        out = wei @ v      # (B, T, T) @ (B, T, C) -> (B, T, C)  out聚合了当前和历史的 Value 信息
        return out

# super simple bigram模型定义
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)     # Token Embedding：将Token索引映射为特征向量，也就是这个token是什么; vocab_size个token，生成n_embd维向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Position Embedding：为每个位置生成位置向量，也就是这个token在哪里; block_size个位置，生成n_embd维向量
        self.sa_head = Head(n_embd)                                       # 单头Self-Attention：聚合Token之间的信息
        self.lm_head = nn.Linear(n_embd, vocab_size)                      # Language Model Head：将特征映射为词表大小的Logits

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx 和 targets 都是(B,T)形状的整数张量
        tok_emb = self.token_embedding_table(idx)                                    # Token内容编码(B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # Token位置编码(T,C)
        x = tok_emb + pos_emb     # Token Embedding + Position Embedding 内容+位置编码 (B,T,C)
        x = self.sa_head(x)       # 经过单头Self-Attention (B,T,C)，每个 Token 都有自己的 QKV 向量
        logits = self.lm_head(x)  # 计算预测下一个Token的Logits (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B,T)，表示当前Token序列
        for _ in range(max_new_tokens):

            # 只使用最近block_size个Token作为上下文
            idx_cond = idx[:, -block_size:]

            # 计算预测结果
            logits, loss = self(idx_cond)

            # 只取最后一个Token位置的预测结果
            logits = logits[:, -1, :]  # (B,C)

            # 转换成概率分布
            probs = F.softmax(logits, dim=-1)  # (B,C)

            # 根据概率采样下一个Token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)

            # 将生成的Token拼接到序列末尾
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)

        return idx

model = BigramLanguageModel()
m = model.to(device)

# 创建优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # 每eval_interval步输出一次当前loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # 获取训练数据
    xb, yb = get_batch('train')

    # 评估损失
    logits, loss = model(xb, yb)            # torch会建立一个计算图，如果没有这一步backward就无法执行
    optimizer.zero_grad(set_to_none=True)   # 梯度清零
    loss.backward()                         # 反向传播计算梯度，并将梯度存储到参数的 .grad 属性中，不会自动改变参数数值
    optimizer.step()                        # 利用优化算法（这里是 AdamW）和已有梯度更新参数

# 生成文本测试
context = torch.zeros((1, 1), dtype=torch.long, device=device)

with torch.no_grad():
    output = m.generate(context, max_new_tokens=500)

generated_text = decode(output[0].tolist())

print(generated_text)

# 保存到 results/v1/sample.txt
result_path = Path(__file__).resolve().parent.parent / "results" / "v1"
result_path.mkdir(parents=True, exist_ok=True)

(result_path / "sample.txt").write_text(
    generated_text,
    encoding="utf-8"
)

# 最终模型评估
final_losses = estimate_loss()

print(f"Final Train Loss: {final_losses['train']:.4f}")
print(f"Final Validation Loss: {final_losses['val']:.4f}")

