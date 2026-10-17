import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# 参数设置
batch_size = 32    # 一次并行处理多少条独立的序列
block_size = 8     # 每个训练样本包含8个token；当前Bigram只根据当前token预测下一个token
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 选取计算设备
eval_iters = 200

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

# Bigram模型定义
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()  # 继承nn.Module父类，并初始化
        # 将孤立的token嵌入到一个高维向量空间，即生成 所有字符 作为 当前字符的下一位字符 的权重表
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # B = batch_size：一批有多少条文本序列
    # T = block_size：每条序列包含多少个 token
    # C = vocab_size：对于每个位置，模型都需要给词表中的每种 token 一个预测分数
    def forward(self, idx, targets=None):
        # idx     (B,T)   输入的字符序列的 Token ID
        # target  (B,T)   目标输出的字符序列的 Token ID
        # logits  (B,T,C) 把每个 Token ID 映射成一个长度为 vocab_size 的向量，从而得到(B,T,C)的 logits  
        logits = self.token_embedding_table(idx)      
        if targets is None:  # 推理/生成分支：没有标签，不计算loss，loss=None
            loss = None
        else:                # 训练分支：有标签，计算loss（不传入target就不会执行这段）
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # cross_entropy平均交叉熵损失
        return logits, loss
    
    # generate函数利用forward函数，不断生成新的字符，从而输出一整段字符序列
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx (B, T)
        for _ in range(max_new_tokens):
            # 文本生成没有标准答案，只关心下一个token，所以不用传target
            logits, loss = self(idx)

            # 我们只需要预测当前序列往后新增的那 1 个token，前面位置已经是已知的历史，不需要再预测
            logits = logits[:, -1, :]                          # (B, T, C) → (B, C)

            # dim 表示沿着张量的哪个维度进行计算，每个样本的 C 个 token 概率相加等于 1
            probs = F.softmax(logits, dim=-1)                  # (B, C)

            # torch.multinomial()根据概率抽样，probs不同token概率，取一个token（并不总是选概率最高的 token）
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 在T这个维度上进行拼接 
            idx = torch.cat((idx, idx_next), dim=1)            # (B, T)→ (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
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

# 保存到 results/v0/sample.txt
result_path = Path(__file__).resolve().parent.parent / "results" / "v0"
result_path.mkdir(parents=True, exist_ok=True)

(result_path / "sample.txt").write_text(
    generated_text,
    encoding="utf-8"
)

# 最终模型评估
final_losses = estimate_loss()

print(f"Final Train Loss: {final_losses['train']:.4f}")
print(f"Final Validation Loss: {final_losses['val']:.4f}")