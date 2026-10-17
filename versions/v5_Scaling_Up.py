import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# 参数设置：V5 扩大上下文、模型宽度与深度
batch_size = 64          # 每批训练样本数量；显存不足可改为16或32
block_size = 256         # 每个训练样本包含的字符级Token数，也是最大上下文窗口
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embd = 384             # 每个Token的特征维度
num_heads = 6            # 6个注意力头，每个Head输出64维
n_layer = 6              # 堆叠6个Transformer Block
dropout = 0.2            # 训练时随机屏蔽部分激活，缓解过拟合
assert n_embd % num_heads == 0

torch.manual_seed(1337)

# 数据链接
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
DATA_PATH = Path(__file__).resolve().parent.parent / "input.txt"
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# 字符级Tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda indices: ''.join(itos[i] for i in indices)

# 训练集和验证集切分
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    source = train_data if split == 'train' else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    # 关闭梯度记录；评估结束后恢复原先的train/eval模式
    out = {}
    was_training = model.training
    model.eval()
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train(was_training)
    return out


# 单头因果自注意力：每个Token只能关注自身及之前的Token
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)       # (B,T,head_size)
        q = self.query(x)     # (B,T,head_size)
        v = self.value(x)     # (B,T,head_size)

        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # 训练时随机屏蔽部分注意力连接
        return wei @ v           # (B,T,head_size)


# 多头注意力：各Head独立学习，拼接后用输出投影混合各头特征
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,n_embd)
        out = self.proj(out)                                  # 输出投影W_O
        return self.dropout(out)


# 前馈网络：逐Token执行非线性特征变换，不直接混合Token位置
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 384 -> 1536
            nn.ReLU(),                      # 非线性激活
            nn.Linear(4 * n_embd, n_embd),  # 1536 -> 384
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# Pre-LN Transformer Block：LayerNorm + Attention/FFN + Residual
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, n_embd // num_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # 注意力子层前的归一化
        self.ln2 = nn.LayerNorm(n_embd)  # 前馈子层前的归一化

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # 先归一化、计算注意力，再与原x残差相加
        x = x + self.ffwd(self.ln2(x))  # 对更新后的x重复归一化、FFN、残差相加
        return x


# 字符级Decoder-only GPT：多个独立Block堆叠，自回归预测下一个Token
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, num_heads) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)            # 所有Block之后的最终归一化
        self.lm_head = nn.Linear(n_embd, vocab_size) # 映射到词表Logits

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )                                           # (T,n_embd)，沿B维广播
        x = tok_emb + pos_emb
        x = self.blocks(x)                          # 依次经过6个独立Transformer Block
        x = self.ln_f(x)
        logits = self.lm_head(x)                    # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]          # 只取最近block_size个Token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]               # 最后一个位置的预测分数
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = TransformerLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(f'Device: {device}, Parameters: {sum(p.numel() for p in model.parameters()):,}')

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 文本生成：评估模式关闭Dropout；no_grad关闭梯度记录
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    output = model.generate(context, max_new_tokens=500)
generated_text = decode(output[0].tolist())
print(generated_text)

# 保存V5生成结果和本次实验配置、最终Loss
result_path = Path(__file__).resolve().parent.parent / 'results' / 'v5'
result_path.mkdir(parents=True, exist_ok=True)
(result_path / 'sample.txt').write_text(generated_text, encoding='utf-8')

final_losses = estimate_loss()
train_loss = final_losses['train'].item()
val_loss = final_losses['val'].item()
print(f'Final Train Loss: {train_loss:.4f}')
print(f'Final Validation Loss: {val_loss:.4f}')

metrics = f'''# V5 — Multi-Layer Decoder-only GPT & Scaling Up

## Model Configuration

| Parameter | Value |
|---|---|
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Vocabulary Size | {vocab_size} |
| Batch Size | {batch_size} |
| Block Size | {block_size} |
| Embedding Dimension | {n_embd} |
| Attention Heads | {num_heads} |
| Head Size | {n_embd // num_heads} |
| Transformer Blocks | {n_layer} |
| FeedForward Hidden Dimension | {4 * n_embd} |
| Attention Output Projection | Yes |
| Residual Connection | Yes |
| LayerNorm | Pre-LN + final LayerNorm |
| Dropout | {dropout} |
| Optimizer | AdamW |
| Learning Rate | {learning_rate} |
| Max Iterations | {max_iters} |
| Evaluation Interval | {eval_interval} |
| Evaluation Iterations | {eval_iters} |
| Random Seed | 1337 |
| Device | {device} |
| Parameter Count | {sum(p.numel() for p in model.parameters())} |

## Training Results

| Metric | Value |
|---|---:|
| Final Train Loss | {train_loss:.4f} |
| Final Validation Loss | {val_loss:.4f} |

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Context Window | Last {block_size} Tokens |
| Sampling | Softmax + Multinomial |

Generated text: [sample.txt](sample.txt)
'''
(result_path / 'metrics.md').write_text(metrics, encoding='utf-8')
