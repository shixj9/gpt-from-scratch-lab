# GPT From Scratch Lab

## 1. Introduction | 项目简介

本项目旨在使用 **PyTorch 从零构建 GPT 语言模型**，通过逐步实现模型的核心组件，深入理解语言模型的工作原理与 Transformer 架构。

项目主要参考 Andrej Karpathy 的 [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) 教学视频。

学习过程中，将结合代码实现、模型训练与实验结果，记录不同模型结构的实现原理、核心功能、特点及局限性。

## 2. Learning Map | 模型学习与实现

### v0 — Bigram Language Model

**模型简介**

Bigram（二元语言模型）是一种简单的语言模型，其核心思想是：**仅根据当前 Token 预测下一个 Token，而不考虑更早的上下文信息。**

本项目采用字符级（Character-level）Tokenization，将文本中的每个字符映射为唯一的整数 ID，并使用 PyTorch 的 `nn.Embedding` 实现可学习的 Token 预测查找表。

**核心实现**

- **Character-level Tokenization**：建立字符与整数 Token ID 之间的双向映射，实现文本编码和解码。
- **Data Batching**：从训练数据中随机抽取连续的 Token 序列，构造模型输入与预测目标。
- **Token Embedding**：使用 `nn.Embedding(vocab_size, vocab_size)` 建立可训练权重矩阵，根据当前 Token 输出下一个 Token 的预测分数（Logits）。
- **Cross Entropy Loss**：计算预测分布与真实下一个 Token 之间的交叉熵损失。
- **AdamW Optimizer**：结合反向传播与梯度更新，优化 Embedding 权重参数。
- **Autoregressive Generation**：通过 Softmax 获取预测概率，使用多项式分布采样，逐步生成新的 Token 序列。

**模型特点**

Bigram 模型的可训练部分只有一个 Embedding 权重矩阵。对于大小为 V 的词表，模型拥有 V × V 个可训练参数。

模型通过学习训练文本中相邻 Token 的统计关系，预测当前 Token 后面最有可能出现的 Token。

**优点**

- 模型结构简单，参数规模小，计算开销较低。
- 容易理解和实现，适合作为语言模型的基础学习案例。
- 能够学习相邻字符之间的统计规律。
- 可以作为后续更复杂语言模型的性能基线（Baseline）。

**局限性**

- **缺乏上下文建模能力**：预测下一个 Token 时，仅依赖当前 Token，无法利用更早的历史信息。
- **无法学习长距离依赖**：难以捕捉单词、句子之间的复杂关系。
- **生成文本连贯性较差**：虽然能够学习局部字符组合规律，但生成的长文本通常缺乏语义一致性。
- **模型表达能力有限**：难以表示复杂语言结构和上下文条件下的不同预测分布。

### v1 — Single-Head Self-Attention Language Model

**模型简介**

v1 在 v0 的字符级语言模型基础上，引入 **Token Embedding、Position Embedding 和单头因果自注意力（Single-Head Causal Self-Attention）**。模型不再只根据当前 Token 预测下一位，而是可以综合当前及之前 Token 的信息，在长度不超过 `block_size` 的上下文窗口中进行预测。

**核心实现**

- **Token Embedding**：使用 `nn.Embedding(vocab_size, n_embd)` 将字符 ID 映射为特征向量。
- **Position Embedding**：使用 `nn.Embedding(block_size, n_embd)` 添加可学习的位置特征。
- **Query / Key / Value**：通过三个独立的线性层，为每个 Token 生成 Q、K、V 向量；同一个注意力头内所有 Token 共享线性层参数。
- **Scaled Dot-Product Attention**：利用 `Q @ Kᵀ / √dₖ` 计算 Token 之间的匹配分数。
- **Causal Mask + Softmax**：屏蔽未来 Token，将允许访问的位置归一化为注意力权重。
- **Value Aggregation**：通过注意力权重与 V 相乘，得到聚合历史上下文后的特征。
- **Language Model Head**：用 `nn.Linear(n_embd, vocab_size)` 输出下一 Token 的预测分数（Logits）。

**模型特点**

对于每个位置，模型可以利用当前位置以及之前位置的特征；Q、K 决定信息聚合权重，V 提供实际被聚合的内容。当前版本只有一个注意力头，尚未实现多头注意力、前馈网络或残差连接。

**优点**

- 从仅利用当前 Token 过渡到利用窗口内的历史上下文。
- 通过可训练的 Q、K、V 映射学习不同 Token 之间的信息关联。
- 通过因果掩码保证预测时不会读取未来 Token。

**局限性**

- `block_size=8`，字符级上下文窗口较短。
- 单头、单层注意力结构较简单，缺少前馈网络、残差连接等 Transformer 组件。
- 生成文本仍有大量不完整词语和不连贯语句。

**实验结果**

| Metric | v0 — Bigram | v1 — Single-Head Attention |
|---|---:|---:|
| Train Loss | 2.4613 | 2.4156 |
| Validation Loss | 2.4897 | 2.4231 |

以上为各次已记录实验结果；不同版本的模型结构和训练设置存在差异，数据只用于当前学习实验的初步对照。完整 V1 实验配置参见 [`results/v1/metrics.md`](results/v1/metrics.md)，生成样本参见 [`results/v1/sample.txt`](results/v1/sample.txt)。

## 3. Dataset | 数据集下载

本项目使用 **Tiny Shakespeare** 数据集进行字符级语言模型训练。

该数据集包含莎士比亚作品的文本内容，适合用于小型语言模型的训练与测试。

**原始数据下载**

[Download input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)


## 4. Run | 运行方式

安装项目依赖后，在项目根目录执行：

```bash
python versions/v0_bigram.py
python versions/v1_single-head-attention.py
```

运行前请确认 `input.txt` 位于项目根目录，且脚本名称与本地仓库一致。实验样本与指标分别保存在 `results/v0/`、`results/v1/` 中。
