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

## 3. Dataset | 数据集下载

本项目使用 **Tiny Shakespeare** 数据集进行字符级语言模型训练。

该数据集包含莎士比亚作品的文本内容，适合用于小型语言模型的训练与测试。

**原始数据下载**

[Download input.txt](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt)
