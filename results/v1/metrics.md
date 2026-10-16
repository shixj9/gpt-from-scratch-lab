# v1 — Single-Head Self-Attention Experiment

## Model Configuration

| Parameter | Value |
|---|---|
| Model | Single-Head Causal Self-Attention Language Model |
| Dataset | Tiny Shakespeare |
| Vocabulary Size | 65 |
| Batch Size | 32 |
| Block Size | 8 |
| Embedding Dimension (`n_embd`) | 32 |
| Attention Heads | 1 |
| Head Size | 32 |
| Max Iterations | 3000 |
| Evaluation Interval | 300 |
| Evaluation Iterations | 200 |
| Learning Rate | 0.01 |
| Optimizer | AdamW |
| Random Seed | 1337 |

## Training Results

| Metric | Value |
|---|---:|
| Final Train Loss | 2.4156 |
| Final Validation Loss | 2.4231 |

以上 Loss 为训练完成后重新随机取批次估计的结果。

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Context Window | Last 8 Tokens |
| Sampling | Softmax + Multinomial |

生成文本见 [`sample.txt`](sample.txt)。

## Observations

相比 v0 的验证损失 2.4897，v1 的验证损失降低到 2.4231。模型可以利用窗口内的历史 Token 信息，但生成文本仍存在不完整单词与缺乏长距离连贯性的问题。当前结果为已记录运行数据，不代表严格控制所有变量的对照实验。
