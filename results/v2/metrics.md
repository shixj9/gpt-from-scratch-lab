# v2 — Multi-Head Self-Attention Experiment

## Model Configuration

| Parameter | Value |
|---|---|
| Model | Multi-Head Causal Self-Attention Language Model |
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Vocabulary Size | 65 |
| Batch Size | 32 |
| Block Size | 8 |
| Embedding Dimension (`n_embd`) | 32 |
| Attention Heads (`num_heads`) | 4 |
| Head Size (`n_embd // num_heads`) | 8 |
| Attention Layers | 1 |
| Max Iterations | 3000 |
| Evaluation Interval | 300 |
| Evaluation Iterations | 200 |
| Learning Rate | 0.01 |
| Optimizer | AdamW |
| Random Seed | 1337 |

## Training Results

| Metric | Value |
|---|---:|
| Final Train Loss | 2.2213 |
| Final Validation Loss | 2.2798 |

最终 Loss 为训练结束后重新随机采样批次计算的估计值。

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Context Window | Last 8 Tokens |
| Sampling | Softmax + Multinomial |

生成文本见 [`sample.txt`](sample.txt)。

## Observations

相比 v1（Train Loss 2.4156，Validation Loss 2.4231），v2（Train Loss 2.2213，Validation Loss 2.2798）的损失均下降；验证损失降低 0.1433。生成文本具有部分英文字符组合与对话格式特征，但仍缺乏完整的语法和长距离语义连贯性。当前结论仅基于已记录的单次实验。
