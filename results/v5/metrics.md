# V5 — Multi-Layer Decoder-only GPT & Scaling Up

## Model Configuration

| Parameter | Value |
|---|---|
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Vocabulary Size | 65 |
| Batch Size | 64 |
| Block Size | 256 |
| Embedding Dimension | 384 |
| Attention Heads | 6 |
| Head Size | 64 |
| Transformer Blocks | 6 |
| FeedForward Hidden Dimension | 1536 |
| Attention Output Projection | Yes |
| Residual Connection | Yes |
| LayerNorm | Pre-LN + final LayerNorm |
| Dropout | 0.2 |
| Optimizer | AdamW |
| Learning Rate | 0.0003 |
| Max Iterations | 5000 |
| Evaluation Interval | 500 |
| Evaluation Iterations | 200 |
| Random Seed | 1337 |
| Device | cuda |
| Parameter Count | 10788929 |

## Training Results

| Metric | Value |
|---|---:|
| Final Train Loss | 1.0502 |
| Final Validation Loss | 1.4896 |

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Context Window | Last 256 Tokens |
| Sampling | Softmax + Multinomial |

Generated text: [sample.txt](sample.txt)
