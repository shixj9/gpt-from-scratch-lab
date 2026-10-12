# v0 — Bigram Experiment

## Model Configuration

| Parameter | Value |
|---|---|
| Model | Bigram Language Model |
| Dataset | Tiny Shakespeare |
| Vocabulary Size | 65 |
| Batch Size | 32 |
| Block Size | 8 |
| Max Iterations | 3000 |
| Learning Rate | 0.01 |
| Optimizer | AdamW |
| Random Seed | 1337 |

## Training Results

| Metric | Value |
|---|---|
| Train Loss | 待填写 |
| Validation Loss | 待填写 |

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Sampling | Softmax + Multinomial |

生成文本见 `sample.txt`。