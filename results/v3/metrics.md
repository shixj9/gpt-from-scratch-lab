# v3 — FeedForward Network Experiment

## Model Configuration

| Parameter | Value |
|---|---|
| Model | Multi-Head Self-Attention + FeedForward Language Model |
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Vocabulary Size | `len(chars)` (determined from `input.txt`) |
| Batch Size | 32 |
| Block Size | 8 |
| Embedding Dimension | 32 |
| Attention Heads | 4 |
| Head Size | 8 |
| FeedForward Hidden Dimension | 128 (4 × 32) |
| FeedForward Activation | ReLU |
| Residual Connections | No |
| Layer Normalization | No |
| Max Iterations | 3000 |
| Evaluation Interval | 300 |
| Evaluation Batches | 200 per split |
| Learning Rate | 0.01 |
| Optimizer | AdamW |
| Random Seed | 1337 |

## Training Results

| Metric | Value |
|---|---:|
| Final Train Loss | 2.1397 |
| Final Validation Loss | 2.2138 |
| Train–Validation Gap | 0.0741 |

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Sampling | Softmax + Multinomial |
| Generated Text | See `sample.txt` |

## Experiment Notes

V3 adds a position-wise FeedForward network (`Linear(32, 128) → ReLU → Linear(128, 32)`) after the four-head causal Self-Attention module. Each token's contextual representation is transformed independently using the same FFN parameters.

Compared with V2 (final train loss 2.2213, validation loss 2.2798), V3 achieved lower losses in these recorded runs: train loss decreased by 0.0816 and validation loss by 0.0660. This is a single-run observation rather than a controlled estimate of the FFN's effect.

The generated sample retains Shakespeare-like dialogue formatting and some recognizable English fragments, but remains largely incoherent. The context window is limited to eight character-level tokens.
