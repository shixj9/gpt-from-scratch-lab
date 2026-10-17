# v4 — Residual Connections & Layer Normalization Experiment

## Model Configuration

| Parameter | Value |
|---|---|
| Model | Character-level Transformer Language Model (1 Pre-LN Block) |
| Dataset | Tiny Shakespeare |
| Tokenization | Character-level |
| Vocabulary Size | 65 |
| Batch Size | 32 |
| Block Size | 8 |
| Embedding Dimension (`n_embd`) | 32 |
| Attention Heads (`num_heads`) | 4 |
| Head Size (`n_embd // num_heads`) | 8 |
| Transformer Blocks | 1 |
| FeedForward Hidden Dimension | 128 (`4 * n_embd`) |
| Residual Connections | 2 per Block (Attention and FeedForward) |
| Layer Normalization | 2 independent `nn.LayerNorm(32)` modules, Pre-LN |
| Max Iterations | 3000 |
| Evaluation Interval | 300 |
| Evaluation Iterations | 200 |
| Learning Rate | 0.01 |
| Optimizer | AdamW |
| Random Seed | 1337 |

## Training Results

| Metric | Value |
|---|---:|
| Final Train Loss | **2.0782** |
| Final Validation Loss | **2.1684** |
| Train–Validation Gap | 0.0902 |

### Training Progress

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.4793 | 4.4905 |
| 300 | 2.3378 | 2.3661 |
| 600 | 2.2350 | 2.2810 |
| 900 | 2.1896 | 2.2404 |
| 1200 | 2.1428 | 2.2118 |
| 1500 | 2.1302 | 2.1943 |
| 1800 | 2.1218 | 2.1905 |
| 2100 | 2.1068 | 2.1909 |
| 2400 | 2.0758 | 2.1715 |
| 2700 | 2.0880 | 2.1734 |

## Generation Configuration

| Parameter | Value |
|---|---|
| Initial Context | Token ID 0 |
| New Tokens | 500 |
| Context Window | Last 8 tokens |
| Sampling | Softmax + Multinomial |
| Generated Sample | [sample.txt](sample.txt) |

## Observations

- V4 adds two residual connections and two independent LayerNorm modules around the attention and feedforward sublayers, forming one Pre-LN Transformer Block.
- Final validation loss decreased from **2.2138 (V3)** to **2.1684 (V4)**, an absolute reduction of **0.0454** (approximately **2.05%**).
- The generated sample imitates some Shakespeare-like dialogue formatting and character names, but remains largely incoherent. A context window of only eight character tokens limits longer-range consistency.
- The train–validation gap is **0.0902**. Results reflect a single run; improvement cannot be conclusively attributed to the new architecture without repeated controlled experiments.

> Note: Final losses are measured after all 3000 optimizer steps, while the training-progress table records evaluations at steps 0 through 2700.
