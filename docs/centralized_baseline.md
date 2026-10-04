# Day 5 — Centralized 1D CNN baseline

## Scope

This is the first centralized ECG beat-classification baseline. It uses the Day 4 training partition for gradient updates and validation only for per-epoch monitoring and checkpoint selection. The test partition was not loaded or evaluated. No augmentation, class balancing, class weighting, federated learning, privacy mechanism, blockchain, or proposed aggregation method was added.

## Data and labels

- Source: local Day 4 group-separated MIT-BIH split, 77,550 train segments and 16,351 validation segments.
- Saved segment shape: `(216 samples, 2 channels)`; model input tensor shape: `(batch, 2 channels, 216 samples)`.
- Output classes: 15 original Day 2 annotation symbols, kept distinct.
- Symbol-to-index order: `/`→0, `A`→1, `E`→2, `F`→3, `J`→4, `L`→5, `N`→6, `Q`→7, `R`→8, `S`→9, `V`→10, `a`→11, `e`→12, `f`→13, `j`→14.
- Training has examples for all 15 labels and is strongly imbalanced (`N`: 53,489; `S`: 2). Validation has no `E`, `Q`, `R`, `S`, or `e` examples. Counts are listed in `data/splits/mitbih_v1/train_class_distribution.csv` and `validation_class_distribution.csv`.

The label order is deterministic and stored in the model checkpoint and training-history JSON. Macro-F1 is reported over all 15 configured labels; labels unsupported in validation receive a zero contribution under `zero_division=0`. Interpret this metric cautiously because five labels are absent in validation.

## Architecture

`src/models/cnn1d.py` implements a small PyTorch Conv1D classifier. The network receives channels-first `(2, 216)` segments.

| Stage | Layers | Temporal effect |
|---|---|---|
| 1 | Conv1D 2→16, kernel 7, padding 3, BatchNorm1D, ReLU, MaxPool1D 2 | 216→108 samples |
| 2 | Conv1D 16→32, kernel 5, padding 2, BatchNorm1D, ReLU, MaxPool1D 2 | 108→54 samples |
| 3 | Conv1D 32→64, kernel 3, padding 1, BatchNorm1D, ReLU | 54 samples |
| Pool/head | Adaptive average pool to length 1, Dropout 0.3, Linear 64→15 | 15 logits |

Convolution biases are disabled because each convolution is followed by batch normalization. ReLU is used after each normalization. The output is unnormalized logits for `CrossEntropyLoss`. The model has **10,127 trainable parameters** with 15 classes.

## Training configuration and procedure

Configuration is in [`configs/training/centralized_cnn_v1.json`](../configs/training/centralized_cnn_v1.json): seed 42, batch size 256, learning rate 0.001, Adam, `CrossEntropyLoss`, five epochs, CPU, eight PyTorch CPU threads, and zero DataLoader workers. Training examples are shuffled with a seeded generator. Validation is evaluated without gradients each epoch. The checkpoint saved is the epoch with the lowest validation loss. No loss weighting or balancing is applied.

The loader explicitly opens only the `train/` and `validation/` directories under `data/splits/mitbih_v1/`. It checks the shard schema, label mapping, dimensions, finite values, and record metadata. It never opens the test directory.

## Observed training results

Training completed all five configured epochs in approximately 30.5 seconds on the available CPU-only PyTorch environment (PyTorch 2.14.1+cpu). Metrics are actual outputs of this run; they are not test metrics.

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy | Validation macro-F1 (15 labels) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7918 | 0.8195 | 1.2786 | 0.6853 | 0.1737 |
| 2 | 0.2780 | 0.9356 | 1.4898 | 0.6263 | 0.0819 |
| 3 | 0.2096 | 0.9487 | 1.4255 | 0.7126 | 0.0893 |
| 4 | 0.1798 | 0.9550 | 1.6772 | 0.6030 | 0.0902 |
| 5 | 0.1595 | 0.9598 | 1.8939 | 0.7004 | 0.1351 |

The minimum validation loss occurred at epoch 1, and that checkpoint was saved. Training loss declined while validation loss was higher and generally worsened after epoch 1; this short run therefore shows an overfitting warning. Validation accuracy varies across epochs and does not account for the absent validation classes. No conclusion about generalization should be made before reviewing the split limitations, final label definition, and further validation protocol. The test set remains untouched.

## Artifacts

- Model state and metadata: `results/models/centralized_cnn_v1.pt`
- Per-epoch history: `results/metrics/centralized_cnn_v1_history.json` and `.csv`
- Training plot: `results/figures/centralized_cnn_v1_training.png`
- Executed notebook: [`notebooks/04_centralized_baseline.ipynb`](../notebooks/04_centralized_baseline.ipynb)
- Model tests: `tests/test_model_cnn1d.py`

The saved checkpoint and metrics are local generated artifacts. No test-set metric is present in them.
