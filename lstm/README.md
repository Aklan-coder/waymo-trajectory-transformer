# LSTM and Constant-Velocity Baseline

This folder holds the **LSTM and constant-velocity baseline** for our CSCE 883 (Machine Learning, Prof. Jianjun Hu) project, *Multimodal Transformer-Based Vehicle Trajectory Prediction Under Uncertain Driving Conditions*. The Transformer models, the data conversion script (`fetch_convert.py`) and the main project README are in the parent folder of this repository.

**Task.** Given 1.1 s of a vehicle's past motion (11 steps at 10 Hz), predict where it will be over the next 3 s (30 steps) on the Waymo Open Motion Dataset.

## Results

Evaluated once on the test set, after all tuning was finished. Errors are in metres. Brackets are 95% bootstrap confidence intervals over cars.

| Model | ADE | FDE |
| --- | --- | --- |
| Constant velocity | 1.430 [1.407, 1.452] | 3.892 [3.832, 3.953] |
| **LSTM (residual, hidden 256)** | **0.631 [0.620, 0.642]** | **1.789 [1.756, 1.822]** |

- Test set: 12,555 vehicles. Validation set: 12,190 vehicles.
- The LSTM lowers ADE by 0.798 m (95% CI -0.820 to -0.778) and FDE by 2.103 m (95% CI -2.161 to -2.047) compared with constant velocity.
- The LSTM is better than constant velocity on about 67% of cars for ADE and 65% for FDE. It is not better on every car.
- ADE is the mean distance error over all 30 future steps. FDE is the distance error at the last step (3 s).

Tables and figures are in [`results/`](results/).

### Tuning (validation set)

Best validation ADE/FDE over training for each configuration.

| Run | Settings | Training data | Val ADE | Val FDE |
| --- | --- | --- | --- | --- |
| `lstm_a` | hidden 128, up to 30 epochs | 50 files | 0.692 | 1.936 |
| `lstm_long` | hidden 128, up to 60 epochs | 50 files | 0.680 | 1.913 |
| `lstm_h256` | hidden 256 | 50 files | 0.673 | 1.896 |
| `lstm_res` | residual, hidden 128 | 50 files | 0.672 | 1.885 |
| `lstm_res_h256` | residual, hidden 256 | 50 files | 0.655 | 1.843 |
| **`lstm_res_h256_big`** | residual, hidden 256 | **200 files** | **0.631** | **1.782** |

The last row is the final model. Model selection used validation ADE only. Training histories are in `results/histories/`.

## Model

`lstm_model.py`, class `LSTMPredictor`:

- Input: `[batch, 11, 6]`, with features x, y, vx, vy, and the sin and cos of the heading change. Inputs are standardised with the training-set mean and standard deviation, which are stored inside the checkpoint.
- Encoder: 2-layer LSTM, hidden size 256, dropout 0.1.
- Head: Linear(256, 256), ReLU, Dropout, Linear(256, 60), reshaped to 30 (x, y) points. All 30 points are predicted in one shot.
- **Residual mode** (`--residual`): the network predicts a correction added to a constant-velocity path computed from the last two observed positions.
- Loss: ADE itself (mean Euclidean distance over all 30 steps).
- About 0.9 million parameters.
- Training: Adam, learning rate 1e-3, halved after 2 epochs without improvement, batch size 1024, gradient clipping at 1.0, early stopping after 6 epochs without improvement in validation ADE.

## Data

The data is built with `fetch_convert.py` from the parent folder, so it is identical in format to the Transformer work:

- Waymo Open Motion Dataset v1.3.1, vehicles only, observed at all 41 steps (11 past, 30 future).
- Each vehicle is centred at its current position and rotated so its heading faces +x.
- Vehicles that move less than 2 m over the 3 s horizon count as parked, and only a random 10% of them are kept (seeded by the file index in the conversion script).
- **Training:** training files 0-199 of 1,000 (818,954 samples). The Transformer work used all 1,000 files.
- **Validation:** validation files 00000-00004 (12,190 vehicles).
- **Test:** validation files 00005-00009 (12,555 vehicles), because Waymo's own test split hides the future.

The Waymo data is **not** included in this repository, because of the dataset license and its size. Register at [waymo.com/open](https://waymo.com/open) to get access.

## Files in this folder

| File | Purpose |
| --- | --- |
| `lstm_model.py` | LSTM model, ADE loss, normalisation, checkpoint saving and loading |
| `train_lstm_arrays.py` | Trains the LSTM on a folder of converted `.npz` files |
| `evaluate_test_arrays.py` | ADE/FDE with bootstrap confidence intervals, constant-velocity baseline, paired differences, figures, saved predictions |
| `array_data.py` | Loads `.npz` data (a single file, two `.npy` files, or a folder of per-file `.npz` files) and checks shapes |
| `baselines.py` | Constant-velocity baselines |
| `common.py` | ADE/FDE and bootstrap helpers |
| `plots.py` | Loss curves, bar charts, trajectory overlays |
| `make_fake_arrays.py` | Makes fake data in the right format, for testing the scripts only |
| `results/` | Result tables, figures, and training histories |

The trained checkpoint (`best.pt`, about a few MB) is not tracked by git because this repository ignores `*.pt` files. Download it here: **[Google Drive link, add before submitting]**.

## How to reproduce

Requirements: Python 3.10+. A GPU is recommended (about 20 s per epoch on a Colab T4 with 200 files).

```bash
pip install torch numpy pandas matplotlib scipy protobuf
```

**1. Build the data.** Run from the repository root (needs Waymo access and `gsutil`):

```bash
python fetch_convert.py --split validation --start 0 --end 5   --out data/val
python fetch_convert.py --split validation --start 5 --end 10  --out data/test
python fetch_convert.py --split training   --start 0 --end 200 --out data/train
```

**2. Train the final model.** Run from this folder:

```bash
cd lstm
python train_lstm_arrays.py --train ../data/train --val ../data/val --out runs/lstm_res_h256_big \
    --epochs 100 --batch_size 1024 --residual --hidden 256
```

**3. Evaluate on the test set (once, after tuning):**

```bash
python evaluate_test_arrays.py --test ../data/test --ckpt runs/lstm_res_h256_big/best.pt --out results/final
```

This writes `summary.csv`, `paired_differences.csv`, `per_example_errors.npz`, `pred_lstm.npy` (shape N x 30 x 2) and the figures. The `.npz` and `.npy` files are not tracked by git.

**Quick check without Waymo data:**

```bash
python make_fake_arrays.py --out fake --n 20000
python train_lstm_arrays.py --train fake/train.npz --val fake/val.npz --out runs/test --epochs 3
```

The fake data only tests that the code runs. Do not use its numbers.

## Limitations

- The LSTM used 200 of the 1,000 training files. More data would probably help, since the gain from 50 to 200 files was about 4% in validation ADE.
- Each configuration was trained with a single random seed, so small differences between tuning runs (for example 0.672 and 0.673) are within noise.
- My rebuilt test set has 12,555 vehicles, while the Transformer test set reported in the parent folder has 12,700. The parked-car sampling is random, and the counts are close but not identical. The LSTM result should be re-scored on the exact same test files before it is compared directly with the Transformer numbers.
- The LSTM predicts a single trajectory. It is comparable with the single-path Transformer, not with the multi-path model scored on minADE@3.
- The test set was used once, after the model had been chosen on validation data.

## Acknowledgements

Data: Waymo Open Motion Dataset. Data conversion and the shared setup: Abdulahi Taiwo. Course: CSCE 883 Machine Learning, Prof. Jianjun Hu.

*Author: Pallavi Aashadapu*
