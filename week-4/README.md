# Caltech101 Image Classifier

## About Caltech101.py
- Dataset is downloaded from Kaggle API (tested on Google Colab).
- Splits the dataset into training, validation, and test sets.
#### Model Architecture
- This model is a deep convolutional neural network designed for classifying Caltech101 images into 102 categories.
- 4 convolutional blocks:  
  -`Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d`, with increasing channels: 64 → 128 → 256 → 512.
- Flattened output passed through:  
  - `Linear(512×5×5 → 4096 → 1024 → 256 → 102)` with `BatchNorm1d`, `LeakyReLU`, and `Dropout` between layers.
 
#### Training Setup
- **Loss Function**: `CrossEntropyLoss`
- **Optimizer**: `Adam` (learning rate = 0.01)
- **Epochs**: Configurable (default: 25)
- **Reproducibility**: Enabled via seeds and deterministic settings
- **Split**: 70% train / 10% val / 20% test

#### Output
- Plots training loss and validation accuracy across epochs
- Prints final test accuracy

#### The Final Plot
![](./Loss&Acc_Plots.jpeg)

> ⚠️ Recommended to run on **Google Colab** for dataset commands to work

## Caltech-101 CNN Model Experiments

| Exp # | Conv Layers | Filters (per layer)                | Kernel / Stride (per layer)          | Fully Connected Layers                          | Activation   | Pooling      | Dropout         | Normalization              | LR Scheduler                   | Val Acc (%) | Test Acc (%) | Notes                                                        |
|:-----:|:-----------:|:----------------------------------:|:------------------------------------:|:-----------------------------------------------:|:------------:|:------------:|:----------------:|:----------------------------:|:-------------------------------:|:-----------:|:------------:|:-------------------------------------------------------------|
| 1     | 5           | 16, 64, 128, 256, 512              | 5/1, 3/1, 5/1, 3/1, 3/1               | 1024 → 256 → 102                                 | ReLU         | Max          | None            | None                       | None                          | -           | 8.5          | Baseline, very low accuracy                                  |
| 2     | 5           | 16, 64, 128, 256, 512              | 5/1, 3/1, 5/1, 3/1, 3/1               | 4096 → 1024 → 256 → 102                          | ReLU         | Max          | None            | None                       | None                          | -           | 8.5          | Larger FC, no improvement                                    |
| 3     | 5           | 16, 64, 128, 256, 512              | 5/1, 3/1, 5/1, 5/1, 3/1               | 4096 → 1024 → 512 → 256 → 102                    | ReLU         | Max          | None            | None                       | None                          | -           | 7.4          | Slightly different conv, still poor                          |
| 4     | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 512 → 256 → 102                    | ReLU         | Max          | None            | None                       | None                          | -           | 7.4          | Larger initial kernel, still poor                            |
| 5     | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 512 → 256 → 102                    | ReLU         | Max          | None            | BatchNorm                  | None                          | 60.6        | 43.9         | BatchNorm added, big improvement                             |
| 6     | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 512 → 256 → 102                    | LeakyReLU    | Max          | None            | BatchNorm                  | None                          | 61.4        | 49.2         | LeakyReLU, higher accuracy                                   |
| 7     | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 512 → 256 → 102                    | LeakyReLU    | Avg/Max      | None            | BatchNorm                  | None                          | 67.9        | 63.7         | AvgPool in some layers                                       |
| 8     | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 512 → 256 → 102                    | LeakyReLU    | Max          | Dropout (varied)| BatchNorm                  | None                          | 68.3        | 66.2         | Dropout added, best accuracy                                 |
| 9     | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 256 → 102                          | LeakyReLU    | Max          | Dropout (varied)| BatchNorm                  | None                          | 68.3        | 66.1         | Reduced FC, similar performance                              |
| 10    | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 256 → 128 → 102                    | LeakyReLU    | Max          | Dropout (varied)| BatchNorm                  | None                          | 67.4        | 65.6         | Extra FC layer, stable                                       |
| 11    | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 512 → 256 → 102                    | LeakyReLU    | Max          | Dropout (higher)| BatchNorm                  | None                          | 68.3        | 66.2         | Higher dropout, stable                                       |
| 12    | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 256 → 128 → 102                    | LeakyReLU    | Max          | Dropout (varied)| BatchNorm                  | None                          | 68.3        | 66.1         | Best result, stable                                          |
| 13    | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 256 → 128 → 102                    | LeakyReLU    | Max          | Dropout (varied)| BatchNorm (input too)      | None                          | 66.0        | 64.4         | Input BatchNorm2d, slightly lower                            |
| 14    | 4           | 64, 128, 256, 512                  | 7/2, 5/1, 5/1, 5/1                    | 4096 → 1024 → 256 → 128 → 102                    | LeakyReLU    | Max          | Dropout (varied)| BatchNorm (input too)      | None                          | 64.3        | 62.6         | Similar as above                                             |
| 15    | 3           | 32, 64, 128                        | 3/1, 3/1, 3/1                         | 256 → 102                                | ReLU         | Max          | None            | None                       | None                          | 58–60       | 57.52        | Baseline, no regularization or normalization                 |
| 16    | 4           | 32, 64, 128, 256                   | 3/1, 3/1, 3/1, 3/1                    | 1024 → 256 → 102                         | LeakyReLU    | Max          | 0.5 (after FC1) | BatchNorm (conv)           | StepLR (5, 0.5)               | 67–68       | 65.40        | Strong generalization with dropout, batch norm, LR scheduling|
| 17    | 4           | 32, 64, 128, 256                   | 3/1, 3/1, 3/1, 3/1                    | 512 → 102                                | ELU          | Avg          | 0.3             | LayerNorm (conv)           | ExponentialLR (0.95)          | 54–56       | 50.90        | Fast-convergence, underperformed due to pooling/norm         |
| 18    | 4           | 32, 64, 128, 256                   | 3/1, 3/1, 3/1, 3/1                    | 1024 → 256 → 102                         | GELU         | Max          | 0.4             | BatchNorm (conv)           | CosineAnnealingLR (T_max=5)   | 62–64       | 60.47        | Modern activation + scheduler, smoother convergence          |

### Experiment Visualizations

####  Exp 15 – Baseline
![Exp 15](./i1.jpeg)

#### 🔹 Exp 16 – LeakyReLU + StepLR + Dropout
![Exp 16](./i2.jpeg)

#### 🔹 Exp 17 – ELU + AvgPool + LayerNorm
![Exp 17](./i3.jpeg)

#### 🔹 Exp 18 – GELU + CosineAnnealingLR
![Exp 18](./i4.jpeg)

