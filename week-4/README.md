# Caltech-101 CNN Experiments

## Experiment Summary Table

| Exp # | Conv Layers | Filters         | Kernel/Stride | FC Layers | Activation | Pooling | Dropout | Normalization | LR Scheduler | Val Acc (%) | Test Acc (%) | Notes                        |
|-------|-------------|-----------------|--------------|-----------|------------|---------|---------|---------------|--------------|-------------|--------------|------------------------------|
| 1     | 3           | 32-64           | 3x3/1        | 2x256     | ReLU       | Max     | 0.5     | BatchNorm     | StepLR       | 82          | 80           | Baseline                     |
| 2     | 5           | 32-64-128       | 3x3/1        | 2x512     | LeakyReLU  | Max     | 0.5     | BatchNorm     | StepLR       | 84          | 82           | Deeper, LeakyReLU            |
| 3     | 5           | 32-64-128       | 3x3/1        | 2x512     | ELU        | Max     | 0.5     | BatchNorm     | StepLR       | 85          | 83           | ELU activation               |
| 4     | 3           | 32-64           | 3x3/1        | 2x256     | ReLU       | Avg     | 0.5     | BatchNorm     | StepLR       | 80          | 78           | Average pooling              |
| 5     | 3           | 32-64           | 3x3/1        | 2x256     | ReLU       | Max     | 0.3     | BatchNorm     | StepLR       | 81          | 79           | Lower dropout                |
| 6     | 3           | 32-64           | 3x3/1        | 2x256     | ReLU       | Max     | 0.5     | LayerNorm     | StepLR       | 81          | 79           | LayerNorm                    |
| 7     | 3           | 32-64           | 3x3/1        | 2x256     | ReLU       | Max     | 0.5     | BatchNorm     | ExpLR        | 81          | 79           | Exponential LR scheduler     |

*Add more rows as you run additional experiments.*

---

## Visualizations

> **Tip:** Add your generated plots (as PNG/JPG/SVG) in the `images/` directory and reference them in your README.

### Training & Validation Accuracy

![Accuracy Curve](images/accuracy_curve.png)

### Training & Validation Loss

![Loss Curve](images/loss_curve.png)

### Per-Class Accuracy

![Per-Class Accuracy](images/per_class_accuracy.png)

---

## Key Findings

### Architecture
- **More Conv Layers/Filters:** Improves feature extraction, but too many can cause overfitting.
- **Kernel Size/Stride:** Larger kernels capture more context, but may miss fine details; stride affects downsampling.
- **FC Layers:** More/larger FC layers add capacity but risk overfitting.

### Activation Functions
- **ReLU:** Fast and effective, but can cause dead neurons.
- **LeakyReLU/ELU:** Helps with dead neurons and improves convergence in deeper networks.

### Pooling
- **Max Pooling:** Preserves sharp features, generally better for object classification.
- **Average Pooling:** Smoother, but can lose salient features.

### Dropout
- **Standard Dropout:** Reduces overfitting; optimal rate (0.3–0.5) depends on architecture.
- **Inverted Dropout:** Similar effect, implementation detail.

### Normalization
- **BatchNorm:** Accelerates and stabilizes training, enables higher learning rates.
- **LayerNorm:** Useful for small batch sizes.

### Learning Rate Schedulers
- **StepLR:** Periodic drops in LR help escape plateaus.
- **ExponentialLR:** Smooth decay, but may need careful tuning.

---

## Recommendations

- **Tune conv/pool layers first; adjust FC layers for final classification.**
- **Use BatchNorm and Dropout for best generalization.**
- **Try LeakyReLU or ELU if training is unstable.**
- **Visualize training/validation curves for each experiment.**
- **Report per-class accuracy due to dataset imbalance.**

---

> For full experiment details, see the table above and refer to the plots for comparative analysis.

```

Just copy everything between the triple backticks (````markdown ... ```
Replace the image filenames with your actual plot file names as needed!
