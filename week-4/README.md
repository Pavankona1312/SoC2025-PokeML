## Caltech101 Model Experiments

| Exp # | Conv Layers | Filters & Kernel/Stride                | FC Layers                  | Activation   | Pooling         | Dropout             | Normalization           | Val Acc (%) | Test Acc (%) | Notes                                  |
|-------|-------------|----------------------------------------|----------------------------|--------------|-----------------|---------------------|-------------------------|-------------|--------------|----------------------------------------|
| 1     | 5           | 3-16-64-128-256-512 <br> 5/3/5/3/3, s=1| 1024-256-102               | ReLU         | Max             | None                | None                    | -           | 8.5          | Baseline, very low accuracy            |
| 2     | 5           | 3-16-64-128-256-512 <br> 5/3/5/3/3, s=1| 4096-1024-256-102          | ReLU         | Max             | None                | None                    | -           | 8.5          | Larger FC, no improvement              |
| 3     | 5           | 3-16-64-128-256-512 <br> 5/3/5/5/3, s=1| 4096-1024-512-256-102      | ReLU         | Max             | None                | None                    | -           | 7.4          | Slightly different conv, still poor    |
| 4     | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-512-256-102      | ReLU         | Max             | None                | None                    | -           | 7.4          | Larger initial kernel, still poor      |
| 5     | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-512-256-102      | ReLU         | Max             | None                | BatchNorm               | 60.6        | 43.9         | BatchNorm added, big improvement       |
| 6     | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-512-256-102      | LeakyReLU    | Max             | None                | BatchNorm               | 61.4        | 49.2         | LeakyReLU, higher accuracy             |
| 7     | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-512-256-102      | LeakyReLU    | Avg/Max         | None                | BatchNorm               | 67.9        | 63.7         | AvgPool in some layers                 |
| 8     | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-512-256-102      | LeakyReLU    | Max             | Dropout (varied)    | BatchNorm               | 68.3        | 66.2         | Dropout added, best accuracy           |
| 9     | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-256-102          | LeakyReLU    | Max             | Dropout (varied)    | BatchNorm               | 68.3        | 66.1         | Reduced FC, similar performance        |
| 10    | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-256-128-102      | LeakyReLU    | Max             | Dropout (varied)    | BatchNorm               | 67.4        | 65.6         | Extra FC layer, stable                 |
| 11    | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-512-256-102      | LeakyReLU    | Max             | Dropout (higher)    | BatchNorm               | 68.3        | 66.2         | Higher dropout, stable                 |
| 12    | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-256-128-102      | LeakyReLU    | Max             | Dropout (varied)    | BatchNorm               | 68.3        | 66.1         | Best result, stable                    |
| 13    | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-256-128-102      | LeakyReLU    | Max             | Dropout (varied)    | BatchNorm (input too)    | 66.0        | 64.4         | Input BatchNorm2d, slightly lower      |
| 14    | 4           | 3-64-128-256-512 <br> 7/5/5/5, s=2/1   | 4096-1024-256-128-102      | LeakyReLU    | Max             | Dropout (varied)    | BatchNorm (input too)    | 64.3        | 62.6         | Similar as above                       |

**Notes:**
- "s" in "Kernel/Stride" means stride.
- All models use nn.Flatten() before FC layers.
- Dropout rates and normalization details are as per experiment.
