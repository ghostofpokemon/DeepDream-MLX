import mlx.core as mx
import mlx.nn as nn

l = nn.Conv2d(3, 64, 3)
print(l.weight.shape)
