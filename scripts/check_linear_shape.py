import mlx.core as mx
import mlx.nn as nn

l = nn.Linear(10, 20)
print(l.weight.shape)
