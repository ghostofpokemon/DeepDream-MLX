import mlx.core as mx
import mlx.nn as nn

model = nn.Sequential(nn.Conv2d(3, 64, 3))
mx.eval(model.parameters())
print(model.parameters())
