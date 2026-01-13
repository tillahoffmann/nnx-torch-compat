# Flax NNX PyTorch Compat

Reproducing machine learning experiments is difficult, in part because different frameworks use different default values and initialization schemes. Flax NNX PyTorch Compat implements a context manager to initialize Flax NNX models in exactly the same way as their corresponding PyTorch model.

```python
from flax import nnx
from nnx_torch_compat import torch_initialization

with torch_initialization():
    model = nnx.Linear(in_features=3, out_features=4)
```
