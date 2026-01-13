import math

import torch
from flax import nnx
from torch.nn import init as torch_init

from .util import register_initializer


def _linear_kernel_factory(**module_kwargs) -> nnx.Initializer:
    """Factory for Linear kernel initialization matching PyTorch.

    PyTorch stores weights as (out_features, in_features), while NNX uses
    (in_features, out_features). We initialize with PyTorch's shape then transpose.
    """

    def _initializer(key, shape, dtype):
        # shape is (in_features, out_features), but PyTorch uses transposed shape
        transposed_shape = (shape[1], shape[0])
        x = torch.empty(transposed_shape)
        torch_init.kaiming_uniform_(x, a=math.sqrt(5))
        return x.T

    return _initializer


def _linear_bias_factory(in_features: int, **module_kwargs) -> nnx.Initializer:
    """Factory for Linear bias initialization matching PyTorch."""

    def _initializer(key, shape, dtype):
        x = torch.empty(shape)
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0
        torch_init.uniform_(x, -bound, bound)
        return x

    return _initializer


register_initializer(
    nnx.Linear,
    {"bias_init": _linear_bias_factory, "kernel_init": _linear_kernel_factory},
)
