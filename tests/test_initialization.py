from typing import Type

import numpy
import pytest
import torch
from flax import nnx
from jax import numpy as jnp
from jax import random
from torch import nn

from nnx_torch_compat import torch_initialization
from nnx_torch_compat.initialization.util import _INITIALIZER_REGISTRY


def _compare_modules(
    nnx_module: nnx.Module,
    nnx_params: dict,
    torch_module: nn.Module,
    torch_params: dict,
    *,
    equal: bool,
) -> tuple[dict, dict]:
    nnx_values = {}
    torch_values = {}

    for (nnx_param, nnx_getter), (torch_param, torch_getter) in zip(
        nnx_params.items(), torch_params.items()
    ):
        nnx_getter = nnx_getter or (lambda _, p=nnx_param: getattr(nnx_module, p))
        nnx_value: numpy.ndarray = numpy.asarray(nnx_getter(nnx_module))

        torch_getter = torch_getter or (
            lambda _, p=torch_param: getattr(torch_module, p)
        )
        torch_value = torch_getter(torch_module)
        if isinstance(torch_value, torch.Tensor):
            torch_value = torch_value.detach().numpy()

        nnx_shape = numpy.shape(nnx_value)
        torch_shape = numpy.shape(torch_value)
        assert nnx_shape == torch_shape, (
            f"Incompatible NNX and PyTorch shapes: {nnx_shape} and {torch_shape}."
        )

        if equal:
            assert (nnx_value == torch_value).all(), (
                f"Parameters '{nnx_module.__class__}.{nnx_param}' and "
                f"'{torch_module.__class__}.{torch_param} differ."
            )
        else:
            assert not (nnx_value == torch_value).any(), (
                f"Parameters '{nnx_module.__class__}.{nnx_param}' and "
                f"'{torch_module.__class__}.{torch_param} are the same."
            )

        nnx_values[nnx_param] = nnx_value
        torch_values[torch_param] = torch_value

    return nnx_values, torch_values


@pytest.mark.parametrize(
    "nnx_module_type, nnx_args, nnx_params, torch_module_type, torch_args, torch_params, input_shape, input_dtype, output_shape",
    [
        (
            nnx.Linear,
            {"in_features": 3, "out_features": 4},
            {"bias": None, "kernel": None},
            nn.Linear,
            {"in_features": 3, "out_features": 4},
            {"bias": None, "weight": lambda torch_module: torch_module.weight.T},
            (10, 3),
            jnp.float32,
            (10, 4),
        ),
        (
            nnx.Embed,
            {"num_embeddings": 10, "features": 7},
            {"embedding": None},
            nn.Embedding,
            {"num_embeddings": 10, "embedding_dim": 7},
            {"weight": None},
            (11, 3),
            jnp.int32,
            (11, 3, 7),
        ),
        (
            nnx.LayerNorm,
            {"num_features": 6},
            {"bias": None, "scale": None, "epsilon": None},
            nn.LayerNorm,
            {"normalized_shape": (6,)},
            {"bias": None, "weight": None, "eps": None},
            (12, 6),
            jnp.float32,
            (12, 6),
        ),
    ],
)
def test_initialization(
    nnx_module_type,
    nnx_args,
    nnx_params,
    torch_module_type,
    torch_args,
    torch_params,
    input_shape,
    input_dtype,
    output_shape,
) -> None:
    default_nnx_module = nnx_module_type(**nnx_args, rngs=nnx.Rngs(14))

    torch.manual_seed(19)
    torch_module = torch_module_type(**torch_args)

    # Ensure that initialization is different before patching. We exclude some modules
    # that already have consistent initialization.
    if nnx_module_type not in {nnx.LayerNorm}:
        _compare_modules(
            default_nnx_module, nnx_params, torch_module, torch_params, equal=False
        )

    torch.manual_seed(19)
    with torch_initialization():
        patched_nnx_module = nnx_module_type(**nnx_args, rngs=nnx.Rngs(14))

    # Ensure that initialization is the same after patching.
    _compare_modules(
        patched_nnx_module, nnx_params, torch_module, torch_params, equal=True
    )

    # Verify shapes and types.
    for nnx_param in nnx_params:
        patched_param = getattr(patched_nnx_module, nnx_param)
        assert isinstance(patched_param, (nnx.Param, float)), (
            f"Parameter '{nnx_param}' is not a Flax parameter or number."
        )
        default_param = getattr(default_nnx_module, nnx_param)
        assert numpy.shape(patched_param) == numpy.shape(default_param)

    key = random.key(42)
    if input_dtype == jnp.float32:
        x = random.normal(key, input_shape)
    elif input_dtype == jnp.int32:
        x = random.randint(key, input_shape, 0, nnx_args.get("num_embeddings", 10))
    else:
        raise NotImplementedError(f"Unsupported dtype: {input_dtype}")

    nnx_y = patched_nnx_module(x)
    assert nnx_y.shape == output_shape

    torch_y = torch_module(torch.as_tensor(x))
    numpy.testing.assert_allclose(nnx_y, torch_y.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_explicit_init_takes_precedence() -> None:
    """Test that explicitly provided initializers are not overwritten by patches."""

    def custom_kernel_init(key, shape, dtype):
        return jnp.ones(shape, dtype=dtype) * 42.0

    def custom_bias_init(key, shape, dtype):
        return jnp.ones(shape, dtype=dtype) * -42.0

    with torch_initialization():
        module = nnx.Linear(
            3,
            4,
            kernel_init=custom_kernel_init,
            bias_init=custom_bias_init,
            rngs=nnx.Rngs(0),
        )

    # Verify custom initializers were used, not the patched ones
    assert (numpy.asarray(module.kernel) == 42.0).all()
    assert (numpy.asarray(module.bias) == -42.0).all()


def _discover_unpatched_module_types():
    exclude = {
        nnx.bridge.Module,
        nnx.RNNCellBase,
    }
    module_types = []
    queue = [nnx.Module]
    while queue:
        cls = queue.pop()
        for subcls in cls.__subclasses__():
            if subcls not in _INITIALIZER_REGISTRY:
                if subcls not in exclude:
                    module_types.append(subcls)
            queue.append(subcls)
    return sorted(module_types, key=lambda x: x.__class__.__name__)


@pytest.mark.parametrize("unpatched_module_type", _discover_unpatched_module_types())
def test_unpatched_modules(unpatched_module_type: Type[nnx.Module]) -> None:
    pytest.skip(f"Module '{unpatched_module_type}' is not patched.")
