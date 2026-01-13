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
        torch_value: numpy.ndarray = torch_getter(torch_module).detach().numpy()

        if equal:
            assert (nnx_value == torch_value).all(), (
                f"Parameters '{nnx_module}.{nnx_param}' and '{torch_module}.{torch_param} differ."
            )
        else:
            assert not (nnx_value == torch_value).any(), (
                f"Parameters '{nnx_module}.{nnx_param}' and '{torch_module}.{torch_param} are the same."
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

    # Ensure that initialization is different before patching.
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
    for nnx_param in nnx_params:
        assert isinstance(getattr(patched_nnx_module, nnx_param), nnx.Param), (
            f"Parameter '{nnx_param}' is not a Flax parameter."
        )

    key = random.key(42)
    if input_dtype == jnp.float32:
        x = random.normal(key, input_shape)
    else:
        raise NotImplementedError

    y = patched_nnx_module(x)
    assert y.shape == output_shape


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


unpatched_module_types = []
queue = [nnx.Module]
while queue:
    cls = queue.pop()
    for subcls in cls.__subclasses__():
        if subcls not in _INITIALIZER_REGISTRY:
            unpatched_module_types.append(subcls)
        queue.append(subcls)


@pytest.mark.parametrize("unpatched_module_type", unpatched_module_types)
def test_unpatched_modules(unpatched_module_type: Type[nnx.Module]) -> None:
    pytest.skip(f"Module '{unpatched_module_type}' is not patched.")
