import contextlib
import functools
import inspect
from typing import Callable, Generator, Type, TypeVar
from unittest.mock import patch

from flax import nnx

InitializerFactory = Callable[..., nnx.Initializer]
_INITIALIZER_REGISTRY: dict[Type[nnx.Module], dict[str, InitializerFactory]] = {}


def register_initializer(
    type: Type[nnx.Module],
    initializer_factories: dict[str, InitializerFactory],
    update: bool = False,
) -> None:
    """Register initializer factories for a module type.

    Args:
        type: Module type to register.
        initializer_factories: Mapping of parameter names to initializer factories.
            Each factory receives the module's constructor kwargs and returns an
            initializer function.
        update: Update initializers if they exist for the module type (raises a
            :class:`ValueError` otherwise).

    Raises:
        ValueError: If an initializer is already registered for the module type and
            :code:`update == False`.
    """
    if type in _INITIALIZER_REGISTRY and not update:
        raise ValueError(f"Initializer is already registered for module type '{type}'.")
    _INITIALIZER_REGISTRY[type] = initializer_factories


M = TypeVar("M", bound=nnx.Module)


def _make_patched_module_init(
    type: Type[M], initializer_factories: dict[str, InitializerFactory]
):
    original_init = type.__init__
    signature = inspect.signature(original_init)

    # Validate initializer names at patch creation time.
    has_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not has_kwargs:
        invalid_names = set(initializer_factories) - set(signature.parameters)
        if invalid_names:
            raise TypeError(
                f"Unexpected arguments for {signature}: {', '.join(sorted(invalid_names))}"
            )

    @functools.wraps(original_init)
    def _patched_init(self, *args, **kwargs):
        bound_args = signature.bind(self, *args, **kwargs)

        # Build kwargs dict from bound arguments with defaults (excluding 'self')
        # because the initializer may depend on default arguments that are not
        # explicitly specified.
        bound_args_with_defaults = signature.bind(self, *args, **kwargs)
        bound_args_with_defaults.apply_defaults()
        module_kwargs = {
            key: value
            for key, value in bound_args_with_defaults.arguments.items()
            if key != "self"
        }

        # Patch initializers if not explicitly provided by caller.
        for name, factory in initializer_factories.items():
            if name not in bound_args.arguments:
                bound_args.arguments[name] = factory(**module_kwargs)

        original_init(*bound_args.args, **bound_args.kwargs)

    return _patched_init


@contextlib.contextmanager
def torch_initialization() -> Generator[None]:
    """Patch default initialization of :class:`flax.nnx.Module`\\s to match PyTorch.
    Explicit initialization always takes precedent.
    """
    stack = contextlib.ExitStack()
    with stack:
        for type, initializers in _INITIALIZER_REGISTRY.items():
            target = f"{type.__module__}.{type.__name__}.__init__"
            stack.enter_context(
                patch(target, _make_patched_module_init(type, initializers))
            )
        yield None
