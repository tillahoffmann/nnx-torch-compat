from flax import nnx

from .util import register_initializer

register_initializer(nnx.LayerNorm, {"epsilon": lambda **_: 1e-5})
