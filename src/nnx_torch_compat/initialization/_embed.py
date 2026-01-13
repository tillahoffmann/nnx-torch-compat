import torch
from flax import nnx
from torch.nn import init as torch_init

from .util import register_initializer


def _embed_embedding_factory(**module_kwargs) -> nnx.Initializer:
    def _initializer(key, shape, dtype):
        x = torch.empty(shape)
        torch_init.normal_(x)
        return x.numpy()

    return _initializer


register_initializer(nnx.Embed, {"embedding_init": _embed_embedding_factory})
