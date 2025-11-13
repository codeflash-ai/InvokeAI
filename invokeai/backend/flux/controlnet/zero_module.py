from typing import TypeVar

import torch

T = TypeVar("T", bound=torch.nn.Module)


def zero_module(module: T) -> T:
    """Initialize the parameters of a module to zero."""
    # Vectorized zero-ing of parameters for efficiency
    params = list(module.parameters())
    if params:
        with torch.no_grad():
            for p in params:
                p.zero_()
    return module
