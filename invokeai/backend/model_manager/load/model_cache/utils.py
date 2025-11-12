import itertools

import torch


def get_effective_device(model: torch.nn.Module) -> torch.device:
    """A utility to infer the 'effective' device of a model.

    This utility handles the case where a model is partially loaded onto the GPU, so is safer than just calling:
    `next(iter(model.parameters())).device`.

    In the worst case, this utility has to check all model parameters, so if you already know the intended model device,
    then it is better to avoid calling this function.
    """
    # Check buffers first, likely smaller, then parameters
    for p in itertools.chain(model.buffers(), model.parameters()):
        if p.device.type != "cpu":
            return p.device

    return torch.device("cpu")
