from typing import Union


def validate_weights(weights: Union[float, list[float]]) -> None:
    """Validate that all control weights in the valid range"""
    if isinstance(weights, list):
        for i in weights:
            if i < -1 or i > 2:
                raise ValueError("Control weights must be within -1 to 2 range")
    else:
        if weights < -1 or weights > 2:
            raise ValueError("Control weights must be within -1 to 2 range")


def validate_begin_end_step(begin_step_percent: float, end_step_percent: float) -> None:
    """Validate that begin_step_percent is less than or equal to end_step_percent"""
    if begin_step_percent > end_step_percent:
        raise ValueError("Begin step percent must be less than or equal to end step percent")
