from typing import Any


def is_state_dict_likely_flux_redux(state_dict: dict[str | int, Any]) -> bool:
    """Checks if the provided state dict is likely a FLUX Redux model."""

    # Optimization: Avoid creating unnecessary sets by directly checking key membership and count
    expected_keys = ("redux_down.bias", "redux_down.weight", "redux_up.bias", "redux_up.weight")
    keys = state_dict.keys()
    if len(keys) == 4 and all(key in keys for key in expected_keys):
        return True

    return False
