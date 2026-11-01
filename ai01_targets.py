"""Reconstruction supervision selects Z targets, never encoder input channels."""


DEFAULT_TARGET_Z = (5, 6, 7)  # Zero-based Z05, Z06, Z07; NOT one-based slices 5-7.


def validate_target_z(target_z, num_channels):
    try:
        values = tuple(target_z)
    except TypeError as error:
        raise ValueError("target_z must be a nonempty sequence of integer Z indices") from error
    if not values or any(not isinstance(z, int) or isinstance(z, bool) for z in values):
        raise ValueError("target_z must be a nonempty sequence of integer Z indices")
    if len(set(values)) != len(values):
        raise ValueError("target_z must not contain duplicate indices")
    if any(z < 0 or z >= num_channels for z in values):
        raise ValueError(f"target_z indices must be in [0, {num_channels - 1}]")
    return tuple(sorted(values))


def checkpoint_model_config(saved, num_channels=11):
    """Old checkpoints without target_z supervised ALL 11 Z, never assume 5-7."""
    result = {"image_size": 256, "target_z": list(range(num_channels)), **saved}
    result["target_z"] = list(validate_target_z(result["target_z"], num_channels))
    return result
