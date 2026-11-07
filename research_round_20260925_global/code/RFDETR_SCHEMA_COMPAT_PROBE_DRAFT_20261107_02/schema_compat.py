"""Narrow legacy state-schema completion. Importing uses only the standard library."""
from collections import OrderedDict
from collections.abc import Mapping


def complete_legacy_schema_state(torch, model, original_state):
    """Return a shallow state copy plus one cloned empty schema buffer and evidence.

    The caller performs strict loading and original-tensor equality verification.
    This helper never loads a state into the model or authorizes model execution.
    """
    key = "_kp_active_mask"
    if not isinstance(original_state, Mapping):
        raise ValueError("original_state must be a named tensor mapping")
    expected = model.state_dict()
    missing = set(expected) - set(original_state)
    unexpected = set(original_state) - set(expected)
    if missing != {key} or unexpected:
        raise ValueError("only the missing top-level _kp_active_mask is compatible")
    if getattr(model, "use_grouppose_keypoints", None) is not False:
        raise ValueError("model must explicitly disable GroupPose keypoints")
    schema = getattr(model, "num_keypoints_per_class", None)
    if not isinstance(schema, list) or schema:
        raise ValueError("model keypoint schema must be an empty list")
    if getattr(model, "keypoint_embed", object()) is not None:
        raise ValueError("model must have no keypoint head")
    try:
        buffer = model.get_buffer(key)
    except AttributeError as error:
        raise ValueError("_kp_active_mask must be a registered model buffer") from error
    for tensor in (buffer, expected[key]):
        if not torch.is_tensor(tensor) or isinstance(tensor, torch.nn.Parameter):
            raise ValueError("_kp_active_mask must be a tensor buffer, not a parameter")
        if tensor.device.type != "cpu" or tensor.dtype != torch.bool:
            raise ValueError("_kp_active_mask must be a CPU bool buffer")
        if tuple(tensor.shape) != (0, 0) or tensor.numel() != 0:
            raise ValueError("_kp_active_mask must have exact empty shape (0, 0)")
    for name, tensor in original_state.items():
        expected_tensor = expected[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor) or not torch.is_tensor(expected_tensor):
            raise ValueError("state entries must be named tensors")
        if tensor.device.type != "cpu" or expected_tensor.device.type != "cpu":
            raise ValueError("existing state tensors and the model must be on CPU: " + name)
        if tuple(tensor.shape) != tuple(expected_tensor.shape) or tensor.dtype != expected_tensor.dtype:
            raise ValueError("existing state tensor shape or dtype differs: " + name)
    completed = OrderedDict(original_state.items())
    if hasattr(original_state, "_metadata"):
        completed._metadata = original_state._metadata
    completed[key] = buffer.clone()
    metadata = {
        "schema_completion": "legacy_empty_keypoint_mask_v1",
        "added_keys": [key],
        "original_missing_keys": [key],
        "original_unexpected_keys": [],
        "use_grouppose_keypoints": False,
        "num_keypoints_per_class": [],
        "added_buffer": {"shape": [0, 0], "dtype": str(buffer.dtype),
                         "device": "cpu", "numel": 0, "learned_parameter": False},
        "strict_original_state_load": False,
        "strict_completed_state_load": False,
        "qualified": False,
    }
    return completed, metadata
