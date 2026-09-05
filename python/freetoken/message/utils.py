from __future__ import annotations

from typing import Any, Dict, Type

import torch


_TYPE_KEY = "__type__"
# Message payloads carry free-form client dicts (tool JSON Schemas, chat_template_kwargs) that
# may legitimately use our tag key as a field name. Wrapping such a dict keeps the decoder from
# reading it as a serialized class -- without this, a request could crash the tokenizer worker.
_RAW_DICT_KEY = "__raw_dict__"


def _serialize_any(value: Any) -> Any:
    if isinstance(value, dict):
        encoded = {k: _serialize_any(v) for k, v in value.items()}
        if _TYPE_KEY in encoded or _RAW_DICT_KEY in encoded:
            return {_RAW_DICT_KEY: encoded}
        return encoded
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value
    else:
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    serialized = {}

    if isinstance(self, torch.Tensor):
        tensor = self.detach().cpu().contiguous()
        serialized["__type__"] = "Tensor"
        # View as bytes instead of routing through NumPy's dtype table: NumPy has
        # no native bfloat16, and multimodal payloads are naturally N-dimensional.
        serialized["buffer"] = tensor.view(torch.uint8).numpy().tobytes()
        serialized["dtype"] = str(tensor.dtype)
        serialized["shape"] = list(tensor.shape)
        return serialized

    # normal type
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    if isinstance(data, dict):
        if len(data) == 1 and _RAW_DICT_KEY in data:
            inner = data[_RAW_DICT_KEY]
            return {k: _deserialize_any(cls_map, v) for k, v in inner.items()}
        if _TYPE_KEY in data:
            return deserialize_type(cls_map, data)
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    type_name = data["__type__"]
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")
        assert isinstance(buffer, bytes)
        dtype = getattr(torch, dtype_str)
        # bytearray owns writable storage, avoiding both NumPy's read-only warning
        # and an extra dtype-specific conversion.
        tensor = torch.frombuffer(bytearray(buffer), dtype=dtype)
        shape = data.get("shape")
        return tensor.reshape(shape) if shape is not None else tensor

    cls = cls_map.get(type_name)
    if cls is None:
        raise ValueError(f"Unknown serialized message type {type_name!r}")
    kwargs = {}
    for k, v in data.items():
        if k == _TYPE_KEY:
            continue
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
