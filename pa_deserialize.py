from __future__ import annotations

import argparse
import datetime as dt
import json
import struct
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Iterable

import pyarrow as pa

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


_CONTINUATION_MARKER = b"\xff\xff\xff\xff"


def _align_64(size: int) -> int:
    return (size + 63) & ~63


def _find_all(haystack: bytes, needle: bytes) -> Iterable[int]:
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return
        yield idx
        start = idx + 1


def _candidate_offsets(data: bytes) -> list[int]:
    candidates: list[int] = [0]
    if data.startswith(b"\x00" * 16):
        # Legacy pyarrow serialization often prepended 16 null bytes.
        candidates.append(16)

    for idx in _find_all(data, _CONTINUATION_MARKER):
        candidates.append(idx)
        candidates.append(idx + 4)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(off for off in candidates if off < len(data)))


def _tensor_count_from_header(data: bytes) -> int:
    if len(data) < 16:
        return 0
    # Legacy layout: first 8 bytes are unused, next 8 bytes is tensor count.
    return int.from_bytes(data[8:16], "little", signed=False)


def _open_embedded_stream(
    data: bytes,
) -> tuple[pa.ipc.RecordBatchStreamReader, pa.BufferReader, int]:
    last_error: Exception | None = None
    for offset in _candidate_offsets(data):
        try:
            source = pa.BufferReader(data[offset:])
            reader = pa.ipc.open_stream(source)
            _ = reader.schema
            return reader, source, offset
        except (pa.ArrowInvalid, OSError, EOFError, ValueError) as exc:
            last_error = exc

    raise ValueError(
        "Could not locate an Arrow IPC stream in this file."
    ) from last_error


def _extract_payload_and_stream_end(data: bytes) -> tuple[Any, int]:
    reader, source, offset = _open_embedded_stream(data)
    payload: Any | None = None

    while True:
        try:
            batch = reader.read_next_batch()
        except StopIteration:
            break

        if batch is None:
            break

        if payload is None and batch.num_columns > 0 and batch.num_rows > 0:
            # Old pa.serialize writes a single top-level value at row 0, column 0.
            payload = batch.column(0)[0].as_py()

    if payload is None:
        raise ValueError("Stream was readable but did not contain payload rows.")

    stream_end = offset + source.tell()
    return payload, stream_end


def _read_tensor_pool(data: bytes, stream_end: int, expected_count: int) -> list[pa.Tensor]:
    if expected_count <= 0:
        return []

    cursor = _align_64(stream_end)
    tensors: list[pa.Tensor] = []

    for idx in range(expected_count):
        if data[cursor:cursor + 4] != _CONTINUATION_MARKER:
            found = data.find(_CONTINUATION_MARKER, cursor)
            if found == -1:
                raise ValueError(
                    f"Tensor #{idx}: could not find continuation marker after byte {cursor}."
                )
            cursor = found

        try:
            source = pa.BufferReader(data[cursor:])
            tensor = pa.ipc.read_tensor(source)
        except Exception as exc:
            raise ValueError(f"Tensor #{idx}: unable to read tensor payload.") from exc

        consumed = source.tell()
        if consumed <= 0:
            raise ValueError(f"Tensor #{idx}: read zero bytes from tensor payload.")

        tensors.append(tensor)
        cursor += _align_64(consumed)

    return tensors


def _restore_python_shape(value: Any) -> Any:
    if isinstance(value, list):
        # Legacy dict encoding: [{"keys": key, "vals": value}, ...]
        if all(
            isinstance(item, dict) and set(item.keys()) == {"keys", "vals"}
            for item in value
        ):
            return {
                _restore_python_shape(item["keys"]): _restore_python_shape(item["vals"])
                for item in value
            }
        return [_restore_python_shape(item) for item in value]

    if isinstance(value, dict):
        return {key: _restore_python_shape(val) for key, val in value.items()}

    return value


def _is_np_array_stub(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("_pytype_") == "np.array"
        and isinstance(value.get("data"), list)
        and len(value["data"]) == 2
        and isinstance(value["data"][0], int)
    )


def _collect_explicit_tensor_refs(value: Any, refs: set[int]) -> None:
    if _is_np_array_stub(value):
        refs.add(value["data"][0])
        return

    if isinstance(value, dict):
        for item in value.values():
            _collect_explicit_tensor_refs(item, refs)
        return

    if isinstance(value, list):
        for item in value:
            _collect_explicit_tensor_refs(item, refs)


def _reshape(flat: list[Any], shape: tuple[int, ...]) -> Any:
    if not shape:
        return flat[0] if flat else None

    if len(shape) == 1:
        return flat

    step = 1
    for dim in shape[1:]:
        step *= dim

    return [
        _reshape(flat[i:i + step], shape[1:])
        for i in range(0, len(flat), step)
    ]


def _dtype_from_tensor_type(tensor_type: pa.DataType) -> str:
    if pa.types.is_int8(tensor_type):
        return "<i1"
    if pa.types.is_uint8(tensor_type):
        return "<u1"
    if pa.types.is_int16(tensor_type):
        return "<i2"
    if pa.types.is_uint16(tensor_type):
        return "<u2"
    if pa.types.is_int32(tensor_type):
        return "<i4"
    if pa.types.is_uint32(tensor_type):
        return "<u4"
    if pa.types.is_int64(tensor_type):
        return "<i8"
    if pa.types.is_uint64(tensor_type):
        return "<u8"
    if pa.types.is_float32(tensor_type):
        return "<f4"
    if pa.types.is_float64(tensor_type):
        return "<f8"
    raise ValueError(f"Unsupported tensor type: {tensor_type}")


def _decode_utf32_fixed_width(data: bytes, dtype_hint: str) -> list[str]:
    # E.g. dtype '<U4' means 4 UTF-32 codepoints per value.
    width = int(dtype_hint.split("U", 1)[1])
    item_bytes = width * 4
    if item_bytes <= 0:
        return []

    out: list[str] = []
    for idx in range(0, len(data), item_bytes):
        raw = data[idx:idx + item_bytes]
        out.append(raw.decode("utf-32-le").rstrip("\x00"))
    return out


def _decode_datetime64_us(data: bytes) -> list[str]:
    values = struct.unpack("<" + "q" * (len(data) // 8), data)
    epoch = dt.datetime(1970, 1, 1)
    return [(epoch + dt.timedelta(microseconds=value)).isoformat() for value in values]


def _tensor_to_python_without_numpy(tensor: pa.Tensor, dtype_hint: str | None = None) -> Any:
    raw = memoryview(tensor).tobytes()

    if dtype_hint and dtype_hint.startswith("<U"):
        return _decode_utf32_fixed_width(raw, dtype_hint)

    if dtype_hint == "<M8[us]":
        return _decode_datetime64_us(raw)

    dtype = _dtype_from_tensor_type(tensor.type)

    if dtype == "<u1":
        values = list(raw)
    else:
        format_map = {
            "<i1": "b",
            "<u1": "B",
            "<i2": "h",
            "<u2": "H",
            "<i4": "i",
            "<u4": "I",
            "<i8": "q",
            "<u8": "Q",
            "<f4": "f",
            "<f8": "d",
        }
        fmt = format_map[dtype]
        item_size = struct.calcsize(fmt)
        count = len(raw) // item_size
        values = list(struct.unpack("<" + fmt * count, raw))

    return _reshape(values, tensor.shape)


def _tensor_to_python(tensor: pa.Tensor, dtype_hint: str | None = None) -> Any:
    if np is None:
        return _tensor_to_python_without_numpy(tensor, dtype_hint)

    if dtype_hint is not None:
        array = np.frombuffer(memoryview(tensor), dtype=np.dtype(dtype_hint))
        return array.copy()

    dtype = np.dtype(_dtype_from_tensor_type(tensor.type))
    array = np.frombuffer(memoryview(tensor), dtype=dtype)
    if tensor.shape:
        array = array.reshape(tensor.shape)
    return array.copy()


def _resolve_explicit_np_array_stubs(
    value: Any,
    tensor_value: Callable[[int, str | None], Any],
) -> Any:
    if _is_np_array_stub(value):
        index, dtype_hint = value["data"]
        return tensor_value(index, dtype_hint)

    if isinstance(value, dict):
        return {key: _resolve_explicit_np_array_stubs(val, tensor_value) for key, val in value.items()}

    if isinstance(value, list):
        return [_resolve_explicit_np_array_stubs(item, tensor_value) for item in value]

    return value


def _resolve_integer_tensor_refs(
    value: Any,
    unresolved_refs: set[int],
    tensor_value: Callable[[int, str | None], Any],
) -> Any:
    if isinstance(value, dict):
        return {
            key: _resolve_integer_tensor_refs(val, unresolved_refs, tensor_value)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_integer_tensor_refs(item, unresolved_refs, tensor_value)
            for item in value
        ]

    if isinstance(value, int) and value in unresolved_refs:
        return tensor_value(value, None)

    return value


def deserialize_legacy_pyarrow_bytes(raw: bytes) -> Any:
    payload, stream_end = _extract_payload_and_stream_end(raw)
    obj = _restore_python_shape(payload)

    tensor_count = _tensor_count_from_header(raw)
    if tensor_count <= 0:
        return obj

    tensors = _read_tensor_pool(raw, stream_end, tensor_count)

    cache: dict[tuple[int, str | None], Any] = {}

    def tensor_value(index: int, dtype_hint: str | None) -> Any:
        key = (index, dtype_hint)
        if key not in cache:
            cache[key] = _tensor_to_python(tensors[index], dtype_hint)
        return cache[key]

    explicit_refs: set[int] = set()
    _collect_explicit_tensor_refs(obj, explicit_refs)

    obj = _resolve_explicit_np_array_stubs(obj, tensor_value)

    unresolved_refs = set(range(len(tensors))) - explicit_refs
    if unresolved_refs:
        obj = _resolve_integer_tensor_refs(obj, unresolved_refs, tensor_value)

    return obj


def deserialize_legacy_pyarrow_file(path: str | Path) -> Any:
    data = Path(path).read_bytes()
    return deserialize_legacy_pyarrow_bytes(data)


def _json_ready(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {key: _json_ready(val) for key, val in value.items()}

    if isinstance(value, list):
        return [_json_ready(item) for item in value]

    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deserialize legacy pyarrow.serialize() binary payloads."
    )
    parser.add_argument("path", nargs="?", default="np.bin", help="Path to .bin file")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output when possible (falls back to strings for unknown types).",
    )
    args = parser.parse_args()

    obj = deserialize_legacy_pyarrow_file(args.path)
    if args.json:
        print(json.dumps(_json_ready(obj), indent=2, ensure_ascii=False, default=str))
    else:
        pprint(obj)


if __name__ == "__main__":
    main()
