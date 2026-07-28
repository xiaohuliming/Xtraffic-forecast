from __future__ import annotations

import os
import pickle
import stat
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from numpy.lib import format as npy_format

try:
    from numpy._core.multiarray import _reconstruct as _RECONSTRUCT
    from numpy._core.multiarray import scalar as _SCALAR
except ModuleNotFoundError:
    from numpy.core.multiarray import _reconstruct as _RECONSTRUCT
    from numpy.core.multiarray import scalar as _SCALAR

_MAX_ARRAY_BYTES = 64 * 1024 * 1024
_MAX_DTYPE_ITEMSIZE = 64
_MAX_NDIM = 4
_NUMERIC_DTYPE_KINDS = frozenset("buifcmM")
_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"): np.dtype,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy._core.multiarray", "_reconstruct"): _RECONSTRUCT,
    ("numpy._core.multiarray", "scalar"): _SCALAR,
    ("numpy.core.multiarray", "_reconstruct"): _RECONSTRUCT,
    ("numpy.core.multiarray", "scalar"): _SCALAR,
}


class _RestrictedNumpyUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(
        self,
        handle: BinaryIO,
        *,
        header_shape: tuple[int, ...],
        source_size: int,
    ) -> None:
        super().__init__(handle)
        self._header_shape = header_shape
        self._source_size = source_size
        self._outer_array: np.ndarray | None = None
        self._reconstructed_array_ids: set[int] = set()
        self._built_array_ids: set[int] = set()
        self._numeric_array_bytes = 0

    def find_class(self, module: str, name: str) -> Any:
        allowed = _ALLOWED_PICKLE_GLOBALS.get((module, name))
        if allowed is None:
            raise pickle.UnpicklingError(f"Forbidden pickle global: {module}.{name}")
        return allowed

    def persistent_load(self, pid: Any) -> Any:
        raise pickle.UnpicklingError(
            f"Persistent pickle identifiers are forbidden: {pid!r}"
        )

    @staticmethod
    def _require_simple_dtype(
        dtype: Any,
        *,
        allow_object: bool,
    ) -> np.dtype[Any]:
        if not isinstance(dtype, np.dtype):
            raise pickle.UnpicklingError("ndarray state has an invalid dtype")
        if dtype.fields is not None or dtype.subdtype is not None:
            raise pickle.UnpicklingError("structured and subarray dtypes are forbidden")
        if dtype.itemsize <= 0 or dtype.itemsize > _MAX_DTYPE_ITEMSIZE:
            raise pickle.UnpicklingError("dtype itemsize exceeds safety limit")
        if dtype.hasobject:
            if not allow_object or dtype != np.dtype("object"):
                raise pickle.UnpicklingError("nested object dtypes are forbidden")
        elif dtype.kind not in _NUMERIC_DTYPE_KINDS:
            raise pickle.UnpicklingError(f"dtype kind {dtype.kind!r} is forbidden")
        return dtype

    @staticmethod
    def _require_shape(shape: Any) -> tuple[int, ...]:
        if not isinstance(shape, tuple) or len(shape) > _MAX_NDIM:
            raise pickle.UnpicklingError("ndarray state has an invalid shape")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise pickle.UnpicklingError("ndarray state has an invalid shape")
        return shape

    @staticmethod
    def _element_count(shape: tuple[int, ...]) -> int:
        count = 1
        for dimension in shape:
            count *= dimension
        return count

    def _validate_dtype_reduce(self, args: Any) -> None:
        if (
            not isinstance(args, tuple)
            or not 1 <= len(args) <= 3
            or not isinstance(args[0], str)
            or len(args[0]) > 32
        ):
            raise pickle.UnpicklingError("invalid NumPy dtype reducer")
        if len(args) >= 2 and type(args[1]) is not bool:
            raise pickle.UnpicklingError("invalid NumPy dtype reducer")
        if len(args) == 3 and type(args[2]) is not bool:
            raise pickle.UnpicklingError("invalid NumPy dtype reducer")

    def _validate_scalar_reduce(self, args: Any) -> None:
        if not isinstance(args, tuple) or len(args) != 2:
            raise pickle.UnpicklingError("invalid NumPy scalar reducer")
        dtype = self._require_simple_dtype(args[0], allow_object=False)
        raw = args[1]
        if not isinstance(raw, bytes) or len(raw) != dtype.itemsize:
            raise pickle.UnpicklingError(
                "NumPy scalar payload size does not match its dtype"
            )

    @staticmethod
    def _validate_reconstruct_reduce(args: Any) -> None:
        if (
            not isinstance(args, tuple)
            or len(args) != 3
            or args[0] is not np.ndarray
            or args[1] != (0,)
            or args[2] != b"b"
        ):
            raise pickle.UnpicklingError("invalid NumPy ndarray reconstruction")

    def load_reduce(self) -> None:
        stack = self.stack
        args = stack.pop()
        func = stack[-1]
        if func is np.ndarray:
            raise pickle.UnpicklingError("direct ndarray construction is forbidden")
        if func is _RECONSTRUCT:
            self._validate_reconstruct_reduce(args)
        elif func is np.dtype:
            self._validate_dtype_reduce(args)
        elif func is _SCALAR:
            self._validate_scalar_reduce(args)
        else:
            raise pickle.UnpicklingError("pickle reducer callable is forbidden")

        value = func(*args)
        if func is np.dtype:
            self._require_simple_dtype(value, allow_object=True)
        elif func is _RECONSTRUCT:
            if not isinstance(value, np.ndarray) or value.size != 0:
                raise pickle.UnpicklingError("invalid NumPy ndarray reconstruction")
            self._reconstructed_array_ids.add(id(value))
            if self._outer_array is None:
                self._outer_array = value
        stack[-1] = value

    dispatch[pickle.REDUCE[0]] = load_reduce

    def load_newobj(self) -> None:
        raise pickle.UnpicklingError(
            "direct ndarray construction through NEWOBJ is forbidden"
        )

    dispatch[pickle.NEWOBJ[0]] = load_newobj

    def load_newobj_ex(self) -> None:
        raise pickle.UnpicklingError(
            "direct ndarray construction through NEWOBJ_EX is forbidden"
        )

    dispatch[pickle.NEWOBJ_EX[0]] = load_newobj_ex

    def load_legacy_constructor(self) -> None:
        raise pickle.UnpicklingError("legacy object construction opcodes are forbidden")

    dispatch[pickle.INST[0]] = load_legacy_constructor
    dispatch[pickle.OBJ[0]] = load_legacy_constructor

    def _validate_dtype_state(self, dtype: np.dtype[Any], state: Any) -> None:
        self._require_simple_dtype(dtype, allow_object=True)
        canonical_state = dtype.__reduce__()[2]
        if state != canonical_state:
            raise pickle.UnpicklingError("noncanonical NumPy dtype state is forbidden")

    def _validate_ndarray_state(
        self,
        array: np.ndarray,
        state: Any,
    ) -> None:
        if id(array) not in self._reconstructed_array_ids:
            raise pickle.UnpicklingError(
                "ndarray was not created by the safe reconstruction path"
            )
        if id(array) in self._built_array_ids:
            raise pickle.UnpicklingError("ndarray state was applied twice")
        if (
            not isinstance(state, tuple)
            or len(state) != 5
            or state[0] != 1
            or type(state[3]) is not bool
        ):
            raise pickle.UnpicklingError("invalid ndarray state")

        shape = self._require_shape(state[1])
        dtype = self._require_simple_dtype(
            state[2],
            allow_object=array is self._outer_array,
        )
        element_count = self._element_count(shape)
        raw = state[4]
        if dtype.hasobject:
            if (
                array is not self._outer_array
                or shape != self._header_shape
                or state[3]
                or not isinstance(raw, list)
                or len(raw) != element_count
                or element_count > self._source_size
            ):
                raise pickle.UnpicklingError(
                    "outer object ndarray state does not match its header"
                )
        else:
            array_bytes = element_count * dtype.itemsize
            if (
                array_bytes > _MAX_ARRAY_BYTES
                or self._numeric_array_bytes + array_bytes > self._source_size
            ):
                raise pickle.UnpicklingError("ndarray state exceeds safety limits")
            if not isinstance(raw, bytes) or len(raw) != array_bytes:
                raise pickle.UnpicklingError(
                    "ndarray payload size does not match its state"
                )
            self._numeric_array_bytes += array_bytes
        self._built_array_ids.add(id(array))

    def load_build(self) -> None:
        state = self.stack[-1]
        instance = self.stack[-2]
        if isinstance(instance, np.ndarray):
            self._validate_ndarray_state(instance, state)
        elif isinstance(instance, np.dtype):
            self._validate_dtype_state(instance, state)
        else:
            raise pickle.UnpicklingError("BUILD target type is forbidden")
        super().load_build()

    dispatch[pickle.BUILD[0]] = load_build


def _read_header(handle: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = npy_format.read_magic(handle)
    if version == (1, 0):
        return npy_format.read_array_header_1_0(handle)
    if version in {(2, 0), (3, 0)}:
        return npy_format.read_array_header_2_0(handle)
    raise ValueError(f"Unsupported NPY format version: {version}")


def load_restricted_object_npy_handle(
    handle: BinaryIO,
    *,
    source_size: int | None = None,
) -> np.ndarray:
    if handle.tell() != 0:
        raise ValueError("Object NPY handle must start at byte zero")
    descriptor_stat = os.fstat(handle.fileno())
    if not stat.S_ISREG(descriptor_stat.st_mode):
        raise ValueError("Object NPY handle must refer to a regular file")
    if source_size is None:
        source_size = descriptor_stat.st_size
    elif source_size != descriptor_stat.st_size:
        raise ValueError("Object NPY source size changed before loading")

    shape, fortran_order, dtype = _read_header(handle)
    if dtype != np.dtype("object"):
        raise ValueError(f"Expected an object NPY, got dtype {dtype!s}")
    if fortran_order:
        raise ValueError("The official incident object array must be C-order")
    if len(shape) > _MAX_NDIM or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ValueError("Object NPY header has an invalid shape")
    value = _RestrictedNumpyUnpickler(
        handle,
        header_shape=shape,
        source_size=source_size,
    ).load()
    if handle.read(1) != b"":
        raise ValueError("Object NPY contains trailing data")

    if not isinstance(value, np.ndarray):
        raise ValueError("Object NPY pickle did not produce a NumPy array")
    if value.shape != shape or value.dtype != dtype:
        raise ValueError("Object NPY payload does not match its declared header")
    return value


def load_restricted_object_npy(path: Path) -> np.ndarray:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for object NPY inputs")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(
            f"Object NPY input could not be opened safely: {source}"
        ) from error
    with os.fdopen(descriptor, "rb") as handle:
        return load_restricted_object_npy_handle(handle)
