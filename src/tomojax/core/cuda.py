"""Allocator-free CUDA kernels on JAX-owned buffers and execution streams.

The CUDA driver owns _loaded modules until process exit; JAX owns every array.
NVRTC compiles only on a cache miss, without importing an array library.
Compiled binaries live beside the TomoJAX JAX cache and follow its
``TOMOJAX_JAX_CACHE`` opt-out.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import ctypes as C
from functools import cache
import hashlib
from importlib.util import find_spec
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol, cast

import numpy as np

_OPTIONS = ("--std=c++17",)


class _Buffer(Protocol):
    __cuda_array_interface__: dict[str, Any]


@cache
def _libraries() -> tuple[C.CDLL, C.CDLL]:
    driver = C.CDLL("libcuda.so.1")
    try:
        spec = find_spec("nvidia.cuda_nvrtc")
    except ModuleNotFoundError:
        spec = None
    path = "libnvrtc.so.12"
    if spec is not None:
        path = str(
            Path(cast("list[str]", spec.submodule_search_locations)[0]) / "lib/libnvrtc.so.12"
        )
    nvrtc = C.CDLL(path)
    # ctypes defaults correctly return the integer CUDA status, but pointers need
    # explicit declarations so handles cannot be silently narrowed to 32 bits.
    signatures = {
        "cuStreamGetCtx": (C.c_void_p, C.POINTER(C.c_void_p)),
        "cuCtxPushCurrent_v2": (C.c_void_p,),
        "cuCtxPopCurrent_v2": (C.POINTER(C.c_void_p),),
        "cuCtxGetDevice": (C.POINTER(C.c_int),),
        "cuDeviceGetAttribute": (C.POINTER(C.c_int), C.c_int, C.c_int),
        "cuModuleLoadData": (C.POINTER(C.c_void_p), C.c_void_p),
        "cuModuleGetFunction": (C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p),
        "cuLaunchKernel": (
            C.c_void_p,
            *(C.c_uint,) * 7,
            C.c_void_p,
            C.POINTER(C.c_void_p),
            C.c_void_p,
        ),
        "cuGetErrorString": (C.c_int, C.POINTER(C.c_char_p)),
    }
    for name, args in signatures.items():
        getattr(driver, name).argtypes = args
    signatures = {
        "nvrtcCreateProgram": (
            C.POINTER(C.c_void_p),
            C.c_char_p,
            C.c_char_p,
            C.c_int,
            C.c_void_p,
            C.c_void_p,
        ),
        "nvrtcCompileProgram": (C.c_void_p, C.c_int, C.POINTER(C.c_char_p)),
        "nvrtcGetCUBINSize": (C.c_void_p, C.POINTER(C.c_size_t)),
        "nvrtcGetCUBIN": (C.c_void_p, C.c_void_p),
        "nvrtcGetProgramLogSize": (C.c_void_p, C.POINTER(C.c_size_t)),
        "nvrtcGetProgramLog": (C.c_void_p, C.c_void_p),
        "nvrtcDestroyProgram": (C.POINTER(C.c_void_p),),
        "nvrtcGetErrorString": (C.c_int,),
        "nvrtcVersion": (C.POINTER(C.c_int), C.POINTER(C.c_int)),
    }
    for name, args in signatures.items():
        getattr(nvrtc, name).argtypes = args
    nvrtc.nvrtcGetErrorString.restype = C.c_char_p
    return driver, nvrtc


def _check(code: int) -> None:
    if code:
        msg = C.c_char_p()
        _libraries()[0].cuGetErrorString(code, C.byref(msg))
        raise RuntimeError(f"CUDA {code}: {(msg.value or b'unknown driver error').decode()}")


def _check_nvrtc(code: int) -> None:
    if code:
        raise RuntimeError(_libraries()[1].nvrtcGetErrorString(code).decode())


@contextmanager
def _current(stream: int) -> Iterator[int]:
    driver, _ = _libraries()
    context = C.c_void_p()
    _check(driver.cuStreamGetCtx(int(stream), C.byref(context)))
    _check(driver.cuCtxPushCurrent_v2(context))
    try:
        yield cast("int", context.value)
    finally:
        popped = C.c_void_p()
        _check(driver.cuCtxPopCurrent_v2(C.byref(popped)))


@cache
def _architecture(_context: int) -> str:
    driver, _ = _libraries()
    dev = C.c_int()
    _check(driver.cuCtxGetDevice(C.byref(dev)))
    major, minor = C.c_int(), C.c_int()
    _check(driver.cuDeviceGetAttribute(C.byref(major), 75, dev))
    _check(driver.cuDeviceGetAttribute(C.byref(minor), 76, dev))
    return f"sm_{major.value}{minor.value}"


def _cache_file(source: str, options: tuple[str, ...], architecture: str) -> Path | None:
    from .compilation_cache import default_cache_dir

    if os.environ.get("TOMOJAX_JAX_CACHE", "").strip().lower() in {"0", "off", "false", "no"}:
        return None
    _, nvrtc = _libraries()
    major, minor = C.c_int(), C.c_int()
    _check_nvrtc(nvrtc.nvrtcVersion(C.byref(major), C.byref(minor)))
    key = repr((source, _OPTIONS, options, architecture, major.value, minor.value)).encode()
    return default_cache_dir().parent / "cuda_cache" / (hashlib.sha256(key).hexdigest() + ".cubin")


@cache
def _compiled_binary(source: str, options: tuple[str, ...], architecture: str) -> bytes:
    path = _cache_file(source, options, architecture)
    if path is not None:
        try:
            return path.read_bytes()
        except OSError:
            pass
    binary = _compile_source(source, options, architecture)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                temporary = Path(tmp.name)
                try:
                    tmp.write(binary)
                    tmp.close()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return binary


def _compile_source(source: str, options: tuple[str, ...], architecture: str) -> bytes:
    _, nvrtc = _libraries()
    prog = C.c_void_p()
    _check_nvrtc(
        nvrtc.nvrtcCreateProgram(C.byref(prog), source.encode(), b"tomojax.cu", 0, None, None)
    )
    try:
        opts = (*_OPTIONS, f"--gpu-architecture={architecture}", *options)
        status = nvrtc.nvrtcCompileProgram(
            prog, len(opts), (C.c_char_p * len(opts))(*(o.encode() for o in opts))
        )
        if status:
            n = C.c_size_t()
            _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(prog, C.byref(n)))
            log = C.create_string_buffer(n.value)
            _check_nvrtc(nvrtc.nvrtcGetProgramLog(prog, log))
            raise RuntimeError(log.value.decode())
        n = C.c_size_t()
        _check_nvrtc(nvrtc.nvrtcGetCUBINSize(prog, C.byref(n)))
        binary = C.create_string_buffer(n.value)
        _check_nvrtc(nvrtc.nvrtcGetCUBIN(prog, binary))
        return binary.raw
    finally:
        _check_nvrtc(nvrtc.nvrtcDestroyProgram(C.byref(prog)))


@cache
def _loaded(
    source: str, name: str, options: tuple[str, ...], context: int
) -> tuple[C.c_void_p, C.c_void_p]:
    driver, _ = _libraries()
    binary = _compiled_binary(source, options, _architecture(context))
    module, func = C.c_void_p(), C.c_void_p()
    _check(driver.cuModuleLoadData(C.byref(module), binary))
    _check(driver.cuModuleGetFunction(C.byref(func), module, name.encode()))
    return module, func


def launch_kernel(
    source: str,
    name: str,
    *,
    options: tuple[str, ...] = (),
    stream: int,
    grid: tuple[int, ...],
    block: tuple[int, ...],
    arguments: tuple[_Buffer | np.int32 | np.float32, ...],
) -> None:
    """Launch CUDA C on the context owning an XLA execution stream.

    Parameters
    ----------
    source : str
        Complete CUDA source, including any private headers' contents.
    name : str
        Name of an ``extern "C"`` kernel.
    options : tuple of str, optional
        Additional NVRTC compiler options.
    stream : int
        CUDA stream handle from a JAX buffer callback's context.
    grid, block : tuple of int
        One to three launch dimensions.
    arguments : tuple
        JAX callback buffers, or explicit NumPy float32/int32 scalars.

    Notes
    -----
    The callback must keep input/output buffers alive through execution. The
    launcher allocates no array storage and restores the calling thread's context.
    """
    args = arguments
    with _current(stream) as context:
        _, func = _loaded(source, name, options, context)
        values: list[C.c_int32 | C.c_float | C.c_void_p] = []
        for a in args:
            if isinstance(a, np.signedinteger):
                values.append(C.c_int32(int(a)))
            elif isinstance(a, np.floating):
                values.append(C.c_float(float(a)))
            else:
                values.append(C.c_void_p(a.__cuda_array_interface__["data"][0]))
        params = (C.c_void_p * len(values))(*(C.addressof(v) for v in values))
        grid = (*grid, 1, 1)[:3]
        block = (*block, 1, 1)[:3]
        _check(_libraries()[0].cuLaunchKernel(func, *grid, *block, 0, int(stream), params, None))
