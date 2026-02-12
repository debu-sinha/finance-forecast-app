"""
Structured input/output logging for the Finance Forecasting Platform.

Provides a @log_io decorator that logs function arguments on entry and
return values on exit at DEBUG level, with smart truncation for large
objects (DataFrames, numpy arrays, long lists/dicts/strings, Pydantic models).

Usage:
    from backend.utils.logging_utils import log_io

    @log_io
    def my_function(x, y):
        return x + y

    @log_io(log_result=False)
    async def fetch_data(url):
        ...
"""
import functools
import inspect
import logging
import time
import traceback
from typing import Any

MAX_STR_LENGTH = 500
MAX_LIST_ITEMS = 5
MAX_DICT_ITEMS = 10
MAX_REPR_LENGTH = 200


class _Lazy:
    """Defers evaluation until __str__ is called by the logging framework.

    Ensures _truncate_value is never invoked when the log message
    would be discarded due to log level filtering.
    """

    __slots__ = ("_fn", "_args")

    def __init__(self, fn, *args):
        self._fn = fn
        self._args = args

    def __str__(self):
        return self._fn(*self._args)


def _truncate_value(value: Any, depth: int = 0) -> str:
    """Smart truncation that preserves useful debugging info without log bloat."""
    if depth > 2:
        return f"<{type(value).__name__}>"

    if value is None:
        return "None"

    # pandas DataFrame
    try:
        import pandas as pd

        if isinstance(value, pd.DataFrame):
            cols = list(value.columns)
            col_preview = cols[:8]
            if len(cols) > 8:
                col_preview.append(f"...+{len(cols) - 8}")
            return f"DataFrame(shape={value.shape}, cols={col_preview})"
        if isinstance(value, pd.Series):
            return f"Series(name={value.name}, len={len(value)}, dtype={value.dtype})"
    except ImportError:
        pass

    # numpy array
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    except ImportError:
        pass

    # Pydantic BaseModel
    try:
        from pydantic import BaseModel

        if isinstance(value, BaseModel):
            d = value.model_dump() if hasattr(value, "model_dump") else value.dict()
            truncated = _truncate_dict(d, depth + 1)
            return f"{type(value).__name__}({truncated})"
    except ImportError:
        pass

    # dict
    if isinstance(value, dict):
        if len(value) > MAX_DICT_ITEMS:
            sample = dict(list(value.items())[:MAX_DICT_ITEMS])
            return f"dict(len={len(value)}, sample={_truncate_dict(sample, depth + 1)})"
        return _truncate_dict(value, depth)

    # list / tuple
    if isinstance(value, (list, tuple)):
        type_name = type(value).__name__
        if len(value) > MAX_LIST_ITEMS:
            sample = [_truncate_value(item, depth + 1) for item in value[:MAX_LIST_ITEMS]]
            return f"{type_name}(len={len(value)}, first_{MAX_LIST_ITEMS}={sample})"
        return f"{type_name}([{', '.join(_truncate_value(item, depth + 1) for item in value)}])"

    # str
    if isinstance(value, str):
        if len(value) > MAX_STR_LENGTH:
            return f"str(len={len(value)}, preview='{value[:MAX_STR_LENGTH]}...')"
        return repr(value)

    # bytes
    if isinstance(value, bytes):
        return f"bytes(len={len(value)})"

    # set / frozenset
    if isinstance(value, (set, frozenset)):
        type_name = type(value).__name__
        return f"{type_name}(len={len(value)})"

    # Everything else: use repr with length cap
    r = repr(value)
    if len(r) > MAX_REPR_LENGTH:
        return r[:MAX_REPR_LENGTH] + "..."
    return r


def _truncate_dict(d: dict, depth: int) -> str:
    items = []
    for k, v in d.items():
        items.append(f"{k}={_truncate_value(v, depth + 1)}")
    return "{" + ", ".join(items) + "}"


def _prepare_logged_params(func, args, kwargs):
    """Prepare args/kwargs for logging, stripping self/cls."""
    try:
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())
    except (ValueError, TypeError):
        param_names = []

    logged_args = []
    start_idx = 0
    if param_names and param_names[0] in ("self", "cls"):
        start_idx = 1

    for i, arg in enumerate(args):
        if i < start_idx:
            continue
        name = param_names[i] if i < len(param_names) else f"arg{i}"
        logged_args.append(f"{name}={_Lazy(_truncate_value, arg)}")

    logged_kwargs = {k: _Lazy(_truncate_value, v) for k, v in kwargs.items()}
    return logged_args, logged_kwargs


def _format_entry(func_qualname, logged_args, logged_kwargs):
    parts = [str(a) for a in logged_args]
    parts.extend(f"{k}={v}" for k, v in logged_kwargs.items())
    return f"[ENTER] {func_qualname}({', '.join(parts)})"


def log_io(fn=None, *, log_args=True, log_result=True, log_level=logging.DEBUG):
    """Decorator that logs function inputs and outputs at DEBUG level.

    Auto-detects sync vs async. Strips self/cls from logged args.
    Uses _Lazy evaluation so truncation never runs when log level filters the message.

    Args:
        log_args: Whether to log function arguments (default True).
        log_result: Whether to log the full return value (default True).
            When False, only the return type is logged.
        log_level: Log level for entry/exit messages (default DEBUG).
    """

    def decorator(func):
        func_logger = logging.getLogger(func.__module__)
        func_qualname = func.__qualname__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                if func_logger.isEnabledFor(log_level):
                    if log_args:
                        logged_args, logged_kwargs = _prepare_logged_params(
                            func, args, kwargs
                        )
                        func_logger.log(
                            log_level,
                            "%s",
                            _Lazy(_format_entry, func_qualname, logged_args, logged_kwargs),
                        )
                    else:
                        func_logger.log(log_level, "[ENTER] %s()", func_qualname)

                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    if func_logger.isEnabledFor(log_level):
                        if log_result:
                            func_logger.log(
                                log_level,
                                "[EXIT]  %s -> %s (%.3fs)",
                                func_qualname,
                                _Lazy(_truncate_value, result),
                                elapsed,
                            )
                        else:
                            func_logger.log(
                                log_level,
                                "[EXIT]  %s -> %s (%.3fs)",
                                func_qualname,
                                type(result).__name__,
                                elapsed,
                            )
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    func_logger.error(
                        "[ERROR] %s raised %s: %s (%.3fs)\n%s",
                        func_qualname,
                        type(e).__name__,
                        str(e)[:500],
                        elapsed,
                        traceback.format_exc()[-1000:],
                    )
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                if func_logger.isEnabledFor(log_level):
                    if log_args:
                        logged_args, logged_kwargs = _prepare_logged_params(
                            func, args, kwargs
                        )
                        func_logger.log(
                            log_level,
                            "%s",
                            _Lazy(_format_entry, func_qualname, logged_args, logged_kwargs),
                        )
                    else:
                        func_logger.log(log_level, "[ENTER] %s()", func_qualname)

                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    if func_logger.isEnabledFor(log_level):
                        if log_result:
                            func_logger.log(
                                log_level,
                                "[EXIT]  %s -> %s (%.3fs)",
                                func_qualname,
                                _Lazy(_truncate_value, result),
                                elapsed,
                            )
                        else:
                            func_logger.log(
                                log_level,
                                "[EXIT]  %s -> %s (%.3fs)",
                                func_qualname,
                                type(result).__name__,
                                elapsed,
                            )
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    func_logger.error(
                        "[ERROR] %s raised %s: %s (%.3fs)\n%s",
                        func_qualname,
                        type(e).__name__,
                        str(e)[:500],
                        elapsed,
                        traceback.format_exc()[-1000:],
                    )
                    raise

            return sync_wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def log_route_io_middleware(app):
    """FastAPI middleware that logs HTTP request/response metadata at DEBUG level."""

    @app.middleware("http")
    async def _log_route_io(request, call_next):
        route_logger = logging.getLogger("backend.routes")
        if not route_logger.isEnabledFor(logging.DEBUG):
            return await call_next(request)

        body_bytes = await request.body()
        body_size = len(body_bytes)
        route_logger.debug(
            "[REQ] %s %s (body=%d bytes, content-type=%s)",
            request.method,
            request.url.path,
            body_size,
            request.headers.get("content-type", "none"),
        )
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        route_logger.debug(
            "[RES] %s %s -> %d (%.3fs)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response
