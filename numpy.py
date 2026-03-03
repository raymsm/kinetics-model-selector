"""Tiny NumPy compatibility shim for offline test environments.

This module intentionally implements only the small subset used by this
repository's tests. It is not a drop-in replacement for NumPy.
"""

from __future__ import annotations

import math
import random as _random
from typing import Iterable, Sequence


class ndarray:
    def __init__(self, data):
        self._data = _to_native(data)

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        return self._data[item]

    @property
    def shape(self):
        if isinstance(self._data, list) and self._data and isinstance(self._data[0], list):
            return (len(self._data), len(self._data[0]))
        if isinstance(self._data, list):
            return (len(self._data),)
        return ()

    @property
    def ndim(self):
        return len(self.shape)

    def tolist(self):
        return _to_native(self._data)

    def __neg__(self):
        return ndarray(_elementwise(self._data, -1.0, lambda a, b: a * b))

    def __add__(self, other):
        return ndarray(_elementwise(self._data, _unwrap(other), lambda a, b: a + b))

    __radd__ = __add__

    def __sub__(self, other):
        return ndarray(_elementwise(self._data, _unwrap(other), lambda a, b: a - b))

    def __rsub__(self, other):
        return ndarray(_elementwise(_unwrap(other), self._data, lambda a, b: a - b))

    def __mul__(self, other):
        return ndarray(_elementwise(self._data, _unwrap(other), lambda a, b: a * b))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return ndarray(_elementwise(self._data, _unwrap(other), lambda a, b: a / b))


def _unwrap(x):
    return x._data if isinstance(x, ndarray) else x


def _to_native(x):
    if isinstance(x, ndarray):
        return _to_native(x._data)
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
        return [_to_native(v) for v in x]
    return x


def _elementwise(a, b, op):
    if isinstance(a, list) and isinstance(b, list):
        return [_elementwise(x, y, op) for x, y in zip(a, b)]
    if isinstance(a, list):
        return [_elementwise(x, b, op) for x in a]
    if isinstance(b, list):
        return [_elementwise(a, y, op) for y in b]
    return op(float(a), float(b))


def array(data, dtype=None):
    return asarray(data, dtype=dtype)


def asarray(data, dtype=None):
    native = _to_native(data)
    if dtype is float:
        native = _elementwise(native, 0.0, lambda a, _: float(a))
    return ndarray(native)


def linspace(start, stop, num):
    if num <= 1:
        return ndarray([float(start)])
    step = (stop - start) / (num - 1)
    return ndarray([float(start + i * step) for i in range(num)])


def polyfit(x, y, deg):
    if deg != 1:
        raise NotImplementedError("Only degree-1 polyfit supported")
    xv = list(_unwrap(x))
    yv = list(_unwrap(y))
    n = len(xv)
    sx = sum(xv)
    sy = sum(yv)
    sxx = sum(v * v for v in xv)
    sxy = sum(a * b for a, b in zip(xv, yv))
    denom = n * sxx - sx * sx
    b = (n * sxy - sx * sy) / denom
    a = (sy - b * sx) / n
    return ndarray([b, a])


def isclose(a, b, rtol=1e-5, atol=1e-8):
    return abs(float(a) - float(b)) <= (atol + rtol * abs(float(b)))


def log(x):
    return ndarray(_elementwise(_unwrap(x), 0.0, lambda a, _: math.log(a))) if isinstance(x, ndarray) or isinstance(x, list) else math.log(x)


def exp(x):
    return ndarray(_elementwise(_unwrap(x), 0.0, lambda a, _: math.exp(a))) if isinstance(x, ndarray) or isinstance(x, list) else math.exp(x)


def abs(x):
    return ndarray(_elementwise(_unwrap(x), 0.0, lambda a, _: math.fabs(a))) if isinstance(x, ndarray) or isinstance(x, list) else math.fabs(x)


def max(x, axis=None):
    data = _unwrap(x)
    if axis is None:
        return builtins_max(_flatten(data))
    if axis == 0:
        cols = len(data[0])
        return ndarray([builtins_max(row[i] for row in data) for i in range(cols)])
    raise NotImplementedError


def mean(x, axis=None):
    data = _unwrap(x)
    if axis is None:
        flat = _flatten(data)
        return sum(flat) / len(flat)
    if axis == 0:
        cols = len(data[0])
        return ndarray([sum(row[i] for row in data) / len(data) for i in range(cols)])
    raise NotImplementedError


def sum(x, axis=None):
    data = _unwrap(x)
    if axis is None:
        return builtins_sum(_flatten(data))
    if axis == 0:
        cols = len(data[0])
        return ndarray([builtins_sum(row[i] for row in data) for i in range(cols)])
    raise NotImplementedError


def var(x, axis=None, ddof=0):
    data = _unwrap(x)
    if axis == 0:
        m = mean(data, axis=0)
        cols = len(data[0])
        out = []
        for i in range(cols):
            vals = [row[i] for row in data]
            mu = m[i]
            out.append(builtins_sum((v - mu) ** 2 for v in vals) / (len(vals) - ddof))
        return ndarray(out)
    flat = _flatten(data)
    mu = builtins_sum(flat) / len(flat)
    return builtins_sum((v - mu) ** 2 for v in flat) / (len(flat) - ddof)


def percentile(x, q, axis=None):
    data = _unwrap(x)
    if axis == 0:
        cols = len(data[0])
        return ndarray([_percentile_1d([row[i] for row in data], q) for i in range(cols)])
    return _percentile_1d(_flatten(data), q)


def _percentile_1d(vals, q):
    s = sorted(float(v) for v in vals)
    if not s:
        raise ValueError("empty data")
    pos = (len(s) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def full_like(a, fill_value, dtype=float):
    data = _unwrap(a)
    return ndarray(_elementwise(data, 0.0, lambda _x, _y: dtype(fill_value)))


def maximum(a, b):
    return ndarray(_elementwise(_unwrap(a), _unwrap(b), lambda x, y: x if x >= y else y))


def _flatten(data):
    if isinstance(data, list):
        if data and isinstance(data[0], list):
            return [v for row in data for v in row]
        return list(data)
    if hasattr(data, "__iter__") and not isinstance(data, (str, bytes, dict)):
        seq = list(data)
        if seq and isinstance(seq[0], list):
            return [v for row in seq for v in row]
        return seq
    return [data]


builtins_sum = __builtins__["sum"] if isinstance(__builtins__, dict) else __builtins__.sum
builtins_max = __builtins__["max"] if isinstance(__builtins__, dict) else __builtins__.max


class _Generator:
    def __init__(self, seed=None):
        self._rng = _random.Random(seed)

    def normal(self, loc=0.0, scale=1.0):
        scale_data = _unwrap(scale)
        if isinstance(scale_data, list):
            return ndarray(_elementwise(scale_data, 0.0, lambda s, _y: self._rng.gauss(loc, s)))
        return self._rng.gauss(loc, scale_data)


class random:
    Generator = _Generator

    @staticmethod
    def default_rng(seed=None):
        return _Generator(seed)
