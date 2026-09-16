"""
The numeric coordinate: a table of runs and the labels it cannot describe.

A coordinate is a table with one row per *run*, held as one structured numpy
array with the fields ``start`` (the run's first label), ``length`` (its
sample count), ``num`` over ``den`` (its spacing as a fraction of ticks), and
``offset`` (the phase of its first sample on that grid). A tick is one unit
of the coordinate's resolution: a nanosecond for a time, one for an integer.
Sample ``k`` of a grid run is labeled ``start + floor((offset + k * num) /
den)``, so a rate with no whole-tick period -- 1024 Hz, say -- never drifts,
and a strided slice keeps the phase it was cut at. A float coordinate has no
tick, so its offsets are zero and its label is ``start + (k * num) / den``.

Where the labels follow no grid the run is *stored*: ``den`` is zero and its
labels are a slice of the ``labels`` array, which holds the labels of the
stored runs concatenated in table order. Runs partition the samples, so
index space has no holes; a hole is a run which does not start where the run
before it would put its next sample.

A grid of ticks labels its samples exactly. A float grid is *fitted*: labels
built by arithmetic -- numpy's own ``arange(n) * 0.1`` -- land a last bit
either side of the correctly rounded grid they name, and a last bit is not a
change of rate. So a stretch of float labels is read as a grid when every
one of them is within a couple of their own last bits of it, measured in the
coordinate's dtype and at the magnitude the arithmetic was done at, and the
simplest fraction which describes them is the one it keeps. `NumericND.values`
answers with that grid's labels, so a float coordinate gives back what it was
read from to within the same couple of bits rather than bit for bit; a sample
further off than that is no part of the grid and keeps its own label.

Half of `NumericND` is the coordinate kernel a downstream wrapper answers
with rather than anything unidas converts through: `NumericND.select`,
`index_of`, `min`, `max`, `empty`, `partial`, `reversed`, `translated`,
`rescaled`, `with_length`, `run_hashes` and `fingerprint` have no caller in
this package. They are kept, and kept compatible, because DASCore's own
coordinate delegates to them.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache, cached_property
from typing import Any

import numpy as np


class CoordinateError(ValueError):
    """Raised when a coordinate cannot be built, or asked what it was asked."""


class _OutOfRangeError(CoordinateError):
    """A grid whose arithmetic would leave the coordinate's own dtype."""


# Nanoseconds per second: the tick of every exact time grid.
NS_PER_SECOND = 10**9
# A float spacing this close to a whole number of steps is on the grid;
# a float grid such as 0.1 cannot be held exactly, an off-grid label can.
_GRID_RTOL = 1e-6
# The dense-array guard. Stored arrays often carry sub-step jitter
# (GPS-stamped DAS time), so run detection would give roughly one run per
# sample; at or past this many samples, an array whose runs would outnumber
# this fraction of them keeps its values as one array, which is faster to
# build, smaller, and no less exact.
_MIN_RUN_GUARD_SIZE = 1_000
_MAX_RUN_FRACTION = 0.1
# The largest denominator a float step is reduced to; a float spacing which
# is not a short fraction keeps an exact binary one instead.
_FLOAT_DENOMINATOR = 10**12
# Float labels this close together sit on one grid; a fuse test cannot ask
# for equality of values which were never computed the same way. The bound
# is in the labels' own last bits rather than relative to their magnitude,
# so a real gap is never swallowed however large the labels are.
_FLOAT_RTOL = 1e-12
_FLOAT_ULPS = 8
# How far a float label may sit from the grid which describes it: a couple
# of its own last bits. Labels built by arithmetic -- numpy's own
# ``arange(n) * 0.1`` -- land either side of the correctly rounded grid
# they name, and a last bit is not a change of rate. Whole ticks have no
# such slack: a grid of ticks describes its labels exactly or not at all.
_FIT_ULPS = 2
# The denominators a fitted float step is tried against, simplest first: a
# rate read off labels which carry only seven digits states nothing finer
# than those digits, and the tolerance above refuses a fraction too simple
# to describe them.
_FIT_DENOMINATORS = (10**3, 10**6, 10**9, _FLOAT_DENOMINATOR)
_INT64_MAX = 2**63
# The run record's integer fields; ``start`` leads them, as a tick or a float.
_RECORD_TAIL = ("length", "num", "den", "offset")
# Numpy's temporal units, coarsest first, so a unit can be told from a
# nanosecond by where it sits.
_TIME_UNITS = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as")
# Built once; a union spelled inside a test makes a type object per call.
_FLOATS = (float, np.floating)


# ------------------------ small numeric helpers


def _unpack(value):
    """Unpack a zero-dimensional array; leave anything else alone."""
    if isinstance(value, np.ndarray) and not value.ndim:
        return value[()]
    return value


def _fraction_step(step) -> Fraction | None:
    """A step given as a Fraction or (numerator, denominator) tuple, else None."""
    if isinstance(step, tuple):
        return Fraction(*step)
    return step if isinstance(step, Fraction) else None


def _is_null(value) -> bool:
    """Whether a scalar is None or a null (NaN, NaT)."""
    if value is None:
        return True
    if _fraction_step(value) is not None:
        return False
    array = np.asarray(_unpack(value))
    if array.ndim:
        return False
    if array.dtype.kind in "mM":
        return bool(np.isnat(array))
    if array.dtype.kind == "f":
        return bool(np.isnan(array))
    return False


def _is_time(value) -> bool:
    """Whether a value is a numpy date or duration."""
    return isinstance(_unpack(value), np.datetime64 | np.timedelta64)


def _time_like(dtype) -> bool:
    """Whether a dtype holds dates or durations."""
    return np.dtype(dtype).kind in "mM"


def _strictly_monotonic(values) -> bool:
    """Whether every label is on the same side of the one before it."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) < 2:
        return True
    return bool(np.all(values[1:] > values[:-1])) or bool(
        np.all(values[1:] < values[:-1])
    )


def _all_close(first, second) -> bool:
    """Whether two label arrays hold the same values, floats within a rounding."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape:
        return False
    if first.dtype.kind == "f" and second.dtype.kind == "f":
        return bool(np.allclose(first, second, equal_nan=True))
    return bool(np.array_equal(first, second))


def _readonly(values: np.ndarray) -> np.ndarray:
    """A read-only view of an array, so a cached result cannot be written."""
    view = np.asarray(values).view()
    view.flags.writeable = False
    return view


def _diffs(values) -> np.ndarray:
    """Neighbour spacings, signed even for unsigned values."""
    values = np.asarray(values)
    if values.dtype.kind == "u":
        values = values.astype(np.int64)
    return np.diff(values)


def _float_meets(expected, actual, step) -> np.ndarray:
    """
    Whether float labels agree to within a few of their own last bits.

    Never to within a sample, however coarse those bits are: at a large
    enough magnitude one last bit is several steps, and a run which starts
    a sample late is a hole rather than the same point.
    """
    expected, actual = np.asarray(expected), np.asarray(actual)
    bound = np.minimum(_FLOAT_ULPS * np.spacing(np.abs(expected)), np.abs(step) / 2)
    return np.abs(expected - actual) <= bound


@cache
def _largest_spacing(dtype) -> float:
    """The widest gap between two labels of this dtype which is a number."""
    info = np.finfo(np.dtype(dtype))
    return float(np.spacing(np.nextafter(info.max, np.asarray(0, dtype))))


def _spacing(magnitude, dtype) -> np.ndarray:
    """
    The gap between neighbouring labels at each magnitude.

    At the largest label a dtype holds, the next one is an infinity, and
    the gap to it is no measure of anything: the widest gap between two
    numbers stands in, so a tolerance stays a tolerance out there.
    """
    with np.errstate(over="ignore"):  # the gap past the last label is one
        out = np.spacing(np.asarray(magnitude, dtype=np.dtype(dtype)))
    if np.all(np.isfinite(out)):
        return out
    return np.where(np.isfinite(out), out, _largest_spacing(dtype))


def labels_meet(expected, actual, dtype, anchor=0.0, step=None) -> bool:
    """
    Whether a grid's labels are the labels it is meant to describe.

    Measured in the dtype the coordinate holds them in, so a float32 axis
    is judged by float32 bits, and at the magnitude the arithmetic was done
    at rather than at the answer's: a grid stepping across zero lands on a
    label whose own last bit is far finer than the rounding which put it
    there, and the run's own anchor says what that rounding was. Never to
    within a sample, though, however coarse those bits are: past 2**53 one
    last bit is several steps, and a label a sample away from the grid is a
    different sample rather than the same one rounded. Two nulls are one
    label; a null beside a number is not.
    """
    dtype = np.dtype(dtype)
    expected = np.asarray(expected).astype(dtype, copy=False)
    actual = np.asarray(actual).astype(dtype, copy=False)
    magnitude = np.maximum(np.abs(expected), np.abs(actual))
    bound = _FIT_ULPS * _spacing(np.maximum(magnitude, abs(anchor)), dtype)
    if step:
        bound = np.minimum(bound, abs(step) / 2)
    with np.errstate(invalid="ignore"):  # a null is not an invalid operation
        close = (expected == actual) | (np.abs(expected - actual) <= bound)
    null = _isnull_array(expected) & _isnull_array(actual)
    return bool(np.all(close | null))


def _scaled(count, num, den) -> np.ndarray:
    """``count * num / den`` for float labels, in float arithmetic throughout."""
    return np.multiply(count, num, dtype=np.float64) / np.maximum(den, 1)


def grid_at(start, offset, num, den, k, ticked: bool):
    """
    Where sample ``k`` of a grid run lands: its label, and its phase there.

    The one expression a run states its labels by: ``start + (offset + k *
    num) // den`` for ticks, ``start + k * num / den`` for floats. A run
    re-anchored at sample ``k`` starts at that label, carrying the phase
    returned beside it, so slicing and reversing read the same arithmetic
    as labelling does. Stored runs (``den == 0``) have no grid, and answer
    with their own start.
    """
    if not ticked:
        return start + _scaled(k, num, den), 0
    if np.ndim(k) == 0 and np.ndim(num) == 0:
        # One sample is python arithmetic: padding steps a run a long way
        # outside itself, where int64 would wrap and label the far end of
        # the range rather than say the grid does not reach.
        carry, phase = divmod(int(offset) + int(k) * int(num), max(int(den), 1))
        return int(start) + carry, phase
    carry, phase = np.divmod(offset + k * num, np.maximum(den, 1))
    return start + carry, phase


def _stacked_ranges(starts, counts, stride: int) -> np.ndarray:
    """The ranges ``starts[i] + stride * arange(counts[i])``, concatenated."""
    counts = np.asarray(counts, np.int64)
    total = int(counts.sum())
    if not total:
        return np.zeros(0, np.int64)
    within = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    return np.repeat(np.asarray(starts, np.int64), counts) + within * stride


# ------------------------ ticks and dtypes


def to_tick(value) -> int:
    """Return a time value as nanoseconds, or an integer value as itself."""
    value = _unpack(value)
    if _is_time(value):
        # A nanosecond is the tick every time is counted in; anything a
        # nanosecond count cannot state is refused rather than rounded.
        return int(as_ns(np.asarray(value)).view("int64")[()])
    if isinstance(value, _FLOATS):
        if not float(value).is_integer():
            msg = f"An integer coordinate cannot hold the non-integer value {value}."
            raise CoordinateError(msg)
        return int(value)
    try:
        tick = int(value)
    except (TypeError, ValueError) as error:
        msg = f"{value!r} is not an integer or time value."
        raise CoordinateError(msg) from error
    if not -_INT64_MAX <= tick < _INT64_MAX:
        msg = f"{value} lies outside the int64 range a coordinate counts in."
        raise CoordinateError(msg)
    return tick


def _out_of_ns(array: np.ndarray, target: np.dtype):
    """The label a nanosecond count cannot hold, or None."""
    flat = np.ravel(array)
    flat = flat[~np.isnat(flat)]
    if not flat.size:
        return None
    for value in (flat.min(), flat.max()):
        try:
            np.asarray(value).astype(target)
        except (OverflowError, ValueError):
            return value
    return None


def as_ns(values) -> np.ndarray:
    """
    Temporal labels counted in nanoseconds, refusing what one cannot hold.

    A nanosecond is the tick every time is counted in, and int64 spans it
    only from 1678 to 2262; a unit finer than a nanosecond can also name an
    instant between two ticks. Either way the conversion would change the
    label rather than restate it, so it is refused by name.
    """
    array = np.asarray(values)
    kind = "datetime64" if array.dtype.kind == "M" else "timedelta64"
    target = np.dtype(f"{kind}[ns]")
    if array.dtype == target or not array.size:
        return array.astype(target, copy=False)
    unit = np.datetime_data(array.dtype)[0]
    finer = _TIME_UNITS.index(unit) > _TIME_UNITS.index("ns")
    try:
        out = array.astype(target)
    except (OverflowError, ValueError) as error:
        bad = _out_of_ns(array, target)
        msg = f"{bad} lies outside the nanosecond range of a coordinate."
        raise CoordinateError(msg) from error
    # Read back rather than trusted: a label past the nanosecond range wraps
    # silently on some numpy versions, and a finer unit rounds. NaT is never
    # equal to itself, so it is not an off-grid label.
    off = (out.astype(array.dtype) != array) & ~np.isnat(array)
    if np.any(off):
        bad = np.ravel(array[off])[0]
        msg = (
            f"{bad} is not a whole number of nanoseconds."
            if finer
            else f"{bad} lies outside the nanosecond range of a coordinate."
        )
        raise CoordinateError(msg)
    return out


def coord_dtype(dtype) -> np.dtype:
    """
    The dtype a run table counts in; a time is always in nanoseconds.

    Every other numeric dtype is kept as it is, so a float32 or int32
    coordinate stays one however the table stores its record.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "M":
        return np.dtype("datetime64[ns]")
    if dtype.kind == "m":
        return np.dtype("timedelta64[ns]")
    return dtype


def _as_dtype(value) -> np.dtype:
    """The coordinate dtype a scalar or array implies."""
    return coord_dtype(np.asarray(value).dtype)


def _check_numeric(dtype) -> np.dtype:
    """Refuse labels a run table has no arithmetic for."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "iufMm":
        msg = (
            f"A coordinate of {dtype} labels is a Categorical; a "
            "NumericND holds numbers, times, and durations."
        )
        raise CoordinateError(msg)
    return dtype


@cache
def _ticked(dtype) -> bool:
    """Whether labels are whole ticks (times and integers) rather than floats."""
    return np.dtype(dtype).kind in "iuMm"


@cache
def _tick_bounds(dtype) -> tuple[int, int]:
    """
    The lowest and highest tick a coordinate of this dtype can label.

    Every tick is counted in int64, so a uint64 coordinate is bounded by
    int64's ceiling rather than its own: past it the vectorised check reads
    the wrap, and this one reads the bound.
    """
    nd = np.dtype(dtype)
    info = np.iinfo(np.int64 if nd.kind in "mM" else nd)
    return max(int(info.min), -_INT64_MAX), min(int(info.max), _INT64_MAX - 1)


def _check_unsigned(values, dtype) -> None:
    """Refuse an unsigned label the signed tick range cannot hold."""
    if np.dtype(dtype).kind != "u" or not np.size(values):
        return
    top = np.max(values)
    if int(top) >= _INT64_MAX:
        msg = f"{top} lies outside the int64 range a coordinate counts in."
        raise CoordinateError(msg)


def _as_coord_values(values) -> np.ndarray:
    """Labels in the dtype a coordinate holds them in."""
    array = np.asarray(values)
    dtype = _as_dtype(array)
    if _time_like(dtype):
        return as_ns(array)
    return array.astype(dtype, copy=False)


def _as_ticks(values, dtype) -> np.ndarray:
    """Labels as integer ticks (nanoseconds, or the integers themselves)."""
    dtype = np.dtype(dtype)
    if dtype.kind in "mM":
        values = as_ns(values)
    values = np.ascontiguousarray(values).astype(dtype, copy=False)
    # A time is already a count of nanoseconds; a narrower integer has to
    # be widened rather than reinterpreted.
    if dtype.kind in "mM":
        return values.view("int64")
    _check_unsigned(values, dtype)
    return values.astype("int64", copy=False)


def _first_anchor(values: np.ndarray, dtype) -> float | int:
    """The first label of an array, as a tick or a float."""
    if not values.size:
        return 0
    flat = np.ravel(values)[:1]
    return (_as_ticks(flat, dtype) if _ticked(dtype) else flat)[0]


# ------------------------ the run record


@cache
def _record_dtype_cached(start: str) -> np.dtype:
    """One of the two record layouts a run table can have."""
    return np.dtype([("start", start), *[(x, "i8") for x in _RECORD_TAIL]])


def record_dtype(dtype) -> np.dtype:
    """The record of one run; only the start field varies with the coordinate."""
    return _record_dtype_cached("i8" if _ticked(dtype) else "f8")


def _row_count(length) -> int:
    """How many runs a length column states."""
    if isinstance(length, np.ndarray):
        return len(length) if length.ndim else 1
    if isinstance(length, list | tuple):
        return len(length)
    return 1


def rows(dtype, start, length, num, den, offset) -> np.ndarray:
    """A run table from its columns."""
    columns = (start, length, num, den, offset)
    record = record_dtype(dtype)
    out = np.empty(_row_count(length), record)
    for name, column in zip(("start", *_RECORD_TAIL), columns, strict=True):
        out[name] = column
    return out


def runs_from_rows(table, dtype) -> np.ndarray:
    """A run table from plain rows of ``(start, length, num, den, offset)``."""
    return np.asarray([tuple(row) for row in table], record_dtype(dtype))


# ------------------------ steps as fractions of ticks


def float_fraction(step: float) -> Fraction:
    """
    A float spacing as the short fraction it is, else its binary one.

    The short form is kept only when it is the same double and fits int64,
    so a spacing below the shortening limit stays itself rather than
    becoming zero; a binary fraction too fine for int64 keeps the closest
    fraction which is still the same double, and one no fraction of ticks
    can state at all -- a spacing past 2**63, say -- is refused.
    """
    short = Fraction(step).limit_denominator(_FLOAT_DENOMINATOR)
    if (
        float(short) == step
        and max(abs(short.numerator), short.denominator) < _INT64_MAX
    ):
        return short
    exact = Fraction(step)
    if max(abs(exact.numerator), exact.denominator) < _INT64_MAX:
        return exact
    close = exact.limit_denominator(_INT64_MAX // 2)
    if float(close) != step or abs(close.numerator) >= _INT64_MAX:
        msg = f"A step of {step} has no fraction of ticks within int64."
        raise CoordinateError(msg)
    return close


def step_terms(step, dtype) -> tuple[int, int]:
    """
    A step of any spelling as a numerator and denominator of ticks.

    The terms are left as they were given: an origin offset is stated
    against this denominator, so reducing the step alone would move the
    phase. Canonicalising the table reduces the two of them together.
    """
    if _fraction_step(step) is not None:
        if isinstance(step, tuple):
            num, den = int(step[0]), int(step[1])
        else:
            num, den = step.numerator, step.denominator
        if den < 1:
            msg = f"A step denominator must be positive, got {den}."
            raise CoordinateError(msg)
        # A fraction is given in coordinate units: seconds for a time.
        return (num * NS_PER_SECOND, den) if _time_like(dtype) else (num, den)
    step = _unpack(step)
    if not _ticked(dtype) and not _is_time(step):
        fraction = float_fraction(float(step))
        return fraction.numerator, fraction.denominator
    return to_tick(step), 1


def _float_terms(spacings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Each float spacing as a numerator over a denominator of ticks.

    A run whose spacing has no such fraction gets a zero denominator, which
    is the sentinel for a run holding its own labels.
    """
    nums = np.zeros(len(spacings), np.int64)
    dens = np.zeros(len(spacings), np.int64)
    # One fraction per distinct spacing: a detected table is usually one
    # rate, and reducing a fraction is not cheap enough to do per run.
    uniq, inverse = np.unique(spacings, return_inverse=True)
    for index, value in enumerate(uniq):
        try:
            fraction = float_fraction(float(value))
        except CoordinateError:
            continue  # no fraction of ticks: the run keeps its own labels
        here = inverse == index
        nums[here] = fraction.numerator
        dens[here] = fraction.denominator
    return nums, dens


def _declared_step(step, dtype):
    """
    A step declared on array values, as the scalar the values are measured in.

    A fraction may only be whole (labels on a fractional grid are floors of
    ideal positions, so they do not state it) and becomes seconds for time
    and an integer otherwise; a zero or non-finite step is no grid.
    """
    if (fraction := _fraction_step(step)) is not None:
        if fraction.denominator != 1:
            msg = (
                f"A fractional step ({fraction}) cannot be declared on array "
                "values; build the run with from_run instead."
            )
            raise CoordinateError(msg)
        whole = fraction.numerator
        step = (
            np.timedelta64(whole * NS_PER_SECOND, "ns") if _time_like(dtype) else whole
        )
    magnitude = np.abs(np.asarray(_unpack(step)))[()]
    if not magnitude or not np.isfinite(_as_float(magnitude)):
        msg = f"A declared step must be a finite non-zero spacing, got {step}."
        raise CoordinateError(msg)
    return step


def _as_float(value) -> float:
    """A number or duration as a float, so it can be asked if it is finite."""
    value = _unpack(value)
    if _is_time(value):
        return float(np.asarray(value).astype("int64"))
    return float(value)


def _on_grid(deltas, step) -> np.ndarray:
    """
    The whole number of steps in each spacing, raising when one is not whole.

    Ticks (time and integer) must divide exactly; floats within a relative
    tolerance of the step.
    """
    deltas, step = np.asarray(deltas), np.asarray(step)
    if deltas.dtype.kind in "mM":
        deltas, step = deltas.astype("timedelta64[ns]").astype(np.int64), to_tick(step)
    if np.issubdtype(deltas.dtype, np.integer):
        counts, remainder = np.divmod(deltas, step)
        off = remainder != 0
    else:
        counts = np.round(deltas / step)
        off = np.abs(deltas - counts * step) > np.abs(step) * _GRID_RTOL
    if np.any(off):
        msg = f"Values are not on a grid of step {step}: spacing {deltas[off][0]}."
        raise CoordinateError(msg)
    return counts.astype(np.int64)


def _as_step(num: int, den: int, dtype):
    """A run's spacing of ``num`` ticks over ``den`` as the scalar step."""
    if not _ticked(dtype):
        # The quotient a Fraction of these terms would round to, without it.
        return num / den
    tick = round(Fraction(num, den))
    return np.timedelta64(tick, "ns") if _time_like(dtype) else tick


# ------------------------ canonical form


def _tick_range_error(row, dtype):
    """The message a run whose arithmetic leaves its dtype is refused with."""
    msg = (
        f"A grid of {row['length']} samples with step {row['num']}/"
        f"{row['den']} ticks from {row['start']} exceeds the {dtype} range."
    )
    raise _OutOfRangeError(msg)


def _one_row_tick_range(row: np.void, dtype) -> None:
    """
    `check_tick_range` for a single run, in python arithmetic.

    One run is the common table, and the vectorised body below is twenty
    times its cost on one: it is what makes canonicalising a range a
    couple of microseconds rather than twenty. The same float64 sums are
    formed here so that the two agree bit for bit, which
    `test_one_row_check_matches_the_vectorised_one` holds them to.
    """
    start, length, num, den, offset = row.item()
    if den <= 0:  # a stored run has no tick arithmetic to leave
        return
    if not _ticked(dtype):  # floats have no tick to leave
        span = abs(float(start)) + float(length) * float(abs(num)) / float(den)
        if math.isfinite(span):
            return
        _tick_range_error(row, dtype)
    if float(length) * float(abs(num)) + float(den) >= _INT64_MAX:
        _tick_range_error(row, dtype)
    # The span fits, so the last label is exact python arithmetic; the
    # array body reads the same overflow off the int64 wrap instead.
    stop = start + (offset + length * num) // den
    low, high = _tick_bounds(dtype)
    if min(start, stop) < low or max(start, stop) > high:
        _tick_range_error(row, dtype)


def check_tick_range(table: np.ndarray, dtype) -> None:
    """Refuse a table whose tick arithmetic would leave its dtype."""
    if len(table) == 1:
        return _one_row_tick_range(table[0], dtype)
    grid = table["den"] > 0
    length = table["length"][grid].astype(np.float64)
    # Widened before it is made positive: int64's floor has no positive of
    # its own, and taking it there would read as a small step.
    num = np.abs(table["num"][grid].astype(np.float64))
    den = table["den"][grid].astype(np.float64)
    if not _ticked(dtype):  # floats have no tick to leave
        bad = ~np.isfinite(np.abs(table["start"][grid]) + length * num / den)
    else:
        bad = length * num + den >= _INT64_MAX
        # The span fits, so the last label is int64 arithmetic; a start far
        # enough out still wraps, which a sign change gives away.
        delta = (table["offset"] + table["length"] * table["num"]) // np.maximum(
            table["den"], 1
        )
        stop = table["start"] + delta
        wrapped = np.where(delta > 0, stop < table["start"], stop > table["start"])
        bad = bad | wrapped[grid]
        # A narrow integer coordinate holds fewer labels than int64 does.
        nd = np.dtype(dtype)
        info = np.iinfo(np.int64 if nd.kind in "mM" else nd)
        low = np.minimum(table["start"], stop)[grid]
        high = np.maximum(table["start"], stop)[grid]
        bad = bad | (low < info.min) | (high > info.max)
    if np.any(bad):
        _tick_range_error(table[grid][np.argmax(bad)], dtype)


def _continues(table: np.ndarray, dtype) -> np.ndarray:
    """Whether each run begins where the run before would put its next sample."""
    before, after = table[:-1], table[1:]
    # A stored run is nothing but its labels, and two of them already lie
    # end to end in ``labels``; joining them loses nothing and keeps a
    # jittered coordinate one run however it was assembled.
    stored = (before["den"] == 0) & (after["den"] == 0)
    same = (
        (before["num"] == after["num"])
        & (before["den"] == after["den"])
        & (before["den"] > 0)
    )
    ticked = _ticked(dtype)
    # Where the run before would put the sample the next run starts at.
    expected, phase = grid_at(
        before["start"],
        before["offset"],
        before["num"],
        before["den"],
        before["length"],
        ticked,
    )
    if not ticked:
        step = _scaled(1, before["num"], before["den"])
        return stored | (same & _float_meets(expected, after["start"], step))
    meets = (expected == after["start"]) & (phase == after["offset"])
    return stored | (same & meets)


def _fuses(head: np.void, length: int, row: np.void, dtype) -> bool:
    """Whether a fused float run gives back every label of the run it takes."""
    k = np.arange(int(row["length"]), dtype=np.int64)
    fused, _ = grid_at(head["start"], 0, head["num"], head["den"], length + k, False)
    own, _ = grid_at(row["start"], 0, row["num"], row["den"], k, False)
    # Within the same last bit or two a float grid is fitted by: the runs
    # were each read within it, and a fuse which stays inside it describes
    # their labels as well as they did.
    anchor = max(abs(float(head["start"])), abs(float(row["start"])))
    step = float(head["num"]) / max(float(head["den"]), 1.0)
    return labels_meet(fused, own, dtype, anchor, step)


def _float_joins(table: np.ndarray, joins: np.ndarray, dtype) -> np.ndarray:
    """
    Which float runs may fuse without moving a label of the run they take.

    A fused run labels every sample from the first run's start, and a float
    grid stepped that far need not land where the run it swallows starts.
    Each run is asked against the run it would join, so a table whose runs
    reproduce their labels keeps them rather than drifting.
    """
    joins = joins.copy()
    head, length = table[0], int(table["length"][0])
    for index in range(1, len(table)):
        row = table[index]
        if joins[index - 1] and (not head["den"] or _fuses(head, length, row, dtype)):
            length += int(row["length"])
            continue
        joins[index - 1] = False
        head, length = row, int(row["length"])
    return joins


def canonical(table: np.ndarray, labels, dtype) -> tuple[np.ndarray, Any]:
    """
    A run table in the one form its labels can take.

    Empty runs are dropped, each grid is put in lowest terms with its phase,
    the tick arithmetic is checked, and runs which continue their neighbour
    on the same grid are fused, so equal labels give an equal table however
    they were assembled.
    """
    if len(table) != 1:  # one row is already the table it states
        table = table[table["length"] > 0]
    if len(table) == 1:
        # The common case, where the vectorised body below is all overhead:
        # one row's phase and lowest terms are python integer arithmetic.
        start, length, num, den, offset = table[0].item()
        if length > 0:
            table = table.copy()
            row = table[0]
            if den > 0:
                carry, offset = divmod(offset, den)
                common = max(math.gcd(abs(num), den), 1)
                if carry or common != 1:  # already in lowest terms and phase
                    row["start"] = start + carry
                    row["num"] = num // common
                    row["den"] = den // common
                    row["offset"] = offset // common
            check_tick_range(table, dtype)
            return table, labels
        table = table[:0]
    if not len(table):
        # The empty coordinate is a run of no samples, not an empty table,
        # so it still carries its dtype and concatenates away.
        return rows(dtype, 0, [0], 0, 1, 0), None
    table = table.copy()
    grid = table["den"] > 0
    # A phase past its denominator is the next tick's phase.
    carry = np.where(grid, table["offset"] // np.maximum(table["den"], 1), 0)
    table["start"] += carry.astype(table["start"].dtype)
    table["offset"] -= carry * table["den"]
    # Lowest terms: the offset's remainder under gcd(num, den) can never
    # carry a floor over an integer boundary, so dividing it is label
    # preserving and leaves one table per set of labels.
    common = np.where(grid, np.gcd(np.abs(table["num"]), table["den"]), 1)
    common = np.maximum(common, 1)
    table["num"] //= common
    table["den"] //= common
    table["offset"] //= common
    check_tick_range(table, dtype)
    if len(table) > 1:
        joins = _continues(table, dtype)
        if not _ticked(dtype) and np.any(joins):
            # A float grid is not exact arithmetic, so a fuse it cannot
            # reproduce is no description of the labels it swallows.
            joins = _float_joins(table, joins, dtype)
        heads = np.flatnonzero(np.concatenate([[True], ~joins]))
        if len(heads) != len(table):
            lengths = np.add.reduceat(table["length"], heads)
            fused = table[heads].copy()
            fused["length"] = lengths
            # Runs which continue each other are one run, and one run has
            # to fit its dtype: labels whose own grid cannot be stepped
            # through are refused rather than held as an ambiguous table.
            check_tick_range(fused, dtype)
            table = fused
    return table, labels


def _one_grid(table: np.ndarray, dtype, num: int, den: int) -> bool:
    """Whether every run begins on the one grid a spacing of num/den makes."""
    if len(table) < 2:
        return True
    before, after = table[:-1], table[1:]
    if not num:
        return bool(np.all(after["start"] == before["start"]))
    if _ticked(dtype):
        rel = (after["start"] - before["start"]) * den
        rel = rel + after["offset"] - before["offset"]
        return bool(np.all(rel % num == 0))
    steps = (after["start"] - before["start"]) * den / num
    return bool(np.allclose(steps, np.round(steps), rtol=_FLOAT_RTOL, atol=0.0))


def _declares(table: np.ndarray, dtype, step) -> bool:
    """Whether a step declared beside the runs is a grid they all sit on."""
    try:
        num, den = step_terms(step, dtype)
    except CoordinateError:
        return False
    return bool(num) and _one_grid(table, dtype, num, den)


def scalar_step(table: np.ndarray, dtype):
    """
    The whole-tick spacing every run shares, or None.

    A run of one sample states no spacing of its own, so among several runs
    only those holding two or more are asked; a lone run is taken at its
    word. The runs must also meet on the grid that spacing makes, or the
    spacing is not one the coordinate as a whole follows.
    """
    if len(table) == 1:
        # One run states its own spacing, and meets no other run on it.
        _, length, num, den, _ = table[0].item()
        if not length or not den:
            return None
        return _as_step(num, den, dtype)
    if not table["length"].sum() or np.any(table["den"] == 0):
        return None
    spacing = table[table["length"] > 1]
    if not len(spacing):
        return None
    num, den = spacing["num"], spacing["den"]
    if not (np.all(num == num[0]) and np.all(den == den[0])):
        return None
    if not _one_grid(table, dtype, int(num[0]), int(den[0])):
        return None
    return _as_step(int(num[0]), int(den[0]), dtype)


# ------------------------ run detection


def detect_runs(ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Where each run starts, which of them are grids, and how many were read.

    A spacing belongs to a run when a neighbouring spacing matches it. The
    samples left over gather into stored runs. Where one run's spacing gives
    way to another's the two runs meet at a sample, which the earlier of them
    keeps: without that cut a change of rate would read as one grid and
    relabel every sample past it.
    """
    diffs = np.diff(ticks)
    equal = diffs[:-1] == diffs[1:]
    in_run = np.zeros(len(diffs), dtype=bool)
    in_run[1:] |= equal
    in_run[:-1] |= equal
    # A spacing which opens a run of its own where the spacing before it
    # closed one: the sample between them ends the earlier run.
    opens = np.zeros(len(diffs), dtype=bool)
    opens[1:] = in_run[1:] & in_run[:-1] & ~equal
    splits = np.flatnonzero(~in_run | opens) + 1
    block_starts = np.concatenate([[0], splits])
    block_lengths = np.diff(np.concatenate([block_starts, [len(ticks)]]))
    # A block of one sample states no grid; only the runs do.
    on_grid = block_lengths > 1
    heads = np.concatenate([[True], on_grid[1:] | on_grid[:-1]])
    return block_starts[heads], on_grid[heads], len(block_starts)


def guard_declines(sample_count: int, run_count: int) -> bool:
    """Whether a dense array's runs are too many to be worth detecting."""
    dense = sample_count >= _MIN_RUN_GUARD_SIZE
    return dense and run_count > _MAX_RUN_FRACTION * sample_count


def _fit_terms(step: float) -> list[tuple[int, int]]:
    """The fractions of units a fitted spacing is tried as, simplest first."""
    out: list[tuple[int, int]] = []
    if not math.isfinite(step):
        return out
    for limit in _FIT_DENOMINATORS:
        short = Fraction(step).limit_denominator(limit)
        terms = (short.numerator, short.denominator)
        if not terms[0] and step:
            # A rate of zero describes a flat stretch and nothing else; a
            # spacing too fine for this denominator is not flat.
            continue
        if terms not in out and max(abs(terms[0]), terms[1]) < _INT64_MAX:
            out.append(terms)
    try:
        exact = float_fraction(step)
    except (ValueError, OverflowError):
        return out  # no fraction of units states this spacing
    if (terms := (exact.numerator, exact.denominator)) not in out:
        out.append(terms)
    return out


def _float_fit(values: np.ndarray, dtype) -> tuple[int, int] | None:
    """
    The fraction of units a stretch of float labels steps by, or None.

    A float grid is fitted rather than read off one of its spacings: the
    span states the rate best, since the rounding in it is shared out over
    every sample where a spacing read from one pair carries all of it. The
    simplest fraction which describes the labels wins, so a rate of a tenth
    read off labels which are a last bit from it is a tenth rather than the
    binary fraction those labels happen to average; where the span
    describes nothing the first spacing is tried beside it, and where
    neither does, these labels follow no grid at all.
    """
    count = len(values)
    if count < 2:
        return None
    first = float(values[0])
    span = (float(values[-1]) - first) / (count - 1)
    k = np.arange(count, dtype=np.int64)
    tried: list[tuple[int, int]] = []
    for terms in _fit_terms(span) + _fit_terms(float(values[1]) - first):
        if terms in tried:
            continue
        tried.append(terms)
        step = terms[0] / terms[1]
        if labels_meet(values[0] + _scaled(k, *terms), values, dtype, first, step):
            return terms
    return None


def _reaches(values, begin: int, first: int, stop: int, terms, dtype) -> bool:
    """Whether the grid a run is on describes the samples after it too."""
    k = np.arange(first - begin, stop - begin, dtype=np.int64)
    expected = values[begin] + _scaled(k, *terms)
    anchor, step = float(values[begin]), terms[0] / terms[1]
    return labels_meet(expected, values[first:stop], dtype, anchor, step)


def _float_table(values: np.ndarray, dtype) -> np.ndarray:
    """
    The runs a stretch of float labels follows.

    One grid where the whole stretch sits on one. Otherwise the spacings
    say where to look: a run is opened on the grid its first stretch fits
    and grows over every stretch that grid still describes, so a rate read
    as a dozen stretches -- its spacings differing in their last bits -- is
    one run, and a gap or a change of rate opens the next. What no grid
    describes keeps its labels.
    """
    terms = _float_fit(values, dtype)
    if terms is not None:
        return rows(dtype, [values[0]], [len(values)], [terms[0]], [terms[1]], 0)
    starts, _on_grid, detected = detect_runs(values)
    if guard_declines(len(values), detected):
        return rows(dtype, [values[0]], [len(values)], 0, 0, 0)
    bounds = np.concatenate([starts, [len(values)]]).astype(np.int64)
    out: list[tuple[int, int, tuple[int, int] | None]] = []
    begin, terms = 0, None
    for index in range(len(starts)):
        first, stop = int(bounds[index]), int(bounds[index + 1])
        if first == begin:  # this stretch opens the run
            terms = _float_fit(values[first:stop], dtype)
            continue
        if terms is not None and _reaches(values, begin, first, stop, terms, dtype):
            continue  # the run's own grid reaches these samples as well
        fresh = _float_fit(values[first:stop], dtype)
        if terms is None and fresh is None:
            continue  # neither states a grid, so the labels stay together
        out.append((begin, first - begin, terms))
        begin, terms = first, fresh
    out.append((begin, len(values) - begin, terms))
    return rows(
        dtype,
        [values[x[0]] for x in out],
        [x[1] for x in out],
        [0 if x[2] is None else x[2][0] for x in out],
        [0 if x[2] is None else x[2][1] for x in out],
        0,
    )


# ------------------------ hashing


# The splitmix64 finalizer's three constants, shared by the array and the
# scalar fold below so the two cannot drift apart.
_MIX_ADD = 0x9E3779B97F4A7C15
_MIX_ONE = 0xBF58476D1CE4E5B9
_MIX_TWO = 0x94D049BB133111EB
_UINT64_MASK = (1 << 64) - 1
# Below this many runs the numpy fold is all overhead: ten calls on
# one-element arrays cost more than the same arithmetic in python ints.
_SMALL_TABLE = 16


def _splitmix(value: np.ndarray) -> np.ndarray:
    """
    The splitmix64 finalizer, over a column of 64 bit words.

    Overflow is the algorithm, so callers wrap this in ``np.errstate``.
    """
    value = value + np.uint64(_MIX_ADD)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(_MIX_ONE)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(_MIX_TWO)
    return value ^ (value >> np.uint64(31))


def _splitmix_int(value: int) -> int:
    """The same finalizer in python integers, word for word."""
    value = (value + _MIX_ADD) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * _MIX_ONE) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _MIX_TWO) & _UINT64_MASK
    return value ^ (value >> 31)


def _blake(payload: bytes) -> int:
    """Eight bytes of a blake2b digest, as an integer."""
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


# ------------------------ the coordinate


@dataclass(frozen=True, eq=False)
class NumericND:
    """
    A numeric coordinate held as a table of runs.

    Build one with `from_run`, `from_array`, `from_rows`, or `concat` rather
    than with the class directly.

    Parameters
    ----------
    runs
        The run table: a structured array with the fields ``start``,
        ``length``, ``num``, ``den``, and ``offset``.
    labels
        The labels of the stored runs (those with ``den == 0``),
        concatenated in table order, or None when there are none. A table
        whose stored run has no labels is a *partial*: a coordinate whose
        shape is known and whose labels are not loaded.
    dtype
        The dtype of the labels; a time is always counted in nanoseconds.
    units
        A portable unit string, or None. Never a quantity object.
    dims
        The array axes the coordinate is attached to; none for a scalar.
    shape
        The shape of the labels.
    attrs
        What a source says about the coordinate beyond its units, carried
        for whichever destination can hold it. Metadata rather than labels,
        so two coordinates which differ only here are equal.
    declared_step
        The grid a source declared its labels sit on, kept only where the
        runs themselves state none -- a stored run of gappy timestamps --
        so that `step` answers the same on both sides of the dense-array
        guard, and a run joined after such labels is measured against the
        grid they were declared on rather than against the spacing they
        happen to keep. It describes the labels rather than stating them,
        so it is no part of equality or of the fingerprint, and a
        coordinate derived from this one does not carry it.

    Examples
    --------
    >>> import numpy as np
    >>> from fractions import Fraction
    >>> from unidas import NumericND
    >>>
    >>> t0 = np.datetime64("2020-01-01", "ns")
    >>> coord = NumericND.from_run(t0, Fraction(1, 1024), 2048)
    >>> assert coord.step == np.timedelta64(976562, "ns")
    >>> assert coord.step_exact == Fraction(1, 1024)
    """

    runs: np.ndarray
    labels: Any = None
    dtype: Any = None
    units: str | None = None
    dims: tuple[str, ...] = ()
    shape: tuple[int, ...] = field(default=None)
    attrs: dict[str, Any] = field(default_factory=dict)
    declared_step: Any = None

    def __post_init__(self):
        """Hold the table in the layout the coordinate's dtype states."""
        set_ = object.__setattr__
        dtype = np.dtype(self.dtype) if self.dtype is not None else None
        if dtype is None:
            dtype = _as_dtype(self.labels) if self.labels is not None else None
        if dtype is None:
            msg = "A coordinate states its dtype, or the labels it reads one from."
            raise CoordinateError(msg)
        set_(self, "dtype", _check_numeric(coord_dtype(dtype)))
        table = np.asarray(self.runs)
        if table.dtype.names is None:
            table = np.asarray(self.runs, record_dtype(self.dtype))
        if table.ndim != 1 or set(table.dtype.names) != {"start", *_RECORD_TAIL}:
            msg = f"A run table needs the fields {('start', *_RECORD_TAIL)}."
            raise CoordinateError(msg)
        set_(self, "runs", table)
        if self.labels is not None:
            set_(self, "labels", np.asarray(self.labels))
        set_(self, "dims", tuple(self.dims))
        set_(self, "attrs", dict(self.attrs))
        if self.units is not None:
            set_(self, "units", str(self.units))
        if self.shape is None:
            set_(self, "shape", (int(table["length"].sum()),))
        set_(self, "shape", tuple(int(x) for x in self.shape))
        if np.any(table["den"] < 0):
            msg = "A run's denominator is positive, or zero for stored labels."
            raise CoordinateError(msg)
        if self.labels is not None:
            stored = int(table["length"][table["den"] == 0].sum())
            if stored != self.labels.size:
                msg = (
                    f"The stored runs hold {stored} samples but "
                    f"{self.labels.size} labels were given."
                )
                raise CoordinateError(msg)
        if self.dims and len(self.dims) != len(self.shape):
            msg = f"A coordinate of shape {self.shape} has no dims {self.dims}."
            raise CoordinateError(msg)
        total = int(table["length"].sum())
        if math.prod(self.shape) != total:
            msg = f"A coordinate of shape {self.shape} holds {total} samples."
            raise CoordinateError(msg)

    # --- construction

    @classmethod
    def _build(
        cls, dtype, table, labels=None, units=None, dims=(), attrs=None, step=None
    ):
        """
        Canonicalise a fresh table and hold it as a coordinate.

        ``step`` is the grid the labels were *declared* to sit on, kept
        only where the runs themselves state none.
        """
        dtype = coord_dtype(dtype)
        # Labels of their own state their own shape, which counting samples
        # would flatten: an N-D run to one axis, and a rank-0 label to a
        # coordinate of one sample it never was.
        nd_shape = None
        if labels is not None and np.ndim(labels) != 1:
            nd_shape = np.shape(labels)
        table, labels = canonical(table, labels, dtype)
        total = int(table["length"].sum())
        shape = nd_shape if nd_shape is not None else (total,)
        declared = None
        if not _is_null(step) and scalar_step(table, dtype) is None:
            declared = step if _declares(table, dtype, step) else None
        return cls(
            runs=table,
            labels=labels,
            dtype=dtype,
            units=units,
            dims=tuple(dims),
            shape=shape,
            attrs=attrs or {},
            declared_step=declared,
        )

    @classmethod
    def from_rows(
        cls,
        runs,
        labels=None,
        dtype=None,
        units=None,
        dims=(),
        attrs=None,
        step=None,
    ):
        """
        Build from a run table.

        Parameters
        ----------
        runs
            A record array of runs, or anything which casts to one.
        labels
            The labels of the stored runs, concatenated in table order.
        dtype
            The coordinate dtype; taken from ``labels`` when not given.
        units
            The units of the labels.
        dims
            The array axes the coordinate is attached to.
        attrs
            What the source says about the coordinate beyond its units.
        step
            The grid a source declares its stored labels sit on, kept only
            where the runs themselves state none.
        """
        if dtype is None:
            if labels is None:
                msg = "A run table states its dtype, or the labels it reads one from."
                raise CoordinateError(msg)
            dtype = _as_dtype(labels)
        dtype = coord_dtype(dtype)
        table = np.asarray(runs)
        if table.dtype.names is None:
            table = np.asarray(runs, record_dtype(dtype))
        if labels is None and np.any((table["den"] == 0) & (table["length"] > 0)):
            # A stored run is nothing but its labels; filling it from its
            # start would silently repeat that one label.
            msg = "A stored run (den == 0) cannot be rebuilt without its labels."
            raise CoordinateError(msg)
        if labels is not None:
            labels = np.asarray(labels)
            # A time is counted in nanoseconds, and one which cannot be is
            # refused rather than truncated into them.
            labels = as_ns(labels) if labels.dtype.kind in "mM" else labels
            labels = labels.astype(dtype)
        return cls._build(dtype, table, labels, units, dims, attrs, step=step)

    @classmethod
    def from_run(
        cls,
        start,
        step,
        shape,
        origin_offset: int = 0,
        units=None,
        dims=(),
        dtype=None,
        attrs=None,
    ):
        """
        Build one evenly sampled run.

        Parameters
        ----------
        start
            The first label.
        step
            The spacing: a number, a duration, a `Fraction`, or a
            ``(numerator, denominator)`` tuple in coordinate units.
        shape
            The sample count.
        origin_offset
            The phase of the first sample, in ``1 / denominator`` ticks.
        units
            The units of the labels.
        dims
            The array axes the coordinate is attached to.
        dtype
            The coordinate dtype; taken from ``start`` when not given. A
            float spacing on integer labels states one the start cannot.
        attrs
            What the source says about the coordinate beyond its units.
        """
        if isinstance(shape, int | np.integer):
            count = int(shape)
        elif isinstance(shape, tuple) and len(shape) == 1:
            count = int(shape[0])
        else:
            count = int(np.prod(shape))
        dtype = _as_dtype(start) if dtype is None else coord_dtype(dtype)
        try:
            num, den = step_terms(step, dtype)
        except CoordinateError:
            if _ticked(dtype):
                raise
            # A subnormal float spacing is no ratio of int64s; the labels
            # themselves are then the only description of the grid.
            labels = np.asarray(start) + np.arange(count) * np.asarray(step)
            return cls.from_array(
                labels.astype(dtype), units=units, dims=dims, detect=False, attrs=attrs
            )
        start = _unpack(start)
        anchor = to_tick(start) if _ticked(dtype) else float(start)
        table = rows(
            dtype,
            [anchor],
            [count],
            [num],
            [den],
            [int(origin_offset) if _ticked(dtype) else 0],
        )
        return cls._build(dtype, table, None, units, dims, attrs)

    @classmethod
    def from_array(
        cls, data, units=None, dims=(), detect: bool = True, step=None, attrs=None
    ):
        """
        Build from labels, keeping the runs they happen to follow.

        Evenly sampled stretches become grid runs and everything else a
        stored run. An array with as many runs as samples -- jittered
        timestamps, say -- is kept whole rather than split into them.

        Parameters
        ----------
        data
            The labels.
        units
            The units of the labels.
        dims
            The array axes the coordinate is attached to.
        detect
            Whether to read runs out of the spacings. Without it the labels
            are kept whole, as one stored run.
        step
            The grid the labels are declared to sit on. Every spacing must
            then be a whole number of steps, and each run of consecutive
            grid positions becomes a run of this step.
        attrs
            What the source says about the coordinate beyond its units.
        """
        values = _as_coord_values(data)
        dtype = _check_numeric(_as_dtype(values))
        if not values.size:
            # No labels state no grid; an N-D emptiness keeps its shape.
            table = rows(dtype, 0, [0], 0, 1, 0)
            nd = values if values.ndim != 1 else None
            return cls._build(dtype, table, nd, units, dims, attrs)
        if step is not None:
            try:
                out = cls._from_declared(
                    values, dtype, step, units, dims, detect, attrs
                )
            except _OutOfRangeError:
                return cls.from_array(
                    values, units=units, dims=dims, detect=False, attrs=attrs
                )
            return cls._exactly(out, values, units, dims, attrs)
        if not detect or values.ndim != 1 or len(values) < 3:
            # Labels which are kept rather than read still have to be labels
            # this coordinate can answer about at all.
            _check_unsigned(values, dtype)
            table = rows(dtype, [_first_anchor(values, dtype)], [values.size], 0, 0, 0)
            return cls._build(dtype, table, values, units, dims, attrs)
        if not _ticked(dtype):
            # A float grid is fitted to its labels within a last bit or
            # two of each of them; whole ticks are read off the spacings
            # themselves, which they state exactly.
            table = _float_table(values, dtype)
        else:
            anchors = _as_ticks(values, dtype)
            starts, on_grid, detected = detect_runs(anchors)
            if guard_declines(len(values), detected):
                starts, on_grid = np.zeros(1, np.int64), np.zeros(1, bool)
            lengths = np.diff(np.concatenate([starts, [len(values)]]))
            follow = np.minimum(starts + 1, len(values) - 1)
            spacings = np.where(on_grid, anchors[follow] - anchors[starts], 0)
            num, den = spacings.astype(np.int64), np.where(on_grid, 1, 0)
            table = rows(dtype, anchors[starts], lengths, num, den, 0)
        stored = values[np.repeat(table["den"] == 0, table["length"])]
        try:
            out = cls._build(
                dtype, table, stored if len(stored) else None, units, dims, attrs
            )
        except _OutOfRangeError:
            # A grid whose arithmetic leaves the dtype -- an int8 axis whose
            # next sample would wrap -- is no description of these labels,
            # however well it fits them.
            return cls.from_array(
                values, units=units, dims=dims, detect=False, attrs=attrs
            )
        return cls._exactly(out, values, units, dims, attrs)

    @classmethod
    def _exactly(cls, out, values, units, dims, attrs):
        """
        The table, or the labels themselves where it does not describe them.

        A grid of ticks labels its samples exactly. A float grid labels
        them within a couple of their last bits -- the rounding of its own
        arithmetic -- and a table which leaves them further than that, such
        as one built on a step a caller declared too loosely or one whose
        runs were fused a bit at a time, is no description of these labels.
        Each run answers for the labels it covers, against its own start:
        what another run is anchored at says nothing about these.
        """
        if _ticked(out.dtype) or _runs_describe(out, values):
            return out
        return cls.from_array(values, units=units, dims=dims, detect=False, attrs=attrs)

    @classmethod
    def _from_declared(cls, values, dtype, step, units, dims, detect=True, attrs=None):
        """Build from labels which are declared to sit on a grid of ``step``."""
        if values.ndim != 1:
            msg = "A declared step needs one-dimensional, monotonic values."
            raise CoordinateError(msg)
        step = _declared_step(step, dtype)
        magnitude = np.abs(np.asarray(_unpack(step)))[()]
        if len(values) > 1 and not _strictly_monotonic(values):
            msg = "A declared step needs one-dimensional, monotonic values."
            raise CoordinateError(msg)
        signed = magnitude if len(values) < 2 or values[-1] > values[0] else -magnitude
        anchors = _as_ticks(values, dtype) if _ticked(dtype) else values
        if len(values) < 2:
            num, den = step_terms(signed, dtype)
            table = rows(dtype, anchors[:1], [len(values)], num, den, 0)
            return cls._build(dtype, table, None, units, dims, attrs)
        counts = _on_grid(_diffs(values), signed)
        splits = np.flatnonzero(counts != 1) + 1
        starts = np.concatenate([[0], splits])
        if not detect or guard_declines(len(values), len(starts)):
            # Too many runs to be worth detecting, or none asked for; the
            # labels are kept as they are, with the grid they declare.
            table = rows(dtype, anchors[:1], [len(values)], 0, 0, 0)
            return cls._build(dtype, table, values, units, dims, attrs, step=signed)
        lengths = np.diff(np.concatenate([starts, [len(values)]]))
        if _ticked(dtype):
            table = rows(dtype, anchors[starts], lengths, to_tick(signed), 1, 0)
        else:
            num, den = step_terms(signed, dtype)
            table = rows(dtype, anchors[starts], lengths, num, den, 0)
        # Runs of a single sample state no spacing of their own, so the
        # grid they were read against travels with them.
        return cls._build(dtype, table, None, units, dims, attrs, step=signed)

    @classmethod
    def partial(cls, shape, dtype="float64", units=None, dims=(), attrs=None):
        """
        A coordinate whose shape is known and whose labels are not loaded.

        Parameters
        ----------
        shape
            The shape the labels will have.
        dtype
            The dtype the labels will have.
        units
            The units of the labels.
        dims
            The array axes the coordinate is attached to.
        """
        shape = (int(shape),) if isinstance(shape, int | np.integer) else tuple(shape)
        dtype = coord_dtype(dtype)
        table = rows(dtype, [0], [math.prod(shape)], 0, 0, 0)
        return cls(
            runs=table,
            labels=None,
            dtype=dtype,
            units=units,
            dims=dims,
            shape=shape,
            attrs=attrs or {},
        )

    # --- the run table

    @property
    def runs_count(self) -> int:
        """How many runs the table holds."""
        return len(self.runs)

    @property
    def size(self) -> int:
        """How many labels the coordinate holds."""
        return math.prod(self.shape)

    @property
    def ndim(self) -> int:
        """How many axes the labels have."""
        return len(self.shape)

    def __len__(self) -> int:
        if self.ndim != 1:
            msg = f"A coordinate of shape {self.shape} has no length."
            raise TypeError(msg)
        return self.shape[0]

    @property
    def partial_labels(self) -> bool:
        """Whether the labels are not loaded (a partial coordinate)."""
        return self.labels is None and bool(np.any(self.runs["den"] == 0))

    @property
    def _ticks(self) -> bool:
        """Whether labels are whole ticks (times and integers) or floats."""
        return _ticked(self.dtype)

    @property
    def _all_stored(self) -> bool:
        """Whether the coordinate is nothing but labels it holds."""
        return self.labels is not None and not np.any(self.runs["den"])

    @cached_property
    def _sample_starts(self) -> np.ndarray:
        """The sample index each run begins at, with the total appended."""
        return _readonly(np.concatenate([[0], np.cumsum(self.runs["length"])]))

    @cached_property
    def _label_starts(self) -> np.ndarray:
        """Where each run's labels begin in `labels`, with the total appended."""
        stored = np.where(self.runs["den"] == 0, self.runs["length"], 0)
        return _readonly(np.concatenate([[0], np.cumsum(stored)]))

    @cached_property
    def _flat_labels(self) -> np.ndarray:
        """The stored labels as one flat array of ticks (or floats)."""
        self._require_labels()
        flat = np.ravel(self.labels)
        return _as_ticks(flat, self.dtype) if self._ticks else flat

    def _require_labels(self) -> None:
        """Refuse an answer which the unloaded labels alone could give."""
        if self.partial_labels:
            msg = "This coordinate's labels are not loaded."
            raise CoordinateError(msg)

    def _from_anchor(self, anchor) -> np.ndarray:
        """Anchors (ticks or floats) as labels of the coordinate dtype."""
        anchor = np.ascontiguousarray(anchor)
        dtype = np.dtype(self.dtype)
        if dtype.kind in "mM":
            return anchor.astype("int64", copy=False).view(dtype)
        return anchor.astype(dtype, copy=False)

    def _dims_for(self, shape) -> tuple[str, ...]:
        """The dims a derived coordinate of this shape keeps."""
        return self.dims if len(self.dims) == len(shape) else ()

    # --- what the grid states

    @property
    def start(self):
        """The first label."""
        return self._from_anchor(self.runs["start"][:1])[0][()]

    @property
    def step(self):
        """The whole-tick spacing every run shares, the declared one, or None."""
        derived = scalar_step(self.runs, self.dtype)
        return self.declared_step if derived is None else derived

    @property
    def step_exact(self) -> Fraction | None:
        """The exact spacing in coordinate units (seconds for time), or None."""
        table = self.runs
        if scalar_step(table, self.dtype) is not None:
            num, den = int(table["num"][0]), int(table["den"][0])
        elif self.declared_step is not None:
            num, den = step_terms(self.declared_step, self.dtype)
        else:
            return None
        fraction = Fraction(num, den)
        return fraction / NS_PER_SECOND if _time_like(self.dtype) else fraction

    @property
    def evenly_sampled(self) -> bool:
        """Whether the labels are one grid run of at least one sample."""
        table = self.runs
        return (
            len(table) == 1 and bool(table["den"][0] > 0) and bool(table["length"][0])
        )

    def _run_labels(self, run, k) -> np.ndarray:
        """The labels of the grid rows ``run`` at their own indices ``k``."""
        table = self.runs
        return grid_at(
            table["start"][run],
            table["offset"][run],
            table["num"][run],
            table["den"][run],
            k,
            self._ticks,
        )[0]

    def labels_at(self, indices) -> np.ndarray:
        """
        The labels at these sample indices.

        Indices outside the coordinate extend the grid of the run they fall
        nearest, which is what padding a range asks for.
        """
        indices = np.asarray(indices, np.int64)
        table = self.runs
        ends = self._sample_starts[1:]
        run = np.clip(
            np.searchsorted(ends, indices.ravel(), side="right"), 0, len(table) - 1
        )
        k = indices.ravel() - self._sample_starts[run]
        out = self._run_labels(run, k)
        stored = table["den"][run] == 0
        if np.any(stored):
            gather = self._label_starts[run[stored]] + k[stored]
            out[stored] = self._flat_labels[gather]
        return self._from_anchor(out).reshape(indices.shape)

    @cached_property
    def values(self) -> np.ndarray:
        """The labels. Cached and shared, so the array is read-only."""
        table = self.runs
        if not self.size and self.ndim != 1:
            # An N-D coordinate emptied along one axis keeps its shape,
            # which a run of no samples cannot state on its own.
            return _readonly(np.empty(self.shape, dtype=self.dtype))
        if self.labels is not None and len(table) == 1:
            return _readonly(self.labels)
        self._require_labels()
        lengths = table["length"]
        total = int(lengths.sum())
        k = np.arange(total, dtype=np.int64) - np.repeat(
            self._sample_starts[:-1], lengths
        )
        den = np.maximum(table["den"], 1)
        if np.all(table["num"] == table["num"][0]) and np.all(den == den[0]):
            # One rate across the table needs no column of its own.
            num, den = int(table["num"][0]), int(den[0])
        else:
            num, den = np.repeat(table["num"], lengths), np.repeat(den, lengths)
        offset = np.repeat(table["offset"], lengths) if self._ticks else 0
        out, _ = grid_at(
            np.repeat(table["start"], lengths), offset, num, den, k, self._ticks
        )
        if self.labels is not None:
            out[np.repeat(table["den"] == 0, lengths)] = self._flat_labels
        return _readonly(self._from_anchor(out))

    # --- order and limits

    @cached_property
    def _run_ends(self) -> np.ndarray:
        """The last label (tick or float) of each run."""
        table = self.runs
        k = table["length"] - 1
        out = self._run_labels(np.arange(len(table)), k)
        stored = table["den"] == 0
        if np.any(stored):
            out[stored] = self._flat_labels[self._label_starts[1:][stored] - 1]
        return out

    @cached_property
    def _direction(self) -> int:
        """1 for increasing labels, -1 for decreasing, 0 for neither."""
        table = self.runs
        if self.ndim != 1 or not self.size or self.partial_labels:
            return 0
        grid = table["den"] > 0
        num = table["num"][grid]
        # A zero-step run is flat, so it goes either way.
        up, down = bool(np.all(num >= 0)), bool(np.all(num <= 0))
        if self.labels is not None and len(self._flat_labels) > 1:
            flat = self._flat_labels
            inner = np.ones(len(flat) - 1, dtype=bool)
            cuts = np.unique(self._label_starts)
            inner[cuts[(cuts > 0) & (cuts < len(flat))] - 1] = False
            try:
                # Compared rather than differenced: two int64 labels a whole
                # range apart difference into a wrap, and say the opposite.
                rising, falling = flat[1:] > flat[:-1], flat[1:] < flat[:-1]
            except TypeError:
                # Labels which cannot be compared say nothing about order.
                return 0
            up &= bool(np.all(rising[inner]))
            down &= bool(np.all(falling[inner]))
        if len(table) > 1:  # runs may overlap, which their boundaries show
            ends, starts = self._run_ends[:-1], table["start"][1:]
            up &= bool(np.all(ends < starts))
            down &= bool(np.all(ends > starts))
        return 1 if up else (-1 if down else 0)

    @property
    def sorted(self) -> bool:
        """Whether every label is greater than the one before."""
        return self._direction == 1

    @property
    def reverse_sorted(self) -> bool:
        """Whether every label is less than the one before."""
        return self._direction == -1

    def min(self):
        """The smallest label."""
        if not self.size:
            return _nullish(self.dtype)
        if self.reverse_sorted:
            return self._from_anchor(self._run_ends[-1:])[0][()]
        # Labels in no order are read out; a missing one is not the
        # smallest label, it is no label at all.
        return self.start if self.sorted else _nanmin(self.values)

    def max(self):
        """The largest label."""
        if not self.size:
            return _nullish(self.dtype)
        if self.reverse_sorted:
            return self.start
        if self.sorted:
            return self._from_anchor(self._run_ends[-1:])[0][()]
        return _nanmax(self.values)

    def _run_spacing(self, index: int):
        """
        The spacing a stored run's own labels keep, in anchor space, or None.

        A stored run states no grid, so what one of its steps looks like is
        only what its labels do -- unless a step was declared beside them,
        which outranks what they happen to keep.
        """
        if (step := self.step) is not None:
            return to_tick(step) if self._ticks else float(step)
        first, last = self._label_starts[index], self._label_starts[index + 1]
        labels = self._flat_labels[first:last]
        if len(labels) < 2:
            return None  # one label states no spacing at all
        # Differenced before it is widened: a nanosecond tick past 2**53 has
        # no float of its own, but the spacings between such ticks do.
        median = float(np.median(np.abs(np.diff(labels))))
        return median if labels[-1] >= labels[0] else -median

    @cached_property
    def _hole_boundaries(self) -> np.ndarray:
        """
        Whether each boundary between runs opens a hole.

        A run which changes rate at the sample the last one stopped at is
        not a hole. A stored run states no next position, so the spacing its
        own labels keep stands in for one; the boundary is a hole where the
        next run starts more than one sample past its last label. Rounding
        is what keeps jitter from reading as a hole: a run whose own
        spacings vary states nothing finer than a sample.
        """
        table = self.runs
        if len(table) < 2:
            return np.zeros(0, dtype=bool)
        before, after = table[:-1], table[1:]
        # Where the run before would put the sample the next run starts at.
        expected, _ = grid_at(
            before["start"],
            before["offset"],
            before["num"],
            before["den"],
            before["length"],
            self._ticks,
        )
        if self._ticks:
            gap = expected != after["start"]
        else:
            step = _scaled(1, before["num"], before["den"])
            gap = ~_float_meets(expected, after["start"], step)
        # A boundary one side of which is not a label at all -- a NaN or a
        # NaT among stored values -- states no distance, so no hole.
        ends = self._run_ends
        known = ~(
            _isnull_array(self._from_anchor(ends[:-1]))
            | _isnull_array(self._from_anchor(after["start"].copy()))
        )
        gap &= known
        for index in np.flatnonzero((before["den"] == 0) & known):
            spacing = self._run_spacing(int(index))
            if not spacing or not np.isfinite(spacing):
                gap[index] = False
                continue
            # differenced in the anchors' own type, which for a tick is an
            # exact integer where its float would have rounded
            delta = float(after["start"][index] - ends[index])
            gap[index] = round(delta / spacing) > 1
        return gap

    @property
    def holes(self) -> bool:
        """Whether any run begins past where the run before it would end."""
        if self.partial_labels:
            return False
        return bool(np.any(self._hole_boundaries))

    # --- slicing

    def __getitem__(self, item):
        if self.ndim != 1:  # a stored N-D run is indexed as its labels are
            out = self.values[item]
            if not np.ndim(out):
                return out[()]
            return self.from_array(
                out,
                units=self.units,
                dims=self._dims_for(np.shape(out)),
                attrs=self.attrs,
            )
        if isinstance(item, int | np.integer):
            if item >= len(self) or item < -len(self):
                raise IndexError(f"{item} exceeds the coordinate's length.")
            index = item + len(self) if item < 0 else item
            return self.labels_at(np.asarray(index))[()]
        if isinstance(item, slice):
            indices = range(len(self))[item]
            if not len(indices):
                return self.empty()
            return self._sliced(indices.start, indices.step, len(indices))
        out = self.values[item]
        if not np.ndim(out):  # one label, not a coordinate of one
            return out[()]
        return self.from_array(
            out, units=self.units, dims=self._dims_for(np.shape(out)), attrs=self.attrs
        )

    def with_length(self, length) -> NumericND:
        """
        The same grid over another number of samples.

        Only a coordinate which is one grid run can be given a length it
        never held: the labels a stored run keeps say nothing about the
        samples beside them.

        Parameters
        ----------
        length
            How many samples the result holds.
        """
        count = int(length)
        if count != length:
            msg = f"A coordinate holds a whole number of samples, not {length}."
            raise CoordinateError(msg)
        if count < 0:
            msg = f"A coordinate cannot hold {count} samples."
            raise CoordinateError(msg)
        if not self.evenly_sampled:
            msg = "Only a single grid run can be given another length."
            raise CoordinateError(msg)
        return self._sliced(0, 1, count)

    def empty(self) -> NumericND:
        """A coordinate of no samples, keeping this one's dtype and units."""
        if self.ndim > 1:
            shape = (0,) * self.ndim
            return self.from_array(
                np.empty(shape, dtype=self.dtype),
                units=self.units,
                dims=self.dims,
                attrs=self.attrs,
            )
        table = rows(self.dtype, 0, [0], 0, 1, 0)
        return self._build(self.dtype, table, None, self.units, self.dims, self.attrs)

    def _sliced(self, first: int, stride: int, count: int) -> NumericND:
        """
        The coordinate of samples ``first, first + stride, ...`` (``count``).

        A single grid run may be sliced past either end, which extends the
        grid; padding a coordinate asks for exactly that.
        """
        if stride < 0:
            last = first + stride * (count - 1)
            return self._sliced(last, -stride, count).reversed()
        table = self.runs
        if len(table) == 1 and table["den"][0] > 0:
            row = table[0]
            num, den = int(row["num"]), int(row["den"])
            start, offset = grid_at(
                row["start"], row["offset"], num, den, first, self._ticks
            )
            new = rows(self.dtype, [start], [count], [num * stride], [den], [offset])
            return self._build(self.dtype, new, None, self.units, self.dims, self.attrs)
        self._require_labels()
        starts = self._sample_starts
        # The first selected sample inside each run, and how many it holds,
        # computed per run rather than per sample.
        base = np.maximum(starts[:-1], first)
        picked = first + -((first - base) // stride) * stride
        limit = np.minimum(starts[1:], first + stride * count)
        counts = np.maximum(-(-(limit - picked) // stride), 0)
        keep = counts > 0
        new = table[keep].copy()
        k = picked[keep] - starts[:-1][keep]
        new["start"], new["offset"] = grid_at(
            new["start"], new["offset"], new["num"], new["den"], k, self._ticks
        )
        new["length"] = counts[keep]
        new["num"] *= stride
        labels = None
        if (stored := new["den"] == 0).any():
            gather = _stacked_ranges(
                self._label_starts[:-1][keep][stored] + k[stored],
                new["length"][stored],
                stride,
            )
            flat = self._flat_labels[gather]
            heads = np.cumsum(new["length"][stored]) - new["length"][stored]
            new["start"][stored] = flat[heads]
            labels = self._from_anchor(flat)
        return self._build(self.dtype, new, labels, self.units, self.dims, self.attrs)

    def reversed(self) -> NumericND:
        """The same samples in the opposite order."""
        table = self.runs
        self._require_labels()
        new = table.copy()
        k = table["length"] - 1
        # Each run is re-anchored at its last sample and stepped backwards.
        new["start"], new["offset"] = grid_at(
            table["start"], table["offset"], table["num"], table["den"], k, self._ticks
        )
        new["num"] = -table["num"]
        labels = None
        if (stored := table["den"] == 0).any():
            order = np.flatnonzero(stored)[::-1]
            gather = _stacked_ranges(
                self._label_starts[1:][order] - 1, table["length"][order], -1
            )
            new["start"][stored] = self._flat_labels[self._label_starts[1:][stored] - 1]
            labels = self._from_anchor(self._flat_labels[gather])
        return self._build(
            self.dtype, new[::-1], labels, self.units, self.dims, self.attrs
        )

    # --- lookup by value

    @cached_property
    def _reverse_view(self) -> NumericND:
        """The coordinate the other way round, for lookup on a sorted table."""
        return self.reversed()

    def _compatible_value(self, value):
        """A query bound in the coordinate's own terms, or None."""
        if value is None:
            return None
        array = np.asarray(value)
        if array.ndim:
            msg = "A bound is one value, not an array of them."
            raise CoordinateError(msg)
        if _is_null(array[()]):
            return None
        if _time_like(self.dtype) != (array.dtype.kind in "mM"):
            msg = f"{value!r} is not a bound a {self.dtype} coordinate can be asked."
            raise CoordinateError(msg)
        return as_ns(array)[()] if array.dtype.kind in "mM" else array[()]

    def _bound_tick(self, value, forward: bool) -> int:
        """The integer tick a query bound is equivalent to."""
        if _is_time(value):
            return to_tick(value)
        value = _unpack(value)
        if isinstance(value, _FLOATS):
            value = math.ceil(value) if forward else math.floor(value)
        return int(min(max(int(value), -_INT64_MAX), _INT64_MAX - 1))

    def _float_index(self, start, length, num, den, value, forward: bool) -> int:
        """The index a bound maps to inside a float grid run."""
        if not num:
            if forward:
                return 0 if value <= start else length
            return length - 1 if value >= start else -1
        # Rounded before it is taken up or down, as a float range has always
        # done it: a bound a rounding away from a label is that label, not
        # the one past it.
        position = round((value - start) * den / num, 10)
        if not math.isfinite(position):
            return length if position > 0 else -1
        k = math.ceil(position) if forward else math.floor(position)
        # Inverting the grid in floats loses what the subtraction cancels,
        # so the estimate can land a sample either side of the label it was
        # taken from; the labels around it settle which one the bound is. A
        # guess a whole sample away from the estimate is no such rounding:
        # a bound past either end keeps the position its grid gives it,
        # outside the coordinate, exactly as the ticked path leaves it.
        bound = abs(num / den) * 1e-10
        for guess in (k - 1, k, k + 1):
            if not 0 <= guess < length or abs(guess - position) >= 1:
                continue
            # Compared as the coordinate holds it: a float32 label is the
            # cast of that position, which is not the position itself.
            label = np.asarray([start + _scaled(guess, num, den)])
            delta = float(self._from_anchor(label)[0]) - value
            if forward and delta >= -bound:
                return guess
            if not forward and delta <= bound:
                k = guess
        return k

    def _index_sorted(self, value, forward: bool) -> int:
        """
        The index a query value maps to, for a table of increasing labels.

        Forward: the first index whose label is at or past the value; else
        the last index whose label is at or before it. A value past either
        end of a grid gets the position that grid would give it, outside the
        coordinate, which is how the caller tells an open bound from a bound
        which merely lands on the last sample.
        """
        table = self.runs
        if isinstance(value, _FLOATS) and not math.isfinite(value):
            return len(self) if value > 0 else -1
        anchor = self._bound_tick(value, forward) if self._ticks else float(value)
        run = int(np.searchsorted(table["start"], anchor, side="right")) - 1
        if run < 0:
            # Before every run. A first run which is a grid still says where
            # the value would sit; stored labels can only say that the value
            # is before the first of them.
            if table["den"][0] == 0:
                return -1
            run = 0
        start, length, num, den, offset = table[run].item()
        if den == 0:
            piece = self._flat_labels[
                self._label_starts[run] : self._label_starts[run] + length
            ]
            side = "left" if forward else "right"
            index = int(np.searchsorted(piece, anchor, side=side))
            k = index if forward else index - 1
            if not forward and index == length and anchor > piece[-1]:
                # Past every label the run holds, which the caller reads as
                # a bound the coordinate does not reach.
                k = length
        elif not self._ticks:
            k = self._float_index(start, length, num, den, value, forward)
        elif not num:  # every label of a flat run is its start
            k = (
                (0 if anchor <= start else length)
                if forward
                else (length - 1 if anchor >= start else -1)
            )
        else:
            rel = (anchor - start) * den - offset
            # label(k) >= tick  <=>  (offset + k num) / den >= tick - start
            # label(k) <= tick  <=>  (offset + k num) / den <  tick - start + 1
            k = -((-rel) // num) if forward else -((-(rel + den)) // num) - 1
        base = 0 if run == 0 else int(self._sample_starts[run])
        k = int(k)
        grid = den != 0
        first, last = run == 0, run == len(table) - 1
        if forward:
            if k >= length:  # past this run
                return base + k if last else int(self._sample_starts[run + 1])
            if k < 0:  # in the space before this run
                return base + k if (first and grid) else base
            return base + k
        if k >= length:
            return base + k if last else base + length - 1
        if k < 0:
            return base + k if (first and grid) else base - 1
        return base + k

    def _index_one(self, value, forward: bool) -> int:
        """The index one value maps to, on a table of either direction."""
        if self.reverse_sorted:
            index = self._reverse_view._index_sorted(value, not forward)
            return len(self) - 1 - index
        return self._index_sorted(value, forward)

    def index_of(self, value, forward: bool = True):
        """
        The index a query value maps to.

        Forward: the first index whose label is at or past the value in the
        coordinate's direction; else the last index whose label is at or
        before it. None when the value is null or lies past the open end.

        Parameters
        ----------
        value
            The label to look for.
        forward
            Whether to take the index at or past the value rather than the
            one at or before it.
        """
        if (value := self._compatible_value(value)) is None:
            return None
        if not (self.sorted or self.reverse_sorted):
            msg = (
                "Lookup by value needs a sorted coordinate; select on an "
                "unsorted one matches its labels instead."
            )
            raise CoordinateError(msg)
        out = self._index_one(value, forward)
        if (forward and out < 0) or (not forward and out >= len(self)):
            return None
        return out

    def _select_by_mask(self, low, high):
        """Select from an unsorted coordinate by matching its labels."""
        values = self.values
        mask = np.ones(values.shape, dtype=bool)
        if low is not None:
            mask &= values >= low
        if high is not None:
            mask &= values <= high
        if not np.any(mask):
            return self.empty(), mask
        if np.all(mask):
            return self, slice(None, None)
        return (
            self.from_array(
                values[mask], units=self.units, dims=self.dims, attrs=self.attrs
            ),
            mask,
        )

    def select(self, low=None, high=None):
        """
        Select a value window; the result keeps only the runs inside it.

        Parameters
        ----------
        low
            The lowest label to keep, or None for the coordinate's own start.
        high
            The highest label to keep, or None for its own end.

        Returns
        -------
        The selected coordinate and the index which selects it from the data.
        """
        if self.ndim != 1:
            msg = "Selection needs a one dimensional coordinate."
            raise CoordinateError(msg)
        self._require_labels()
        low, high = self._compatible_value(low), self._compatible_value(high)
        if low is not None and high is not None and high < low:
            low, high = high, low
        if not self.size:  # nothing to keep, and no order to keep it by
            return self, slice(0, 0)
        if not (self.sorted or self.reverse_sorted):
            return self._select_by_mask(low, high)
        start = self.index_of(low, forward=self.sorted)
        stop = self.index_of(high, forward=self.reverse_sorted)
        if self.reverse_sorted:
            start, stop = stop, start
        start = None if start == 0 else start
        item = slice(start, (stop + 1) if stop is not None else stop)
        if _degenerate(item, len(self)):
            return self.empty(), slice(0, 0)
        return self[item], item

    # --- identity

    @cached_property
    def run_hashes(self) -> np.ndarray:
        """A 64 bit hash of each run, independent of where it sits."""
        table = self.runs
        seed = np.uint64(_blake(str(np.dtype(self.dtype)).encode()))
        out = np.full(len(table), seed, np.uint64)
        anchor = table["start"]
        if not self._ticks:  # -0.0 and 0.0 are one label
            anchor = np.where(anchor == 0, 0.0, anchor)
        columns = [np.ascontiguousarray(anchor).view(np.uint64)]
        columns += [
            np.ascontiguousarray(table[x]).view(np.uint64) for x in _RECORD_TAIL
        ]
        with np.errstate(over="ignore"):
            if len(table) <= _SMALL_TABLE:
                words = [column.tolist() for column in columns]
                for index in range(len(table)):
                    acc = int(seed)
                    for column in words:
                        acc = _splitmix_int(acc ^ _splitmix_int(column[index]))
                    out[index] = acc
            else:
                for column in columns:
                    out = _splitmix(out ^ _splitmix(column))
            if self.labels is not None:
                bounds = self._label_starts
                for index in np.flatnonzero(table["den"] == 0):
                    payload = self._flat_labels[bounds[index] : bounds[index + 1]]
                    out[index] = _splitmix(
                        out[index] ^ np.uint64(_blake(payload.tobytes()))
                    )
        return _readonly(out)

    def fingerprint(self) -> str:
        """A stable identifier whose matches imply the coordinates are equal."""
        payload = (
            "unidas.NumericND",
            self.units,
            str(np.dtype(self.dtype)),
            list(self.shape),
            hashlib.blake2b(
                np.ascontiguousarray(self.run_hashes).tobytes(), digest_size=16
            ).hexdigest(),
        )
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def __eq__(self, other) -> bool:
        """
        Whether two coordinates hold the same labels in the same units.

        Equal tables are equal coordinates; two which differ are different
        coordinates, except where both are nothing but stored labels, which
        are compared the way float arrays are -- closely rather than
        exactly. Which axis a
        coordinate is attached to, and what a source said about it beyond
        its units, are metadata rather than labels: `dims` and `attrs` are
        no part of this, nor of the fingerprint.
        """
        if not isinstance(other, NumericND):
            return NotImplemented
        if self.shape != other.shape or self.units != other.units:
            return False
        if np.dtype(self.dtype) != np.dtype(other.dtype):
            return False
        if self.partial_labels or other.partial_labels:
            return self.partial_labels and other.partial_labels
        if self.fingerprint() == other.fingerprint():
            return True
        # A grid says what it is in a handful of scalars, and two which
        # differ in any of them are different coordinates. Only coordinates
        # which are nothing but stored labels are compared the way float
        # arrays are, closely rather than exactly.
        if not (self._all_stored and other._all_stored):
            return False
        return _all_close(self.values, other.values)

    def __repr__(self) -> str:
        step = self.step_exact
        parts = [
            f"shape={self.shape}",
            f"dtype={np.dtype(self.dtype)}",
            f"runs={self.runs_count}",
        ]
        if step is not None:
            parts.append(f"step={step}")
        if self.units is not None:
            parts.append(f"units={self.units!r}")
        if self.dims:
            parts.append(f"dims={self.dims}")
        return f"NumericND({', '.join(parts)})"

    # --- updates

    def translated(self, delta) -> NumericND:
        """Shift every label; the grids move with them."""
        self._require_labels()
        table = self.runs.copy()
        labels = self.labels
        if self._ticks:
            shift = to_tick(delta)
            table["start"] += shift
            if labels is not None:
                moved = self._from_anchor(self._flat_labels + shift)
                labels = moved.reshape(labels.shape)
        else:
            table["start"] += float(delta)
            if labels is not None:
                labels = labels + delta
        return self._build(self.dtype, table, labels, self.units, self.dims, self.attrs)

    def rescaled(self, factor, offset=0.0, units=None) -> NumericND:
        """
        Every label through ``label * factor + offset``.

        The affine map a unit conversion is, done a run at a time: a grid run
        maps its start and its spacing and keeps its count, so only the
        labels a run actually stores are ever spelled out. What is mapped is
        therefore the grid, not each label it was rounded to: where the
        labels have lost the grid's own resolution -- a float run whose
        spacing is below its magnitude's last bit -- the scaled labels are
        the ones the mapped grid gives, which are not the mapped labels. The
        result holds floats, since scaling a grid of ticks rarely lands on
        one.

        Parameters
        ----------
        factor
            What every label is multiplied by.
        offset
            What is added to every scaled label.
        units
            The units the rescaled labels are in; the coordinate's own
            units when none are given, since a factor alone says nothing
            about what the labels now measure.
        """
        units = self.units if units is None else units
        if _time_like(self.dtype):
            msg = "A time is counted in nanoseconds and cannot be rescaled."
            raise CoordinateError(msg)
        self._require_labels()
        table = self.runs
        if self._ticks and np.any(table["den"] > 1):
            # The labels of a fractional tick grid are floors of its ideal
            # positions, which the ideal positions scaled are not.
            values = self.values.astype(np.float64) * factor + offset
            return self.from_array(
                values, units=units, dims=self.dims, detect=False, attrs=self.attrs
            )
        grid = table["den"] > 0
        starts = table["start"].astype(np.float64) * factor + offset
        spacings = np.where(grid, _scaled(1, table["num"], table["den"]), 0.0) * factor
        num = np.zeros(len(table), np.int64)
        den = np.zeros(len(table), np.int64)
        num[grid], den[grid] = _float_terms(spacings[grid])
        if np.any(den[grid] == 0):
            # A scaled spacing with no fraction of its own; the labels
            # themselves are then the only description of the grid.
            values = self.values.astype(np.float64) * factor + offset
            return self.from_array(
                values, units=units, dims=self.dims, detect=False, attrs=self.attrs
            )
        labels = None
        if self.labels is not None:
            labels = self.labels.astype(np.float64) * factor + offset
        dtype = np.dtype(np.float64)
        new = rows(dtype, starts, table["length"], num, den, 0)
        return self._build(dtype, new, labels, units, self.dims, self.attrs)


def _runs_describe(out: NumericND, values) -> bool:
    """Whether every run of a float table describes the labels it covers."""
    table, flat, got = out.runs, np.ravel(values), np.ravel(out.values)
    bounds = np.concatenate([[0], np.cumsum(table["length"])])
    for index in range(len(table)):
        first, stop = int(bounds[index]), int(bounds[index + 1])
        row = table[index]
        step = float(row["num"]) / max(float(row["den"]), 1.0)
        if not labels_meet(
            got[first:stop],
            flat[first:stop],
            out.dtype,
            float(row["start"]),
            step,
        ):
            return False
    return True


def _degenerate(item: slice, length: int) -> bool:
    """Whether a slice of a coordinate selects nothing at all."""
    start, stop = item.start, item.stop
    between = start is not None and start == stop
    bad_start = start is not None and (start < 0 or start >= length)
    bad_stop = stop is not None and stop <= 0
    return between or bad_start or bad_stop


def _isnull_array(values) -> np.ndarray:
    """Which labels of an array are nulls (NaN or NaT)."""
    values = np.asarray(values)
    if values.dtype.kind in "mM":
        return np.isnat(values)
    if values.dtype.kind in "fc":
        return np.isnan(values)
    return np.zeros(values.shape, dtype=bool)


def _nullish(dtype):
    """The null label of a dtype: NaT for a time, NaN for anything else."""
    dtype = np.dtype(dtype)
    if dtype.kind in "mM":
        return np.asarray("NaT", dtype=dtype)[()]
    return np.nan


def _nanmin(values):
    """The smallest label, ignoring nulls."""
    values = np.asarray(values)
    if values.dtype.kind in "mM":
        known = values[~np.isnat(values)]
        return _nullish(values.dtype) if not known.size else known.min()
    return np.nanmin(values) if values.dtype.kind == "f" else values.min()


def _nanmax(values):
    """The largest label, ignoring nulls."""
    values = np.asarray(values)
    if values.dtype.kind in "mM":
        known = values[~np.isnat(values)]
        return _nullish(values.dtype) if not known.size else known.max()
    return np.nanmax(values) if values.dtype.kind == "f" else values.max()


def concat(*coords: NumericND) -> NumericND:
    """
    Join run tables end to end into one table.

    A run which continues the one before it on the same grid, phase
    included, is fused into it, so equal samples give an equal coordinate
    however they were assembled. Anything else starts a new run.

    Parameters
    ----------
    *coords
        The coordinates to join; all must share a dtype and units.
    """
    if not coords:
        msg = "There is nothing to concatenate."
        raise CoordinateError(msg)
    first, *rest = coords
    if any(np.dtype(x.dtype) != np.dtype(first.dtype) for x in rest):
        msg = f"Runs must share a dtype, got {[str(x.dtype) for x in coords]}."
        raise CoordinateError(msg)
    if any(x.units != first.units for x in rest):
        msg = "Runs must share units."
        raise CoordinateError(msg)
    if any(x.partial_labels for x in coords):
        msg = "A coordinate whose labels are not loaded cannot be joined."
        raise CoordinateError(msg)
    table = np.concatenate([x.runs for x in coords])
    pieces = [x.labels for x in coords if x.labels is not None]
    labels = None
    if pieces:
        labels = np.concatenate([np.ravel(x) for x in pieces])
    # A declared grid survives only when every coordinate states the same
    # one; a run which states none says its labels may sit anywhere. The
    # exact spacings are what agree or differ: two runs of 2 and of 3/2
    # ticks both round to a step of 2, and are not one grid.
    exact = [x.step_exact for x in coords]
    stated = all(x is not None for x in exact) and len(set(exact)) == 1
    step = first.step if stated else None
    if step is not None and not _declares(table, first.dtype, step):
        # The runs as given do not all sit on the grid they declare.
        step = None
    return first._build(
        first.dtype, table, labels, first.units, first.dims, first.attrs, step=step
    )
