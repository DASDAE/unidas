"""
The categorical coordinate: codes into a sorted table of categories.

Labels which are not numbers -- station names, flags, anything a grid has no
arithmetic for -- are held as `NumericND` codes into an array of unique
categories. The codes are a run table like any other, so a coordinate which
is piecewise constant (a hundred samples of "north", then a hundred of
"south") costs two runs rather than two hundred labels.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

from .numeric import CoordinateError, NumericND

# One NaN and one NaT, so that two of either are one category: a dict looks
# a key up by identity before it asks whether it is equal, and no null is
# equal to itself.
_NAN = float("nan")
_NULL_TIMES = {"M": np.datetime64("NaT", "ns"), "m": np.timedelta64("NaT", "ns")}

# Where each kind of category sorts among the others when they are mixed.
_NUMBER, _TEXT, _OTHER = 0, 1, 2


def _unwrap(value):
    """A numpy scalar as the python value it holds; times are left alone."""
    if isinstance(value, np.generic) and value.dtype.kind not in "mM":
        return value.item()
    if isinstance(value, np.ndarray) and not value.ndim:
        return _unwrap(value[()])
    return value


def _normalized(value):
    """
    One label in the form a category is held in.

    Bytes are decoded, ``-0.0`` is the same label as ``0.0``, and every null
    is the one null, so that a category is not repeated for each of them.
    """
    value = _unwrap(value)
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, float):
        if math.isnan(value):
            return _NAN
        return 0.0 if value == 0.0 else value
    if isinstance(value, np.datetime64 | np.timedelta64) and np.isnat(value):
        return _NULL_TIMES[value.dtype.kind]
    return value


def _is_number(value) -> bool:
    """Whether a value is a number the sort order ranks as one."""
    return isinstance(value, bool | int | float | np.number | np.bool_)


def _number_key(value) -> tuple[int, int, Fraction]:
    """
    Where a number sorts, and what tells it from another: exactly.

    A NaN is not below or above anything, so it is put last among the
    numbers rather than compared with them, and an infinity sits at the end
    it names. Everything else is the ratio it is, so that ``1``, ``1.0`` and
    ``True`` are one number while two integers no float can tell apart stay
    two.
    """
    if isinstance(value, float | np.floating):
        if math.isnan(value):
            return (1, 0, Fraction(0))
        if math.isinf(value):
            return (0, 1 if value > 0 else -1, Fraction(0))
        # A python float is its own binary ratio; a wider or narrower one
        # is the decimal it prints, which numpy writes back exactly.
        exact = Fraction(value) if isinstance(value, float) else Decimal(str(value))
        return (0, 0, Fraction(exact))
    return (0, 0, Fraction(int(value)))


def _canonical_number(value):
    """
    One number in the one spelling a category is held under.

    ``1``, ``1.0`` and ``True`` are one category, so which of them a
    coordinate shows must not depend on which was read first: an integral
    number is held as an `int` and every other as the `float` its exact key
    names -- unless no float names it, in which case the label keeps its own
    spelling rather than being merged into the number it rounds to.
    """
    key = _number_key(value)
    null, infinite, exact = key
    if null:
        return _NAN
    if infinite:
        return math.inf if infinite > 0 else -math.inf
    if exact.denominator == 1:
        return int(exact)
    plain = float(exact)
    # A number no double tells from another keeps its own spelling: the
    # category it names is the one its exact key states, not the one the
    # rounded double would be merged into.
    return plain if _number_key(plain) == key else value


def _sort_key(value):
    """Where one category sorts: numbers, then text, then anything else."""
    if _is_number(value):
        return (_NUMBER, *_number_key(value), "")
    if isinstance(value, str):
        return (_TEXT, 0, 0, Fraction(0), value)
    return (_OTHER, 0, 0, Fraction(0), repr(value))


def _token(value):
    """One category as the pair a fingerprint and an equality test read it by."""
    value = _normalized(value)
    if _is_number(value):
        null, infinite, exact = _number_key(value)
        return ("number", f"{null}:{infinite}:{exact}")
    if isinstance(value, str):
        return ("text", value)
    return ("other", repr(value))


def _family(value) -> str:
    """Which typed array a category could live in, if they all agree."""
    if isinstance(value, str):
        return "text"
    if isinstance(value, np.datetime64):
        return "datetime"
    if isinstance(value, np.timedelta64):
        return "timedelta"
    if _is_number(value):
        return "number"
    return "other"


def _typed(flat: list) -> np.ndarray | None:
    """
    The typed array these labels make, or None when it would change one.

    Numpy reads ``[1.5, 10**17 + 1]`` as floats, which is a label the array
    cannot hold; labels numpy would round are kept as objects instead.
    """
    typed = np.asarray(flat)
    if typed.dtype.kind == "O":
        return None
    same = (a == b or (a != a and b != b) for a, b in zip(typed.tolist(), flat))
    return typed if all(same) else None


def _typed_categories(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The unique labels of a typed array, sorted, and each label's code."""
    if values.dtype.kind == "f":
        # -0.0 and 0.0 are one label, and numpy sorts them as one anyway.
        values = np.where(values == 0, 0.0, values)
    categories, codes = np.unique(values, return_inverse=True)
    return categories, codes.reshape(values.shape)


def _key(value):
    """What tells two labels apart; a label with no hash is told by its repr."""
    try:
        hash(value)
    except TypeError:
        return ("<unhashable>", repr(value))
    return value


def _object_categories(flat: list) -> tuple[np.ndarray, np.ndarray]:
    """The unique labels of mixed types, in the order the rule states."""
    flat = [_canonical_number(x) if _is_number(x) else x for x in flat]
    unique = list({_key(x): x for x in reversed(flat)}.values())
    unique.sort(key=_sort_key)
    index = {_key(value): position for position, value in enumerate(unique)}
    categories = np.empty(len(unique), dtype=object)
    for position, value in enumerate(unique):
        categories[position] = value
    return categories, np.asarray([index[_key(x)] for x in flat], dtype=np.int64)


@dataclass(frozen=True, eq=False)
class Categorical:
    """
    A coordinate of labels which are not numbers.

    Parameters
    ----------
    codes
        The index of each sample's category, as a `NumericND` of integers.
    categories
        The unique categories, in canonical order.
    dims
        The array axes the coordinate is attached to; none for a scalar.
    attrs
        What a source says about the coordinate, carried for whichever
        destination can hold it.

    Examples
    --------
    >>> from unidas import Categorical
    >>> coord = Categorical.from_labels(["north"] * 3 + ["south"] * 3)
    >>> assert list(coord.categories) == ["north", "south"]
    >>> assert coord.codes.runs_count == 2  # a run each, not a label each
    """

    codes: NumericND
    categories: np.ndarray
    dims: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Hold the categories as an array and check they match the codes."""
        set_ = object.__setattr__
        set_(self, "categories", np.asarray(self.categories))
        set_(self, "dims", tuple(self.dims))
        set_(self, "attrs", dict(self.attrs))
        if self.categories.ndim != 1:
            msg = "The categories of a coordinate are a one dimensional array."
            raise CoordinateError(msg)
        if np.dtype(self.codes.dtype).kind not in "iu":
            msg = f"Codes are integers, not {self.codes.dtype}."
            raise CoordinateError(msg)
        if self.dims and len(self.dims) != len(self.shape):
            msg = f"A coordinate of shape {self.shape} has no dims {self.dims}."
            raise CoordinateError(msg)

    @classmethod
    def from_labels(cls, values, dims=(), attrs=None):
        """
        Build from labels, which become codes into their sorted categories.

        Labels of one type keep that type and numpy's order, nulls last.
        Labels of mixed types are held as objects and ordered numbers, then
        text, then everything else by its repr; ``1``, ``1.0`` and ``True``
        are one category, while ``1`` and ``"1"`` are two.

        Parameters
        ----------
        values
            The labels.
        dims
            The array axes the coordinate is attached to.
        attrs
            What the source says about the coordinate.
        """
        # An array states its own type; anything else is read label by
        # label, since numpy would make one string array of ``[1, "a"]``.
        raw = values if isinstance(values, np.ndarray) else np.asarray(values, object)
        if raw.dtype.kind == "S":
            raw = raw.astype(str)
        if raw.dtype.kind in "UMmbiuf":
            categories, codes = _typed_categories(raw)
        else:
            flat = [_normalized(x) for x in raw.ravel().tolist()]
            families = {_family(x) for x in flat}
            typed = None
            if len(families) == 1 and families != {"other"}:
                # One kind of label after all; numpy holds and orders it.
                typed = _typed(flat)
            if typed is not None:
                categories, codes = _typed_categories(typed)
            else:
                categories, codes = _object_categories(flat)
            codes = codes.reshape(raw.shape)
        return cls(
            codes=NumericND.from_array(np.asarray(codes, dtype=np.int64)),
            categories=categories,
            dims=dims,
            attrs=attrs or {},
        )

    # --- what it holds

    @property
    def units(self) -> None:
        """Categories have no units; the property answers so uniformly."""
        return None

    @property
    def values(self) -> np.ndarray:
        """The label of every sample, as `NumericND.values` is."""
        # asarray, so that a rank-0 coordinate answers with a rank-0 array
        # rather than with the numpy scalar fancy indexing hands back.
        return np.asarray(self.categories[self.codes.values])

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the labels."""
        return self.codes.shape

    @property
    def ndim(self) -> int:
        """How many axes the labels have."""
        return self.codes.ndim

    @property
    def size(self) -> int:
        """How many labels the coordinate holds."""
        return self.codes.size

    def __len__(self) -> int:
        return len(self.codes)

    def __getitem__(self, item):
        picked = self.codes[item]
        if not isinstance(picked, NumericND):
            return self.categories[picked]
        return Categorical(
            codes=picked,
            categories=self.categories,
            dims=self.dims if len(self.dims) == len(picked.shape) else (),
            attrs=self.attrs,
        )

    def reversed(self) -> Categorical:
        """The same samples in the opposite order."""
        return Categorical(
            codes=self.codes.reversed(),
            categories=self.categories,
            dims=self.dims,
            attrs=self.attrs,
        )

    # --- matching

    def _mask_of(self, picked) -> np.ndarray:
        """Which samples carry one of these category indices."""
        return np.isin(np.asarray(self.codes.values), np.asarray(picked, np.int64))

    def isin(self, values) -> np.ndarray:
        """
        Which samples carry one of the labels given.

        Parameters
        ----------
        values
            The labels to look for; one label, or an iterable of them.
        """
        if isinstance(values, str | bytes) or np.ndim(values) == 0:
            values = [values]
        wanted = {_token(x) for x in values}
        picked = [i for i, x in enumerate(self.categories) if _token(x) in wanted]
        return self._mask_of(picked)

    def match(self, glob: str) -> np.ndarray:
        """
        Which samples carry a label matching a glob pattern.

        Parameters
        ----------
        glob
            A pattern such as ``"north*"``, matched case sensitively against
            each category's string form.
        """
        picked = [
            i
            for i, x in enumerate(self.categories)
            if fnmatch.fnmatchcase(str(x), glob)
        ]
        return self._mask_of(picked)

    # --- identity

    @property
    def _tokens(self) -> list[tuple[str, str]]:
        """The categories as the pairs identity is read from."""
        return [list(_token(x)) for x in self.categories]

    def fingerprint(self) -> str:
        """A stable identifier whose matches imply the coordinates are equal."""
        payload: tuple[Any, ...] = (
            "unidas.Categorical",
            self._tokens,
            self.codes.fingerprint(),
        )
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def __eq__(self, other) -> bool:
        """Whether two coordinates carry the same labels in the same order."""
        if not isinstance(other, Categorical):
            return NotImplemented
        if self.shape != other.shape or self._tokens != other._tokens:
            return False
        return self.codes == other.codes

    def __repr__(self) -> str:
        parts = [
            f"shape={self.shape}",
            f"categories={len(self.categories)}",
            f"runs={self.codes.runs_count}",
        ]
        if self.dims:
            parts.append(f"dims={self.dims}")
        return f"Categorical({', '.join(parts)})"
