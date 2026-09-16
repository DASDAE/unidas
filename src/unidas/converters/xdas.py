"""Conversions to and from xdas' DataArray."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..core import (
    BASE_DAS_KEY,
    BaseDAS,
    Converter,
    converts_to,
    coord_from_labels,
    optional_import,
)
from ..numeric import (
    NS_PER_SECOND,
    CoordinateError,
    NumericND,
    concat,
    float_fraction,
    grid_at,
    step_terms,
    to_tick,
)


def _tie_point_runs(coord):
    """
    Read a table of tie points as ``(start, count, step)`` runs, or None.

    A pair of tie points one index apart is a hole, and the second starts a
    new run. A sample left alone between holes states no spacing, and a span
    of times or integers which is not whole ticks per sample has labels the
    tie points only round to; either is read as labels instead.
    """
    indices = np.asarray(coord.tie_indices, dtype=np.int64)
    values = np.asarray(coord.tie_values)
    if not indices.size:
        return None
    ticked = values.dtype.kind != "f"
    if not ticked:
        values = values.astype(float)
    elif values.dtype.kind in "iu":
        # Differenced as python integers: two narrow labels a whole range
        # apart wrap, and the check for whole ticks would read the wrap.
        values = values.astype(object)
    runs, covered = [], 0  # `covered` is the first sample not yet in a run.
    # The indices are python integers beside the labels, so that a span
    # which no integer dtype can hold is divided rather than wrapped.
    heads, tails = indices[:-1].tolist(), indices[1:].tolist()
    last = int(indices[-1])
    pairs = zip(heads, tails, values[:-1], values[1:], strict=True)
    for i0, i1, v0, v1 in pairs:
        if i1 - i0 == 1:
            if covered <= i0:
                return None
            covered = i1
            continue
        step = (v1 - v0) // (i1 - i0) if ticked else (v1 - v0) / (i1 - i0)
        if ticked and step * (i1 - i0) != v1 - v0:
            return None
        first = max(covered, i0)
        runs.append((v0 + step * (first - i0), i1 - first + 1, step))
        covered = i1 + 1
    return runs if covered > last else None


def _from_tie_points(coord, dims):
    """The coordinate a table of tie points states, or None."""
    runs = _tie_point_runs(coord)
    if not runs:
        return None
    dtype = np.asarray(coord.tie_values).dtype
    ticked = dtype.kind != "f"
    labels, pieces, offset = None, [], 0
    for start, count, step in runs:
        run = NumericND.from_run(start, step, count, dims=dims, dtype=dtype)
        if not ticked:
            # A float grid is start + k * num / den, which need not land on
            # the label xdas interpolates; such a stretch keeps its labels.
            if labels is None:
                labels = np.asarray(coord.values)
            piece = labels[offset : offset + count]
            if not np.array_equal(run.values, piece):
                run = NumericND.from_array(piece, detect=False, dims=dims)
        pieces.append(run)
        offset += count
    return concat(*pieces)


def _ratio_step(numerator, denominator, dtype):
    """An xdas sampling ratio as the exact spacing in coordinate units."""
    if np.dtype(dtype).kind in "mM":
        return Fraction(to_tick(numerator), int(denominator) * NS_PER_SECOND)
    return float_fraction(float(numerator)) / int(denominator)


def _from_sampling_ratio(coord, dims):
    """
    The grid an xdas sampling ratio states, when it is the labels.

    xdas rounds a label to the nearest tick where the run table takes the
    tick at or below it, so the grid is only kept when it gives back every
    label the coordinate holds; the phase which states rounding is tried
    beside the one which states flooring.
    """
    data = getattr(coord, "data", {})
    numerator = data.get("sampling_numerator")
    denominator = data.get("sampling_denominator")
    values = np.asarray(coord.tie_values)
    if numerator is None or denominator is None or not values.size:
        return None
    step = _ratio_step(numerator, denominator, values.dtype)
    if not step:
        return None
    count = len(coord)
    den = step_terms(step, np.asarray(values[:1]).dtype)[1]
    labels = np.asarray(coord.values)
    for offset in dict.fromkeys((0, den // 2)):
        run = NumericND.from_run(
            values[0], step, count, origin_offset=offset, dims=dims
        )
        if np.array_equal(run.values, labels):
            return run
    return None


def from_xdas_coord(coord, dims):
    """Read an xdas coordinate as the shape unidas holds it in."""
    xdas = optional_import("xdas")
    if isinstance(coord, xdas.InterpCoordinate):
        for reader in (_from_tie_points, _from_sampling_ratio):
            try:
                out = reader(coord, dims)
            except CoordinateError:
                # A spacing no run can hold -- one past what a fraction of
                # ticks can state -- describes nothing; the labels do.
                out = None
            if out is not None:
                return out
    return coord_from_labels(coord.values, dims=dims)


def _states_tie_points(coord) -> bool:
    """Whether a coordinate is runs xdas can state as pairs of tie points."""
    if not isinstance(coord, NumericND) or coord.ndim != 1 or len(coord) < 2:
        return False
    if not coord.sorted or np.any(coord.runs["den"] == 0):
        # Tie points interpolate between increasing knots; labels which do
        # neither are the only description they have.
        return False
    dtype = np.dtype(coord.dtype)
    if dtype.kind not in "iuMm":
        return True
    # A grid of fractional ticks labels its samples with the tick at or
    # below each one, which one spacing between a pair cannot state; and a
    # narrow integer wraps the arithmetic xdas does between two of them.
    return bool(np.all(coord.runs["den"] == 1)) and dtype.itemsize >= 8


def _one_step_after(table, position: int, ticked: bool) -> bool:
    """
    Whether a run starts one of its own steps after the run before it.

    Not the kernel's `_continues`, which asks whether a run carries on the
    grid and phase of the one before: a run which changes rate at the
    sample the last one stopped at answers yes here and no there, and that
    is the junction xdas states with a shared tie point.
    """
    before = table[position - 1]
    last, _ = grid_at(
        before["start"],
        before["offset"],
        before["num"],
        before["den"],
        int(before["length"]) - 1,
        ticked,
    )
    row = table[position]
    if ticked:
        return int(row["start"]) == int(last) + int(row["num"])
    step = float(row["num"]) / max(int(row["den"]), 1)
    return bool(np.isclose(float(row["start"]), float(last) + step, rtol=1e-12))


def _tie_points(coord):
    """The tie points a table of runs states, a pair per run."""
    table = coord.runs
    ticked = np.dtype(coord.dtype).kind in "iuMm"
    firsts = np.cumsum(table["length"]) - table["length"]
    indices = []
    for position, (first, length) in enumerate(zip(firsts, table["length"])):
        ends = [0] if length == 1 else [0, length - 1]
        # A run continuing the last at a new rate shares its tie point,
        # since a pair one index apart would say there is a hole.
        if position and _one_step_after(table, position, ticked):
            ends = ends[1:]
        indices.extend(int(first) + end for end in ends)
    return np.asarray(indices, dtype=np.int64)


def _interpolates_exactly(coord, out) -> bool:
    """Whether xdas gives back the labels its tie points were built from."""
    dtype = np.dtype(coord.dtype)
    if dtype.kind in "iuMm":
        # Whole ticks interpolate exactly in the int64 xdas counts them in.
        return True
    # A float knot pair states one spacing across a run, which need not land
    # on the label the run's own fraction gives.
    return bool(np.array_equal(np.asarray(out.values), coord.values))


def to_xdas_coord(coord, dims=()):
    """Write a coordinate as xdas holds it: tie points, labels, or a scalar."""
    xdas = optional_import("xdas")
    if coord.ndim > 1:
        raise ValueError("XDAS does not support multidimensional coordinates.")
    dim = dims[0] if len(dims) == 1 else None
    if not coord.ndim:
        return xdas.ScalarCoordinate(data=np.asarray(coord.values)[()])
    if _states_tie_points(coord):
        indices = _tie_points(coord)
        data = {
            "tie_indices": indices,
            "tie_values": coord.labels_at(indices),
        }
        if (step := coord.step) is not None:
            # One rate across every run is a rate xdas can state beside them.
            data["sampling_interval"] = step
        try:
            out = xdas.InterpCoordinate(data=data, dim=dim)
        except (TypeError, ValueError):
            # Tie points xdas will not hold state nothing; the labels do.
            out = None
        if out is not None and _interpolates_exactly(coord, out):
            return out
    return xdas.DenseCoordinate(data=np.asarray(coord.values), dim=dim)


class XDASConverter(Converter):
    """Converter for xdas DataArrays."""

    name = "xdas.DataArray"

    @converts_to(BASE_DAS_KEY)
    def to_base(self, data_array) -> BaseDAS:
        """Convert an xdas data array to the base representation."""
        coords = {}
        for name, coord in data_array.coords.items():
            dims = (coord.dim,) if coord.dim is not None else ()
            coords[name] = from_xdas_coord(coord, dims)
        attrs = {} if data_array.attrs is None else dict(data_array.attrs)
        return BaseDAS(
            data=data_array.data,
            dims=data_array.dims,
            coords=coords,
            attrs=attrs,
            name=data_array.name,
        )
