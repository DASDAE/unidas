"""Conversions to and from DASCore's Patch."""

from __future__ import annotations

import numpy as np

from ..categorical import Categorical
from ..core import BASE_DAS_KEY, BaseDAS, Converter, converts_to, optional_import
from ..numeric import CoordinateError, NumericND, labels_meet


def _refuse_this_dascore() -> None:
    """Refuse a DASCore which states no run table, by version."""
    version = getattr(optional_import("dascore"), "__version__", "unknown")
    msg = (
        "unidas 0.3 converts DASCore coordinates through the run-table API "
        f"(NumericND, DASCore 1.0); found {version}"
    )
    raise CoordinateError(msg)


def from_dascore_coord(coord, dims, units=None, attrs=None):
    """
    Read a DASCore coordinate as the shape unidas holds it in.

    The coordinate is read from its run table rather than from its labels,
    so a range describing a long acquisition crosses without every label
    being spelled out.

    Parameters
    ----------
    coord
        The DASCore coordinate, or any coordinate stating a run table.
    dims
        The dimensions the coordinate is associated with.
    units
        The units, when a source states them beside the coordinate.
    attrs
        Coordinate metadata other than units.
    """
    # A portable unit string keeps a destination's attrs serializable, and
    # DASCore parses one back to a quantity, unit scale included. DASCore
    # states seconds beside every time and derives them again from the
    # dtype, so reading its own units off one would only make a coordinate
    # which has been to DASCore differ from one which has not; what the
    # source said beside the coordinate is not DASCore's to drop.
    if units is None and np.dtype(coord.dtype).kind not in "mM":
        if getattr(coord, "units", None) is not None:
            units = str(coord.units)
    if np.dtype(coord.dtype).kind not in "iufMm":
        return Categorical.from_labels(coord.values, dims=dims, attrs=attrs)
    if getattr(coord, "runs", None) is None:
        _refuse_this_dascore()
    return NumericND.from_rows(
        coord.runs,
        labels=coord.labels,
        dtype=coord.dtype,
        units=units,
        dims=dims,
        attrs=attrs,
        # A step DASCore states for labels its runs do not describe; one
        # its runs do stays with them, and is derived again here.
        step=getattr(coord, "step", None),
    )


def _as_runs(coord, dc_core):
    """The DASCore coordinate a run table states."""
    try:
        return dc_core.get_coord(
            runs=coord.runs,
            labels=coord.labels,
            dtype=coord.dtype,
            units=coord.units,
            # The grid a source declared for labels the runs do not
            # describe; DASCore says which samples are missing from it.
            step=coord.declared_step,
        )
    except TypeError:
        # A DASCore with no run table in it is one which cannot be handed
        # one; anything else is its own error to report.
        if not hasattr(optional_import("dascore.core.coords"), "NumericND"):
            _refuse_this_dascore()
        raise


def to_dascore_coord(coord, dims=()):
    """Write a coordinate as DASCore holds it: its runs, or its labels."""
    dc_core = optional_import("dascore.core")
    if isinstance(coord, Categorical):
        return dc_core.get_coord(data=np.asarray(coord.values))
    out = _as_runs(coord, dc_core)
    ticked = np.dtype(coord.dtype).kind != "f"
    if ticked or coord.runs_count < 2 or coord.partial_labels:
        return out
    # DASCore joins adjacent float runs within a tolerance of its own,
    # relabelling the second from the first's start, which can move a label
    # further than the last bit or two a float grid here is fitted within.
    # Labels it would not give back are handed over as themselves, as they
    # are to any destination which cannot state them.
    if labels_meet(np.asarray(out.values), coord.values, coord.dtype):
        return out
    stored = NumericND.from_array(coord.values, units=coord.units, detect=False)
    return _as_runs(stored, dc_core)


class DASCorePatchConverter(Converter):
    """
    Converter for DASCore's Patch.
    """

    name = "dascore.Patch"
    # PatchAttrs fields which describe the patch's coordinates rather than its
    # acquisition. BaseDAS keeps that information in its own coords and dims,
    # and this copy goes stale as soon as anything changes the extent of an
    # axis, so it must not be dumped into the attrs dict.
    _structural_attrs = frozenset({"coords", "dims"})

    @converts_to(BASE_DAS_KEY)
    def to_base(self, patch) -> BaseDAS:
        """Convert dascore patch to base representation."""
        coords = patch.coords
        base_coords = {
            i: from_dascore_coord(v, coords.dim_map[i])
            for i, v in coords.coord_map.items()
            # A coordinate stating only its shape has no labels to carry, so
            # its dimension stays unlabeled rather than gaining null ones.
            if not getattr(v, "_partial", False)
        }
        attrs = patch.attrs.model_dump(exclude=self._structural_attrs)
        out = {
            "data": patch.data,
            "dims": patch.dims,
            "coords": base_coords,
            # An attr holding nothing says nothing, and a destination which
            # stores its attributes may refuse the null outright.
            "attrs": {i: v for i, v in attrs.items() if v is not None},
        }
        return BaseDAS(**out)
