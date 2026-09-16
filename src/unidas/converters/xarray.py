"""Conversions to and from xarray's DataArray."""

from __future__ import annotations

import numpy as np

from ..core import (
    BASE_DAS_KEY,
    TIME_KINDS,
    BaseDAS,
    Converter,
    converts_to,
    coord_from_labels,
    optional_import,
)
from .dascore import from_dascore_coord


def to_xarray_coord(coord, dims=()):
    """Write a coordinate as an xarray variable of its labels."""
    xr = optional_import("xarray")
    attrs = dict(coord.attrs)
    array = coord.values
    # Xarray reads a units attribute on time values as CF encoding, so only
    # values which are not times get one.
    if coord.units is not None and np.asarray(array).dtype.kind not in TIME_KINDS:
        attrs["units"] = coord.units
    return xr.Variable(dims, array, attrs=attrs)


class XArrayConverter(Converter):
    """Converter for xarray DataArrays, including non-dimensional coordinates."""

    name = "xarray.DataArray"

    @converts_to(BASE_DAS_KEY)
    def to_base(self, data_array) -> BaseDAS:
        """Preserve the DataArray's values, dimensions, and descriptive metadata."""
        coords = {}
        for name, coord in data_array.coords.items():
            attrs = dict(coord.attrs)
            # A stated null unit is metadata of its own, and stays put.
            units = attrs.get("units")
            if units is not None:
                attrs.pop("units")
            # An index which states a run table -- runs with gaps between
            # them, or a grid of exact ticks -- is read from that, and
            # computes no labels to be read from. One saying anything else
            # has labels which say it as well.
            source = getattr(data_array.xindexes.get(name), "coordinate", None)
            if getattr(source, "runs", None) is not None:
                coords[name] = from_dascore_coord(source, coord.dims, units, attrs)
                continue
            coords[name] = coord_from_labels(coord.data, coord.dims, units, attrs)
        return BaseDAS(
            data=data_array.data,
            dims=data_array.dims,
            coords=coords,
            attrs=dict(data_array.attrs),
            name=data_array.name,
        )
