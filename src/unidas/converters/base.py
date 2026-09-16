"""Conversions from the base representation to each library's structure."""

from __future__ import annotations

from ..core import (
    BASE_DAS_KEY,
    BaseDAS,
    Converter,
    _as_number,
    converts_to,
    optional_import,
    time_to_datetime,
    time_to_float,
)
from .dascore import to_dascore_coord
from .xarray import to_xarray_coord
from .xdas import to_xdas_coord


class UnidasBaseDASConverter(Converter):
    """
    Class for converting from the base representation to other library structures.
    """

    name = BASE_DAS_KEY

    @converts_to("dascore.Patch")
    def to_dascore_patch(self, base_das: BaseDAS):
        """Convert to a dascore patch."""
        dc = optional_import("dascore")
        out = base_das.to_dict("dascore", to_dascore_coord, with_dims=True, name=False)
        return dc.Patch(**out)

    @converts_to("xdas.DataArray")
    def to_xdas_dataarray(self, base_das: BaseDAS):
        """Convert to a xdas data array."""
        xdas = optional_import("xdas")
        return xdas.DataArray(**base_das.to_dict("xdas", to_xdas_coord))

    @converts_to("xarray.DataArray")
    def to_xarray_dataarray(self, base_das: BaseDAS):
        """Convert to an xarray data array."""
        xr = optional_import("xarray")
        return xr.DataArray(**base_das.to_dict("xarray", to_xarray_coord))

    @converts_to("daspy.Section")
    def to_daspy_section(self, base_das: BaseDAS):
        """Convert to a daspy section."""
        daspy = optional_import("daspy")
        dasdt = daspy.DASDateTime
        sampling = base_das.get_sampling("daspy.Section")
        time_start, time_step = sampling["time"]
        distance_start, distance_step = sampling["distance"]
        start_time = time_to_datetime(time_start)
        # Coordinates and data are authoritative when metadata contains stale
        # structural fields such as fs, dx, or start_time.
        kwargs = dict(base_das.attrs)
        kwargs.update(
            data=base_das.transpose("distance", "time").data,
            # Divided before it becomes a float, so a rate which is exactly
            # representable stays exact: 1/49 s as a float reads as
            # 49.000000001 Hz, which will not join a section recorded at 49.
            fs=_as_number(1 / time_to_float(time_step)),
            dx=_as_number(distance_step),
            start_distance=distance_start,
            start_time=dasdt.from_datetime(start_time),
        )
        return daspy.Section(**kwargs)

    @converts_to("lightguide.Blast")
    def to_lightguide_blast(self, base_das: BaseDAS):
        """Convert to a lightguide blast."""
        lg_blast = optional_import("lightguide.blast")
        sampling = base_das.get_sampling("lightguide.Blast")
        time_start, time_step = sampling["time"]
        distance_start, distance_step = sampling["distance"]
        start_channel = round(distance_start / distance_step)
        return lg_blast.Blast(
            data=base_das.transpose("distance", "time").data,
            start_time=time_to_datetime(time_start),
            sampling_rate=_as_number(1 / time_to_float(time_step)),
            start_channel=start_channel,
            channel_spacing=_as_number(distance_step),
        )
