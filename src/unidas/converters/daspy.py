"""Conversions to and from DASPy's Section."""

from __future__ import annotations

from ..core import (
    BASE_DAS_KEY,
    BaseDAS,
    Converter,
    converts_to,
    extract_attrs,
    sampled_coord,
)


class DASPySectionConverter(Converter):
    """
    Converter for DASpy sections
    """

    name = "daspy.Section"
    # The attributes of section that get stashed in the attrs dict.
    _section_attrs = (
        "start_channel",
        "origin_time",
        "data_type",
        "source",
        "source_type",
        "gauge_length",
    )

    @converts_to(BASE_DAS_KEY)
    def to_base(self, section) -> BaseDAS:
        """Convert a daspy section to the base representation."""
        dims = ("distance", "time")
        start_time = section.start_time.utc().to_datetime()
        channels, samples = section.data.shape
        coords = {
            "distance": sampled_coord(
                section.start_distance, section.dx, channels, dims=("distance",)
            ),
            "time": sampled_coord(start_time, section.dt, samples, dims=("time",)),
        }
        attrs = extract_attrs(section, self._section_attrs)
        return BaseDAS(data=section.data, dims=dims, coords=coords, attrs=attrs)
