"""Conversions to and from Lightguide's Blast."""

from __future__ import annotations

from ..core import (
    BASE_DAS_KEY,
    BaseDAS,
    Converter,
    converts_to,
    extract_attrs,
    sampled_coord,
)


class LightGuideConverter(Converter):
    """
    Converter for Lightguide Blasts.
    """

    name = "lightguide.Blast"

    _attrs_to_extract = "unit"

    def _get_coords(self, blast):
        """Get base coordinates from Blast."""
        # Need to convert channel numbers to distance.
        start_distance = blast.start_channel * blast.channel_spacing
        channels, samples = blast.data.shape
        return {
            "distance": sampled_coord(
                start_distance, blast.channel_spacing, channels, dims=("distance",)
            ),
            "time": sampled_coord(
                blast.start_time, blast.delta_t, samples, dims=("time",)
            ),
        }

    @converts_to(BASE_DAS_KEY)
    def to_base(self, blast) -> BaseDAS:
        """Convert a lightguide blast to the base representation."""
        # From the plot on lightguide's readme it appears the dims are
        # (channel, time). We need to check if this is always true.
        dims = ("distance", "time")
        return BaseDAS(
            data=blast.data,
            dims=dims,
            coords=self._get_coords(blast),
            attrs=extract_attrs(blast, self._attrs_to_extract),
        )
