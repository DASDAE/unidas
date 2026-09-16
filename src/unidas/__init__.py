"""
Unidas: A DAS Compatibility Package.
"""

from .categorical import Categorical
from .converters import (
    DASCorePatchConverter,
    DASPySectionConverter,
    LightGuideConverter,
    UnidasBaseDASConverter,
    XArrayConverter,
    XDASConverter,
)
from .core import (
    BaseDAS,
    Converter,
    adapter,
    convert,
    converts_to,
    optional_import,
    time_to_datetime,
    time_to_float,
)
from .numeric import CoordinateError, NumericND, concat

# Keep the version hardcoded so vendored copies report their own version
# without requiring installed package metadata.
__version__ = "0.3.0"

# Explicitly defines unidas' public API.
__all__ = (
    "BaseDAS",
    "Categorical",
    "Converter",
    "CoordinateError",
    "NumericND",
    "adapter",
    "concat",
    "convert",
    "converts_to",
    "optional_import",
)
