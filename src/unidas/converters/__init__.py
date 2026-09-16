"""
The converters between the base representation and each supported library.

Importing this package registers every converter with `Converter`, which is
what `convert` and `adapter` look a conversion path up in.
"""

from .dascore import DASCorePatchConverter
from .daspy import DASPySectionConverter
from .lightguide import LightGuideConverter
from .xarray import XArrayConverter
from .xdas import XDASConverter

# Last: it reads how each library above writes one coordinate.
from .base import UnidasBaseDASConverter

__all__ = (
    "DASCorePatchConverter",
    "DASPySectionConverter",
    "LightGuideConverter",
    "UnidasBaseDASConverter",
    "XArrayConverter",
    "XDASConverter",
)
