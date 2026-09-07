"""
Pytest configuration and global fixtures for unidas.
"""

import importlib.util
import platform

import dascore as dc
import daspy
import numpy as np
import pooch
import pytest
import xarray as xr
from xdas.synthetics import wavelet_wavefronts


@pytest.fixture(scope="session")
def dascore_patch():
    """Get a dascore patch for testing."""
    return dc.get_example_patch()


@pytest.fixture(scope="session")
def daspy_section():
    """Get a daspy section for testing."""
    return daspy.read()


@pytest.fixture(scope="session")
# Lightguide currently has platform and Python-version compatibility limits.
def lightguide_blast():
    """Get a Blast from lightguide."""
    if platform.system().lower() == "windows":
        pytest.skip("Lightguide is not supported on Windows")
    if importlib.util.find_spec("lightguide") is None:
        pytest.skip("Lightguide is not installed")

    from lightguide.blast import Blast

    # Use pooch to download lightguide's example data.
    hash = "9e1ef3731cb2cfa1024b8eb36b2c8b78ab7687f4ef74b2aaee8d96cf4d5f2d85"

    file_path = pooch.retrieve(
        # URL to one of Pooch's test files
        url="https://data.pyrocko.org/testing/lightguide/VSP-DAS-G1-120.mseed",
        known_hash=hash,
    )
    return Blast.from_miniseed(file_path)


@pytest.fixture(scope="session")
def xdas_dataarray():
    """Load an xdas data array."""
    dar = wavelet_wavefronts().load()
    return dar


@pytest.fixture(scope="session")
def xarray_dataarray():
    """Create a DataArray independently of the other converters."""
    time = np.datetime64("2020-01-01T00:00:00.123456789") + np.arange(
        6, dtype=np.int64
    ) * np.timedelta64(2, "ms")
    return xr.DataArray(
        np.arange(18, dtype=np.float32).reshape(3, 6),
        dims=("distance", "time"),
        coords={
            "distance": ("distance", np.arange(3) * 0.1, {"units": "m"}),
            "time": time,
        },
        attrs={"description": "Synthetic DAS data"},
        name="strain_rate",
    )
