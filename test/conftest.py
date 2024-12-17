"""
Pytest configuration and global fixtures for unidas.
"""

import platform

import dascore as dc
import daspy
import pooch
import pytest
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
# Currently, lightguide doesn't install on windows in CI. Just skip.
def lightguide_blast():
    """Get a Blast from lightguide."""
    if platform.system().lower() == "windows":
        pytest.skip("Lightguide is not supported on Windows")

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
