"""
Tests for core functionality of unidas.
"""

import datetime
import platform

import dascore as dc
import daspy
import numpy as np
import pandas as pd
import pytest
import unidas
from dascore.examples import EXAMPLE_PATCHES
from unidas import BaseDAS, Converter, adapter, convert, optional_import
from xdas.core.dataarray import DataArray

try:
    from lightguide.blast import Blast

except ImportError:

    class Blast:
        """A dummy blast."""


ON_WINDOWS = platform.system().lower() == "windows"

# A tuple of format names for testing generic conversions.
NAME_CLASS_MAP = {
    "dascore.Patch": dc.Patch,
    "xdas.DataArray": DataArray,
    "daspy.Section": daspy.Section,
    "lightguide.Blast": Blast,
}
BASE_FORMATS = tuple(NAME_CLASS_MAP)
DASCORE_EXAMPLE_NAMES = tuple(EXAMPLE_PATCHES)
DASPY_COMPATIBLE_DASCORE_EXAMPLES = (
    "random_das",
    "patch_with_null",
    "random_patch_with_lat_lon",
    "random_patch_with_xyz",
    "sin_wav",
    "chirp",
    "example_event_1",
    "deformation_rate_event_1",
    "dispersion_event",
)
DASPY_UNSUPPORTED_DASCORE_EXAMPLES = {
    "wacky_dim_coords_patch": "evenly sampled",
    "example_event_2": "absolute datetime",
    "forge_dss": "evenly sampled",
    "forge_dts": "evenly sampled",
    "ricker_moveout": "absolute datetime",
}


# --- Tests for unidas utilities.


def assert_array_equal_with_nan(array_1, array_2):
    """Assert arrays are equal, treating floating NaNs as equal."""
    array_1 = np.asarray(array_1)
    array_2 = np.asarray(array_2)
    if np.issubdtype(array_1.dtype, np.floating):
        assert np.allclose(array_1, array_2, equal_nan=True)
    else:
        assert np.array_equal(array_1, array_2)


@pytest.fixture(params=BASE_FORMATS)
def format_name(request):
    """Fixture for returning format names."""
    name = request.param
    if ON_WINDOWS and name.startswith("lightguide"):
        pytest.skip("waveguide does not support windows")
    return request.param


class TestOptionalImport:
    """Test suite for optional imports."""

    def test_import_installed_module(self):
        """Test to ensure an installed module imports."""
        import functools

        mod = optional_import("functools")
        assert mod is functools

    def test_missing_module_raises(self):
        """Ensure a module which is missing raises the appropriate Error."""
        with pytest.raises(ImportError, match="boblib4"):
            optional_import("boblib4")


class TestMisc:
    """Miscellaneous tests for unidas."""

    def test_version(self):
        """Simply ensure unidas has a version attribute."""
        assert hasattr(unidas, "__version__")
        assert isinstance(unidas.__version__, str)

    def test_time_to_float_datetime(self):
        """Ensure Python datetimes convert to timestamps."""
        time = datetime.datetime(1970, 1, 1, 0, 0, 1, tzinfo=datetime.UTC)

        out = unidas.time_to_float(time)

        assert out == 1


class TestCoordinate:
    """Test suite for coordinate base behavior."""

    def test_base_coordinate_methods_raise(self):
        """Ensure base coordinate methods are abstract."""
        coord = unidas.Coordinate()
        msg = "Not implemented"

        with pytest.raises(NotImplementedError, match=msg):
            coord.to_dascore_coord()
        with pytest.raises(NotImplementedError, match=msg):
            coord.to_xdas_coord()
        with pytest.raises(NotImplementedError, match=msg):
            coord.get_step()
        with pytest.raises(NotImplementedError, match=msg):
            coord.get_start()

    def test_evenly_sampled_coordinate_with_gaps_to_dascore_raises(self):
        """Ensure gapped coordinates cannot convert to DASCore."""
        coord = unidas.EvenlySampledCoordinate(
            tie_values=(0, 1, 3),
            tie_indices=(0, 2, 4),
            step=1,
            dims=("distance",),
        )

        with pytest.raises(NotImplementedError, match="gaps"):
            coord.to_dascore_coord()


class TestConverterBase:
    """Test suite for converter base behavior."""

    def test_subclass_without_name_raises(self):
        """Ensure converter subclasses must define a name."""
        with pytest.raises(ValueError, match="must define a name"):

            class BadConverter(Converter):
                """Converter missing a name."""


# --------- Tests for unidas conversions.


class TestDASCorePatch:
    """Test suite for converting DASCore Patches."""

    @pytest.fixture(scope="class")
    def dascore_base_das(self, dascore_patch):
        """The converted DASCore patch."""
        return convert(dascore_patch, "unidas.BaseDAS")

    def test_to_base_das(self, dascore_base_das):
        """Ensure we can convert DASCore patch to BaseDAS."""
        assert isinstance(dascore_base_das, BaseDAS)

    def test_from_base_das(self, dascore_base_das, dascore_patch):
        """Test the conversion back to DASCore Patch from BaseDAS."""
        out = convert(dascore_base_das, "dascore.Patch")
        assert isinstance(out, dc.Patch)
        assert out == dascore_patch

    def test_to_xdas_time_coord(self, dascore_patch):
        """
        Ensure we can convert to xdas DataArray and the time coords are equal.
        """
        out = convert(dascore_patch, "xdas.DataArray")
        time_coord1 = dascore_patch.get_array("time")
        time_coord2 = out.coords["time"].values
        assert np.all(time_coord1 == time_coord2)

    def test_convert_patch_to_other(self, dascore_patch, format_name):
        """Test that the base patch can be converted to all formats."""
        out = convert(dascore_patch, to=format_name)
        assert isinstance(out, NAME_CLASS_MAP[format_name])

    @pytest.mark.parametrize("example_name", DASCORE_EXAMPLE_NAMES)
    def test_example_patch_to_base_das(self, example_name):
        """Ensure all DASCore generated examples convert to BaseDAS."""
        patch = dc.get_example_patch(example_name)

        out = convert(patch, to="unidas.BaseDAS")

        assert isinstance(out, BaseDAS)
        assert out.dims == patch.dims
        assert out.data.shape == patch.shape
        out.validate()

    @pytest.mark.parametrize("example_name", DASCORE_EXAMPLE_NAMES)
    def test_example_patch_round_trip_to_dascore(self, example_name):
        """Ensure all DASCore generated examples round-trip through BaseDAS."""
        patch = dc.get_example_patch(example_name)
        base = convert(patch, to="unidas.BaseDAS")

        out = convert(base, to="dascore.Patch")

        assert isinstance(out, dc.Patch)
        assert out.dims == patch.dims
        assert out.shape == patch.shape
        assert_array_equal_with_nan(out.data, patch.data)
        assert set(out.coords.coord_map) == set(patch.coords.coord_map)
        for coord_name in patch.coords.coord_map:
            assert out.coords.dim_map[coord_name] == patch.coords.dim_map[coord_name]
            assert_array_equal_with_nan(
                out.get_array(coord_name),
                patch.get_array(coord_name),
            )

    @pytest.mark.parametrize("example_name", DASCORE_EXAMPLE_NAMES)
    def test_example_patch_to_xdas_dataarray(self, example_name):
        """Ensure all DASCore generated examples convert to XDAS."""
        patch = dc.get_example_patch(example_name)

        out = convert(patch, to="xdas.DataArray")

        assert isinstance(out, DataArray)
        assert out.dims == patch.dims
        assert out.shape == patch.shape
        assert_array_equal_with_nan(out.data, patch.data)
        assert set(out.coords) == set(patch.coords.coord_map)
        for coord_name in patch.coords.coord_map:
            assert out.coords[coord_name].dim == patch.coords.dim_map[coord_name][0]
            assert_array_equal_with_nan(
                out.coords[coord_name].values,
                patch.get_array(coord_name),
            )

    def test_time_distance_patch_to_daspy_section_shape(self, dascore_patch):
        """Ensure DASPy sections always use channel/time data order."""
        patch = dascore_patch.transpose("time", "distance")

        out = convert(patch, to="daspy.Section")

        assert isinstance(out, daspy.Section)
        assert out.data.shape == dascore_patch.shape

    @pytest.mark.parametrize("example_name", DASPY_COMPATIBLE_DASCORE_EXAMPLES)
    def test_example_patch_to_daspy_section(self, example_name):
        """Ensure DASCore example patches can convert to DASPy sections."""
        patch = dc.get_example_patch(example_name)
        expected_shape = patch.transpose("distance", "time").shape

        out = convert(patch, to="daspy.Section")

        assert isinstance(out, daspy.Section)
        assert out.data.shape == expected_shape

    @pytest.mark.parametrize(
        ("example_name", "message"),
        DASPY_UNSUPPORTED_DASCORE_EXAMPLES.items(),
    )
    def test_unsupported_example_patch_to_daspy_section(self, example_name, message):
        """Ensure unsupported DASCore examples fail with expected errors."""
        patch = dc.get_example_patch(example_name)

        with pytest.raises(ValueError, match=message):
            convert(patch, to="daspy.Section")

    def test_single_distance_patch_to_daspy_section(self):
        """Ensure singleton distance coordinates default to dx=1."""
        time = dc.to_datetime64("2020-01-01") + dc.to_timedelta64(np.arange(10))
        patch = dc.Patch(
            data=np.zeros((1, 10)),
            coords={"distance": [0], "time": time},
            dims=("distance", "time"),
        )

        out = convert(patch, to="daspy.Section")

        assert isinstance(out, daspy.Section)
        assert out.data.shape == patch.shape
        assert out.dx == 1

    def test_array_coordinates_to_daspy_section(self):
        """Ensure evenly sampled array coordinates can convert to DASPy."""
        time = dc.to_datetime64("2020-01-01") + dc.to_timedelta64(np.arange(4))
        distance = np.arange(3) * 2
        base_das = BaseDAS(
            data=np.zeros((3, 4)),
            coords={
                "distance": unidas.ArrayCoordinate(
                    data=distance,
                    dims=("distance",),
                ),
                "time": unidas.ArrayCoordinate(
                    data=time,
                    dims=("time",),
                ),
            },
            attrs={},
            dims=("distance", "time"),
        )

        out = convert(base_das, to="daspy.Section")

        assert isinstance(out, daspy.Section)
        assert out.data.shape == base_das.data.shape
        assert out.dx == 2
        assert out.fs == 1


class TestDASPySection:
    """Test suite for converting DASPy sections."""

    @pytest.fixture(scope="class")
    def daspy_base_das(self, daspy_section):
        """The default daspy section converted to BaseDAS instance."""
        return convert(daspy_section, "unidas.BaseDAS")

    def test_to_base_das(self, daspy_base_das):
        """Ensure the base section can be converted to BaseDAS."""
        assert isinstance(daspy_base_das, BaseDAS)

    def test_from_base_das(self, daspy_base_das, daspy_section):
        """Ensure the default section can round-trip."""
        out = convert(daspy_base_das, "daspy.Section")
        # TODO these objects aren't equal but their strings are.
        # We need to fix this.
        # assert out == daspy_section
        assert str(out) == str(daspy_section)
        assert np.all(out.data == daspy_section.data)

    def test_convert_section(self, daspy_section, format_name):
        """Test that the base section can be converted to all formats."""
        out = convert(daspy_section, to=format_name)
        assert isinstance(out, NAME_CLASS_MAP[format_name])


class TestXdasDataArray:
    """Tests for converting xdas DataArrays."""

    @pytest.fixture(scope="class")
    def xdas_base_das(self, xdas_dataarray):
        """Converted xdas section to BaseDAS."""
        return convert(xdas_dataarray, "unidas.BaseDAS")

    def test_to_base_das(self, xdas_base_das):
        """Ensure the example data_array can be converted to BaseDAS."""
        assert isinstance(xdas_base_das, BaseDAS)

    def test_convert_data_array_to_other(self, xdas_dataarray, format_name):
        """Test that the base data array can be converted to all formats."""
        out = convert(xdas_dataarray, to=format_name)
        assert isinstance(out, NAME_CLASS_MAP[format_name])

    def test_from_base_das(self, xdas_base_das, xdas_dataarray):
        """Ensure xdas DataArray can round trip."""
        out = convert(xdas_base_das, "xdas.DataArray")
        assert np.all(out.data == xdas_dataarray.data)
        # TODO the str rep of coords are equal but not coords themselves.
        # We need to look into this.
        assert str(out.coords) == str(xdas_dataarray.coords)
        attr1, attr2 = out.attrs, xdas_dataarray.attrs
        assert attr1 == attr2 or (not attr1 and not attr2)
        assert out.dims == xdas_dataarray.dims

    def test_dense_coordinate_to_base_das(self):
        """Ensure XDAS dense coordinates convert to array coordinates."""
        xdas = optional_import("xdas")
        data_array = xdas.DataArray(
            np.zeros((3, 2)),
            coords={
                "time": [0, 1, 2],
                "distance": [0, 1],
                "quality": ("time", [1, 2, 3]),
            },
            dims=("time", "distance"),
        )

        out = convert(data_array, to="unidas.BaseDAS")

        assert isinstance(out.coords["quality"], unidas.ArrayCoordinate)
        assert out.coords["quality"].dims == ("time",)
        assert np.array_equal(out.coords["quality"].data, [1, 2, 3])

    def test_gapped_interp_coordinate_to_base_das_raises(self):
        """Ensure gapped XDAS coordinates are rejected."""
        xdas = optional_import("xdas")
        data_array = xdas.DataArray(
            np.zeros((5, 2)),
            coords={
                "time": {"tie_indices": [0, 2, 4], "tie_values": [0.0, 1.0, 3.0]},
                "distance": [0, 1],
            },
            dims=("time", "distance"),
        )

        with pytest.raises(NotImplementedError, match="gaps"):
            convert(data_array, to="unidas.BaseDAS")


class TestLightGuideBlast:
    """Tests for Blast Conversions."""

    @pytest.fixture(scope="class", autouse=True)
    def skip_on_windows(self):
        """Skip tests if on windows."""
        if ON_WINDOWS:
            pytest.skip("Lightguide doesn't support windows")

    @pytest.fixture(scope="class")
    def lightguide_base_das(self, lightguide_blast):
        """Converted lightguide blast to BaseDAS."""
        return convert(lightguide_blast, "unidas.BaseDAS")

    def test_base_das(self, lightguide_base_das):
        """Ensure the example blast can be converted to BaseDAS."""
        assert isinstance(lightguide_base_das, BaseDAS)

    def test_from_base_das(self, lightguide_base_das, lightguide_blast):
        """Ensure lightguide Blast can round trip."""
        out = convert(lightguide_base_das, "lightguide.Blast")
        # TODO here the objects also do not compare equal. Need to figure out
        # why. For now just do weaker checks.
        # assert out == lightguide_blast
        assert out.start_time == lightguide_blast.start_time
        assert np.all(out.data == lightguide_blast.data)
        assert out.unit == lightguide_blast.unit
        assert out.channel_spacing == lightguide_blast.channel_spacing
        assert out.start_channel == lightguide_blast.start_channel
        assert out.sampling_rate == lightguide_blast.sampling_rate

    def test_convert_blast_to_other(self, lightguide_blast, format_name):
        """Test that the base blast can be converted to all formats."""
        out = convert(lightguide_blast, to=format_name)
        assert isinstance(out, NAME_CLASS_MAP[format_name])


class TestConvert:
    """Generic tests for the convert function."""

    def test_bad_path_raises(self, dascore_patch):
        """Ensure a bad target raises a ValueError."""
        msg = "No conversion path"
        with pytest.raises(ValueError, match=msg):
            convert(dascore_patch, "notadaslibrary.NotAClass")


class TestAdapter:
    """Tests for adapter decorator."""

    def test_conversion(self, dascore_patch):
        """Simple conversion test."""

        @adapter("daspy.Section")
        def section_function(sec):
            """Dummy section function."""
            assert isinstance(sec, daspy.Section)
            return sec

        patch = dascore_patch.transpose("distance", "time")
        out = section_function(patch)
        assert isinstance(out, dc.Patch)

    def test_wrapping(self):
        """
        Ensure a function is only wrapped once for each target, and that
        the original function is accessible.
        """

        @adapter("dascore.Patch")
        def my_patch_func(patch):
            """A dummy patch function."""
            return patch

        assert hasattr(my_patch_func, "raw_function")
        assert my_patch_func is not my_patch_func.raw_function
        # This should simply return the original function
        new = adapter("dascore.Patch")(my_patch_func)
        assert new is my_patch_func
        # But this should wrap it again.
        new2 = adapter("daspy.Section")(my_patch_func)
        assert new2 is not my_patch_func
        # The raw function should remain unchanged.
        assert new2.raw_function is my_patch_func.raw_function
        assert new.raw_function is my_patch_func.raw_function

    def test_different_return_type(self, daspy_section):
        """Ensure wrapped functions that return different types still work."""

        @adapter("dascore.Patch")
        def dummy_func(patch):
            """Dummy function that returns dataframe."""
            return dc.spool(patch).get_contents()

        out = dummy_func(daspy_section)
        assert isinstance(out, pd.DataFrame)


class TestIntegrations:
    """Tests for integrating different data structures."""

    def test_readme_1(self):
        """First test for readme examples."""
        if ON_WINDOWS:
            pytest.skip("Lightguide doesn't support windows")
        sec = daspy.read()
        blast = unidas.convert(sec, to="lightguide.Blast")
        blast.afk_filter(exponent=0.8)
        sec_out = unidas.convert(blast, to="daspy.Section")
        assert isinstance(sec_out, daspy.Section)

    def test_readme_2(self, dascore_patch):
        """Second test for readme examples."""
        from xdas.signal import hilbert

        dascore_hilbert = unidas.adapter("xdas.DataArray")(hilbert)

        patch_hilberto = dascore_hilbert(dascore_patch)
        assert patch_hilberto.shape == dascore_patch.shape

        # The dimensions and coordinates should not have been changed.
        assert dascore_patch.dims == patch_hilberto.dims
        for dim in dascore_patch.dims:
            coord_1 = dascore_patch.get_array(dim)
            coord_2 = patch_hilberto.get_array(dim)
            assert np.all(coord_1 == coord_2)

    def test_readme_3(self, dascore_patch):
        """The third tests for readme code."""

        @unidas.adapter("daspy.Section")
        def daspy_function(sec, **kwargs):
            """A useful daspy function"""
            return sec

        out = daspy_function(dascore_patch)
        assert isinstance(out, dc.Patch)
