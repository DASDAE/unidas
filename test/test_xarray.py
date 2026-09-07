"""Tests for xarray data and metadata interoperability."""

import datetime
import importlib
import subprocess
import sys

import dascore as dc
import numpy as np
import pytest
import xarray as xr

from unidas import ArrayCoordinate, BaseDAS, EvenlySampledCoordinate, adapter, convert


def assert_round_trip(data_array):
    """Check both BaseDAS validity and exact xarray metadata round-tripping."""
    base = convert(data_array, "unidas.BaseDAS")
    base.validate()
    out = convert(base, "xarray.DataArray")
    xr.testing.assert_identical(out, data_array)
    assert out.dtype == data_array.dtype
    return base


def test_rich_round_trip():
    """Preserve arbitrary dimensions, coordinate associations, and metadata."""
    data = np.arange(24, dtype=float).reshape(3, 2, 4)
    data[0, 0, 0] = np.nan
    array = xr.DataArray(
        data,
        dims=("time", "station", "unlabeled"),
        coords={
            "time": ("time", [0.0, 1.0, 3.0], {"units": "s", "long_name": "Time"}),
            "station": ["A", "B"],
            "quality": ("time", [1, 2, 3], {"units": None}),
            "location": (("time", "station"), np.arange(6).reshape(3, 2)),
            "source": "example",
        },
        attrs={"units": "strain/s", "description": "test", "name": "an attribute"},
        name="measurements",
    )
    base = assert_round_trip(array)
    assert "unlabeled" not in base.coords
    out = convert(base.transpose("unlabeled", "station", "time"), "xarray.DataArray")
    # BaseDAS permutes data axes while retaining each coordinate's dimension order.
    expected = array.transpose("unlabeled", "station", "time", transpose_coords=False)
    xr.testing.assert_identical(out, expected)
    out.attrs["description"] = "changed"
    out.coords["time"].attrs["long_name"] = "changed"
    assert array.attrs["description"] == "test"
    assert base.attrs["description"] == "test"
    assert base.coords["time"].attrs["long_name"] == "Time"


@pytest.mark.parametrize("shape", [(), (0,), (1,), (2, 0), (1, 3, 2)])
def test_unlabeled_shapes(shape):
    """Unlabeled scalar, empty, singleton, and higher-rank arrays remain so."""
    array = xr.DataArray(np.zeros(shape, dtype=np.int16), name="data")
    base = assert_round_trip(array)
    assert not base.coords


@pytest.mark.parametrize(
    "values",
    [
        np.array([], dtype="datetime64[ns]"),
        np.array(["2020-01-01T00:00:00.123456789"], dtype="datetime64[ns]"),
        np.array([0, 2, 7], dtype="timedelta64[ns]"),
        np.array([4.0, 1.0, -2.0]),
    ],
)
def test_coordinate_values(values):
    """Dense coordinates retain dtype, order, and exact time precision."""
    array = xr.DataArray(np.zeros(len(values)), dims="axis", coords={"axis": values})
    assert_round_trip(array)


def test_lazy_data_round_trip():
    """Passing through BaseDAS does not compute Dask data."""
    da = pytest.importorskip("dask.array")
    from dask import delayed

    @delayed
    def fail_if_computed():
        raise AssertionError("Conversion computed the data")

    data = da.from_delayed(fail_if_computed(), shape=(3,), dtype=float)
    array = xr.DataArray(data, dims="x")
    base = convert(array, "unidas.BaseDAS")
    base.validate()
    out = convert(base, "xarray.DataArray")
    assert base.data is data
    assert out.data is data


@pytest.mark.parametrize("example_name", tuple(dc.examples.EXAMPLE_PATCHES))
def test_dascore_examples_to_xarray(example_name):
    """DASCore examples preserve values, dimension order, and all coordinates."""
    patch = dc.get_example_patch(example_name)
    out = convert(patch, "xarray.DataArray")
    assert out.dims == patch.dims
    np.testing.assert_allclose(out.data, patch.data, equal_nan=True)
    assert set(out.coords) == set(patch.coords.coord_map)
    for name, coord in patch.coords.coord_map.items():
        assert out.coords[name].dims == patch.coords.dim_map[name]
        values = out.coords[name].values
        if values.dtype.kind in "mM":
            np.testing.assert_array_equal(values, patch.get_array(name))
        else:
            np.testing.assert_allclose(values, patch.get_array(name), equal_nan=True)
        units = str(coord.units) if coord.units is not None else None
        assert out.coords[name].attrs.get("units") == units


def test_xdas_scalar_coordinates_and_name(xarray_dataarray):
    """XDAS preserves scalar associations, dimensionless arrays, and names."""
    array = xarray_dataarray.assign_coords(source="example")
    out = convert(convert(array, "xdas.DataArray"), "xarray.DataArray")
    xr.testing.assert_equal(out, array)
    assert out.name == array.name
    assert out.coords["source"].dims == ()
    scalar = xr.DataArray(3.0, coords={"source": "example"}, name="value")
    out = convert(convert(scalar, "xdas.DataArray"), "xarray.DataArray")
    xr.testing.assert_identical(out, scalar)


def test_xdas_multidimensional_coordinate_error():
    """An unsupported coordinate error identifies its name and destination."""
    array = xr.DataArray(
        np.zeros((2, 3)),
        dims=("x", "y"),
        coords={"location": (("x", "y"), np.zeros((2, 3)))},
    )
    with pytest.raises(ValueError, match="xdas.*location.*multidimensional"):
        convert(array, "xdas.DataArray")


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
@pytest.mark.parametrize("missing", ["time", "distance"])
def test_missing_physical_coordinates(xarray_dataarray, target, missing):
    """Sampled destinations must not invent physical coordinate labels."""
    pytest.importorskip(target.split(".")[0])
    array = xarray_dataarray.drop_vars(missing)
    with pytest.raises(ValueError, match=f"{target}.*{missing!r} coordinate"):
        convert(array, target)


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
@pytest.mark.parametrize(
    "case", ["irregular", "empty", "relative", "zero_step", "dimensions"]
)
def test_sampled_target_errors(xarray_dataarray, target, case):
    """Unsupported sampling fails with a useful ValueError."""
    pytest.importorskip(target.split(".")[0])
    if case == "irregular":
        array = xarray_dataarray.assign_coords(distance=[0, 1, 3])
        message = "distance.*evenly sampled"
    elif case == "empty":
        array = xarray_dataarray.isel(time=slice(0, 0))
        message = "time.*nonempty"
    elif case == "relative":
        array = xarray_dataarray.assign_coords(time=np.arange(6))
        message = "absolute datetime"
    elif case == "zero_step":
        array = xarray_dataarray.assign_coords(distance=[0, 0, 0])
        message = "distance.*nonzero"
    else:
        array = xarray_dataarray.rename(time="frequency")
        message = "time and distance dimensions"
    with pytest.raises(ValueError, match=message):
        convert(array, target)


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("count", [10, 100, 1000])
def test_sampled_targets_float_spacing_and_transpose(
    xarray_dataarray, target, dtype, count
):
    """Floating-point spacing and reversed axis order convert correctly."""
    pytest.importorskip(target.split(".")[0])
    array = xarray_dataarray.reindex(
        distance=np.arange(count, dtype=dtype) * dtype(0.1), fill_value=0
    )
    out = convert(array.transpose(), target)
    np.testing.assert_array_equal(out.data, array.data)
    if target == "daspy.Section":
        assert out.fs == 500
        assert out.dx == pytest.approx(0.1)
    else:
        assert out.sampling_rate == 500
        assert out.channel_spacing == pytest.approx(0.1)


def test_adapter_uses_changed_xarray_coordinates(dascore_patch):
    """Xarray slicing and transposition are reflected in the returned Patch."""

    @adapter("xarray.DataArray")
    def trim_and_transpose(array):
        return array.isel(time=slice(0, 10)).transpose()

    out = trim_and_transpose(dascore_patch)
    expected = dascore_patch.select(time=(0, 10), samples=True).transpose()
    assert out.dims == expected.dims
    np.testing.assert_array_equal(out.data, expected.data)
    np.testing.assert_array_equal(out.get_array("time"), expected.get_array("time"))


def test_adapter_returns_xarray(xarray_dataarray):
    """A DASCore function accepts xarray and returns its changed coordinates."""

    @adapter("dascore.Patch")
    def trim(patch):
        return patch.select(time=(0, 3), samples=True)

    out = trim(xarray_dataarray)
    assert isinstance(out, xr.DataArray)
    expected = xarray_dataarray.isel(time=slice(0, 3))
    np.testing.assert_array_equal(out.data, expected.data)
    np.testing.assert_array_equal(out.time, expected.time)


def test_xarray_identity_and_dataset_exclusion(xarray_dataarray):
    """Identity conversion works and Dataset is not registered."""
    assert convert(xarray_dataarray, "xarray.DataArray") is xarray_dataarray
    with pytest.raises(ValueError, match="No conversion path"):
        convert(xarray_dataarray.to_dataset(), "unidas.BaseDAS")


def test_missing_xarray_dependency(monkeypatch, xarray_dataarray):
    """Missing xarray raises the existing actionable optional-import error."""
    base = convert(xarray_dataarray, "unidas.BaseDAS")
    original = importlib.import_module

    def import_without_xarray(name, *args, **kwargs):
        if name == "xarray":
            raise ImportError("xarray unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", import_without_xarray)
    with pytest.raises(
        ImportError, match="xarray.*not installed.*https://docs.xarray.dev"
    ):
        convert(base, "xarray.DataArray")


def test_import_does_not_load_xarray():
    """Importing unidas retains its NumPy-only runtime requirement."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import unidas; assert 'xarray' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("start", "step", "expected"),
    [
        (
            datetime.datetime(
                2020, 1, 1, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
            ),
            0.000001,
            np.array(
                ["2019-12-31T22:00:00.000000", "2019-12-31T22:00:00.000001"],
                dtype="datetime64[ns]",
            ),
        ),
        (
            np.datetime64("2020-01-01T00:00:00.123456789"),
            np.timedelta64(1, "ns"),
            np.array(
                ["2020-01-01T00:00:00.123456789", "2020-01-01T00:00:00.123456790"],
                dtype="datetime64[ns]",
            ),
        ),
    ],
)
def test_sampled_datetime_precision(start, step, expected):
    """Expanding sampled datetimes uses exact integer offsets and UTC."""
    coord = EvenlySampledCoordinate(
        step=step, tie_values=(start, start), tie_indices=(0, 1), dims=("time",)
    )
    np.testing.assert_array_equal(coord.to_xarray_coord().values, expected)


def test_sampled_singleton_and_gaps():
    """A singleton keeps its value; gapped coordinates retain their rejection."""
    coord = EvenlySampledCoordinate(
        step=np.nan, tie_values=(3.0, 3.0), tie_indices=(0, 0), dims=("x",)
    )
    np.testing.assert_array_equal(coord.to_xarray_coord().values, [3.0])
    coord.tie_values = (0, 1, 3)
    with pytest.raises(NotImplementedError, match="gaps"):
        coord.to_xarray_coord()


def test_validate_auxiliary_shape():
    """BaseDAS catches incorrectly associated multidimensional coordinates."""
    base = BaseDAS(
        data=np.zeros((2, 3)),
        coords={"aux": ArrayCoordinate(data=np.zeros((3, 2)), dims=("x", "y"))},
        dims=("x", "y"),
        attrs={},
    )
    with pytest.raises(AssertionError):
        base.validate()


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
@pytest.mark.parametrize("axis", ["time", "distance"])
def test_nonfinite_singleton_axis(xarray_dataarray, target, axis):
    """A single invalid coordinate cannot define physical sampling."""
    pytest.importorskip(target.split(".")[0])
    array = xarray_dataarray.isel({axis: slice(0, 1)})
    value = np.datetime64("NaT", "ns") if axis == "time" else np.nan
    array = array.assign_coords({axis: [value]})
    with pytest.raises(ValueError, match=f"{axis}.*finite"):
        convert(array, target)


def test_string_time_axis_error(xarray_dataarray):
    """String labels must not be interpreted as physical time samples."""
    array = xarray_dataarray.assign_coords(time=list("abcdef"))
    with pytest.raises(ValueError, match="daspy.*time.*numeric or time axis"):
        convert(array, "daspy.Section")


@pytest.mark.parametrize("coordinate", ["source", "location"])
def test_dascore_coordinate_support(coordinate):
    """Honor native support, or identify the coordinate when DASCore rejects it."""
    values = np.asarray(3.0) if coordinate == "source" else np.zeros((2, 3))
    coord_dims = () if coordinate == "source" else ("x", "y")
    coords = {"x": [0, 1], "y": [0, 1, 2], coordinate: (coord_dims, values)}
    array = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"), coords=coords)
    # DASCore's ability to construct these coordinates varies by version.
    try:
        expected = dc.Patch(data=array.data, dims=array.dims, coords=coords)
    except (TypeError, ValueError):
        with pytest.raises(ValueError, match=f"dascore.*{coordinate!r}"):
            convert(array, "dascore.Patch")
    else:
        out = convert(array, "dascore.Patch")
        np.testing.assert_array_equal(
            out.get_array(coordinate), expected.get_array(coordinate)
        )
        assert out.coords.dim_map[coordinate] == expected.coords.dim_map[coordinate]


def test_dascore_coordinate_units_are_portable(dascore_patch):
    """DASCore units are serializable strings and still round-trip as units."""
    array = convert(dascore_patch, "xarray.DataArray")
    assert isinstance(array.distance.attrs["units"], str)
    # Test numeric-coordinate I/O without imposing a CF time-encoding policy.
    assert array.distance.to_netcdf()
    out = convert(array, "dascore.Patch")
    for name, coord in dascore_patch.coords.coord_map.items():
        assert out.coords.coord_map[name].units == coord.units


def test_daspy_uses_coordinates_over_conflicting_attrs(xarray_dataarray):
    """User metadata cannot override the physical sampling or data buffer."""
    array = xarray_dataarray.assign_attrs(
        data="metadata",
        fs=123,
        dx=99,
        start_distance=200,
        start_time="metadata",
        gauge_length=25,
    )
    out = convert(array, "daspy.Section")
    np.testing.assert_array_equal(out.data, array.data)
    assert out.fs == 500
    assert out.dx == pytest.approx(0.1)
    assert out.start_distance == 0
    assert out.start_time.utc().to_datetime() == datetime.datetime(
        2020, 1, 1, microsecond=123456, tzinfo=datetime.UTC
    )
    assert out.gauge_length == 25


def test_lightguide_rounds_channel_offset(xarray_dataarray):
    """Lightguide retains its documented integer-channel starting offset."""
    pytest.importorskip("lightguide")
    array = xarray_dataarray.assign_coords(distance=[0.04, 0.14, 0.24])
    blast = convert(array, "lightguide.Blast")
    assert blast.start_channel == 0
    out = convert(blast, "xarray.DataArray")
    np.testing.assert_allclose(out.distance, [0, 0.1, 0.2])


@pytest.mark.parametrize(("count", "span"), [(2, 10), (4, 10), (4, 12)])
def test_xdas_interpolation_preserves_labels(count, span):
    """Short and rounded datetime interpolation retains native labels exactly."""
    xdas = pytest.importorskip("xdas")
    ties = np.datetime64("2020-01-01", "ns") + np.array(
        [0, span], dtype="timedelta64[ns]"
    )
    array = xdas.DataArray(
        np.arange(count),
        dims=("time",),
        coords={"time": {"tie_indices": [0, count - 1], "tie_values": ties}},
    )
    out = convert(array, "xarray.DataArray")
    np.testing.assert_array_equal(out.time.values, array.coords["time"].values)
    np.testing.assert_array_equal(out.data, array.data)


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
def test_sampled_targets_reject_swapped_associations(target):
    """Coordinate names cannot override their actual dimension associations."""
    pytest.importorskip(target.split(".")[0])
    array = xr.DataArray(
        np.zeros((2, 3)),
        dims=("distance", "time"),
        coords={
            "time": (
                "distance",
                np.datetime64("2020-01-01", "ns")
                + np.arange(2) * np.timedelta64(1, "s"),
            ),
            "distance": ("time", [0, 1, 2]),
        },
    )
    with pytest.raises(ValueError, match="time.*associated with dimension 'time'"):
        convert(array, target)


@pytest.mark.parametrize(
    ("values", "step"),
    [
        (np.array([2, 1, 0], dtype="uint64"), -1),
        (np.array([-128, 0], dtype="int8"), 128),
    ],
)
def test_integer_spacing_does_not_overflow(values, step):
    """Coordinate differences retain their mathematical value across dtypes."""
    assert ArrayCoordinate(values).get_step() == step


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
def test_sampled_targets_integer_spacing(xarray_dataarray, target):
    """Narrow integer coordinates do not become negative sampling intervals."""
    pytest.importorskip(target.split(".")[0])
    array = xarray_dataarray.isel(distance=slice(0, 2)).assign_coords(
        distance=np.array([-128, 0], dtype="int8")
    )
    out = convert(array, target)
    step = out.dx if target == "daspy.Section" else out.channel_spacing
    assert step == 128


def test_float32_jitter_is_not_rounding():
    """A coordinate perturbation larger than storage rounding stays irregular."""
    values = np.arange(100, dtype=np.float32) * np.float32(0.1)
    values[50] += 0.001
    with pytest.raises(ValueError, match="evenly sampled"):
        ArrayCoordinate(values).get_step()


def test_scalar_coordinate_named_like_dimension():
    """A scalar coordinate may share a dimension name without labeling its axis."""
    assert_round_trip(xr.DataArray([1, 2], dims="x", coords={"x": 3.0}))


@pytest.mark.parametrize("rate", [1024, 3000])
@pytest.mark.parametrize("source", ["xarray", "xdas"])
@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
def test_fractional_nanosecond_sampling(rate, source, target):
    """Common sampling rates survive nanosecond rounding of timestamp labels."""
    pytest.importorskip(target.split(".")[0])
    count = rate + 1
    start = np.datetime64("2020-01-01", "ns")
    if source == "xdas":
        xdas = pytest.importorskip("xdas")
        array = xdas.DataArray(
            np.zeros((2, count)),
            dims=("distance", "time"),
            coords={
                "distance": [0, 1],
                "time": {
                    "tie_indices": [0, count - 1],
                    "tie_values": [start, start + np.timedelta64(1, "s")],
                },
            },
        )
    else:
        offsets = np.rint(np.arange(count) * 1_000_000_000 / rate).astype(
            "timedelta64[ns]"
        )
        array = xr.DataArray(
            np.zeros((2, count)),
            dims=("distance", "time"),
            coords={"distance": [0, 1], "time": start + offsets},
        )
    out = convert(array, target)
    out_rate = out.fs if target == "daspy.Section" else out.sampling_rate
    assert out_rate == pytest.approx(rate)
    np.testing.assert_array_equal(out.data, array.data)


def test_datetime_sampling_rejects_actual_jitter(xarray_dataarray):
    """Clock jitter larger than timestamp quantization is still irregular."""
    time = xarray_dataarray.time.values.copy()
    time[3] += np.timedelta64(10, "us")
    array = xarray_dataarray.assign_coords(time=time)
    with pytest.raises(ValueError, match="time.*evenly sampled"):
        convert(array, "daspy.Section")


def test_dascore_unknown_units_identify_coordinate():
    """An unknown coordinate unit reports both the destination and coordinate."""
    array = xr.DataArray(
        np.zeros(3),
        dims="distance",
        coords={"distance": ("distance", [0, 1, 2], {"units": "not_a_unit"})},
    )
    with pytest.raises(ValueError, match="dascore.*distance"):
        convert(array, "dascore.Patch")
