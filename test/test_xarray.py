"""Tests for xarray data and metadata interoperability."""

import datetime
import importlib
import subprocess
import sys
from fractions import Fraction

import dascore as dc
import numpy as np
import pytest
import xarray as xr
from conftest import needs_run_tables

from unidas import (
    BaseDAS,
    Categorical,
    CoordinateError,
    NumericND,
    adapter,
    concat,
    convert,
)
from unidas.converters.dascore import from_dascore_coord, to_dascore_coord
from unidas.converters.xarray import to_xarray_coord
from unidas.converters.xdas import to_xdas_coord
from unidas.core import coord_from_labels, coord_step, sampled_coord, time_to_float


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
@needs_run_tables
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
        # A time states its resolution in its dtype, and xarray spends the
        # units attribute on saying how to store one.
        expected = None if values.dtype.kind in "mM" else units
        assert out.coords[name].attrs.get("units") == expected


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
    with pytest.raises(ValueError, match=r"xdas.*location.*multidimensional"):
        convert(array, "xdas.DataArray")


@pytest.mark.parametrize("target", ["daspy.Section", "lightguide.Blast"])
@pytest.mark.parametrize("missing", ["time", "distance"])
def test_missing_physical_coordinates(xarray_dataarray, target, missing):
    """Sampled destinations must not invent physical coordinate labels."""
    pytest.importorskip(target.split(".")[0])
    array = xarray_dataarray.drop_vars(missing)
    with pytest.raises(ValueError, match=rf"{target}.*{missing!r} coordinate"):
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


@needs_run_tables
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
        ImportError, match=r"xarray.*not installed.*https://docs.xarray.dev"
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
    coord = sampled_coord(start, step, 2, dims=("time",))
    variable = to_xarray_coord(coord, ("time",))
    np.testing.assert_array_equal(variable.values, expected)


def test_sampled_singleton_keeps_its_value():
    """A single sample is its own label, whatever spacing is claimed for it."""
    coord = NumericND.from_array(np.asarray([3.0]), dims=("x",))
    np.testing.assert_array_equal(to_xarray_coord(coord, ("x",)).values, [3.0])


def test_validate_auxiliary_shape():
    """BaseDAS catches incorrectly associated multidimensional coordinates."""
    base = BaseDAS(
        data=np.zeros((2, 3)),
        coords={"aux": NumericND.from_array(np.zeros((3, 2)), dims=("x", "y"))},
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
    with pytest.raises(ValueError, match=rf"{axis}.*finite"):
        convert(array, target)


def test_string_time_axis_error(xarray_dataarray):
    """String labels must not be interpreted as physical time samples."""
    array = xarray_dataarray.assign_coords(time=list("abcdef"))
    with pytest.raises(ValueError, match=r"daspy.*time.*numeric or time axis"):
        convert(array, "daspy.Section")


@pytest.mark.parametrize("coordinate", ["source", "location"])
@needs_run_tables
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
        with pytest.raises(ValueError, match=rf"dascore.*{coordinate!r}"):
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
    with pytest.raises(ValueError, match=r"time.*associated with dimension 'time'"):
        convert(array, target)


@pytest.mark.parametrize(
    ("values", "step"),
    [
        (np.array([3, 2, 1], dtype="uint64"), -1),
        (np.array([-128, 0], dtype="int8"), 128),
    ],
)
def test_integer_spacing_does_not_overflow(values, step):
    """Coordinate differences retain their mathematical value across dtypes."""
    assert coord_step(NumericND.from_array(values)) == step


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
        coord_step(NumericND.from_array(values))


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
        offsets = np.rint(
            np.arange(count, dtype=np.int64) * 1_000_000_000 / rate
        ).astype("timedelta64[ns]")
        assert offsets[-1] == np.timedelta64(1, "s")
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
    with pytest.raises(ValueError, match=r"time.*evenly sampled"):
        convert(array, "daspy.Section")


def test_dascore_unknown_units_identify_coordinate():
    """An unknown coordinate unit reports both the destination and coordinate."""
    array = xr.DataArray(
        np.zeros(3),
        dims="distance",
        coords={"distance": ("distance", [0, 1, 2], {"units": "not_a_unit"})},
    )
    with pytest.raises(ValueError, match=r"dascore.*distance"):
        convert(array, "dascore.Patch")


def exact_labels(start, rate, count, phase=0, stride=1):
    """The labels of a grid, quantized with exact arithmetic in the test."""
    period = Fraction(1_000_000_000, rate)
    ideal = (Fraction(phase) + index * stride * period for index in range(count))
    ticks = [int(x) for x in ideal]
    return np.asarray(start).astype("datetime64[ns]") + np.asarray(
        ticks, dtype="timedelta64[ns]"
    )


class StubCoordinate:
    """
    A provider's coordinate, which unidas reads without knowing its library.

    The coordinate is stated as the run table a provider holds: one row per
    run, with its spacing as a numerator over a denominator of the
    coordinate's own ticks and the phase of its first sample.
    """

    values = property(lambda self: pytest.fail("The labels were spelled out."))

    def __init__(self, coord):
        self.runs = coord.runs
        self.labels = coord.labels
        self.dtype = coord.dtype
        self.units = coord.units


class StubIndex(xr.Index):
    """An xarray index which states the coordinate it serves."""

    def __init__(self, coordinate, variables):
        self.coordinate = coordinate
        self._variables = variables

    @classmethod
    def from_variables(cls, variables, *, options):
        """Build the index from the variables it is set on."""
        return cls(options["coordinate"], dict(variables))

    def create_variables(self, variables=None):
        """Return the variables this index stands for."""
        return self._variables


def indexed_by(coordinate, labels, name="time", attrs=None):
    """A DataArray whose index states the coordinate given."""
    array = xr.DataArray(
        np.zeros(len(labels)),
        dims=(name,),
        coords={name: (name, labels, dict(attrs or {}))},
    )
    return array.drop_indexes(name).set_xindex(name, StubIndex, coordinate=coordinate)


class StubStepCoordinate:
    """A provider's coordinate which states a step and no run table."""

    def __init__(self, step):
        self.step = step


class StubOldCoordinate:
    """A DASCore coordinate from before run tables: labels and no runs."""

    def __init__(self, values, units=None):
        self.values = np.asarray(values)
        self.dtype = self.values.dtype
        self.units = units
        self.step = None


def test_exact_grid_labels_do_not_drift():
    """A rate with no whole-tick period is counted from the origin."""
    start = np.datetime64("2020-01-01", "ns")
    coord = NumericND.from_run(start, Fraction(1, 1024), 2048)
    labels = coord.values
    np.testing.assert_array_equal(labels, exact_labels(start, 1024, 2048))
    # The rounded step is 976562 ns, so stepping by it loses a microsecond
    # over this many samples; the exact grid does not.
    assert labels[-1] != start + np.timedelta64(976562, "ns") * 2047


def test_exact_grid_keeps_the_phase_it_was_sliced_at():
    """A slice of a fractional grid stays on the grid it was cut from."""
    # Every third sample of a 1024 Hz grid, from the seventh: the first label
    # is the seventh, and the origin sits half a tick before the grid's own.
    origin = np.datetime64("2020-01-01", "ns")
    coord = NumericND.from_run(origin, Fraction(1, 1024), 2048)[7::3]
    start = exact_labels(origin, 1024, 8)[-1]
    assert coord.start == start
    expected = exact_labels(start, 1024, len(coord), phase=Fraction(1, 2), stride=3)
    np.testing.assert_array_equal(coord.values, expected)
    # A grid stated as 1953125/1 with no phase carries the same labels but a
    # different origin, which is what a neighbouring run is fused against.
    assert tuple(coord.runs[0])[2:] == (5859375, 2, 1)


def segmented_of(start, step, size, count, apart):
    """A coordinate of runs, each `size` long and `apart` steps from the next."""
    runs = [
        NumericND.from_run(start + step * apart * i, step, size, dims=("time",))
        for i in range(count)
    ]
    return concat(*runs)


def test_segmented_coordinate_describes_itself():
    """A coordinate of runs states its length and where it begins."""
    start = np.datetime64("2020-01-01", "ns")
    coord = segmented_of(start, np.timedelta64(4, "ms"), 10, 3, 20)
    assert len(coord) == 30
    assert coord.shape == (30,)
    assert coord.runs_count == 3 and coord.holes
    assert coord.start == start


@needs_run_tables
def test_runs_reach_dascore_as_runs():
    """A coordinate with holes crosses to DASCore as the runs it is."""
    start = np.datetime64("2020-01-01", "ns")
    coord = segmented_of(start, np.timedelta64(4, "ms"), 10, 3, 20)
    out = to_dascore_coord(coord)
    np.testing.assert_array_equal(np.asarray(out.values), coord.values)
    assert out.runs_count == 3
    # And back again, run for run.
    assert np.array_equal(from_dascore_coord(out, ("time",)).runs, coord.runs)


@needs_run_tables
def test_stored_runs_and_categories_reach_dascore():
    """Labels which follow no grid, and labels which are not numbers, cross."""
    start = np.datetime64("2020-01-01", "ns")
    jitter = start + np.asarray([0, 3, 7, 8, 14]) * np.timedelta64(1, "ns")
    stored = concat(
        NumericND.from_run(start - np.timedelta64(1, "s"), np.timedelta64(1, "ms"), 4),
        NumericND.from_array(jitter),
    )
    out = to_dascore_coord(stored)
    np.testing.assert_array_equal(np.asarray(out.values), stored.values)
    assert from_dascore_coord(out, ("time",)) == stored
    labels = Categorical.from_labels(["n1", "n1", "s2"])
    back = from_dascore_coord(to_dascore_coord(labels), ("station",))
    assert isinstance(back, Categorical)
    np.testing.assert_array_equal(back.values, labels.values)


def test_a_fractional_grid_reaches_xdas_as_its_labels():
    """Tie points state one spacing, which a fractional grid has not."""
    xdas = pytest.importorskip("xdas")
    start = np.datetime64("2020-01-01", "ns")
    coord = NumericND.from_run(start, Fraction(1, 1024), 6, dims=("time",))
    out = to_xdas_coord(coord, ("time",))
    assert isinstance(out, xdas.DenseCoordinate)
    np.testing.assert_array_equal(np.asarray(out.values), exact_labels(start, 1024, 6))


def test_segmented_coordinate_states_its_runs():
    """A coordinate with a hole keeps the runs on either side of it."""
    start = np.datetime64("2020-01-01", "ns")
    step = np.timedelta64(4, "ms")
    coord = segmented_of(start, step, 10, 2, 20)
    assert len(coord) == 20
    expected = np.concatenate(
        [start + np.arange(10) * step, start + (20 + np.arange(10)) * step]
    )
    np.testing.assert_array_equal(coord.values, expected)
    variable = to_xarray_coord(coord, ("time",))
    np.testing.assert_array_equal(variable.values, expected)


def test_extraction_reads_structure_not_labels():
    """A compact coordinate is read from what it states, not from its values."""
    start = np.datetime64("2020-01-01", "ns")
    grid = NumericND.from_run(start, Fraction(1, 1024), 8)
    out = from_dascore_coord(StubCoordinate(grid), ("time",))
    assert out.dims == ("time",)
    assert tuple(out.runs[0])[2:] == (1953125, 2, 0)
    np.testing.assert_array_equal(out.values, exact_labels(start, 1024, 8))
    # A phase is carried as stated, not recomputed from the reduced step.
    phased = NumericND.from_run(start, Fraction(1, 1024), 2048)[7::3]
    assert tuple(from_dascore_coord(StubCoordinate(phased), ()).runs[0])[2:] == (
        5859375,
        2,
        1,
    )
    # A table of several runs is read as the several runs it is.
    runs = from_dascore_coord(
        StubCoordinate(segmented_of(start, np.timedelta64(1, "ms"), 3, 2, 10)), ()
    )
    assert runs.runs_count == 2


def test_extraction_of_an_integer_grid():
    """A grid which is not a time counts ticks of its own units."""
    coord = from_dascore_coord(
        StubCoordinate(NumericND.from_run(0, 3, 4)), ("channel",)
    )
    np.testing.assert_array_equal(coord.values, np.arange(4) * 3)
    assert coord_step(coord) == 3
    # Halves of its own units, rather than of a second: every other sample
    # of a grid spaced three apart, labeled by the whole unit below.
    half = from_dascore_coord(
        StubCoordinate(NumericND.from_run(0, Fraction(3, 2), 5)), ("channel",)
    )
    assert coord_step(half) == Fraction(3, 2)
    np.testing.assert_array_equal(half.values, [0, 1, 3, 4, 6])


def test_an_index_is_read_from_the_coordinate_it_states():
    """A lazy index hands over its coordinate instead of its labels."""
    start = np.datetime64("2020-01-01", "ns")
    labels = exact_labels(start, 1024, 6)
    stub = StubCoordinate(NumericND.from_run(start, Fraction(1, 1024), 6))
    base = convert(indexed_by(stub, labels), "unidas.BaseDAS")
    coord = base.coords["time"]
    assert tuple(coord.runs[0])[2:] == (1953125, 2, 0)
    np.testing.assert_array_equal(coord.values, labels)


@pytest.mark.parametrize("stating", ["its labels", "a step"])
def test_an_index_stating_no_runs_keeps_its_labels(stating):
    """Labels are only given up for a run table which can state them."""
    # Float labels are not ticks of anything, so a range over them describes
    # them less exactly than they describe themselves, float32 above all.
    labels = np.float32(0.5) + np.arange(5, dtype=np.float32) * np.float32(0.1)
    # An index whose coordinate is a provider object stating a spacing and
    # no table says nothing a run can be read from either.
    coordinate = labels if stating == "its labels" else StubStepCoordinate(0.1)
    base = convert(indexed_by(coordinate, labels, "distance"), "unidas.BaseDAS")
    out = base.coords["distance"].values
    assert out.dtype == labels.dtype
    np.testing.assert_array_equal(out, labels)


def test_an_index_keeps_the_units_the_source_stated():
    """Units xarray states beside a coordinate are not DASCore's to drop."""
    labels = (np.arange(5) * np.timedelta64(1, "s")).astype("timedelta64[ns]")
    stub = StubCoordinate(NumericND.from_array(labels))
    stated = indexed_by(stub, labels, "time", attrs={"units": "s"})
    lazy = convert(stated, "unidas.BaseDAS").coords["time"]
    plain = coord_from_labels(labels, ("time",), units="s")
    # The lazy index and the labels beside it answer alike.
    assert lazy.units == plain.units == "s"


def test_a_dascore_without_run_tables_is_refused_by_version():
    """A coordinate stating no runs names the DASCore which cannot."""
    with pytest.raises(CoordinateError, match=r"DASCore 1\.0"):
        from_dascore_coord(StubOldCoordinate(np.arange(5)), ("x",))
    # Text is read from its labels, which every DASCore states.
    out = from_dascore_coord(StubOldCoordinate(np.asarray(["a", "b"])), ("x",))
    assert isinstance(out, Categorical)


def test_the_installed_dascore_is_read_or_named():
    """Whatever DASCore is installed, a patch is read or refused by name."""
    patch = dc.get_example_patch()
    labels = np.arange(3) * 1.5
    array = xr.DataArray(np.zeros(3), dims="x", coords={"x": labels})
    if hasattr(dc.core.coords, "NumericND"):
        assert convert(patch, "unidas.BaseDAS").coords
        assert convert(array, "dascore.Patch").coords
        return
    # Both directions name the version rather than the keyword or the
    # attribute the older DASCore happens to be missing.
    with pytest.raises(CoordinateError, match=r"DASCore 1\.0"):
        convert(patch, "unidas.BaseDAS")
    with pytest.raises(ValueError, match=r"DASCore 1\.0"):
        convert(array, "dascore.Patch")


def test_an_index_stating_runs_is_read_from_them():
    """Runs with gaps between them are what labels alone cannot say."""
    start = np.datetime64("2020-01-01", "ns")
    step = np.timedelta64(4, "ms")
    coord = segmented_of(start, step, 3, 2, 10)
    base = convert(indexed_by(StubCoordinate(coord), coord.values), "unidas.BaseDAS")
    assert base.coords["time"].runs_count == 2
    assert base.coords["time"].holes


def test_a_time_is_counted_in_nanoseconds():
    """A time arrives in whatever unit it likes and is held in one."""
    start = np.zeros(1, dtype="datetime64[10us]")[0]
    labels = start + np.arange(6).astype("timedelta64[10us]")
    coord = NumericND.from_array(labels)
    assert coord.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(coord.values, labels.astype("datetime64[ns]"))
    # Five halves of a ten-microsecond tick is 25 us, which is 40 kHz.
    rate = NumericND.from_run(start, Fraction(1, 40000), 6)
    assert 1 / time_to_float(coord_step(rate)) == 40000


def test_an_exactly_representable_rate_stays_exact():
    """A period is divided while it is still rational."""
    start = np.datetime64("2020-01-01", "ns")
    coord = NumericND.from_run(start, Fraction(1, 49), 98)
    # 1/49 s is 20408163.27 ns; through a float period the rate reads as
    # 49.00000000000001, which will not join a recording made at 49.
    assert float(1 / coord_step(coord)) == 49.0


@needs_run_tables
def test_float_runs_dascore_would_join_travel_as_their_labels():
    """A join which moves a label further than a bit or two is not one."""
    # Two runs a ten-billionth apart: unidas keeps them apart, DASCore
    # joins them within a tolerance of its own and relabels the second.
    coord = concat(
        NumericND.from_run(1000.0, 0.5, 5),
        NumericND.from_run(1000.0 + 2.5 + 1e-10, 0.5, 5),
    )
    assert coord.runs_count == 2
    out = to_dascore_coord(coord)
    # The labels cross exactly; what cannot cross is the table which
    # states them, so DASCore is given the labels themselves instead.
    np.testing.assert_array_equal(np.asarray(out.values), coord.values)
    back = from_dascore_coord(out, ("x",))
    assert back.labels is not None
    np.testing.assert_array_equal(back.values, coord.values)


@needs_run_tables
def test_a_declared_grid_travels_with_the_labels_it_describes():
    """The step a source declared is what says which samples are missing."""
    steps = np.where(np.arange(2000) % 2, 10, 20)
    values = np.concatenate([[0], np.cumsum(steps)]).astype("int64")
    coord = NumericND.from_array(values, step=10)
    assert coord.runs_count == 1 and coord.step == 10
    out = to_dascore_coord(coord)
    assert out.step == 10
    back = from_dascore_coord(out, ("distance",))
    assert back.step == 10 and back == coord
    np.testing.assert_array_equal(back.values, values)


@needs_run_tables
def test_a_whole_tick_grid_travels_as_a_grid():
    """A grid of whole nanoseconds is stated, and its labels are its labels."""
    start = np.datetime64("2020-01-01", "ns")
    step = np.timedelta64(4_000_000, "ns")
    labels = start + np.arange(6) * step
    stub = StubCoordinate(NumericND.from_run(start, step, 6))
    coord = convert(indexed_by(stub, labels), "unidas.BaseDAS").coords["time"]
    out = to_dascore_coord(coord)
    np.testing.assert_array_equal(np.asarray(out.values), labels)
    assert out.step == step


@needs_run_tables
def test_a_fractional_grid_travels_as_a_grid():
    """The resolution DASCore counts in is stated, not spelled out."""
    start = np.datetime64("2020-01-01", "ns")
    labels = exact_labels(start, 1024, 6)
    stub = StubCoordinate(NumericND.from_run(start, Fraction(1, 1024), 6))
    coord = convert(indexed_by(stub, labels), "unidas.BaseDAS").coords["time"]
    out = to_dascore_coord(coord)
    assert (out.step_numerator, out.step_denominator) == (1953125, 2)
    np.testing.assert_array_equal(out.values, labels)


def test_time_coordinates_state_no_units_attribute():
    """Xarray reserves a time's units attribute for how it is stored."""
    start = np.datetime64("2020-01-01", "ns")
    coord = NumericND.from_array(
        start + np.arange(3) * np.timedelta64(4, "ms"), units="1 s", dims=("time",)
    )
    assert "units" not in to_xarray_coord(coord, ("time",)).attrs
    numeric = NumericND.from_array(np.arange(3.0), units="1 m", dims=("distance",))
    assert to_xarray_coord(numeric, ("distance",)).attrs["units"] == "1 m"


def dascore_grid(rate=1024, size=2048):
    """A DASCore grid stating a rate with no whole-tick period, or skip."""
    start = dc.to_datetime64("2020-01-01")
    try:
        coord = dc.core.get_coord(start=start, step=Fraction(1, rate), shape=(size,))
    except (TypeError, ValueError):
        coord = None
    if getattr(coord, "step_exact", None) != Fraction(1, rate):
        pytest.skip("This DASCore does not state exact coordinate grids.")
    return coord


def dascore_patch_along(coord):
    """A DASCore patch whose time coordinate is the one given."""
    coords = {"distance": np.arange(3), "time": coord}
    data = np.zeros((3, len(coord)))
    return dc.Patch(data=data, dims=("distance", "time"), coords=coords)


@pytest.mark.parametrize(
    "index", [slice(None), slice(7, 1000, 3), slice(None, None, -1)]
)
@needs_run_tables
def test_dascore_exact_grid_round_trip(index):
    """A fractional rate returns as the grid it was, phase included."""
    coord = dascore_grid()[index]
    base = convert(dascore_patch_along(coord), "unidas.BaseDAS")
    assert convert(base, "dascore.Patch").get_coord("time") == coord
    # A destination without exact grids still receives exact labels.
    np.testing.assert_array_equal(base.coords["time"].values, coord.values)


@needs_run_tables
def test_dascore_segmented_round_trip():
    """A gap in a DASCore coordinate is carried across as a gap."""
    step, start = dc.to_timedelta64(0.004), dc.to_datetime64("2020-01-01")
    first = dc.core.get_coord(start=start, step=step, shape=(100,))
    second = dc.core.get_coord(start=first.max() + 10 * step, step=step, shape=(50,))
    concat_coords = getattr(dc.core.coords, "concat_coords", None)
    coord = None if concat_coords is None else concat_coords(first, second)
    if getattr(coord, "runs_count", 1) < 2:
        pytest.skip("This DASCore does not state coordinate runs.")
    base = convert(dascore_patch_along(coord), "unidas.BaseDAS")
    assert base.coords["time"].runs_count == 2
    assert convert(base, "dascore.Patch").get_coord("time") == coord


@needs_run_tables
def test_dascore_lazy_index_is_read_from_its_coordinate():
    """A lazy index states its coordinate, and is read from that."""
    coord = dascore_grid()
    patch = dascore_patch_along(coord)
    try:
        array = patch.io.to_xarray(lazy_coords=True)
    except (AttributeError, TypeError):
        pytest.skip("This DASCore does not serve lazy xarray indexes.")
    if getattr(array.xindexes.get("time"), "coordinate", None) is None:
        pytest.skip("This DASCore does not serve lazy xarray indexes.")
    out = convert(array, "dascore.Patch").get_coord("time")
    assert out == coord


def test_null_patch_attributes_are_omitted(dascore_patch):
    """An attribute holding nothing says nothing, and some stores refuse it."""
    null = [i for i, v in dict(dascore_patch.attrs).items() if v is None]
    assert null, "The patch fixture states no null attribute to drop."
    array = convert(dascore_patch, "xarray.DataArray")
    assert not [i for i, v in array.attrs.items() if v is None]


@needs_run_tables
def test_unlabeled_dimension_stays_unlabeled():
    """A dimension DASCore only knows the length of gains no null labels."""
    patch = dc.Patch(
        data=np.zeros((3, 4)),
        dims=("distance", "time"),
        coords={"distance": np.arange(3)},
    )
    array = convert(patch, "xarray.DataArray")
    assert "time" not in array.coords
    assert convert(array, "dascore.Patch").dims == patch.dims


def test_adapter_converts_every_operand(dascore_patch):
    """A function of two patches takes two of whatever the caller holds."""

    @adapter("dascore.Patch")
    def add(patch, other):
        """Add two patches."""
        return patch + other

    array = convert(dascore_patch, "xarray.DataArray")
    for out in (add(array, array), add(array, other=array)):
        assert isinstance(out, xr.DataArray)
        np.testing.assert_allclose(out.data, dascore_patch.data * 2)


def test_adapter_passes_other_arguments_through(dascore_patch):
    """
    An argument which is not the caller's data is handed over as it is.

    A library's array type is used for ordinary options too -- a taper window
    is an `xarray.DataArray` like any other -- so only another operand of the
    kind the call was made on is taken to be more of the same data.
    """
    window = xr.DataArray(np.hanning(3), dims=("distance",))

    @adapter("dascore.Patch")
    def scale(patch, factor, taper, samples, kind=None):
        """Multiply a patch by a number."""
        assert isinstance(taper, xr.DataArray)
        assert isinstance(samples, np.ndarray)
        assert kind is dc.Patch
        return patch * factor

    out = scale(dascore_patch, 2.0, window, np.arange(3), kind=dc.Patch)
    np.testing.assert_allclose(out.data, dascore_patch.data * 2)


def test_adapter_leaves_an_option_the_target_cannot_hold(dascore_patch):
    """Being the caller's type is not being the caller's data."""

    @adapter("daspy.Section")
    def bandpass(section, taper):
        """Filter a section with a window."""
        return type(taper).__name__

    pytest.importorskip("daspy")
    array = convert(dascore_patch, "xarray.DataArray")
    # A window is an array of the caller's own class, and no section.
    window = xr.DataArray(np.hanning(8), dims=("window",))
    assert bandpass(array, window) == "DataArray"


def test_adapter_returns_a_class_of_the_target_unchanged(dascore_patch):
    """A class names a type; it is not an instance of one to convert."""

    @adapter("dascore.Patch")
    def get_type(patch):
        """Return the class the target names."""
        return dc.Patch

    assert get_type(dascore_patch) is dc.Patch


def test_runs_reach_xdas_as_tie_points_with_a_hole_between():
    """Whole-tick runs are pairs of tie points; a pair one apart is a hole."""
    start = np.datetime64("2020-01-01", "ns")
    step = np.timedelta64(4, "ms")
    coord = segmented_of(start, step, 10, 2, 20)
    xdas = pytest.importorskip("xdas")
    out = to_xdas_coord(coord, ("time",))
    assert isinstance(out, xdas.InterpCoordinate)
    assert list(out.tie_indices) == [0, 9, 10, 19]
    np.testing.assert_array_equal(np.asarray(out.values), coord.values)
    # One rate across every run is a rate xdas is told as well.
    assert out.sampling_interval == step
    # And they come back as the runs they were, not as one rate across a hole.
    back = convert(xdas.DataArray(np.zeros(20), coords={"time": out}), "unidas.BaseDAS")
    assert back.coords["time"].runs["length"].tolist() == [10, 10]
    np.testing.assert_array_equal(back.coords["time"].values, coord.values)
    # A rate across a hole is a claim about samples which are not there, so
    # a target needing one is told why rather than told nothing.
    with pytest.raises(ValueError, match="evenly sampled"):
        coord_step(coord)


def test_xdas_states_a_sampling_ratio_which_seeds_a_grid():
    """A rate xdas states beside its tie points is read back as the grid."""
    xdas = pytest.importorskip("xdas")
    start = np.datetime64("2020-01-01", "ns")
    step = np.timedelta64(4, "ms")
    # A lone sample states no spacing between tie points; the ratio does.
    lonely = xdas.InterpCoordinate.from_block(start, 1, step, dim="time")
    if "sampling_numerator" not in lonely.data:
        pytest.skip("This xdas states no sampling ratio beside its tie points.")
    out = convert(
        xdas.DataArray(np.zeros(1), coords={"time": lonely}), "unidas.BaseDAS"
    ).coords["time"]
    assert out.evenly_sampled and out.step == step
    # A rate whose period is not whole ticks labels its samples by rounding
    # where the run table floors, so the grid is only kept when it gives
    # back every label; here it does not, and the labels travel instead.
    fractional = xdas.InterpCoordinate.from_block(
        start, 1025, (np.timedelta64(1953125, "ns"), 2), dim="time"
    )
    out = convert(
        xdas.DataArray(np.zeros(1025), coords={"time": fractional}), "unidas.BaseDAS"
    ).coords["time"]
    np.testing.assert_array_equal(out.values, np.asarray(fractional.values))


def test_xdas_tie_points_no_run_can_hold_are_read_as_labels():
    """A spacing past any fraction of ticks describes nothing; labels do."""
    xdas = pytest.importorskip("xdas")
    huge = xdas.InterpCoordinate(
        {"tie_indices": [0, 2], "tie_values": [0.0, 1e308]}, dim="x"
    )
    out = convert(
        xdas.DataArray(np.zeros(3), coords={"x": huge}), "unidas.BaseDAS"
    ).coords["x"]
    assert out.labels is not None
    np.testing.assert_array_equal(out.values, np.asarray(huge.values))


# Coordinates of every shape a label can take, for the round trips below.
LABEL_CASES = {
    "float_tenths": np.arange(20) * 0.1,
    "float32": np.arange(20, dtype=np.float32) * np.float32(0.1),
    "int8": np.asarray([-100, -36, 28], dtype="int8"),
    "uint16": np.asarray([0, 20000, 40000], dtype="uint16"),
    "whole_tick_grid": np.datetime64("2020-01-01", "ns")
    + np.arange(20) * np.timedelta64(4, "ms"),
    "fractional_rate": np.datetime64("2020-01-01", "ns")
    + (np.arange(20) * 1953125 // 2).astype("timedelta64[ns]"),
    "jitter": np.datetime64("2020-01-01", "ns")
    + np.asarray([0, 3, 7, 8, 14]) * np.timedelta64(1, "ns"),
    "unsorted": np.asarray([3.0, 1.0, 2.0]),
    "text": np.asarray(["a", "b", "a"]),
    "days": np.datetime64("2020-01-01", "D") + np.arange(5).astype("timedelta64[D]"),
    "microseconds": np.datetime64("2020-01-01", "us")
    + np.arange(5).astype("timedelta64[us]"),
    "seconds": np.datetime64("2020-01-01", "s") + np.arange(5).astype("timedelta64[s]"),
    # Labels whose own description would leave the dtype it is held in.
    "int8_to_its_edge": np.asarray([125, 126, 127], dtype="int8"),
    "int64_far_apart": np.asarray([-9 * 10**18, 0, 9 * 10**18], dtype="int64"),
    "floats_past_their_bits": float(2**50) + np.arange(6) * 0.25,
}


def assert_labels_match(actual, expected):
    """
    The labels came back as they went out.

    A float coordinate is a grid fitted to its labels within a couple of
    their last bits, and its values are that grid's -- so float labels come
    back within the same tolerance rather than bit for bit. Every other
    dtype is exact.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    if expected.dtype.kind != "f":
        np.testing.assert_array_equal(actual, expected)
        return
    assert actual.shape == expected.shape
    bound = 2 * np.spacing(np.abs(expected))
    off = np.abs(actual - expected) > bound
    assert not np.any(off), f"{actual[off]} is not {expected[off]}"


@pytest.mark.parametrize("case", sorted(LABEL_CASES))
@pytest.mark.parametrize("target", ["xarray.DataArray", "xdas.DataArray"])
def test_every_label_survives_its_round_trip(case, target):
    """Whatever a destination makes of a coordinate, the labels come back."""
    pytest.importorskip(target.split(".")[0])
    labels = LABEL_CASES[case]
    array = xr.DataArray(np.zeros(len(labels)), dims="x", coords={"x": labels})
    # Through the hub explicitly: convert() hands back an object which is
    # already its target, so an xarray leg would otherwise never run.
    base = convert(array, "unidas.BaseDAS")
    out = convert(convert(base, target), "xarray.DataArray")
    assert_labels_match(out.x.values, labels)


@pytest.mark.parametrize(
    "target",
    [
        "xarray.DataArray",
        "xdas.DataArray",
        pytest.param("dascore.Patch", marks=needs_run_tables),
    ],
)
@pytest.mark.parametrize("rank", ["scalar", "two dimensional"])
def test_a_coordinate_of_any_rank_crosses_or_is_named(target, rank):
    """A scalar and an auxiliary coordinate cross, or are refused by name."""
    pytest.importorskip(target.split(".")[0])
    scalar = rank == "scalar"
    values = np.arange(6).reshape(2, 3) * 1.5
    coords = {"source": "station"} if scalar else {"location": (("x", "y"), values)}
    array = xr.DataArray(np.zeros((2, 3)), dims=("x", "y"), coords=coords)
    if not scalar and target == "xdas.DataArray":
        # XDAS holds one axis per coordinate, and says which it cannot.
        with pytest.raises(ValueError, match=r"location.*multidimensional"):
            convert(array, target)
        return
    base = convert(array, "unidas.BaseDAS")
    out = convert(convert(base, target), "xarray.DataArray")
    if scalar:
        assert out.coords["source"].shape == ()
        assert out.coords["source"].values[()] == "station"
    else:
        np.testing.assert_array_equal(out.coords["location"].values, values)


@pytest.mark.parametrize(
    "target",
    [
        "xarray.DataArray",
        "xdas.DataArray",
        pytest.param("dascore.Patch", marks=needs_run_tables),
    ],
)
def test_categories_reach_every_target_as_their_labels(target):
    """Text has no run table, so a destination is given the labels."""
    pytest.importorskip(target.split(".")[0])
    labels = np.asarray(["n1", "n1", "s2"])
    array = xr.DataArray(np.zeros(3), dims="x", coords={"x": labels})
    base = convert(array, "unidas.BaseDAS")
    assert isinstance(base.coords["x"], Categorical)
    out = convert(convert(base, target), "xarray.DataArray")
    np.testing.assert_array_equal(np.asarray(out.x.values), labels)


def test_a_multidimensional_categorical_is_refused_by_xdas():
    """Categories of two axes are no more an xdas coordinate than numbers."""
    pytest.importorskip("xdas")
    array = xr.DataArray(
        np.zeros((2, 2)),
        dims=("x", "y"),
        coords={"station": (("x", "y"), np.asarray([["a", "b"], ["c", "d"]]))},
    )
    with pytest.raises(ValueError, match=r"station.*multidimensional"):
        convert(array, "xdas.DataArray")


def test_a_descending_grid_reaches_xdas_as_its_labels():
    """Tie points interpolate between rising knots, which these are not."""
    xdas = pytest.importorskip("xdas")
    start = np.datetime64("2020-01-01", "ns")
    labels = start + np.arange(6)[::-1] * np.timedelta64(4, "ms")
    coord = NumericND.from_array(labels)
    assert coord.reverse_sorted and coord.evenly_sampled
    out = to_xdas_coord(coord, ("time",))
    assert isinstance(out, xdas.DenseCoordinate)
    np.testing.assert_array_equal(np.asarray(out.values), labels)
    array = xr.DataArray(np.zeros(6), dims="time", coords={"time": labels})
    back = convert(convert(array, "xdas.DataArray"), "xarray.DataArray")
    np.testing.assert_array_equal(np.asarray(back.time.values), labels)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        (np.asarray([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype="uint64"), "int64 range"),
        (
            np.asarray(
                ["1000-01-01", "1000-01-02", "1000-01-03"], dtype="datetime64[D]"
            ),
            "nanosecond range",
        ),
    ],
)
def test_labels_past_what_a_coordinate_counts_are_refused(labels, message):
    """A label no tick can count is named where a user meets it."""
    array = xr.DataArray(np.zeros(3), dims="x", coords={"x": labels})
    with pytest.raises(CoordinateError, match=message):
        convert(array, "unidas.BaseDAS")


@pytest.mark.parametrize("unit", ["us", "ms", "s", "D"])
def test_a_grid_of_coarser_ticks_keeps_its_instants(unit):
    """A coarser grid is counted in nanoseconds, naming the same instants."""
    labels = np.datetime64("2020-01-01", unit) + np.arange(5).astype(
        f"timedelta64[{unit}]"
    )
    coord = coord_from_labels(labels, ("time",))
    assert coord.dtype == np.dtype("datetime64[ns]")
    assert coord.evenly_sampled
    np.testing.assert_array_equal(coord.values, labels.astype("datetime64[ns]"))
    array = xr.DataArray(np.zeros(5), dims="time", coords={"time": labels})
    out = convert(convert(array, "unidas.BaseDAS"), "xarray.DataArray")
    # The instants are the same; the dtype a coordinate counts in is ns.
    np.testing.assert_array_equal(np.asarray(out.time.values), labels)
    assert out.time.dtype == np.dtype("datetime64[ns]")


# DASCore re-reads a stored coordinate from its labels when a patch is built
# on it, and its own range inference overflows on integer labels this near
# the edge of their dtype -- `dc.Patch(coords={"x": dc.core.get_coord(
# data=np.asarray([125, 126, 127], dtype="int8"))})` raises without unidas
# in the picture. Those labels reach DASCore as themselves; it is what
# DASCore then does with them that this cannot assert.
DASCORE_REREADS = ("int8_to_its_edge", "int64_far_apart")


@needs_run_tables
@pytest.mark.parametrize(
    "case", [x for x in sorted(LABEL_CASES) if x not in DASCORE_REREADS]
)
def test_every_label_survives_dascore(case):
    """A DASCore patch hands back the labels it was given, run table and all."""
    labels = LABEL_CASES[case]
    array = xr.DataArray(np.zeros(len(labels)), dims="x", coords={"x": labels})
    out = convert(convert(array, "dascore.Patch"), "xarray.DataArray")
    assert_labels_match(out.x.values, labels)


def test_xdas_narrow_integer_labels_survive_the_arithmetic():
    """Two int8 labels a whole range apart neither wrap nor round."""
    xdas = pytest.importorskip("xdas")
    ties = xdas.InterpCoordinate(
        {"tie_indices": [0, 2], "tie_values": np.asarray([-120, 120], dtype="int8")},
        dim="x",
    )
    coord = convert(
        xdas.DataArray(np.zeros(3), coords={"x": ties}), "unidas.BaseDAS"
    ).coords["x"]
    np.testing.assert_array_equal(coord.values, np.asarray(ties.values))
    assert coord.dtype == np.dtype("int8")
    # A narrow grid which does fit is still written as its labels, since the
    # arithmetic xdas would do between two tie points is its own dtype's.
    grid = NumericND.from_array(np.asarray([-100, -36, 28], dtype="int8"))
    assert grid.evenly_sampled
    out = to_xdas_coord(grid, ("x",))
    assert isinstance(out, xdas.DenseCoordinate)
    np.testing.assert_array_equal(np.asarray(out.values), grid.values)


def test_xdas_tie_points_no_integer_can_difference():
    """Labels a whole int64 range apart are divided, not wrapped."""
    xdas = pytest.importorskip("xdas")
    values = np.asarray([-9 * 10**18, 9 * 10**18], dtype="int64")
    ties = xdas.InterpCoordinate({"tie_indices": [0, 2], "tie_values": values}, dim="x")
    coord = convert(
        xdas.DataArray(np.zeros(3), coords={"x": ties}), "unidas.BaseDAS"
    ).coords["x"]
    np.testing.assert_array_equal(coord.values, np.asarray(ties.values))


@pytest.mark.parametrize("start", [0.0, 1e9])
def test_xdas_float_export_states_labels_it_cannot_interpolate(start):
    """A float grid xdas would interpolate differently goes out dense."""
    xdas = pytest.importorskip("xdas")
    coord = NumericND.from_run(start, 0.1, 4, dims=("x",))
    assert coord.evenly_sampled  # a grid, so the rule is what sends it dense
    out = to_xdas_coord(coord, ("x",))
    assert isinstance(out, xdas.DenseCoordinate)
    np.testing.assert_array_equal(np.asarray(out.values), coord.values)
    # A grid of whole ticks interpolates exactly, and still travels as one.
    ticks = NumericND.from_run(
        np.datetime64("2020-01-01", "ns"), np.timedelta64(4, "ms"), 50, dims=("time",)
    )
    ticked = to_xdas_coord(ticks, ("time",))
    assert isinstance(ticked, xdas.InterpCoordinate)
    np.testing.assert_array_equal(np.asarray(ticked.values), ticks.values)


def test_xdas_float_stretches_keep_labels_a_grid_cannot_give():
    """A float run is a grid only when the grid reproduces its own labels."""
    xdas = pytest.importorskip("xdas")
    # A tenth is exactly what a float grid of 1/10 gives back.
    clean = xdas.InterpCoordinate(
        {"tie_indices": [0, 10], "tie_values": [0.0, 1.0]}, dim="x"
    )
    out = convert(
        xdas.DataArray(np.zeros(11), coords={"x": clean}), "unidas.BaseDAS"
    ).coords["x"]
    np.testing.assert_array_equal(out.values, np.asarray(clean.values))
    assert out.labels is None  # a grid gives these back, so it is kept
    # A span whose interpolation lands off any grid of its own spacing
    # keeps the labels xdas computed.
    rough = xdas.InterpCoordinate(
        {"tie_indices": [0, 11], "tie_values": [1.0, 2.0]}, dim="x"
    )
    out = convert(
        xdas.DataArray(np.zeros(12), coords={"x": rough}), "unidas.BaseDAS"
    ).coords["x"]
    assert out.labels is not None  # the grid of elevenths is not these labels
    np.testing.assert_array_equal(out.values, np.asarray(rough.values))


@pytest.mark.parametrize(
    "coord_kwargs",
    [
        # A float range: its step may be a whole number while its labels are
        # not, so it states no grid of ticks and must not be given one.
        {"start": 0.5, "step": 2, "shape": (4,)},
        # A time counted in days rather than nanoseconds.
        {
            "start": np.datetime64("2020-01-01", "D"),
            "step": np.timedelta64(1, "D"),
            "shape": (4,),
        },
        # A duration rather than a date.
        {
            "start": np.timedelta64(0, "h"),
            "step": np.timedelta64(1, "h"),
            "shape": (4,),
        },
    ],
)
@needs_run_tables
def test_coordinates_which_state_no_grid(coord_kwargs):
    """A coordinate not counted in ticks keeps the labels it has."""
    coord = dc.core.get_coord(**coord_kwargs)
    patch = dascore_patch_along(coord)
    array = convert(patch, "xarray.DataArray")
    np.testing.assert_array_equal(array.coords["time"].values, coord.values)
    out = convert(array, "dascore.Patch")
    np.testing.assert_array_equal(out.get_array("time"), coord.values)


def test_an_exact_rate_reaches_a_sampled_target_exactly():
    """A target stating one rate is given the exact one, not the rounded one."""
    daspy = pytest.importorskip("daspy")
    coord = dascore_grid(rate=48000, size=4801)
    patch = dascore_patch_along(coord).transpose("distance", "time")
    try:
        array = patch.io.to_xarray(lazy_coords=True)
    except (AttributeError, TypeError):
        pytest.skip("This DASCore does not serve lazy xarray indexes.")
    section = convert(array, "daspy.Section")
    assert isinstance(section, daspy.Section)
    # 1/48000 s is 20833.33 ns; the rounded tick reads as 48000.77 Hz.
    assert section.fs == 48000.0
    xdas = pytest.importorskip("xdas")
    out = convert(array, "xdas.DataArray")
    assert isinstance(out, xdas.DataArray)
    np.testing.assert_array_equal(np.asarray(out.coords["time"].values), coord.values)
