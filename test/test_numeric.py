"""Tests for the numeric coordinate: the run table and its kernel."""

import math
from fractions import Fraction

import numpy as np
import pytest

from unidas import CoordinateError, NumericND, concat
from unidas.numeric import (
    canonical,
    check_tick_range,
    detect_runs,
    float_fraction,
    record_dtype,
    runs_from_rows,
    to_tick,
)

T0 = np.datetime64("2020-01-01", "ns")
MS = np.timedelta64(1, "ms")
NS = np.timedelta64(1, "ns")
STEP = np.timedelta64(4, "ms")

# The labels of the stored run every mixed table below carries.
JITTER = (
    T0 + STEP * np.asarray([20, 23, 27, 28, 34]) + np.asarray([0, 1, -3, 2, 0]) * NS
)


def grid_1024(count=2048, start=T0):
    """A 1024 Hz coordinate, whose period is 976562.5 ns."""
    return NumericND.from_run(start=start, step=Fraction(1, 1024), shape=count)


def exact_labels(start, count, stride=1, phase=Fraction(0)):
    """Labels of a 1024 Hz grid, as the floor of each ideal position."""
    ideal = [phase + Fraction(i * stride * 1953125, 2) for i in range(count)]
    return start + np.asarray([int(x // 1) for x in ideal]).astype("timedelta64[ns]")


def table(coord):
    """The run table as plain numbers, for comparing two coordinates."""
    return coord.runs.tolist()


def assert_within_a_bit(actual, expected, ulps=2, anchor=None, coord=None):
    """
    A float grid labels its samples within a couple of their last bits.

    Measured against each run's own start where the coordinate is given,
    which is the scale its labels were computed at; an exact match and two
    nulls always pass, and anything else -- a null beside a number above
    all -- does not.
    """
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    scale = np.abs(expected)
    if anchor is not None:
        scale = np.maximum(scale, anchor)
    if coord is not None:
        table = coord.runs
        starts = np.repeat(np.abs(table["start"]), table["length"])
        scale = np.maximum(scale, starts.reshape(scale.shape))
    with np.errstate(over="ignore", invalid="ignore"):
        # In the labels' own dtype: a float32 axis is judged by float32 bits.
        bound = np.spacing(np.asarray(scale, dtype=expected.dtype))
        near = (actual == expected) | (np.abs(actual - expected) <= ulps * bound)
    both_null = np.isnan(actual) & np.isnan(expected)
    off = ~(near | both_null)
    assert not np.any(off), f"{actual[off]} is not {expected[off]}"


@pytest.fixture(scope="module")
def holed():
    """Three runs of ten at 4 ms, each a hole of ten samples apart."""
    runs = [NumericND.from_run(T0 + STEP * 20 * i, STEP, 10) for i in range(3)]
    return concat(*runs)


@pytest.fixture(scope="module")
def mixed():
    """A grid run, a stored run of jittered labels, then a grid run."""
    return concat(
        NumericND.from_run(T0, STEP, 10),
        NumericND.from_array(JITTER),
        NumericND.from_run(T0 + STEP * 50, STEP, 8),
    )


class TestSingleRun:
    """One run is an ordinary range."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(start=0.0, step=0.1, shape=50),
            dict(start=3, step=2, shape=10),
            dict(start=T0, step=STEP, shape=100),
            dict(start=np.timedelta64(0, "s"), step=np.timedelta64(1, "s"), shape=5),
        ],
    )
    def test_values_match_arange(self, kwargs):
        """The labels are those the same start, step, and count describe."""
        out = NumericND.from_run(**kwargs)
        assert out.runs_count == 1
        assert out.evenly_sampled
        expected = kwargs["start"] + np.arange(kwargs["shape"]) * kwargs["step"]
        if out.dtype.kind in "mM":
            # A time is counted in nanosecond ticks whatever unit it came in.
            assert out.dtype == np.dtype(f"{out.dtype.kind}8[ns]")
            np.testing.assert_array_equal(out.values, expected.astype(out.dtype))
        elif out.dtype.kind == "f":
            # A grid multiplies the count by the step; adding the step that
            # many times, as arange does, need not give the same last bit.
            np.testing.assert_allclose(out.values, expected, rtol=1e-15)
        else:
            np.testing.assert_array_equal(out.values, expected)

    def test_exact_grid_does_not_drift(self):
        """A rate with no whole-tick period is counted from the origin."""
        coord = grid_1024()
        np.testing.assert_array_equal(coord.values, exact_labels(T0, 2048))
        assert coord.step == np.timedelta64(976562, "ns")
        assert coord.step_exact == Fraction(1, 1024)
        assert table(coord)[0][2:] == (1953125, 2, 0)

    def test_limits(self):
        """Min and max come from the ends without computing the labels."""
        coord = grid_1024(10)
        assert coord.min() == T0
        assert coord.max() == exact_labels(T0, 10)[-1]
        assert coord.sorted and not coord.reverse_sorted
        assert not coord.holes

    def test_integer_start_refuses_a_fractional_float_step(self):
        """An integer coordinate cannot hold labels between integers."""
        with pytest.raises(CoordinateError, match="non-integer"):
            NumericND.from_run(start=0, step=0.5, shape=3)

    def test_label_at_index(self):
        """An integer index gives one label, counted from either end."""
        coord = grid_1024(10)
        assert coord[3] == coord.values[3]
        assert coord[-1] == coord.values[-1]
        with pytest.raises(IndexError):
            coord[10]

    def test_dtype_states_the_step(self):
        """A float step on integer labels states a grid the start cannot."""
        coord = NumericND.from_run(0, 0.5, 4, dtype="float64")
        np.testing.assert_array_equal(coord.values, [0.0, 0.5, 1.0, 1.5])


class TestCanonical:
    """Equal labels give one table, whatever made them."""

    def test_reduced_grid_equals_whole_tick_grid(self):
        """A stride-two 1024 Hz slice is the 2048 Hz-period grid it labels."""
        odd = grid_1024(5000)[1::2]
        fresh = NumericND.from_run(odd.start, np.timedelta64(1953125, "ns"), 2500)
        np.testing.assert_array_equal(odd.values, fresh.values)
        assert table(odd) == table(fresh)
        assert odd.fingerprint() == fresh.fingerprint()
        assert odd == fresh

    def test_unreduced_input_keeps_its_labels(self):
        """A step stated in unreduced terms is reduced with its phase."""
        coord = NumericND.from_run(0, (6, 4), 5, origin_offset=2)
        assert table(coord)[0][2:] == (3, 2, 1)
        np.testing.assert_array_equal(coord.values, [0, 2, 3, 5, 6])

    @pytest.mark.parametrize("step", [(3906250, 2), (6, 4), (12, 8), (10, 4), (9, 6)])
    def test_reduction_preserves_every_label(self, step):
        """Reducing a grid moves no label, at any phase it can hold."""
        for offset in range(step[1]):
            coord = NumericND.from_run(0, step, 50, origin_offset=offset)
            expected = [(offset + i * step[0]) // step[1] for i in range(50)]
            np.testing.assert_array_equal(coord.values, expected)

    def test_phase_past_its_denominator(self):
        """A phase of more than one tick is carried into the start."""
        coord = NumericND.from_run(0, (3, 2), 4, origin_offset=5)
        np.testing.assert_array_equal(coord.values, [2, 4, 5, 7])

    def test_a_derived_table_is_already_canonical(self):
        """A slice, reversal or shift of a canonical table is canonical."""
        coord = grid_1024(500)
        derived = [
            coord[3::7],
            coord[::-1],
            coord[100:200],
            coord.translated(MS),
            coord[::2][::3],
            NumericND.from_run(0.0, 0.1, 50)[2::5],
        ]
        for out in derived:
            again, _ = canonical(out.runs, out.labels, out.dtype)
            assert np.array_equal(again, out.runs)

    @pytest.mark.parametrize("cut", [1, 7, 100, 499])
    def test_split_and_concat_is_the_same_table(self, cut):
        """Cutting a coordinate anywhere and rejoining it changes nothing."""
        coord = grid_1024(500)
        rejoined = concat(coord[:cut], coord[cut:])
        assert table(rejoined) == table(coord)
        assert rejoined.fingerprint() == coord.fingerprint()

    @pytest.mark.parametrize("stride", [2, 3, 7])
    def test_stride_and_fuse(self, stride):
        """A strided slice fuses back onto the grid it was cut from."""
        strided = grid_1024(500)[::stride]
        halves = concat(strided[:20], strided[20:])
        assert halves.runs_count == 1
        assert table(halves) == table(strided)


# (dtype, start, length, num, den, offset): grids which fit and grids which
# leave their dtype at either end.
RANGE_CASES = [
    ("datetime64[ns]", 0, 10, 1, 1, 0),
    ("datetime64[ns]", 0, 100, 3, 2, 1),
    ("datetime64[ns]", 0, 2**62, 8, 1, 0),
    ("datetime64[ns]", 2**62, 2**62, 4, 1, 0),
    ("datetime64[ns]", -(2**62), 10, -(2**60), 1, 0),
    ("timedelta64[ns]", 5, 3, 1, 1024, 3),
    ("int64", 0, 10, 0, 1, 0),
    # A step which is int64's floor: it has no positive of its own, so a
    # magnitude taken in int64 would read it as a small step.
    ("int64", 0, 3, -(2**63), 1, 0),
    ("int8", 0, 100, 1, 1, 0),
    ("int8", 100, 100, 1, 1, 0),
    ("int8", -100, 100, -1, 1, 0),
    ("uint8", 200, 100, 1, 1, 0),
    # A uint64 label past int64's ceiling: every tick is counted in
    # int64, so it is out of range whatever uint64 could hold.
    ("uint64", 2**63 - 10, 20, 1, 1, 0),
    ("uint64", 2**62, 20, 1, 1, 0),
    ("float64", 0.0, 10, 1, 1, 0),
    ("float64", np.inf, 10, 1, 1, 0),
    ("float64", 1e308, 10, 1, 1, 0),
    ("float64", 0.0, 10, 0, 0, 0),  # a stored run states no grid
]


class TestOneRowArithmetic:
    """One run is canonicalised in python; it must answer as the table does."""

    def _raised(self, func) -> bool:
        """Whether a check refused the run it was handed."""
        try:
            func()
        except CoordinateError:
            return True
        return False

    @pytest.mark.parametrize("case", RANGE_CASES)
    def test_one_row_check_matches_the_vectorised_one(self, case):
        """The scalar tick-range check refuses exactly what the array one does."""
        name, *fields = case
        dtype = np.dtype(name)
        rows = np.asarray([tuple(fields)] * 2, record_dtype(dtype))
        scalar = self._raised(lambda: check_tick_range(rows[:1], dtype))
        # The same row twice, so that it goes through the vectorised body,
        # which each row is checked by on its own.
        vector = self._raised(lambda: check_tick_range(rows, dtype))
        assert scalar == vector

    @pytest.mark.parametrize("length", [0, -1])
    def test_a_run_of_no_samples_is_dropped(self, length):
        """A run holding nothing leaves an empty coordinate, not a bad shape."""
        coord = NumericND.from_run(0, 1, length)
        assert coord.shape == (0,) and len(coord) == 0

    def test_a_shape_the_runs_do_not_fill(self):
        """A coordinate holds as many labels as its shape states."""
        table = runs_from_rows([(0, 4, 1, 1, 0)], "int64")
        with pytest.raises(CoordinateError, match="holds 4 samples"):
            NumericND(runs=table, dtype="int64", shape=(5,))

    def test_rows_from_plain_tuples(self):
        """A table states its runs the way an index row spells them."""
        coord = NumericND.from_rows(
            runs_from_rows([(0, 4, 1, 1, 0), (10, 3, 2, 1, 0)], "int64"),
            dtype="int64",
        )
        np.testing.assert_array_equal(coord.values, [0, 1, 2, 3, 10, 12, 14])


class TestSlicing:
    """Slices stay on the grid, phase included."""

    def test_stride_keeps_phase(self):
        """Every third sample of a 1024 Hz grid, from the seventh, stays on it."""
        coord = grid_1024()
        out = coord[7:1000:3]
        assert out.runs_count == 1
        np.testing.assert_array_equal(out.values, coord.values[7:1000:3])

    def test_stride_across_holes_gives_each_run_its_phase(self):
        """Runs cut at different phases keep their own."""
        a, b = grid_1024(100), grid_1024(100, T0 + np.timedelta64(1, "s"))
        coord = concat(a, b)
        out = coord[1:200:3]
        assert out.runs_count == 2
        np.testing.assert_array_equal(out.values, coord.values[1:200:3])

    @pytest.mark.parametrize(
        "item",
        [
            slice(7, None, 3),
            slice(1, 400, 2),
            slice(None, None, -1),
            slice(3, None, -2),
            slice(None, None, 1024),
        ],
    )
    def test_slices_match_their_labels(self, item):
        """A slice of the table holds the labels that slice of the values does."""
        coord = grid_1024(5000)
        np.testing.assert_array_equal(coord[item].values, coord.values[item])

    def test_reversed(self):
        """A negative stride reverses the runs and their steps."""
        a, b = grid_1024(10), grid_1024(10, T0 + np.timedelta64(1, "s"))
        coord = concat(a, b)
        out = coord[::-1]
        assert out.reverse_sorted
        np.testing.assert_array_equal(out.values, coord.values[::-1])
        assert table(out[::-1]) == table(coord)
        assert out.min() == coord.min() and out.max() == coord.max()

    def test_index_array(self):
        """An index array gives the labels it picks, in that order."""
        coord = grid_1024(10)
        picked = coord[np.array([1, 5, 2])]
        np.testing.assert_array_equal(picked.values, coord.values[[1, 5, 2]])

    def test_empty_slice(self):
        """A slice of nothing is a coordinate of no samples."""
        empty = grid_1024(10)[5:2]
        assert len(empty) == 0
        assert empty.dtype == np.dtype("datetime64[ns]")

    def test_padding_extends_the_grid(self):
        """A single run may be sliced before its start, which padding needs."""
        coord = grid_1024(100)
        padded = coord._sliced(-10, 1, 120)
        assert len(padded) == 120
        np.testing.assert_array_equal(padded.values[10:110], coord.values)
        # The label before the first is the floor of its ideal position,
        # half a tick further out than the rounded step.
        assert padded.values[9] == T0 - np.timedelta64(976563, "ns")
        assert padded[10:110] == coord

    def test_labels_at_outside_the_coordinate(self):
        """Indices past either end extend the grid of the nearest run."""
        coord = grid_1024(10)
        picked = coord.labels_at(np.asarray([-2, 0, 11]))
        assert picked[1] == T0
        assert picked[0] == exact_labels(T0, 1, phase=Fraction(-1953125))[0]
        assert picked[2] == exact_labels(T0, 12)[11]


class TestRuns:
    """Holes and changes of rate are runs in one table."""

    def test_hole_is_a_second_run(self):
        """Runs with a hole between them stay two runs."""
        a = NumericND.from_run(T0, STEP, 100)
        b = NumericND.from_run(T0 + STEP * 110, STEP, 50)
        coord = concat(a, b)
        assert coord.runs_count == 2
        assert not coord.evenly_sampled
        assert coord.holes
        assert coord.step == STEP  # one rate, still, across the hole
        np.testing.assert_array_equal(
            coord.values, np.concatenate([a.values, b.values])
        )

    def test_contiguous_runs_fuse(self):
        """Equal samples give an equal coordinate however they were assembled."""
        whole = grid_1024(300)
        parts = [whole[:100], whole[100:250], whole[250:]]
        rebuilt = concat(*parts)
        assert rebuilt.runs_count == 1
        assert rebuilt == whole
        assert rebuilt.fingerprint() == whole.fingerprint()
        # A slice with a phase fuses only onto the grid it came from.
        strided = whole[1::2]
        assert concat(strided[:50], strided[50:]) == strided

    def test_change_of_rate_without_a_hole(self):
        """A run at a new rate directly after another is two runs, no hole."""
        a = NumericND.from_run(0, 2, 5)
        b = NumericND.from_run(10, 3, 4)
        coord = concat(a, b)
        assert coord.runs_count == 2
        assert coord.step is None
        assert not coord.holes
        np.testing.assert_array_equal(coord.values, [0, 2, 4, 6, 8, 10, 13, 16, 19])

    def test_mixed_dtypes_refused(self):
        """Runs must share a dtype."""
        with pytest.raises(CoordinateError, match="dtype"):
            concat(
                NumericND.from_run(0, 1, 3),
                NumericND.from_run(0.0, 1.0, 3),
            )

    def test_mixed_units_refused(self):
        """Runs must share units."""
        with pytest.raises(CoordinateError, match="units"):
            concat(
                NumericND.from_run(0, 1, 3, units="m"),
                NumericND.from_run(3, 1, 3, units="ft"),
            )

    def test_nothing_to_concatenate(self):
        """Joining no coordinates states what it was asked, not an IndexError."""
        with pytest.raises(CoordinateError, match="nothing"):
            concat()


class TestMixedRuns:
    """Stored labels and grids live in one table."""

    def test_values(self, mixed):
        """The table holds the labels of all three runs."""
        assert mixed.runs_count == 3
        assert len(mixed) == 23
        assert mixed.sorted and mixed.holes
        assert mixed.step is None  # a stored run states no rate
        assert mixed.runs["den"].tolist() == [1, 0, 1]  # the middle is stored
        assert len(mixed.labels) == 5
        expected = np.concatenate(
            [
                T0 + STEP * np.arange(10),
                JITTER,
                T0 + STEP * (50 + np.arange(8)),
            ]
        )
        np.testing.assert_array_equal(mixed.values, expected)

    def test_select_across_runs(self, mixed):
        """A window over the stored run keeps the samples inside it."""
        values = mixed.values
        out, indexer = mixed.select(values[8], values[12])
        assert indexer == slice(8, 13)
        np.testing.assert_array_equal(out.values, values[8:13])

    def test_select_between_stored_labels(self, mixed):
        """A bound between two stored labels selects from the next one in."""
        values = mixed.values
        out, _ = mixed.select(values[11] + NS, values[13] - NS)
        np.testing.assert_array_equal(out.values, values[12:13])

    def test_strided_slice(self, mixed):
        """A stride cuts every run, stored labels included."""
        out = mixed[1::4]
        np.testing.assert_array_equal(out.values, mixed.values[1::4])

    def test_reversed(self, mixed):
        """Reversing turns the stored labels round with the grids."""
        out = mixed[::-1]
        np.testing.assert_array_equal(out.values, mixed.values[::-1])
        assert out.reverse_sorted
        assert table(out[::-1]) == table(mixed)

    def test_fingerprint_is_stable(self, mixed):
        """Splitting and rejoining gives back the same table and fingerprint."""
        # Grid runs fuse back together; a stored run keeps its own bounds.
        rebuilt = concat(mixed[:7], mixed[7:15], mixed[15:])
        assert table(rebuilt) == table(mixed)
        assert rebuilt.fingerprint() == mixed.fingerprint()

    def test_labels_at(self, mixed):
        """Labels can be asked for at scattered indices."""
        picked = np.asarray([0, 11, 22, 13])
        np.testing.assert_array_equal(mixed.labels_at(picked), mixed.values[picked])


class TestFromArray:
    """Labels state their own runs, or are kept as they are."""

    def test_even_array_is_one_run(self):
        """An exactly evenly sampled array is a grid run."""
        coord = NumericND.from_array(np.arange(10) * 3)
        assert coord.runs_count == 1
        assert coord.evenly_sampled
        assert coord.step == 3
        np.testing.assert_array_equal(coord.values, np.arange(10) * 3)

    def test_hole_gives_two_runs(self):
        """An array with a hole in it becomes two grid runs."""
        values = np.asarray([0.0, 1, 2, 3, 10, 11, 12, 13])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 2
        assert coord.holes
        np.testing.assert_array_equal(coord.values, values)

    def test_stray_labels_are_stored(self):
        """Labels which state no grid are kept, beside the runs which do."""
        values = np.asarray([0, 1, 2, 3, 20, 33, 34, 35, 36])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 3
        assert coord.labels.tolist() == [20]
        np.testing.assert_array_equal(coord.values, values)

    def test_jitter_is_one_stored_run(self):
        """A dense jittered array is kept whole rather than split per sample."""
        rng = np.random.default_rng(42)
        count = 100_000
        values = T0 + np.arange(count) * MS + rng.integers(-500, 500, count) * NS
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1
        assert coord.labels is not None
        assert coord.step is None
        np.testing.assert_array_equal(coord.values, values)

    def test_declared_step_reads_the_grid(self):
        """Labels declared to sit on a grid become runs of that step."""
        coord = NumericND.from_array(np.asarray([1, 3, 4, 10, 11, 12]), step=1)
        assert coord.runs_count == 3
        assert coord.step == 1
        np.testing.assert_array_equal(coord.values, [1, 3, 4, 10, 11, 12])

    def test_declared_step_refuses_labels_off_its_grid(self):
        """A declared grid is a claim the labels must actually keep."""
        with pytest.raises(CoordinateError, match="not on a grid"):
            NumericND.from_array(np.asarray([0, 1, 2, 4, 5]), step=2)

    def test_two_dimensional(self):
        """An array of any shape is one stored run carrying its shape."""
        coord = NumericND.from_array(np.zeros((3, 4)))
        assert coord.shape == (3, 4)
        assert coord.ndim == 2
        assert coord.runs_count == 1
        assert coord.runs["length"][0] == 12
        np.testing.assert_array_equal(coord[1], np.zeros(4))
        assert coord[:2].shape == (2, 4)
        with pytest.raises(TypeError, match="no length"):
            len(coord)

    def test_float_grid_below_one_is_detected(self):
        """A float spacing is a fraction, not a truncated whole tick."""
        values = np.concatenate([np.arange(10) * 0.5, 100 + np.arange(10) * 0.5])
        coord = NumericND.from_array(values, detect=True)
        assert coord.runs_count == 2
        assert np.all(coord.runs["den"] == 2) and np.all(coord.runs["num"] == 1)
        assert coord.step == 0.5
        np.testing.assert_array_equal(coord.values, values)

    def test_a_spacing_with_no_fraction_is_stored(self):
        """Labels whose spacing has no fraction of ticks keep themselves."""
        values = np.asarray([0.0, 1e20, 2e20, 1e21, 1.1e21, 1.2e21])
        coord = NumericND.from_array(values, detect=True)
        assert coord.labels is not None
        np.testing.assert_array_equal(coord.values, values)

    def test_rank_zero_stays_rank_zero(self):
        """A scalar label is a coordinate of no axis, not one of one sample."""
        coord = NumericND.from_array(np.array(3.0))
        assert coord.shape == () and coord.ndim == 0
        assert coord.size == 1
        assert coord.values.shape == ()
        assert coord.values[()] == 3.0

    def test_unsorted_labels(self):
        """Arbitrary labels are stored, and hashed as they are."""
        coord = NumericND.from_array(np.asarray([3.0, 1.0, 2.0]))
        assert not coord.sorted and not coord.reverse_sorted
        assert coord.min() == 1.0 and coord.max() == 3.0
        ordered = NumericND.from_array(np.asarray([1.0, 2.0, 3.0]))
        assert coord.fingerprint() != ordered.fingerprint()

    def test_monotonic_select(self):
        """A sorted stored run selects by searching its labels."""
        coord = NumericND.from_array(np.asarray([1.0, 2.5, 4.0, 8.0]))
        out, indexer = coord.select(2.0, 5.0)
        np.testing.assert_array_equal(out.values, [2.5, 4.0])
        assert indexer == slice(1, 3)

    @pytest.mark.parametrize(
        "labels",
        [
            np.asarray(["a", "b"]),
            np.asarray([0.5, 1.5]).astype(object),
            np.asarray([True, False]),
        ],
    )
    def test_labels_a_categorical_holds_are_refused(self, labels):
        """A table has no arithmetic for labels which are not numbers."""
        with pytest.raises(CoordinateError, match="Categorical"):
            NumericND.from_array(labels)
        with pytest.raises(CoordinateError, match="Categorical"):
            NumericND.from_rows(
                runs_from_rows([(0, 2, 0, 0, 0)], "float64"),
                labels=labels,
                dtype=labels.dtype,
            )


class TestPartial:
    """A coordinate whose shape is known and whose labels are not loaded."""

    def test_states_its_shape(self):
        """A partial answers about its shape without any labels."""
        coord = NumericND.partial((3, 4), dtype="float64")
        assert coord.shape == (3, 4) and coord.size == 12
        assert coord.partial_labels
        assert coord.labels is None
        assert not coord.holes

    def test_labels_are_refused(self):
        """Anything which needs the labels says they are not there."""
        coord = NumericND.partial(10, dtype="datetime64[ns]")
        calls = (
            lambda: coord.values,
            lambda: coord.select(T0, T0),
            lambda: coord.translated(NS),
            lambda: coord.reversed(),
        )
        for call in calls:
            with pytest.raises(CoordinateError, match="not loaded"):
                call()

    def test_two_partials_of_a_shape_are_equal(self):
        """A partial states its shape, dtype and units, and nothing else."""
        first = NumericND.partial(10, dtype="int64")
        assert first == NumericND.partial(10, dtype="int64")
        assert first != NumericND.partial(11, dtype="int64")
        assert first != NumericND.from_run(0, 1, 10)

    def test_from_rows_refuses_a_stored_run_without_labels(self):
        """A stored run is nothing but its labels, so it needs them."""
        rows = runs_from_rows([(0, 4, 0, 0, 0)], "float64")
        with pytest.raises(CoordinateError, match="without its labels"):
            NumericND.from_rows(rows, dtype="float64")

    def test_concat_refuses_a_partial(self):
        """Runs cannot be laid end to end when one holds no labels."""
        with pytest.raises(CoordinateError, match="not loaded"):
            concat(NumericND.from_run(0, 1, 3), NumericND.partial(3, dtype="int64"))


class TestOverlappingRuns:
    """Runs which overlap are not sorted, and are selected by their labels."""

    @pytest.fixture
    def overlapping(self):
        """Ten samples from zero, then ten more from five."""
        first = NumericND.from_run(0, 1, 10)
        return concat(first, NumericND.from_run(5, 1, 10))

    def test_not_sorted(self, overlapping):
        """An overlap is not a sorted coordinate, however each run reads."""
        assert not overlapping.sorted
        assert not overlapping.reverse_sorted

    def test_select_matches_the_labels(self, overlapping):
        """Selection falls back to matching every label, in both runs."""
        np.testing.assert_array_equal(overlapping.values, [*range(10), *range(5, 15)])
        out, indexer = overlapping.select(3, 7)
        assert np.flatnonzero(indexer).tolist() == [3, 4, 5, 6, 7, 10, 11, 12]
        np.testing.assert_array_equal(out.values, [3, 4, 5, 6, 7, 5, 6, 7])

    def test_lookup_refuses(self, overlapping):
        """Looking one value up needs an order the coordinate does not have."""
        with pytest.raises(CoordinateError, match="sorted"):
            overlapping.index_of(4)


class TestDegenerate:
    """Flat, short, and empty coordinates behave."""

    def test_zero_step(self):
        """A step of zero labels every sample the same."""
        coord = NumericND.from_run(5, 0, 4)
        np.testing.assert_array_equal(coord.values, [5, 5, 5, 5])
        assert coord.sorted and not coord.reverse_sorted
        assert coord.step == 0
        out, indexer = coord.select(5, 5)
        assert len(out) == 4 and indexer == slice(None, 4)
        out, _ = coord.select(6, 7)
        assert len(out) == 0

    def test_single_sample(self):
        """A run of one sample selects, reverses, and states its ends."""
        coord = grid_1024(1)
        assert len(coord) == 1
        assert coord.min() == coord.max() == T0
        np.testing.assert_array_equal(coord[::-1].values, coord.values)
        out, indexer = coord.select(T0, T0)
        assert len(out) == 1 and indexer == slice(None, 1)

    def test_min_and_max_ignore_null_labels(self):
        """A missing label is no label at all, not the smallest one."""
        nat = np.datetime64("NaT", "ns")
        labels = np.asarray([T0 + 5 * NS, nat, T0, T0 + 2 * NS])
        coord = NumericND.from_array(labels, detect=False)
        assert not (coord.sorted or coord.reverse_sorted)
        assert coord.min() == T0 and coord.max() == T0 + 5 * NS
        # Nothing but nulls names no instant, and neither does nothing.
        blank = NumericND.from_array(np.asarray([nat, nat]), detect=False)
        assert np.isnat(blank.min()) and np.isnat(blank.max())
        assert np.isnan(NumericND.from_array(np.zeros(0)).min())

    def test_select_on_an_unsorted_coordinate(self):
        """Labels in no order are matched rather than looked up."""
        coord = NumericND.from_array(np.asarray([3.0, 1.0, 2.0]))
        out, indexer = coord.select(0.0, 9.0)
        assert out is coord and indexer == slice(None, None)
        out, mask = coord.select(9.0, 10.0)
        assert len(out) == 0 and not mask.any()
        out, mask = coord.select(1.5, 9.0)
        np.testing.assert_array_equal(out.values, [3.0, 2.0])

    def test_empty(self):
        """An empty coordinate is a run of no samples, not an empty table."""
        empty = grid_1024(10)[5:2]
        assert len(empty) == 0
        assert empty.runs_count == 1
        assert empty.values.dtype == np.dtype("datetime64[ns]")
        assert empty.step is None
        assert len(empty[:]) == 0
        assert np.isnat(empty.min()) and np.isnat(empty.max())

    def test_empty_concatenates_away(self):
        """An empty run leaves nothing behind in a table it joins."""
        whole = grid_1024(300)
        parts = [whole[:100], whole[100:100], whole[100:], whole[300:]]
        assert concat(*parts) == whole

    def test_empty_select(self):
        """Selecting on an empty coordinate selects nothing."""
        out, indexer = grid_1024(10)[5:2].select(T0, T0 + MS)
        assert len(out) == 0
        assert indexer == slice(0, 0)

    def test_empty_of_a_higher_rank(self):
        """An N-D coordinate keeps its rank when it is emptied."""
        coord = NumericND.from_array(np.zeros((3, 4)))
        assert coord.empty().shape == (0, 0)


class TestOverflow:
    """Grids which leave int64 are refused rather than silently wrapped."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(start=0, step=10**12, shape=10**7),
            dict(start=T0, step=np.timedelta64(1, "h"), shape=3_000_000),
            dict(start=2**62, step=2**61, shape=4),
        ],
    )
    def test_refused(self, kwargs):
        """A grid whose ticks do not fit cannot be built."""
        with pytest.raises(CoordinateError, match="exceeds"):
            NumericND.from_run(**kwargs)

    def test_a_time_outside_the_nanosecond_range(self):
        """A date no nanosecond count can hold is refused by name."""
        with pytest.raises(CoordinateError, match="nanosecond range"):
            NumericND.from_run(np.datetime64("1500-01-01", "s"), 1, 3)

    def test_a_sub_nanosecond_label(self):
        """An instant between two nanoseconds is refused rather than rounded."""
        values = np.asarray([0, 1, 2], dtype="timedelta64[ps]")
        with pytest.raises(CoordinateError, match="whole number of nanoseconds"):
            NumericND.from_array(values)


class TestFloats:
    """A float coordinate is the same table without the floor."""

    def test_short_step_is_a_short_fraction(self):
        """A float spacing is stored as the short fraction it is."""
        coord = NumericND.from_run(0.0, 0.1, 11)
        assert table(coord)[0][2:4] == (1, 10)
        assert coord.step == 0.1
        assert coord.step_exact == Fraction(1, 10)
        assert coord.values[3] == 0.3
        assert coord.values[-1] == 1.0

    def test_binary_step_round_trips(self):
        """A spacing which is no short fraction keeps its binary one."""
        coord = NumericND.from_run(0.0, math.pi, 5)
        assert coord.values[-1] == 4 * math.pi
        assert float(coord.step_exact) == math.pi

    def test_tiny_step_does_not_vanish(self):
        """A spacing below the shortening limit stays itself."""
        coord = NumericND.from_run(0.0, 1e-15, 5)
        assert float(coord.step_exact) == 1e-15
        assert coord.values[1] == 1e-15
        np.testing.assert_allclose(coord.values, np.arange(5) * 1e-15, rtol=1e-15)

    def test_a_step_with_no_fraction_of_ticks(self):
        """A spacing past what int64 can state as a ratio is refused."""
        with pytest.raises(CoordinateError, match="no fraction of ticks"):
            float_fraction(1e300)

    def test_slice_keeps_its_first_label(self):
        """
        A slice starts at exactly the label its parent gave that sample.

        A float slice is re-anchored on that label, so the labels after it
        are computed from a new origin and may differ from the parent's in
        their last bit. That is the contract, not an accident: only the
        first label is promised exactly.
        """
        coord = NumericND.from_run(1e6, 0.1, 1000)
        for first in (0, 11, 137):
            sliced = coord[first : first + 9]
            assert sliced.values[0] == coord.values[first]
            np.testing.assert_allclose(
                sliced.values, coord.values[first : first + 9], rtol=1e-12, atol=0
            )

    def test_fuse_within_a_tolerance(self):
        """Float runs which continue each other fuse back into one."""
        coord = NumericND.from_run(0.0, 0.5, 6)
        pieces = (NumericND.from_run(0.0, 0.5, 3), NumericND.from_run(1.5, 0.5, 3))
        rejoined = concat(*pieces)
        assert rejoined.runs_count == 1
        assert rejoined == coord

    def test_a_narrow_float_fuses_on_the_labels_it_holds(self):
        """A float32 label is the cast of its position, not the position."""
        coord = NumericND.from_run(-0.3, 0.1, 6, dtype="float32")
        rejoined = concat(coord[:3], coord[3:])
        np.testing.assert_array_equal(rejoined.values, coord.values)
        assert rejoined.runs_count == 1 and rejoined == coord

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    @pytest.mark.parametrize("cut", [1, 3, 5])
    def test_a_sliced_float_grid_joins_back_into_one(self, dtype, cut):
        """
        A float slice is re-anchored, and re-anchoring is within a last bit.

        The pieces are one grid again, rather than two runs describing what
        the arithmetic happened to give each of them.
        """
        coord = NumericND.from_run(-0.3, 0.1, 6, dtype=dtype)
        rejoined = concat(coord[:cut], coord[cut:])
        assert rejoined.runs_count == 1
        assert rejoined == coord
        assert_within_a_bit(rejoined.values, coord.values)

    @pytest.mark.parametrize("forward", [True, False])
    @pytest.mark.parametrize("count", [1, 2, 10])
    @pytest.mark.parametrize("step", [1.0, 0.25, -1.0])
    def test_a_bound_answers_as_the_same_grid_of_ticks_does(self, forward, count, step):
        """
        A float grid places a bound where a grid of ticks would place it.

        Past either end that is the position the grid gives it, outside the
        coordinate, which is how a caller tells an open bound from one
        landing on the last sample.
        """
        floats = NumericND.from_run(0.0, step, count)
        ticks = NumericND.from_run(0, int(step * 4), count, dtype="int64")
        span = np.arange(-2.0, count + 2.0, 0.5)
        for value in span:
            assert floats.index_of(value, forward=forward) == ticks.index_of(
                value * 4, forward=forward
            ), value

    def test_select(self):
        """Float labels select on their fraction step within a tolerance."""
        coord = NumericND.from_run(0.0, 0.1, 20)
        out, _ = coord.select(0.3, 0.7)
        np.testing.assert_array_equal(out.values, np.arange(3, 8) / 10)

    def test_select_near_a_label(self):
        """A bound a hair inside a label picks the next label in."""
        coord = NumericND.from_run(1e6, 0.1, 1000)
        values = coord.values
        out, _ = coord.select(values[10], values[20])
        np.testing.assert_array_equal(out.values, values[10:21])
        out, _ = coord.select(values[10] + 1e-6, values[20] - 1e-6)
        # Re-anchoring a float slice may move a label by its last bit.
        np.testing.assert_allclose(out.values, values[11:20], rtol=1e-15)


class TestFittedFloatGrids:
    """A float grid is fitted to its labels within a couple of last bits."""

    @pytest.mark.parametrize("count", [20, 1000, 100000])
    def test_a_tenth_is_a_tenth(self, count):
        """The grid numpy's own arithmetic names is the grid it is read as."""
        values = np.arange(count) * 0.1
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1 and coord.evenly_sampled
        assert coord.step == 0.1 and coord.step_exact == Fraction(1, 10)
        assert_within_a_bit(coord.values, values, ulps=1)

    def test_a_span_divided_into_samples(self):
        """Labels laid out by linspace sit on the grid they were laid on."""
        values = np.linspace(0, 99.9, 1000)
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1 and coord.step_exact == Fraction(1, 10)
        assert_within_a_bit(coord.values, values, ulps=1)

    def test_a_narrow_float_grid(self):
        """A float32 axis is judged by float32 bits, and fits the same way."""
        values = (np.arange(20) * np.float32(0.1)).astype(np.float32)
        coord = NumericND.from_array(values)
        assert coord.dtype == np.dtype("float32")
        assert coord.runs_count == 1 and coord.step_exact == Fraction(1, 10)
        assert_within_a_bit(coord.values, values, ulps=1)

    def test_a_grid_which_steps_across_zero(self):
        """A label at a zero crossing is judged by the step, not by itself."""
        values = -1.0 + np.arange(21) * 0.1
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1 and coord.step_exact == Fraction(1, 10)
        # The label at the crossing is judged against the grid's own scale;
        # its own last bit is far finer than the step which reached it.
        assert_within_a_bit(coord.values, values, anchor=1.0)

    def test_a_grid_finer_than_the_simplest_fractions(self):
        """A rate of zero describes a flat stretch and nothing else."""
        values = np.asarray([1.0, np.nextafter(1.0, 2), np.nextafter(1.0, 2)])
        values[2] = np.nextafter(values[1], 2)
        coord = NumericND.from_array(values)
        assert coord.evenly_sampled
        np.testing.assert_array_equal(coord.values, values)
        # A stretch which really is flat still states a step of zero.
        flat = NumericND.from_array(np.full(5, 2.5))
        assert flat.evenly_sampled and flat.step == 0.0

    def test_a_gap_is_never_swallowed_by_a_coarse_last_bit(self):
        """Past 2**53 a last bit is several steps, and a gap is still a gap."""
        steps = np.full(19, 2.0)
        steps[9] = 10.0  # four samples missing, where one bit is two of them
        values = 1e16 + np.concatenate([[0.0], np.cumsum(steps)])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 2 and coord.holes
        assert coord.step == 2.0
        np.testing.assert_array_equal(coord.values, values)

    def test_a_run_answers_for_its_own_labels_only(self):
        """A far-away run's start is no licence for this one's rounding."""
        # Labels jittered by a last bit or two each: fitting spends the
        # tolerance and fusing would spend it again, which the table has to
        # answer for against each run's own start rather than the largest.
        values = np.arange(19, 27) * 0.1
        values += np.asarray([-1, -1, -2, 1, -2, 0, 2, 3]) * np.spacing(values)
        values = np.concatenate([values, [100.0]])
        coord = NumericND.from_array(values)
        np.testing.assert_array_equal(coord.values, values)
        # The same, through a step declared far from where the labels sit.
        declared = np.concatenate(
            [[-1e6, -1e6 + 1, -1e6 + 2], np.arange(3) * (1 + 1e-10)]
        )
        out = NumericND.from_array(declared, step=1.0)
        np.testing.assert_array_equal(out.values, declared)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_the_largest_label_is_no_licence_either(self, dtype):
        """Past the last label the gap is an infinity, which is no bound."""
        largest = np.asarray(np.finfo(dtype).max, dtype=dtype)
        values = np.asarray([largest, largest, 0], dtype=dtype)
        coord = NumericND.from_array(values)
        np.testing.assert_array_equal(coord.values, values)

    def test_a_null_among_the_labels_keeps_the_grids_beside_it(self):
        """A label which is no number says nothing about the ones which are."""
        values = np.concatenate([np.arange(4) * 0.5, [np.nan]])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 2 and coord.evenly_sampled is False
        assert tuple(coord.runs[0])[1:4] == (4, 1, 2)
        np.testing.assert_array_equal(coord.values[:4], values[:4])
        assert np.isnan(coord.values[4])

    def test_labels_no_grid_describes_keep_themselves(self):
        """Jitter is not a rate, and is stored rather than fitted."""
        rng = np.random.default_rng(0)
        values = np.arange(50) * 0.1 + rng.uniform(-0.01, 0.01, 50)
        coord = NumericND.from_array(values)
        assert coord.labels is not None
        np.testing.assert_array_equal(coord.values, values)

    @pytest.mark.parametrize("dtype", ["float64", "float32"])
    def test_every_float_array_keeps_the_contract(self, dtype):
        """Whatever the labels, the table is within a bit or two of them."""
        rng = np.random.default_rng(3)
        for _ in range(60):
            count = int(rng.choice([3, 17, 60, 1200]))
            base = float(rng.uniform(-1e4, 1e4))
            step = float(rng.choice([1.0, 0.1, 0.25, 1 / 1024, -0.5]))
            values = base + np.arange(count) * step
            kind = rng.integers(4)
            if kind == 1 and count > 4:  # a gap
                values[count // 2 :] += 13 * abs(step)
            elif kind == 2:  # jitter, which is no grid at all
                values = values + rng.normal(0, abs(step) / 3, count)
            elif kind == 3 and count > 6:  # a change of rate
                tail = np.arange(1, count - count // 2 + 1) * step * 3
                values[count // 2 :] = values[count // 2 - 1] + tail
            values = values.astype(dtype)
            coord = NumericND.from_array(values)
            assert_within_a_bit(coord.values, values, coord=coord)

    def test_a_rate_change_is_still_two_runs(self):
        """A fitted grid does not swallow a stretch at another rate."""
        values = np.concatenate([np.arange(10) * 0.1, 0.9 + np.arange(1, 11) * 0.3])
        coord = NumericND.from_array(values)
        assert coord.runs_count == 2
        assert [tuple(x[2:4]) for x in table(coord)] == [(1, 10), (3, 10)]
        assert_within_a_bit(coord.values, values)


class TestLabelsAreNotTolerances:
    """A table describes labels only while it gives them back."""

    def test_a_float_grid_off_its_own_grid_is_still_that_grid(self):
        """
        Labels a last bit either side of a tenth are labels of a tenth.

        Numpy's own arithmetic puts them there; reading each last bit as a
        change of rate would leave a dozen runs describing one grid, and
        reading it as a label of its own would leave sixty stored labels.
        """
        values = 899.6560488246 + np.arange(60) * 0.1
        coord = NumericND.from_array(values)
        assert coord.runs_count == 1 and coord.evenly_sampled
        assert coord.step_exact == Fraction(1, 10)
        assert_within_a_bit(coord.values, values)
        # Every way in reads the same table from the same labels.
        assert table(NumericND.from_rows(coord.runs, dtype=coord.dtype)) == table(coord)
        pieces = [NumericND.from_array(values[i : i + 10]) for i in range(0, 60, 10)]
        assert concat(*pieces).runs_count == 1
        assert_within_a_bit(concat(*pieces).values, values)
        assert table(coord[:]) == table(coord)

    def test_a_label_further_off_is_not_on_the_grid(self):
        """A label a few last bits out is one no grid of the rest describes."""
        values = np.arange(20) * 0.1
        moved = values.copy()
        for _ in range(3):  # three last bits, where the fit allows two
            moved[9] = np.nextafter(moved[9], 1)
        coord = NumericND.from_array(moved)
        assert coord.runs_count > 1 and coord.labels is not None
        # The sample which is off the grid keeps its own label exactly,
        # rather than being absorbed into a grid which does not name it.
        assert coord.values[9] == moved[9]
        assert_within_a_bit(coord.values, moved)
        assert NumericND.from_array(values).runs_count == 1  # put back

    def test_a_step_no_magnitude_holds_is_refused_in_any_table(self):
        """A step which is int64's floor leaves the dtype, one row or two."""
        rows = [(0, 3, -(2**63), 1, 0), (5, 2, 1, 1, 0)]
        for table_rows in (rows[:1], rows):
            with pytest.raises(CoordinateError, match="exceeds the int64 range"):
                NumericND.from_rows(table_rows, dtype="int64")

    def test_padding_past_the_range_says_so(self):
        """A grid stepped outside int64 is not the far end of int64."""
        coord = NumericND.from_run(-(2**63) + 100, 1, 10)
        with pytest.raises((OverflowError, CoordinateError)):
            coord._sliced(-200, 1, 3)

    def test_a_fused_run_leaving_its_dtype_is_refused_by_name(self):
        """A join whose arithmetic leaves int64 is a coordinate error."""
        big = 2**61
        rows = [(-3 * big, 3, big, 1, 0), (0, 3, big, 1, 0)]
        first = NumericND.from_rows(rows[:1], dtype="int64")
        second = NumericND.from_rows(rows[1:], dtype="int64")
        for call in (
            lambda: concat(first, second),
            lambda: NumericND.from_rows(rows, dtype="int64"),
        ):
            with pytest.raises(CoordinateError, match="exceeds the int64 range"):
                call()

    def test_a_fused_run_has_to_fit_its_dtype(self):
        """
        Runs which continue each other are one run, which has to fit.

        Two runs which fit their dtype can make one which does not. They
        are one run all the same -- keeping them apart would make the table
        depend on how the labels were assembled rather than on what they
        are -- so such a join is refused by name instead.
        """
        num = 4 * 10**18 + 1
        first = NumericND.from_run(0, (num, 2), 2)
        second = NumericND.from_run(num, (num, 2), 2)
        with pytest.raises(CoordinateError, match="exceeds the"):
            concat(first, second)

    @pytest.mark.parametrize("cut", [1, 3, 7, 12])
    def test_one_table_however_the_labels_were_assembled(self, cut):
        """A coordinate is the table its labels make, not the join order."""
        whole = concat(
            NumericND.from_run(0, 2, 8),
            NumericND.from_run(100, 2, 4),
            NumericND.from_array(np.asarray([200, 203, 207])),
        )
        pieces = [whole[:cut], whole[cut : cut + 4], whole[cut + 4 :]]
        rebuilt = concat(*[x for x in pieces if len(x)])
        assert table(rebuilt) == table(whole)
        assert rebuilt.fingerprint() == whole.fingerprint()

    def test_a_gap_of_whole_samples_is_never_a_rounding(self):
        """A run which starts a sample late is a hole at any magnitude."""
        big = float(2**50)
        first = NumericND.from_run(big, 0.25, 3)
        late = NumericND.from_run(big + 2.75, 0.25, 3)
        joined = concat(first, late)
        assert joined.runs_count == 2 and joined.holes
        np.testing.assert_array_equal(
            joined.values, np.concatenate([first.values, late.values])
        )

    def test_a_declared_grid_which_leaves_its_dtype(self):
        """Labels an int8 grid cannot be stepped past are kept as they are."""
        values = np.asarray([125, 126, 127], dtype="int8")
        coord = NumericND.from_array(values, step=1)
        np.testing.assert_array_equal(coord.values, values)
        assert coord.dtype == np.dtype("int8")

    def test_a_grid_which_leaves_its_dtype_is_no_description(self):
        """Labels an int8 grid cannot be stepped through are kept as they are."""
        values = np.asarray([-120, 0, 120], dtype="int8")
        coord = NumericND.from_array(values)
        assert coord.labels is not None
        np.testing.assert_array_equal(coord.values, values)
        assert coord.dtype == np.dtype("int8")

    def test_a_float_gap_is_not_fused_away(self):
        """Two float runs meet only where their own last bits say they do."""
        joined = concat(
            NumericND.from_run(1e9, 1.0, 3), NumericND.from_run(1e9 + 3.0001, 1.0, 3)
        )
        assert joined.runs_count == 2 and joined.holes
        assert joined.values[3] == 1e9 + 3.0001

    def test_a_declared_step_does_not_move_a_label(self):
        """A label a rounding off the grid it is declared on stays itself."""
        values = np.asarray([0.0, 1.0000001, 2.0])
        coord = NumericND.from_array(values, step=1.0)
        np.testing.assert_array_equal(coord.values, values)

    def test_a_fractional_integer_grid_rescales_its_labels(self):
        """The labels of a floored grid are not its ideal positions."""
        coord = NumericND.from_run(0, (3, 2), 5)
        np.testing.assert_array_equal(coord.values, [0, 1, 3, 4, 6])
        np.testing.assert_allclose(coord.rescaled(2).values, [0, 2, 6, 8, 12])

    def test_labels_which_are_no_order_apart(self):
        """Two int64 labels a whole range apart are compared, not differenced."""
        values = np.asarray([-(2**63) + 1, 2**63 - 1])
        coord = NumericND.from_array(values, detect=False)
        assert coord.sorted and not coord.reverse_sorted
        assert coord.min() == values[0] and coord.max() == values[1]
        out, _ = coord.select(values[0], values[1])
        np.testing.assert_array_equal(out.values, values)

    def test_a_time_past_the_nanosecond_range(self):
        """A date no nanosecond count can hold is refused, not wrapped."""
        values = np.asarray(["2300-01-01", "2301-01-01"], dtype="datetime64[D]")
        with pytest.raises(CoordinateError, match="nanosecond range"):
            NumericND.from_array(values)

    def test_stored_labels_are_restated_in_nanoseconds(self):
        """Labels handed to a table keep the instants they name, or are refused."""
        table = runs_from_rows([(0, 3, 0, 0, 0)], "timedelta64[ns]")
        labels = np.asarray([0, 1001, 2002], dtype="timedelta64[ps]")
        with pytest.raises(CoordinateError, match="whole number of nanoseconds"):
            NumericND.from_rows(table, labels=labels, dtype="timedelta64[ns]")
        whole = np.asarray([0, 1000, 2000], dtype="timedelta64[ps]")
        coord = NumericND.from_rows(table, labels=whole, dtype="timedelta64[ns]")
        np.testing.assert_array_equal(coord.values, whole.astype("timedelta64[ns]"))


class TestLookup:
    """The table answers value windows the way its labels do."""

    @pytest.mark.parametrize("reverse", [False, True])
    def test_random_windows(self, holed, reverse):
        """Random windows give the samples a mask over the labels gives."""
        coord = holed[::-1] if reverse else holed
        values = np.asarray(coord.values)
        rng = np.random.default_rng(0)
        bounds = [*values, *(values + MS), *(values - MS), None]
        for _ in range(100):
            low, high = (bounds[i] for i in rng.integers(0, len(bounds), 2))
            if low is not None and high is not None and low > high:
                low, high = high, low
            out, indexer = coord.select(low, high)
            mask = np.ones(len(values), dtype=bool)
            if low is not None:
                mask &= values >= low
            if high is not None:
                mask &= values <= high
            np.testing.assert_array_equal(out.values, values[mask])
            np.testing.assert_array_equal(values[indexer], values[mask])

    def test_window_across_a_hole(self, holed):
        """A window spanning a hole keeps the samples on both sides."""
        values = holed.values
        out, indexer = holed.select(values[5], values[15])
        np.testing.assert_array_equal(out.values, values[5:16])
        assert indexer == slice(5, 16)
        assert out.runs_count == 2

    def test_window_inside_a_hole(self, holed):
        """A window which falls entirely in a hole selects nothing."""
        values = holed.values
        out, indexer = holed.select(values[9] + MS, values[10] - MS)
        assert len(out) == 0
        assert indexer == slice(0, 0)

    def test_open_ends(self, holed):
        """An open bound reaches the coordinate's end."""
        values = holed.values
        out, _ = holed.select(values[12], None)
        np.testing.assert_array_equal(out.values, values[12:])
        out, _ = holed.select(None, values[12])
        np.testing.assert_array_equal(out.values, values[:13])

    def test_a_bound_which_is_a_label_of_its_own_coordinate(self):
        """A window of one label keeps that label, at any float magnitude."""
        coord = NumericND.from_run(1e6, 0.1, 1000)
        for index in (1, 2, 3, 17, 999):
            value = coord.values[index]
            assert coord.index_of(value) == index
            assert coord.index_of(value, forward=False) == index
            out, _ = coord.select(value, value)
            np.testing.assert_array_equal(out.values, [value])

    @pytest.mark.parametrize("dtype", ["float64", "float32", "float16"])
    @pytest.mark.parametrize(
        ("start", "step", "count"),
        [(0.0, 0.1, 50), (1e6, 0.1, 50), (-5.0, 0.25, 20), (0.0, 1 / 3, 40)],
    )
    def test_every_float_label_finds_itself(self, start, step, count, dtype):
        """Each label of a float grid is the index the grid gives it back."""
        kind = np.dtype(dtype).type
        if abs(start) > np.finfo(dtype).max:
            pytest.skip(f"{start} is no {dtype} label")
        coord = NumericND.from_run(kind(start), kind(step), count, dtype=dtype)
        values = np.asarray(coord.values)
        # A narrower float holds the label, not the position it came from,
        # so the query is the label itself and the tolerance is nothing.
        bound = abs(step) * 1e-10 if dtype == "float64" else 0.0
        for index in range(count):
            value = float(values[index])
            forward = next(j for j in range(count) if values[j] >= value - bound)
            backward = max(j for j in range(count) if values[j] <= value + bound)
            assert (forward, backward) == (index, index)
            assert coord.index_of(value) == forward
            assert coord.index_of(value, forward=False) == backward

    def test_bounds_between_samples(self):
        """A bound between two labels selects from the next label inward."""
        coord = grid_1024(100)
        values = coord.values
        out, _ = coord.select(values[10] + NS, values[20] - NS)
        np.testing.assert_array_equal(out.values, values[11:20])

    def test_reverse_sorted(self, holed):
        """Selection on a reversed coordinate mirrors the forward one."""
        values = holed.values
        out, _ = holed[::-1].select(values[5], values[15])
        np.testing.assert_array_equal(out.values, values[5:16][::-1])

    def test_index_of_a_bound_past_the_open_end(self):
        """A bound past the end the search runs towards has no index."""
        coord = grid_1024(10)
        second = np.timedelta64(1, "s")
        # Forward is the first label at or past the bound, so a bound before
        # every label is an open end rather than an index.
        assert coord.index_of(coord.values[0] - second) is None
        assert coord.index_of(coord.values[-1] + second, forward=False) is None
        assert coord.index_of(None) is None

    def test_a_bound_of_the_wrong_kind(self):
        """A time cannot be looked up in a coordinate of numbers."""
        with pytest.raises(CoordinateError, match="not a bound"):
            NumericND.from_run(0, 1, 5).select(T0, None)

    def test_bounds_are_ordered(self):
        """A window given the other way round is the same window."""
        coord = grid_1024(10)
        values = coord.values
        out, _ = coord.select(values[6], values[2])
        np.testing.assert_array_equal(out.values, values[2:7])


class TestHoles:
    """A hole is a run which does not continue the one before it."""

    def test_a_change_of_rate_is_no_hole(self):
        """A change of rate at the next sample is not a hole."""
        assert not concat(
            NumericND.from_run(0, 2, 5), NumericND.from_run(10, 3, 4)
        ).holes

    def test_a_null_label_is_no_distance(self):
        """A NaN at a boundary states no spacing, so it opens no hole."""
        grid = NumericND.from_run(10.0, 1.0, 3)
        for values in ([0.0, 1.0, np.nan], [0.0, np.nan, 2.0]):
            stored = NumericND.from_array(np.asarray(values), detect=False)
            assert not concat(stored, grid).holes
        stated = NumericND.from_array(np.asarray([0.0, 1.0, 2.1]), detect=False)
        assert concat(stated, grid).holes

    def test_a_late_tick_run_after_a_stored_one_is_a_hole(self):
        """Ticks past 2**53 still difference exactly, so the hole is seen."""
        stored = NumericND.from_array(T0 + np.asarray([0, 10, 21]) * NS, detect=False)
        far = NumericND.from_run(T0 + 100 * NS, np.timedelta64(10, "ns"), 4)
        assert concat(stored, far).holes
        near = NumericND.from_run(T0 + 31 * NS, np.timedelta64(10, "ns"), 4)
        assert not concat(stored, near).holes


# Generated with the DASCore branch's NumericND.run_fingerprints; they are
# the identity a spool index stores one run under, so unidas must reproduce
# them bit for bit.
GOLDEN_RUN_HASHES = {
    "grid_1024": [5871592723391482875],
    "grid_1024_strided": [2509450830044298997],
    "grid_1024_reversed": [9108969104947110774],
    "two_runs_with_a_hole": [13023371710683040167, 994928767540319717],
    "float_tenths": [13753832695304875904],
    "integer_grid": [17276449886061474500],
    "stored_jitter": [8718111513327086440],
    "mixed_grid_and_stored": [
        17841300044571141381,
        8718111513327086440,
        14517662770452402747,
    ],
    "many_runs": [
        17841300044571141381,
        7287613892410693493,
        7029901093473631130,
        8175125493887339375,
        16313530902577049983,
        14916759949766888719,
        16057962133813436127,
        6014240484198604076,
        15185305833524573422,
        17199031704384190948,
        5551128981773662467,
        12600156178743854687,
        9211258193784855422,
        14951510686032451158,
        6359583499851523704,
        10556017160131122707,
        14748694045594974773,
        17431083965012731679,
        3028768942990624328,
        5594497855869343929,
        991579678663827079,
        5686757527644516425,
        2114965618128938722,
        3752753713960163646,
        13417836774415794844,
        12824306618518178844,
        3426923138019833837,
        11060819356645534518,
        4921749479193103447,
        9188189946509992084,
        16439332684247809018,
        16553498740583803437,
        17536877509394091495,
        14137853921573753328,
        11798853741205251215,
        8636802343555858915,
        9469137403373392502,
        6171474592309546701,
        18367101711747642589,
        18101931185682512712,
    ],
}


class TestPlatformIntegers:
    """Python integers count in int64 on every platform."""

    def test_python_integers_are_int64(self):
        """A run or labels given as Python ints state int64, as on Linux."""
        assert NumericND.from_run(3, 2, 10).dtype == np.dtype("int64")
        assert NumericND.from_array([1, 2, 4]).dtype == np.dtype("int64")
        assert NumericND.from_array([1, 2, 4]).values.dtype == np.dtype("int64")

    def test_a_join_widens_within_a_kind(self):
        """int32 beside int64 joins as int64; float32 beside float64 as float64."""
        narrow = NumericND.from_array(np.array([200, 203, 207], dtype="int32"))
        joined = concat(NumericND.from_run(0, 2, 8), narrow)
        assert joined.dtype == np.dtype("int64")
        np.testing.assert_array_equal(joined.values, [*range(0, 16, 2), 200, 203, 207])
        floats = concat(
            NumericND.from_run(np.float32(0), 0.5, 4), NumericND.from_run(10.0, 0.5, 2)
        )
        assert floats.dtype == np.dtype("float64")
        with pytest.raises(CoordinateError, match="share a dtype"):
            concat(NumericND.from_run(0, 1, 3), NumericND.from_run(0.0, 1.0, 3))

    def test_a_narrow_integer_array_keeps_its_dtype(self):
        """An array which states int32 stays int32."""
        narrow = np.array([1, 2, 4], dtype="int32")
        assert NumericND.from_array(narrow).dtype == np.dtype("int32")
        assert NumericND.from_run(np.int32(3), 2, 10).dtype == np.dtype("int32")


def build_tables(cls, join):
    """
    The coordinates the golden vectors were generated from.

    Built from the arguments rather than from a table, so that DASCore can
    be handed the same ones and the two are compared on what they make of
    them as well as on how they hash it.
    """
    grid = cls.from_run(start=T0, step=Fraction(1, 1024), shape=2048)
    return {
        "grid_1024": grid,
        "grid_1024_strided": grid[7::3],
        "grid_1024_reversed": grid[::-1],
        "two_runs_with_a_hole": join(
            cls.from_run(T0, STEP, 100), cls.from_run(T0 + STEP * 110, STEP, 50)
        ),
        "float_tenths": cls.from_run(0.0, 0.1, 50),
        "integer_grid": cls.from_run(3, 2, 10),
        "stored_jitter": cls.from_array(JITTER),
        "mixed_grid_and_stored": join(
            cls.from_run(T0, STEP, 10),
            cls.from_array(JITTER),
            cls.from_run(T0 + STEP * 50, STEP, 8),
        ),
        # Past the small-table fold, so the vectorised one is pinned too.
        "many_runs": join(
            *[cls.from_run(T0 + STEP * 100 * i, STEP, 10) for i in range(40)]
        ),
    }


@pytest.fixture(scope="module")
def tables():
    """The coordinates the golden vectors were generated from."""
    return build_tables(NumericND, concat)


class TestHashing:
    """Runs hash where they are and the coordinate hashes their order."""

    @pytest.mark.parametrize("name", sorted(GOLDEN_RUN_HASHES))
    def test_golden_run_hashes(self, tables, name):
        """The run hashes are the ones the DASCore branch states."""
        assert [int(x) for x in tables[name].run_hashes] == GOLDEN_RUN_HASHES[name]

    def test_matches_dascore(self, tables):
        """The same inputs make the same table, and it hashes the same."""
        dascore_coords = pytest.importorskip("dascore.core.coords")
        if not hasattr(dascore_coords, "NumericND"):
            pytest.skip("This DASCore has no run table to compare against.")
        theirs = build_tables(dascore_coords.NumericND, dascore_coords.concat_tables)
        for name, coord in tables.items():
            other = theirs[name]
            assert np.array_equal(np.asarray(coord.runs), np.asarray(other.runs)), name
            expected = [int(x) for x in np.asarray(other.run_fingerprints)]
            assert [int(x) for x in coord.run_hashes] == expected, name

    def test_a_table_past_the_small_fold(self, tables):
        """The forty-run vector really is folded the vectorised way."""
        from unidas import numeric

        assert tables["many_runs"].runs_count > numeric._SMALL_TABLE

    def test_run_hash_is_position_independent(self):
        """The same run hashes the same wherever it sits in a table."""
        early = NumericND.from_run(T0, STEP, 5)
        late = NumericND.from_run(T0 + STEP * 50, STEP, 5)
        other = NumericND.from_run(T0 + STEP * 200, STEP, 5)
        first = concat(early, late)
        second = concat(early, other, late)
        assert first.run_hashes[-1] == second.run_hashes[-1]
        assert first.run_hashes[0] == second.run_hashes[0]

    @pytest.mark.parametrize("kind", ["time", "float", "int", "timedelta"])
    def test_both_folds_agree(self, monkeypatch, kind, mixed):
        """The small-table integer fold is the vectorised one, bit for bit."""
        from unidas import numeric

        coords = {
            "time": mixed,
            "float": NumericND.from_array(
                np.concatenate([np.arange(10) * 0.5, 100 + np.arange(10) * 0.5])
            ),
            "int": concat(
                NumericND.from_run(0, 3, 5),
                NumericND.from_run(100, 3, 5),
            ),
            "timedelta": NumericND.from_run(
                np.timedelta64(0, "s"), np.timedelta64(2, "s"), 9
            ),
        }
        coord = coords[kind]
        assert coord.runs_count <= numeric._SMALL_TABLE
        small = np.asarray(coord.run_hashes).copy()
        # the same table again, folded by the vectorised path alone
        monkeypatch.setattr(numeric, "_SMALL_TABLE", -1)
        other = NumericND.from_rows(
            coord.runs, labels=coord.labels, dtype=coord.dtype, units=coord.units
        )
        assert other is not coord
        np.testing.assert_array_equal(small, other.run_hashes)

    def test_order_matters(self):
        """The same runs in another order are another coordinate."""
        early = NumericND.from_run(T0, STEP, 5)
        late = NumericND.from_run(T0 + STEP * 50, STEP, 5)
        assert concat(early, late) != concat(late, early)

    def test_stored_labels_are_hashed(self, mixed):
        """Two stored runs of different labels hash differently."""
        moved = concat(mixed[:10], mixed[10:15].translated(NS), mixed[15:])
        assert moved.run_hashes[0] == mixed.run_hashes[0]
        assert moved.run_hashes[1] != mixed.run_hashes[1]
        assert moved != mixed

    def test_equal_after_split_and_concat(self, mixed):
        """Cutting a coordinate up and rejoining it gives the same identity."""
        parts = [mixed[:4], mixed[4:9], mixed[9:20], mixed[20:]]
        assert concat(*parts).fingerprint() == mixed.fingerprint()

    def test_dtype_separates_tables(self):
        """An integer table and a float table of the same numbers differ."""
        ints = NumericND.from_run(0, 1, 5)
        floats = NumericND.from_run(0.0, 1.0, 5)
        assert ints.run_hashes[0] != floats.run_hashes[0]
        assert ints != floats

    def test_units_separate_coordinates(self):
        """The same labels in different units are different coordinates."""
        metres = NumericND.from_run(0.0, 1.0, 5, units="m")
        feet = NumericND.from_run(0.0, 1.0, 5, units="ft")
        assert metres != feet
        assert metres.fingerprint() != feet.fingerprint()

    def test_not_hashable(self):
        """Coordinates state a fingerprint rather than a python hash."""
        with pytest.raises(TypeError):
            hash(grid_1024(10))

    def test_equality_with_something_else(self):
        """A coordinate is not equal to a thing which is not one."""
        assert grid_1024(10) != "a coordinate"


class TestUpdates:
    """Shifting and rescaling a table."""

    def test_translate_a_table(self, mixed):
        """Every label of every run moves by the same amount."""
        moved = mixed.translated(np.timedelta64(7, "ms"))
        np.testing.assert_array_equal(
            moved.values, mixed.values + np.timedelta64(7, "ms")
        )
        assert moved.runs_count == mixed.runs_count

    def test_translate_a_float_table(self):
        """A float coordinate shifts by a number."""
        coord = NumericND.from_run(0.0, 0.5, 5)
        np.testing.assert_allclose(coord.translated(2.0).values, coord.values + 2)

    def test_rescale_a_grid(self):
        """A grid run keeps its count and maps its start and spacing."""
        coord = NumericND.from_run(0, 1, 5, units="m")
        out = coord.rescaled(100.0, units="cm")
        assert out.runs_count == 1
        assert out.units == "cm"
        np.testing.assert_allclose(out.values, np.arange(5) * 100.0)

    def test_rescale_with_an_offset(self):
        """An affine unit adds its offset to every label."""
        coord = NumericND.from_run(0.0, 1.0, 4)
        out = coord.rescaled(1.8, offset=32.0)
        np.testing.assert_allclose(out.values, [32.0, 33.8, 35.6, 37.4])

    def test_rescale_a_stored_run(self):
        """Stored labels are scaled where they lie."""
        coord = NumericND.from_array(np.asarray([0.0, 1.0, 2.5, 9.0]))
        np.testing.assert_allclose(coord.rescaled(2.0).values, [0.0, 2.0, 5.0, 18.0])

    def test_translate_a_float_table_with_stored_labels(self):
        """Stored float labels move with the runs beside them."""
        coord = concat(
            NumericND.from_run(0.0, 1.0, 3),
            NumericND.from_array(np.asarray([10.0, 10.5, 12.25]), detect=False),
        )
        moved = coord.translated(2.5)
        assert moved.labels is not None
        np.testing.assert_allclose(moved.values, coord.values + 2.5)

    def test_rescale_keeps_the_units_it_was_not_given(self):
        """A factor alone says nothing about what the labels now measure."""
        coord = NumericND.from_run(0.0, 1.0, 5, units="m")
        assert coord.rescaled(1000.0).units == "m"
        assert coord.rescaled(1000.0, units="mm").units == "mm"
        assert NumericND.from_run(0.0, 1.0, 5).rescaled(2.0).units is None

    def test_rescale_refuses_a_time(self):
        """A time is counted in nanoseconds, which is not a unit to convert."""
        with pytest.raises(CoordinateError, match="nanoseconds"):
            grid_1024(10).rescaled(2.0)

    def test_rescale_falls_back_to_labels(self):
        """A scaled spacing with no fraction of ticks keeps its labels."""
        coord = NumericND.from_run(0.0, 1.0, 4)
        out = coord.rescaled(1e300)
        np.testing.assert_allclose(out.values, np.arange(4) * 1e300)


class TestDeclaredStep:
    """A grid a source states beside labels the runs cannot describe."""

    def _gappy(self):
        """Labels on a grid of ten, every other position missing."""
        steps = np.where(np.arange(2000) % 2, 10, 20)
        return np.concatenate([[0], np.cumsum(steps)]).astype("int64")

    def test_a_declared_step_survives_the_dense_array_guard(self):
        """Labels too gappy to be read as runs keep the grid they declare."""
        values = self._gappy()
        coord = NumericND.from_array(values, step=10)
        assert coord.runs_count == 1 and coord.labels is not None
        assert coord.step == 10 and coord.step_exact == Fraction(10)
        np.testing.assert_array_equal(coord.values, values)
        # None was declared, so nothing states one.
        assert NumericND.from_array(values).step is None

    def test_a_declared_step_states_what_a_hole_is(self):
        """A stored run is held to the step declared on it, not its median."""
        # Every other position of a grid of ten, so the labels' own median
        # spacing is twenty and only the declared step calls the next run
        # a hole. The declared grid travels with them through the join.
        values = np.asarray([0, 20, 40, 60, 80])
        late = NumericND.from_run(100, 10, 3)
        declared = NumericND.from_array(values, detect=False, step=10)
        assert declared.declared_step == 10
        joined = concat(declared, late)
        assert joined.step == 10 and joined.holes
        plain = NumericND.from_array(values, detect=False)
        assert plain.step is None
        assert not concat(plain, NumericND.from_run(100, 20, 3)).holes

    def test_a_join_states_a_step_only_when_the_rates_agree(self):
        """Two runs of 2 and of 3/2 ticks both round to 2, and are not one."""
        joined = concat(
            NumericND.from_run(0, 2, 3), NumericND.from_run(8, Fraction(3, 2), 3)
        )
        np.testing.assert_array_equal(joined.values, [0, 2, 4, 8, 9, 11])
        assert joined.step is None and joined.step_exact is None

    def test_one_grid_spelled_two_ways_is_one_grid(self):
        """A step of 1 and of 1.0 declare the same grid, so they join as one."""
        later = NumericND.from_run(6.0, 1.0, 3)
        answers = set()
        for step in (1, 1.0):
            first = NumericND.from_array(
                np.asarray([0.0, 2.0, 4.0]), step=step, detect=False
            )
            joined = concat(first, later)
            answers.add((float(joined.step), joined.holes))
        assert answers == {(1.0, True)}

    def test_a_step_which_contradicts_the_labels_is_refused(self):
        """Labels off the grid they are declared on are not on it."""
        with pytest.raises(CoordinateError, match="not on a grid"):
            NumericND.from_array(np.asarray([0, 1, 2, 4, 5]), step=2)

    @pytest.mark.parametrize("step", [0, np.inf, np.nan, -0.0])
    def test_a_step_which_is_no_spacing(self, step):
        """A zero or non-finite step describes no grid at all."""
        with pytest.raises(CoordinateError, match="finite non-zero spacing"):
            NumericND.from_array(np.asarray([0.0, 1.0, 2.0]), step=step)

    def test_a_fractional_step_is_not_declared_on_labels(self):
        """Labels of a fractional grid are floors, so they do not state it."""
        with pytest.raises(CoordinateError, match="from_run instead"):
            NumericND.from_array(np.asarray([0.0, 0.5, 1.0]), step=Fraction(1, 2))

    def test_a_whole_fraction_is_the_step_it_names(self):
        """A fraction of one denominator is the whole step it states."""
        coord = NumericND.from_array(np.asarray([0, 2, 4]), step=Fraction(2, 1))
        assert coord.step == 2 and coord.evenly_sampled

    @pytest.mark.parametrize("values", [np.zeros((2, 2)), np.asarray([0, 2, 1])])
    def test_a_declared_step_needs_one_monotonic_axis(self, values):
        """A grid runs along one axis, in one direction."""
        with pytest.raises(CoordinateError, match="one-dimensional, monotonic"):
            NumericND.from_array(values, step=1)

    def test_a_declared_step_on_one_value(self):
        """One label states no spacing, so the declared one is its grid."""
        coord = NumericND.from_array(np.asarray([5.0]), step=1)
        assert coord.step == 1 and len(coord) == 1
        np.testing.assert_array_equal(coord.values, [5.0])


class TestWithLength:
    """The same grid over another number of samples."""

    @pytest.mark.parametrize("length", [0, 1, 4, 5, 9])
    def test_a_grid_is_stepped_to_its_new_length(self, length):
        """A grid answers for the samples it never held."""
        coord = NumericND.from_run(3, 2, 5)
        out = coord.with_length(length)
        assert len(out) == length
        np.testing.assert_array_equal(out.values, 3 + np.arange(length) * 2)

    def test_a_fractional_grid_keeps_its_phase(self):
        """Lengthening a 1024 Hz grid does not round its period."""
        out = grid_1024(10).with_length(2048)
        np.testing.assert_array_equal(out.values, exact_labels(T0, 2048))

    def test_only_one_grid_run_can_be_given_a_length(self):
        """Labels a run holds say nothing about the samples beside them."""
        stored = NumericND.from_array(JITTER)
        with pytest.raises(CoordinateError, match="single grid run"):
            stored.with_length(9)
        joined = concat(NumericND.from_run(0, 1, 3), NumericND.from_run(10, 1, 3))
        with pytest.raises(CoordinateError, match="single grid run"):
            joined.with_length(9)

    def test_a_length_which_is_no_count(self):
        """A coordinate holds a whole, non-negative number of samples."""
        coord = NumericND.from_run(0, 1, 5)
        with pytest.raises(CoordinateError, match="cannot hold"):
            coord.with_length(-1)
        for length in (4.9, -0.5):
            with pytest.raises(CoordinateError, match="whole number of samples"):
                coord.with_length(length)


class TestTableGuards:
    """A run table has to describe the labels it is handed."""

    def test_a_coordinate_states_its_dtype(self):
        """Without labels to read one from, the dtype has to be given."""
        table = runs_from_rows([(0, 3, 1, 1, 0)], "int64")
        with pytest.raises(CoordinateError, match="states its dtype"):
            NumericND(runs=table, dtype=None, labels=None)

    def test_a_run_table_has_the_fields_of_a_run(self):
        """Plain numbers are no table; the fields are what states a run."""
        with pytest.raises(CoordinateError, match="needs the fields"):
            NumericND(runs=np.zeros((3, 2)), dtype="int64")

    def test_a_denominator_is_never_negative(self):
        """Zero states stored labels; below it states nothing."""
        table = runs_from_rows([(0, 3, 1, -1, 0)], "int64")
        with pytest.raises(CoordinateError, match="denominator is positive"):
            NumericND(runs=table, dtype="int64")

    def test_the_stored_runs_hold_the_labels_given(self):
        """A stored run is as long as the labels it reads."""
        table = runs_from_rows([(0, 3, 0, 0, 0)], "int64")
        with pytest.raises(CoordinateError, match="3 samples but 2 labels"):
            NumericND.from_rows(table, labels=np.asarray([0, 1]), dtype="int64")

    def test_a_coordinate_names_one_dim_per_axis(self):
        """Dims name the axes, so there is one of them for each."""
        with pytest.raises(CoordinateError, match="has no dims"):
            NumericND.from_run(0, 1, 5, dims=("x", "y"))


class TestNarrowDtypes:
    """A coordinate keeps the dtype its labels have."""

    @pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "uint16", "float32"])
    def test_dtype_survives(self, dtype):
        """A narrow coordinate is held in its own dtype, not widened."""
        values = np.arange(5, dtype=dtype)
        coord = NumericND.from_array(values)
        assert coord.dtype == np.dtype(dtype)
        assert coord.values.dtype == np.dtype(dtype)

    def test_unsigned_label_past_the_signed_range_is_refused(self):
        """Every tick is counted in int64, whatever uint64 could hold."""
        values = np.asarray([2**63 + 1, 2**63 + 2], dtype="uint64")
        with pytest.raises(CoordinateError, match="int64 range"):
            NumericND.from_array(values)

    def test_narrow_range_is_guarded(self):
        """A grid which would wrap its own dtype is refused."""
        with pytest.raises(CoordinateError, match="exceeds"):
            NumericND.from_run(np.int8(100), np.int8(1), 100)


class TestRunDetection:
    """The kernel's run detection cuts where the spacing changes."""

    def test_a_change_of_rate_cuts_at_the_shared_sample(self):
        """The sample two runs meet at belongs to the earlier of them."""
        starts, on_grid, count = detect_runs(np.asarray([0, 2, 4, 6, 9, 12, 15]))
        # The label 6 closes the run spaced two apart; 9 opens the one
        # spaced three apart, rather than both claiming the sample between.
        assert starts.tolist() == [0, 4]
        assert on_grid.tolist() == [True, True]
        assert count == 2

    def test_a_lone_sample_states_no_grid(self):
        """A block of one sample is stored, not a run of its own."""
        _, on_grid, _ = detect_runs(np.asarray([0, 1, 2, 30, 60, 61, 62]))
        assert on_grid.tolist() == [True, False, True]

    def test_the_guard_declines_a_jittered_array(self):
        """Too many runs to be worth detecting leaves the labels whole."""
        from unidas.numeric import guard_declines

        assert guard_declines(10_000, 2_000)
        assert not guard_declines(10_000, 20)
        assert not guard_declines(10, 9)


class TestTicks:
    """Every label is a whole tick, or a float which needs none."""

    def test_to_tick_of_a_time(self):
        """A time is a count of nanoseconds."""
        assert to_tick(np.timedelta64(2, "ms")) == 2_000_000

    def test_to_tick_of_a_whole_float(self):
        """A float which names an integer is that integer."""
        assert to_tick(3.0) == 3

    def test_to_tick_refuses_what_is_no_tick(self):
        """Something which is not a number states no tick at all."""
        with pytest.raises(CoordinateError, match="not an integer or time"):
            to_tick("half past three")


class TestOddCorners:
    """The paths a coordinate only takes when it has to."""

    def test_repr_states_what_the_table_is(self):
        """A coordinate says its shape, dtype, runs, step, units and dims."""
        coord = NumericND.from_run(0.0, 0.5, 4, units="m", dims=("distance",))
        text = repr(coord)
        assert "shape=(4,)" in text and "runs=1" in text
        assert "step=1/2" in text and "units='m'" in text
        assert "dims=('distance',)" in text
        assert "step" not in repr(NumericND.from_array(np.asarray([3.0, 1.0, 2.0])))

    def test_a_float_table_reverses(self):
        """Several float runs turn round together."""
        coord = concat(
            NumericND.from_run(0.0, 0.5, 4), NumericND.from_run(100.0, 0.25, 4)
        )
        np.testing.assert_array_equal(coord[::-1].values, coord.values[::-1])

    def test_a_sample_no_grid_describes_is_stored(self):
        """Only the samples a grid cannot give back keep their labels."""
        values = np.asarray([0.0, 0.1, 0.2, 0.3, 10.0, 10.5, 11.0, 11.2])
        coord = NumericND.from_array(values)
        # Two rates and a sample on neither, which is the one stored.
        assert coord.runs_count == 3
        assert coord.labels.tolist() == [11.2]
        np.testing.assert_array_equal(coord.values, values)

    def test_a_subnormal_step_is_no_ratio_of_ticks(self):
        """A spacing with no fraction leaves the labels as the description."""
        coord = NumericND.from_run(0.0, 5e-324, 4)
        assert coord.labels is not None
        np.testing.assert_array_equal(coord.values, np.arange(4) * 5e-324)

    def test_from_rows_needs_a_dtype_or_labels(self):
        """A table of runs says nothing about what its labels are."""
        with pytest.raises(CoordinateError, match="states its dtype"):
            NumericND.from_rows(runs_from_rows([(0, 4, 1, 1, 0)], "int64"))

    def test_a_declared_step_without_detection(self):
        """Labels kept whole still have to sit on the grid they declare."""
        values = np.asarray([1, 2, 3, 9, 10])
        coord = NumericND.from_array(values, step=1, detect=False)
        assert coord.runs_count == 1 and coord.labels is not None
        np.testing.assert_array_equal(coord.values, values)

    def test_a_shape_states_the_sample_count(self):
        """A run takes its count as a number or as a one-axis shape."""
        assert len(NumericND.from_run(0, 1, (5,))) == 5

    def test_selection_needs_one_axis(self):
        """A window over an N-D coordinate is no window at all."""
        with pytest.raises(CoordinateError, match="one dimensional"):
            NumericND.from_array(np.zeros((2, 3))).select(0.0, 1.0)

    def test_a_bound_is_one_value(self):
        """A window is bounded by labels, not by arrays of them."""
        with pytest.raises(CoordinateError, match="one value"):
            grid_1024(10).select(np.zeros(3), None)
