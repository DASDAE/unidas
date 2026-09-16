"""Tests for the categorical coordinate."""

import numpy as np
import pytest

from unidas import Categorical, CoordinateError, NumericND

T0 = np.datetime64("2020-01-01", "ns")


def labels_of(coord):
    """The labels as a plain list, for comparing with what was asked for."""
    return np.asarray(coord.values).ravel().tolist()


class TestCanonicalOrder:
    """The categories are unique, and in the one order the rule states."""

    def test_text_is_sorted(self):
        """Labels of one type keep that type and numpy's order."""
        coord = Categorical.from_labels(["south", "north", "south"])
        assert list(coord.categories) == ["north", "south"]
        assert coord.categories.dtype.kind == "U"
        assert labels_of(coord) == ["south", "north", "south"]

    def test_numbers_are_sorted(self):
        """Numeric labels are held as numbers, not as their text."""
        coord = Categorical.from_labels([3, 1, 2, 1])
        assert list(coord.categories) == [1, 2, 3]
        assert coord.categories.dtype.kind in "iu"

    def test_nulls_sort_last(self):
        """A NaN is not below or above anything, so it goes last."""
        coord = Categorical.from_labels([np.nan, 2.0, 1.0, np.nan])
        assert coord.categories[:2].tolist() == [1.0, 2.0]
        assert np.isnan(coord.categories[-1])
        assert len(coord.categories) == 3
        assert coord.codes.values.tolist() == [2, 1, 0, 2]

    @pytest.mark.parametrize("mixed", [False, True])
    def test_null_times_sort_last(self, mixed):
        """A NaT is one category, and the last of them."""
        values = [T0, np.datetime64("NaT", "ns"), T0 - np.timedelta64(1, "s")]
        # A label of another kind takes the mixed path, which orders and
        # dedupes the same times itself rather than leaving it to numpy.
        extra = ["x"] if mixed else []
        coord = Categorical.from_labels(np.asarray(values + extra, dtype=object))
        times = [x for x in coord.categories if isinstance(x, np.datetime64)]
        assert np.isnat(times[-1])
        assert len(coord.categories) == 3 + len(extra)

    def test_mixed_types_rank_before_text(self):
        """Numbers, then text, then anything else by its repr."""
        coord = Categorical.from_labels([{"a": 1}, "b", 2, "a", np.nan])
        kinds = [type(x) for x in coord.categories]
        assert kinds == [int, float, str, str, dict]
        assert np.isnan(coord.categories[1])
        assert coord.categories[2] == "a"

    def test_one_number_however_it_is_spelled(self):
        """1, 1.0 and True are one category; 1 and "1" are two."""
        coord = Categorical.from_labels([1, 1.0, True, "x"])
        assert len(coord.categories) == 2
        assert labels_of(coord) == [1, 1, 1, "x"]
        two = Categorical.from_labels([1, "1"])
        assert len(two.categories) == 2

    @pytest.mark.parametrize("mixed", [False, True])
    def test_bytes_are_decoded(self, mixed):
        """A byte string is the text it holds."""
        values = [b"north", b"south"]
        if not mixed:
            coord = Categorical.from_labels(np.asarray(values))
            assert coord.categories.dtype.kind == "U"
        else:
            # Beside a number the labels take the mixed path, which decodes
            # each of them itself.
            coord = Categorical.from_labels(np.asarray([*values, 1], dtype=object))
        assert [x for x in coord.categories if isinstance(x, str)] == [
            "north",
            "south",
        ]

    @pytest.mark.parametrize("first", [1, 1.0, True])
    def test_one_number_is_spelled_one_way(self, first):
        """Which spelling was read first does not change the category."""
        coord = Categorical.from_labels([first, 1, 1.0, "x"])
        plain = Categorical.from_labels([1, "x"])
        assert list(coord.categories) == list(plain.categories) == [1, "x"]
        assert [type(x) for x in coord.categories] == [int, str]
        assert (
            coord.fingerprint() == Categorical.from_labels([1, 1, 1, "x"]).fingerprint()
        )

    def test_a_number_no_integer_holds_keeps_its_own_spelling(self):
        """A category which is not whole is the float it names."""
        coord = Categorical.from_labels([2.5, "x"])
        assert list(coord.categories) == [2.5, "x"]
        assert isinstance(coord.categories[0], float)
        # However it was spelled, so that a wider float is not a category
        # of its own beside the double which names the same number.
        assert coord == Categorical.from_labels([np.float32(2.5), "x"])

    def test_a_number_no_double_holds_is_not_merged_into_one(self):
        """A wider float which is not the double it rounds to stays itself."""
        wide = np.longdouble("1.0000000000000000001")
        if wide == np.longdouble(1):
            pytest.skip("This platform's long double is a double.")
        coord = Categorical.from_labels([wide, 1, "x"])
        assert len(coord.categories) == 3
        np.testing.assert_array_equal(coord.isin(wide), [True, False, False])
        np.testing.assert_array_equal(coord.isin(1), [False, True, False])

    def test_negative_zero_is_zero(self):
        """-0.0 and 0.0 are one label."""
        coord = Categorical.from_labels([-0.0, 0.0, 1.0])
        assert len(coord.categories) == 2
        assert str(coord.categories[0]) == "0.0"

    def test_a_label_numpy_would_round_stays_itself(self):
        """An integer no float can hold is kept, not read as a float."""
        big = 10**17 + 1
        coord = Categorical.from_labels([1.5, big, 1.5])
        assert coord.categories.dtype.kind == "O"
        assert labels_of(coord) == [1.5, big, 1.5]

    def test_an_unhashable_label(self):
        """A label with no hash is told from another by its repr."""
        coord = Categorical.from_labels([["a"], "b", ["a"]])
        assert len(coord.categories) == 2
        assert coord.codes.values.tolist() == [1, 0, 1]


class TestShape:
    """The codes carry the shape, whatever its rank."""

    def test_scalar(self):
        """A single label is a coordinate of no axis."""
        coord = Categorical.from_labels("example")
        assert coord.shape == () and coord.ndim == 0 and coord.size == 1
        assert coord.values[()] == "example"

    def test_two_dimensional(self):
        """An N-D array of labels keeps its shape."""
        values = np.asarray([["a", "b"], ["c", "a"]])
        coord = Categorical.from_labels(values, dims=("x", "y"))
        assert coord.shape == (2, 2) and coord.ndim == 2
        np.testing.assert_array_equal(coord.values, values)
        assert coord[0].shape == (2,)
        assert coord[0].dims == ()

    def test_run_length_coded(self):
        """Piecewise constant labels cost a run each, not a label each."""
        coord = Categorical.from_labels(["north"] * 500 + ["south"] * 500)
        assert coord.codes.runs_count == 2
        assert coord.codes.labels is None  # a grid run holds no labels
        assert len(coord) == 1000
        assert labels_of(coord)[499:501] == ["north", "south"]

    def test_slicing_keeps_the_categories(self):
        """A slice is a coordinate of the same categories."""
        coord = Categorical.from_labels(["a", "b", "c", "d"], dims=("x",))
        out = coord[1:3]
        assert labels_of(out) == ["b", "c"]
        assert out.dims == ("x",)
        assert list(out.categories) == list(coord.categories)
        assert coord[1] == "b"

    def test_reversed(self):
        """Reversing turns the codes round."""
        coord = Categorical.from_labels(["a", "b", "c"])
        assert labels_of(coord.reversed()) == ["c", "b", "a"]

    def test_repr_states_what_it_holds(self):
        """A coordinate says its shape, how many categories, and its runs."""
        coord = Categorical.from_labels(["a"] * 3 + ["b"] * 3, dims=("x",))
        text = repr(coord)
        assert "shape=(6,)" in text and "categories=2" in text
        assert "runs=2" in text and "dims=('x',)" in text

    def test_units_are_never_stated(self):
        """A category is not measured in anything."""
        assert Categorical.from_labels(["a"]).units is None


class TestMatching:
    """Selecting samples by their labels."""

    @pytest.fixture
    def stations(self):
        """Six samples over three stations."""
        return Categorical.from_labels(["n1", "n2", "s1", "n1", "s2", "s1"])

    def test_isin_one_label(self, stations):
        """One label gives the samples which carry it."""
        mask = stations.isin("n1")
        assert mask.tolist() == [True, False, False, True, False, False]

    def test_isin_several(self, stations):
        """Several labels give the samples which carry any of them."""
        mask = stations.isin(["s1", "s2"])
        assert mask.tolist() == [False, False, True, False, True, True]

    def test_isin_a_label_which_is_not_there(self, stations):
        """A label no sample carries selects nothing."""
        assert not stations.isin("nowhere").any()

    def test_isin_across_spellings(self):
        """A number is the label it is, however the query spells it."""
        coord = Categorical.from_labels([1, 2, 1])
        assert coord.isin(1.0).tolist() == [True, False, True]

    def test_match(self, stations):
        """A glob matches the categories, and the samples follow."""
        assert stations.match("n*").tolist() == [
            True,
            True,
            False,
            True,
            False,
            False,
        ]
        assert stations.match("?1").tolist() == [
            True,
            False,
            True,
            True,
            False,
            True,
        ]

    def test_match_is_case_sensitive(self, stations):
        """A pattern matches the label as it is written."""
        assert not stations.match("N*").any()


class TestIdentity:
    """Equality and fingerprints read the labels, not how they were built."""

    def test_equal_however_built(self):
        """The same labels give the same coordinate from any spelling."""
        first = Categorical.from_labels(["a", "b", "a"])
        second = Categorical.from_labels(np.asarray(["a", "b", "a"]))
        third = Categorical.from_labels(np.asarray([b"a", b"b", b"a"]))
        assert first == second == third
        assert first.fingerprint() == second.fingerprint() == third.fingerprint()

    def test_numbers_are_one_category_however_spelled(self):
        """A category reached through 1 and through 1.0 is one category."""
        first = Categorical.from_labels([1, "a"])
        second = Categorical.from_labels([1.0, "a"])
        assert first == second
        assert first.fingerprint() == second.fingerprint()

    def test_two_integers_no_float_can_tell_apart_stay_two(self):
        """A category is the number it is, not the float it rounds to."""
        big = 2**53
        first = Categorical.from_labels([big, "x"])
        second = Categorical.from_labels([big + 1, "x"])
        assert first != second
        assert first.fingerprint() != second.fingerprint()
        assert not first.isin(big + 1).any()

    def test_numbers_order_exactly_whatever_holds_them(self):
        """Labels no float can tell apart sort and compare as themselves."""
        big = 2**53
        first = Categorical.from_labels([1, big, big + 1])
        second = Categorical.from_labels([1.0, big, big + 1])
        assert first == second
        assert first.fingerprint() == second.fingerprint()
        assert [int(x) for x in first.categories] == [1, big, big + 1]

    def test_a_wider_float_is_the_number_it_prints(self):
        """An extended precision label is not truncated into its category."""
        wide = Categorical.from_labels([np.longdouble("1.5"), "x"])
        assert wide != Categorical.from_labels([1, "x"])
        assert wide == Categorical.from_labels([1.5, "x"])
        assert not wide.isin(1).any()

    @pytest.mark.parametrize("mixed", [False, True])
    def test_infinities_sit_at_the_ends(self, mixed):
        """An infinity is ordered where it points, and a NaN after both."""
        values = [np.inf, 1.0, -np.inf, np.nan]
        # Beside text the numbers are ordered by the mixed path's own key
        # rather than by numpy's sort.
        extra = ["x"] if mixed else []
        coord = Categorical.from_labels(values + extra)
        assert list(coord.categories[:3]) == [-np.inf, 1.0, np.inf]
        assert np.isnan(coord.categories[3])
        assert list(coord.categories[4:]) == extra

    def test_order_matters(self):
        """The same labels in another order are another coordinate."""
        first = Categorical.from_labels(["a", "b"])
        assert first != Categorical.from_labels(["b", "a"])
        assert first.fingerprint() != Categorical.from_labels(["b", "a"]).fingerprint()

    def test_unused_categories_are_not_carried(self):
        """Only the labels which are there are categories."""
        coord = Categorical.from_labels(["a", "a"])
        assert list(coord.categories) == ["a"]

    def test_nulls_do_not_break_equality(self):
        """Two coordinates of the same nulls are equal, though NaN is not."""
        first = Categorical.from_labels([np.nan, 1.0])
        assert first == Categorical.from_labels([np.nan, 1.0])
        assert (
            first.fingerprint() == Categorical.from_labels([np.nan, 1.0]).fingerprint()
        )

    def test_not_a_categorical(self):
        """A coordinate is not equal to a thing which is not one."""
        assert Categorical.from_labels(["a"]) != "a"
        assert Categorical.from_labels(["a"]) != NumericND.from_run(0, 1, 1)


class TestValidation:
    """A categorical states what it is made of."""

    def test_codes_must_be_integers(self):
        """The codes index the categories, so they are whole numbers."""
        with pytest.raises(CoordinateError, match="Codes are integers"):
            Categorical(
                codes=NumericND.from_run(0.0, 1.0, 2),
                categories=np.asarray(["a", "b"]),
            )

    def test_categories_are_one_dimensional(self):
        """A table of categories is a list of them."""
        with pytest.raises(CoordinateError, match="one dimensional"):
            Categorical(
                codes=NumericND.from_run(0, 1, 2),
                categories=np.zeros((2, 2)),
            )

    def test_dims_match_the_shape(self):
        """A coordinate is attached to as many axes as its labels have."""
        with pytest.raises(CoordinateError, match="no dims"):
            Categorical.from_labels(["a", "b"], dims=("x", "y"))
