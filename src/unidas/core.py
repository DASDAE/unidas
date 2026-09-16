"""
The base representation every conversion passes through, and the registry.
"""

from __future__ import annotations

import datetime
import importlib
import inspect
import zoneinfo
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import cache, wraps
from types import ModuleType
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np

from .categorical import Categorical
from .numeric import NS_PER_SECOND, NumericND, float_fraction

# Define the urls to each project to provide helpful error messages.
PROJECT_URLS = {
    "dascore": "https://github.com/dasdae/dascore",
    "daspy": "https://github.com/HMZ-03/DASPy",
    "lightguide": "https://github.com/pyrocko/lightguide",
    "xarray": "https://docs.xarray.dev/en/stable/getting-started-guide/installing.html",
    "xdas": "https://github.com/xdas-dev/xdas",
}

# A generic type variable.
T = TypeVar("T")

# The numpy dtype kinds of a datetime64 and a timedelta64.
TIME_KINDS = frozenset({"m", "M"})
# The dtype kinds a sampling rate can be read from: numbers and times.
SAMPLED_KINDS = frozenset({"i", "u", "f"}) | TIME_KINDS

# ------------------------ Utility functions


def optional_import(package_name: str) -> ModuleType:
    """
    Import a module and return the module object if installed, else raise error.

    Parameters
    ----------
    package_name
        The name of the package which may or may not be installed. Can
        also be sub-packages/modules (eg dascore.core).

    Raises
    ------
    ImportError if the package is not installed.

    Examples
    --------
    >>> from unidas import optional_import
    >>> # import a module (this is the same as import dascore as dc)
    >>> dc = optional_import('unidas')
    >>> try:
    ...     optional_import('boblib5')  # doesn't exist so this raises
    ... except ImportError:
    ...     pass
    """
    try:
        mod = importlib.import_module(package_name)
    except ImportError:
        url = PROJECT_URLS.get(package_name)
        help_str = f" See {url} for installation instructions." if url else ""
        msg = (
            f"{package_name} is not installed but is required for the "
            f"requested functionality.{help_str}"
        )
        raise ImportError(msg)
    return mod


def converts_to(target: str):
    """
    Marks a method on a `Converter` as a conversion function.

    Parameters
    ----------
    target
        The name of the output target. Should be "{module}.{class_name}".
    """

    def decorator(func):
        # Just add a private string to the method so it can be easily
        # detected later.
        func._unidas_convert_to = target
        return func

    return decorator


def get_class_key(object_class) -> str:
    """
    Get a string which defines the class's identifier.

    The general format is "{package_name}.{class_name}".
    """
    module_name = object_class.__module__.split(".")[0]
    class_name = object_class.__name__
    return f"{module_name}.{class_name}"


def extract_attrs(obj, attrs_names):
    """Extract attributes from an object ot a dict."""
    out = {x: getattr(obj, x) for x in attrs_names if hasattr(obj, x)}
    return out


def time_to_float(obj):
    """Converts a datetime or numpy datetime object to a float (timestamp)."""
    if isinstance(obj, np.datetime64) or isinstance(obj, np.timedelta64):
        obj = obj.astype("timedelta64") / np.timedelta64(1, "s")
    elif hasattr(obj, "timestamp"):
        obj = obj.timestamp()
    return obj


def time_to_datetime(obj):
    """Convert a time-like object to a datetime object."""
    if isinstance(obj, np.datetime64):
        obj = obj.astype("datetime64[us]").item()
    elif isinstance(obj, np.timedelta64) or not isinstance(obj, datetime.datetime):
        msg = "DASPy conversion requires an absolute datetime time coordinate."
        raise ValueError(msg)
    # Lightguide expects a timezone to be attached, so attach UTC to naive values.
    utc = zoneinfo.ZoneInfo("UTC")
    obj = obj.replace(tzinfo=utc) if obj.tzinfo is None else obj.astimezone(utc)
    return obj


def to_stripped_utc(time: datetime.datetime):
    """Convert a datetime to UTC then strip timezone info."""
    out = time.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
    return out


def as_datetime64(time) -> np.datetime64:
    """A python datetime, of any timezone, as the instant numpy names."""
    if isinstance(time, datetime.datetime):
        return np.datetime64(to_stripped_utc(time))
    return np.asarray(time).astype("datetime64[ns]")[()]


def seconds_step(step):
    """
    A spacing given in seconds as the exact fraction a time grid steps by.

    A rate such as 1024 Hz has no whole-nanosecond period, so a step stated
    as a float has to stay a ratio until the grid is built from it; rounding
    it to whole nanoseconds first is what makes such a grid drift.
    """
    if isinstance(step, Fraction):
        return step
    if isinstance(step, datetime.timedelta):
        step = np.timedelta64(step)
    array = np.asarray(step)
    if array.dtype.kind == "m":
        ticks = int(array.astype("timedelta64[ns]").astype("int64"))
        return Fraction(ticks, NS_PER_SECOND)
    return float_fraction(float(step))


def sampled_coord(start, step, count, dims=()):
    """
    One evenly sampled run, in the dtype its start and step share.

    A destination which states a start and a rate rather than labels -- a
    DASPy section, a Lightguide blast -- is read back through this, so a
    rate with no whole-tick period keeps its exact spacing.
    """
    if isinstance(start, datetime.datetime | np.datetime64):
        return NumericND.from_run(
            as_datetime64(start), seconds_step(step), count, dims=dims
        )
    dtype = np.result_type(np.asarray(start), np.asarray(step))
    return NumericND.from_run(start, step, count, dims=dims, dtype=dtype)


def _as_number(value):
    """A rational spacing as the number a destination stores; else unchanged."""
    return float(value) if isinstance(value, Fraction) else value


@runtime_checkable
class ArrayLike(Protocol):
    """
    Simple definition of an array.
    """

    def __array__(self):
        """A method which returns an array."""

    def __len__(self):
        """Arrays have a length."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""


# ------------------------ Reading one sampling rate off a coordinate


def coord_from_labels(data, dims=(), units=None, attrs=None):
    """
    Read labels as the shape which can hold them.

    Numbers and times become a run table -- evenly sampled stretches as
    grids, everything else as the labels themselves -- and anything else
    becomes categories.
    """
    array = np.asarray(data)
    if array.dtype.kind in "iufMm":
        return NumericND.from_array(array, units=units, dims=dims, attrs=attrs)
    return Categorical.from_labels(array, dims=dims, attrs=attrs)


def coord_start(coord):
    """The first label, for a destination which states one."""
    if coord.ndim != 1 or not coord.size:
        raise ValueError("Sampling requires a nonempty one-dimensional axis.")
    if isinstance(coord, Categorical):
        raise ValueError("Sampling requires a nonempty numeric or time axis.")
    return coord.start


def coord_step(coord):
    """
    The spacing a destination needing one sampling rate reads off a coordinate.

    A single grid run states its spacing exactly, fraction of a tick and all.
    Anything else is fitted from the labels, as a coordinate which only ever
    held its labels always was.
    """
    if isinstance(coord, Categorical):
        raise ValueError("Sampling requires a nonempty numeric or time axis.")
    if coord.evenly_sampled:
        # A spacing of whole ticks is the scalar it is; only one a scalar
        # would round -- 1/1024 of a second, say -- stays a fraction, since
        # a consumer dividing by it would otherwise get a rounded rate.
        if coord.runs["den"][0] == 1:
            return coord.step
        return coord.step_exact
    if coord.runs_count > 1 and coord.holes and np.all(coord.runs["den"] > 0):
        # Runs the source itself stated, with samples missing between them:
        # no rate describes those, and the labels need not be spelled out.
        msg = "A coordinate with gaps is not evenly sampled."
        raise ValueError(msg)
    return step_from_labels(coord.values)


def step_from_labels(data):
    """
    The one spacing an array of labels keeps, or raise saying it keeps none.

    The labels are fitted rather than differenced so that storage rounding --
    a float32 axis, or timestamps quantized to whole nanoseconds -- is not
    read as jitter, while jitter larger than that still is.
    """
    data = np.asarray(data)
    if data.ndim != 1 or not data.size or data.dtype.kind not in SAMPLED_KINDS:
        raise ValueError("Sampling requires a nonempty numeric or time axis.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Sampling requires finite coordinate values.")
    if len(data) == 1:
        return 1
    if data.dtype.kind == "f":
        # Fit the endpoints in at least double precision, then compare the
        # grid to the labels at their stored precision. Differencing alone
        # amplifies float32 rounding as the coordinate magnitude increases.
        values = data.astype(np.result_type(data.dtype, np.float64), copy=False)
        step = (values[-1] - values[0]) / (len(data) - 1)
        expected = values[0] + np.arange(len(data)) * step
        tolerance = 2 * np.abs(np.spacing(data))
        regular = np.all(np.abs(values - expected) <= tolerance)
        if regular and np.isfinite(step):
            return step
    else:
        # Avoid wraparound when differencing unsigned or narrow integers.
        integer = data.dtype.kind in "iu"
        diff = np.diff(data.astype(object) if integer else data)
        if np.all(diff == diff[0]) and (integer or np.all(np.isfinite(diff))):
            return diff[0]
        if data.dtype.kind in TIME_KINDS:
            # Rates such as 1024 or 3000 Hz require fractional nanoseconds.
            # Fit relative ticks so epoch magnitude cannot erase precision.
            offsets = (data - data[0]).astype(np.float64)
            step = offsets[-1] / (len(data) - 1)
            expected = np.arange(len(data)) * step
            if np.isfinite(step) and np.all(np.abs(offsets - expected) <= 1):
                unit = np.datetime_data(data.dtype)[0]
                return step * time_to_float(np.timedelta64(1, unit))
    msg = "Array coordinates must be evenly sampled."
    raise ValueError(msg)


# ------------------------ The base representation


@dataclass()
class BaseDAS:
    """
    The base representation of DAS data for unidas.

    This should only be used internally because it is subject to change
    between versions.
    """

    data: ArrayLike
    coords: dict[str, NumericND | Categorical]
    attrs: dict[str, Any]
    dims: tuple[str, ...]
    name: Any = None

    def validate(self):
        """Run simple validation checks on BaseDAS."""
        assert isinstance(self.attrs, Mapping)
        assert isinstance(self.coords, Mapping)
        assert len(self.dims) == len(self.data.shape)
        sizes = dict(zip(self.dims, self.data.shape, strict=True))
        for name, coord in self.coords.items():
            dims = self._get_coord_dims(name, coord)
            assert all(dim in sizes for dim in dims)
            if dims:
                assert coord.shape == tuple(sizes[dim] for dim in dims)

    def _get_coord_dims(self, name, coord):
        """Infer legacy array associations without reassigning scalar coordinates."""
        return coord.dims or ((name,) if name in self.dims and coord.shape else ())

    def coord_dict(self, target: str, writer, with_dims: bool = False):
        """
        Convert every coordinate for one destination.

        Parameters
        ----------
        target
            The name of the destination, for the error a coordinate it
            cannot hold is refused with.
        writer
            A function of the coordinate and its dims which returns what the
            destination holds it as.
        with_dims
            Whether each value is paired with the dims it is attached to.
        """
        out = {}
        for name, coord in self.coords.items():
            dims = self._get_coord_dims(name, coord)
            try:
                converted = writer(coord, dims)
            except (TypeError, ValueError, AttributeError) as exc:
                raise ValueError(
                    f"{target} cannot represent coordinate {name!r}: {exc}"
                ) from exc
            out[name] = (dims, converted) if with_dims else converted
        return out

    def to_dict(self, target: str, writer, with_dims: bool = False, name: bool = True):
        """
        Convert the base representation to the kwargs a destination takes.

        Parameters
        ----------
        target
            The name of the destination.
        writer
            How one coordinate is written; see `coord_dict`.
        with_dims
            Whether each coordinate is paired with its dims.
        name
            Whether the destination holds the array's name.
        """
        out = dict(self.__dict__)
        if not name:
            out.pop("name")
        out["coords"] = self.coord_dict(target, writer, with_dims=with_dims)
        return out

    def transpose(self, *dims):
        """
        Transpose the BaseDAS to the desired dimensional order.

        Parameters
        ----------
        dims
            The dimension names for desired order.
        """
        axes = tuple(self.dims.index(x) for x in dims)
        new_data = self.data.transpose(axes)
        return BaseDAS(
            data=new_data,
            coords=self.coords,
            attrs=self.attrs,
            dims=dims,
            name=self.name,
        )

    def get_sampling(self, target):
        """Get sampled time and distance axes required by DASPy and Lightguide."""
        if len(self.dims) != 2 or set(self.dims) != {"time", "distance"}:
            raise ValueError(f"{target} requires time and distance dimensions.")
        out = {}
        for name in ("time", "distance"):
            if name not in self.coords:
                raise ValueError(f"{target} requires a {name!r} coordinate.")
            coord = self.coords[name]
            try:
                if self._get_coord_dims(name, coord) != (name,):
                    raise ValueError(
                        f"Coordinate must be associated with dimension {name!r}."
                    )
                start, step = coord_start(coord), coord_step(coord)
                if time_to_float(step) == 0:
                    raise ValueError("Sampling step must be nonzero.")
            except ValueError as exc:
                raise ValueError(
                    f"{target} cannot represent coordinate {name!r}: {exc}"
                ) from exc
            out[name] = (start, step)
        return out


# The registry key of the base representation, read off the class so a
# vendored copy of this package -- imported under any name -- converts back
# out of its own BaseDAS rather than out of one called "unidas".
BASE_DAS_KEY = get_class_key(BaseDAS)


# ------------------------ The converter registry


class Converter:
    """
    A base class used convert between object types.

    To use this, simply define a subclass and create the appropriate
    conversion methods with the `converts_to` decorator.
    """

    name: str = None  # should be "{module}.{class_name}" see get_class_key.
    _registry: ClassVar[dict[str, Converter]] = {}
    _graph: ClassVar[dict[str, list[str]]] = defaultdict(list)
    _converters: ClassVar[dict[str, callable]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Runs when subclasses are defined.

        This registers the class and their conversion functions.
        """
        name = cls.name
        if name is None:
            msg = f"Converter subclass {cls} must define a name."
            raise ValueError(msg)
        instance = cls()
        cls._registry[name] = instance
        # Iterate the methods and add conversion functions/names to the graph.
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        for method_name, method in methods:
            convert_target = getattr(method, "_unidas_convert_to", None)
            if convert_target:
                cls._graph[name].append(convert_target)
                # Store the method.
                method = getattr(instance, method_name)
                cls._converters[f"{name}__{convert_target}"] = method

    def post_conversion(self, input_obj: T, output_obj: T) -> T:
        """
        Apply some modifications to the input/output objects.

        Some conversions are lossy. This optional method allows subclasses
        to modify the output of `convert` before it gets returned. This might
        be useful to re-attach lost metadata for example. It doesn't work with
        the `convert` function (in that case it needs to be applied manually).

        Parameters
        ----------
        input_obj
            The original object before conversion.
        output_obj
            The resulting object

        Returns
        -------
        An object of the same type and input and output.
        """
        return output_obj

    @classmethod
    @cache
    def get_shortest_path(cls, start, target):
        """
        Simple breadth first search for getting the shortest path.

        Based on this code: https://stackoverflow.com/a/77539683/3645626

        Parameters
        ----------
        start
            The starting node.
        target
            The node to find.

        Returns
        -------
        A tuple of the nodes in the shortest path.
        """
        queue = deque()
        queue.append(start)
        visited = {start: None}
        graph = cls._graph

        while queue:
            current = queue.popleft()
            if current == target:  # A path has been found.
                path = []  # backtrack to get path.
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return tuple(path[::-1])
            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        # No path found, raise exception.
        msg = (
            f"No conversion path from {start} to {target} found. "
            f"{target} may not be a valid conversion target. Valid targets "
            f"are: {sorted(list(Converter._registry.keys()))}."
        )
        raise ValueError(msg)


def _convert_operand(obj, to: str, kind: type):
    """Convert another operand of the caller's own kind; leave anything else."""
    # A class names a type rather than holding data, and a library's array
    # type is also used for ordinary options -- a taper window is an
    # xarray.DataArray too. Only another operand of the kind the call was
    # made on is data this function is being handed a second helping of.
    if inspect.isclass(obj) or type(obj) is not kind:
        return obj
    try:
        return convert(obj, to)
    except (ValueError, TypeError):
        # Being the caller's type is not being the caller's data: a window or
        # a mask is an array of the same class. One the target cannot hold is
        # handed over as it arrived, for the function to use as it meant to.
        return obj


def adapter(to: str):
    """
    A decorator to make the wrapped function able to accept multiple DAS inputs.

    Parameters
    ----------
    to
        The DAS data structure expected as the first argument of the
        wrapped function.

    Returns
    -------
    The wrapped function able to accept multiple DAS inputs.

    Notes
    -----
    - The original function can be accessed via the 'raw_function' attribute.

    """

    def _outer(func):
        # Check if the appropriate decorator has already been applied and
        # just return if so.
        if getattr(func, "_unidas_to", None) == to:
            return func

        @wraps(func)
        def _decorator(obj, *args, **kwargs):
            """Simple decorator for wrapping."""
            # Convert the incoming object to target. This should do nothing
            # if it is already the correct format.
            cls = obj if inspect.isclass(obj) else type(obj)
            key = get_class_key(cls)
            conversion_class: Converter = Converter._registry[key]
            input_obj = convert(obj, to)
            # The other operands are data too: a function of two sections
            # takes two of whatever the caller holds. Everything else --
            # scalars, arrays, options -- is passed along untouched.
            args = tuple(_convert_operand(x, to, cls) for x in args)
            kwargs = {i: _convert_operand(v, to, cls) for i, v in kwargs.items()}
            func_out = func(input_obj, *args, **kwargs)
            # Sometimes a function can return a different type than its input
            # e.g., a dataframe. In this case just return output. A class is
            # a type rather than an instance of one, and is never converted.
            if inspect.isclass(func_out) or get_class_key(type(func_out)) != to:
                return func_out
            # The first argument says which library the caller works in, so
            # that is the one the result is returned in.
            output_obj = convert(func_out, key)
            # Apply class specific logic to compensate for lossy conversion.
            out = conversion_class.post_conversion(input_obj, output_obj)
            return out

        # Following the convention of pydantic, we attach the raw function
        # in case it needs to be accessed later. Also ensures to keep the
        # original function if it is already wrapped. It goes on the wrapper;
        # the function handed in belongs to the caller.
        _decorator.func = _decorator.raw_function = getattr(func, "raw_function", func)
        # Also attach a private flag indicating the function has already
        # been wrapped. We don't want to allow this more than once.
        _decorator._unidas_to = to

        return _decorator

    return _outer


def convert(obj, to: str):
    """
    Convert an object to something else.

    Parameters
    ----------
    obj
        An input object which has Converter class.
    to
        The name of the output class.

    Returns
    -------
    The input object converted to the specified format.
    """
    obj_class = obj if inspect.isclass(obj) else type(obj)
    key = get_class_key(obj_class)
    # No conversion needed, simply return object.
    if key == to:
        return obj
    # Otherwise, find the path from one object to the target and apply
    # the conversion functions until we reach the target type.
    path = Converter.get_shortest_path(key, to)
    assert len(path) > 1, "path should have at least 2 nodes."
    for num, node in enumerate(path[1:]):
        previous = path[num]
        funct_str = f"{previous}__{node}"
        func = Converter._converters[funct_str]
        obj = func(obj)
    return obj
