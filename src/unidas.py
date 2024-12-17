"""
Core functionality for unidas.

Currently, the base representation is a dictionary of the following form.
"""

from __future__ import annotations

# Unidas version indicator. When incrementing, be sure to update
# pyproject.toml as well.
__version__ = "0.0.0"

# Explicitly defines unidas' public API.
# https://peps.python.org/pep-0008/#public-and-internal-interfaces
__all__ = ("adapter", "convert")

import datetime
import importlib
import inspect
import zoneinfo
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from functools import cache, wraps
from types import ModuleType
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np

# Define the urls to each project to provide helpful error messages.
PROJECT_URLS = {
    "dascore": "https://github.com/dasdae/dascore",
    "daspy": "https://github.com/HMZ-03/DASPy",
    "lightguide": "https://github.com/pyrocko/lightguide",
    "xdas": "https://github.com/xdas-dev/xdas",
}

DT_PRECISION = datetime.datetime.resolution.total_seconds()

# A generic type variable.
T = TypeVar("T")

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
    MissingOptionalDependency if the package is not installed.

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
    A decorator which marks a method as conversion function.

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


def get_object_key(object_class):
    """Get the tuple which defines the objects unique id."""
    module_name = object_class.__module__.split(".")[0]
    class_name = object_class.__name__
    return f"{module_name}.{class_name}"


def extract_attrs(obj, attrs_names):
    """Extract attributes from an object ot a dict."""
    # TODO maybe just use __dict__, but this wont trigger properties.
    out = {x: getattr(obj, x) for x in attrs_names if hasattr(obj, x)}
    return out


def time_to_float(obj):
    """Converts a datetime or numpy datetime object to float."""
    if isinstance(obj, np.datetime64) or isinstance(obj, np.timedelta64):
        obj = obj.astype("timedelta64") / np.timedelta64(1, "s")
    elif hasattr(obj, "timestamp"):
        obj = obj.timestamp()
    return obj


def time_to_datetime(obj):
    """Convert a time-like object to a datetime object."""
    if isinstance(obj, np.datetime64) and DT_PRECISION > 1e-9:
        # On python 3.10 this can fail since the default time precision is
        # for datetime.datetime is us not ns. Need to truncate to us precision.
        # TODO: need to look into daspy's DASUTC to see if it can handle ns.
        obj = obj.astype("datetime64[us]")
    if not isinstance(obj, datetime.datetime):
        # Lightguide expects a timezone to be attached, so we attach utc.
        utc = zoneinfo.ZoneInfo("UTC")
        obj = datetime.datetime.fromisoformat(str(obj)).astimezone(utc)
    return obj


def to_stripped_utc(time: datetime.datetime):
    """Convert a datetime to UTC then strip timezone info"""
    out = time.astimezone(zoneinfo.ZoneInfo("UTC")).replace(tzinfo=None)
    return out


@runtime_checkable
class ArrayLike(Protocol, Sized):
    """
    Simple definition of an array for now.
    """

    def __array__(self):
        """A method which returns an array."""

    def __len__(self):
        """Arrays have a length."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the array."""


class Coordinate(ArrayLike):
    """Base class for representing coordinates."""

    def to_dict(self, flavor=None):
        if flavor == "dascore":
            out = self.to_dascore_coord()
        elif flavor == "xdas":
            out = self.to_xdas_coord()
        else:
            out = self.__dict__
        return out

    def to_dascore_coord(self):
        """Method to convert to DAScore coordinates."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")

    def to_xdas_coord(self):
        """Method to convert to xdas coordinate."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")


@dataclass
class EvenlySampledCoordinate(Coordinate):
    """
    A coordinate which is evenly sampled and contiguous.

    Parameters
    ----------
    tie_values
        The values of the first and last element of the coordinate.
    tie_indices
        The indices of the first and last coordinate.
    step
        The increment between elements.
    units
        The units of the coordinate.
    dims
        The dimensions with which the coordinate is associated.
    """

    step: Any
    tie_values: Sequence
    tie_indices: Sequence[int]
    units: Any = None
    dims: tuple[str] = ()

    def to_dascore_coord(self):
        """Convert to a dascore coordinate."""
        dc = optional_import("dascore")
        dc_core = optional_import("dascore.core")

        if len(self.tie_values) > 2:
            msg = "DASCore doesn't support gaps in coordinates."
            raise NotImplementedError(msg)

        start, stop, step = self.tie_values[0], self.tie_values[-1], self.step

        if isinstance(start, datetime.datetime):
            start = dc.to_datetime64(start)
            stop = dc.to_datetime64(stop)
            step = dc.to_timedelta64(step)

        out = dc_core.get_coord(
            start=start, stop=stop + step, step=step, units=self.units
        )
        return out.change_length(len(self))

    def to_xdas_coord(self):
        """Convert to an XDAS coordinate."""
        xcoords = optional_import("xdas.core.coordinates")
        # Currently, xdas expects a number or numpy datatime, need to convert
        # python datetimes to numpy.
        tie_values = self.tie_values
        if isinstance(self.tie_values[0], datetime.datetime):
            tie_values = [np.datetime64(to_stripped_utc(x)) for x in tie_values]
        data = {"tie_indices": self.tie_indices, "tie_values": tie_values}
        out = xcoords.InterpCoordinate(data=data)
        return out

    def __len__(self):
        return self.tie_indices[-1] - self.tie_indices[0] + 1


@dataclass
class ArrayCoordinate(Coordinate):
    """
    A coordinate which is not evenly sampled and contiguous.

    Parameters
    ----------
    data
        An array of coordinate values.
    units
        The units of the coordinate.
    dims
        The dimensions with which the coordinate is associated.
    """

    data: ArrayLike
    units: Any = None
    dims: tuple[str] = ()

    def to_dascore_coord(self):
        """Convert to a dascore coordinate."""
        dc_core = optional_import("dascore.core")
        return dc_core.get_coord(**self.to_dict())

    def __len__(self):
        return len(self.data)


@dataclass()
class BaseDAS:
    """
    The base representation of DAS data for unidas.

    This should only be used internally because it is subject to change
    between versions.
    """

    data: ArrayLike
    coords: dict[str, Coordinate]
    attrs: dict[str, Any]
    dims: tuple[str, ...]

    def validate(self):
        """Run simple validation checks on BaseDAS."""
        # First ensure shapes are consistent with coordinates.
        for num, name in enumerate(self.dims):
            data_len = self.data.shape[num]
            coord_len = len(self.coords[name])
            assert data_len == coord_len
        # Ensure attrs and coords are mappings
        assert isinstance(self.attrs, Mapping)
        assert isinstance(self.coords, Mapping)

    def _coord_to_dict(self, flavor=None):
        """Convert the coordinates to a dictionary."""
        out = {}
        for name, coord in self.coords.items():
            if flavor == "dascore":
                dims = (name,) if not coord.dims else coord.dims
                out[name] = (dims, coord.to_dict(flavor=flavor))
            elif flavor in {"xdas", "simple", None}:
                out[name] = coord.to_dict(flavor=flavor)
        return out

    def to_dict(self, flavor: str | None = None):
        """
        Convert base das to dict.

        Parameters
        ----------
        flavor
            The target for the output.
        """
        out = dict(self.__dict__)
        out["coords"] = self._coord_to_dict(flavor=flavor)
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
        return BaseDAS(data=new_data, coords=self.coords, attrs=self.attrs, dims=dims)


# ------------------------ Dataformat converters


class Converter:
    """
    A base class used convert between object types.

    To use this, simply define a subclass and create the appropriate
    conversion methods with the `converts_to` decorator.
    """

    name: str = None  # should be "{module}.{class_name}" see get_object_key.
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
        be useful to re-attach lost metadata for example.

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
        msg = f"No conversion path from {start} to {target} found."
        raise ValueError(msg)


class UnidasBaseDASConverter(Converter):
    """
    Class for converting from the base representation to other library structures.
    """

    name = "unidas.BaseDAS"

    @converts_to("dascore.Patch")
    def to_dascore_patch(self, base_das: BaseDAS):
        """Convert to a dascore patch."""
        dc = optional_import("dascore")
        out = base_das.to_dict(flavor="dascore")
        return dc.Patch(**out)

    @converts_to("xdas.DataArray")
    def to_xdas_dataarray(self, base_das: BaseDAS):
        """Convert to a xdas data array."""
        xdas = optional_import("xdas")
        out = base_das.to_dict(flavor="xdas")
        return xdas.DataArray(**out)

    @converts_to("daspy.Section")
    def to_daspy_section(self, base_das: BaseDAS):
        """Convert to a daspy section."""
        daspy = optional_import("daspy")
        dasdt = daspy.DASDateTime
        out = base_das.transpose("time", "distance").to_dict(flavor="simple")
        time, dist = out["coords"]["time"], out["coords"]["distance"]
        start_time = time_to_datetime(time["tie_values"][0])
        section = daspy.Section(
            data=base_das.data,
            fs=1 / time_to_float(time["step"]),  # This is sampling rate in Hz
            dx=dist["step"],
            start_distance=dist["tie_values"][0],
            start_time=dasdt.from_datetime(start_time),
            **out["attrs"],
        )
        return section

    @converts_to("lightguide.Blast")
    def to_lightguide_blast(self, base_base: BaseDAS):
        """Convert to a lightguide blast."""
        lg_blast = optional_import("lightguide.blast")

        data_dict = base_base.to_dict(flavor="simple")
        coords = data_dict["coords"]
        dist_step = coords["distance"]["step"]
        start_channel = round(coords["distance"]["tie_values"][0] / dist_step)

        out = lg_blast.Blast(
            data=data_dict["data"],
            start_time=time_to_datetime(coords["time"]["tie_values"][0]),
            sampling_rate=1 / time_to_float(coords["time"]["step"]),
            start_channel=start_channel,
            channel_spacing=coords["distance"]["step"],
            # **data_dict["attrs"],
        )
        return out


class DASCorePatchConverter(Converter):
    """
    Converter for DASCore's Patch.
    """

    name = "dascore.Patch"

    def _to_base_coords(self, coord, dims):
        """Convert a coordinate to base coordinates."""
        if coord.evenly_sampled:
            tie_inds = (0, len(coord) - 1)
            tie_vals = (coord.start, coord.stop)
            return EvenlySampledCoordinate(
                tie_values=tie_vals,
                tie_indices=tie_inds,
                units=coord.units,
                dims=dims,
                step=coord.step,
            )
        else:
            return ArrayCoordinate(array=coord.array, units=coord.units)

    @converts_to("unidas.BaseDAS")
    def to_base(self, patch) -> BaseDAS:
        """Convert dascore patch to base representation."""
        coords = patch.coords
        base_coords = {
            i: self._to_base_coords(v, dims=coords.dim_map[i])
            for i, v in patch.coords.coord_map.items()
        }
        out = {
            "data": patch.data,
            "dims": patch.dims,
            "coords": base_coords,
            "attrs": patch.attrs.model_dump(),
        }
        return BaseDAS(**out)


class DASPySectionConverter(Converter):
    """
    Converter for DASpy sections
    """

    name = "daspy.Section"
    # The attributes of section that get stashed in the attrs dict.
    _section_attrs = (
        "start_channel",
        "origin_time",
        "data_type",
        "source",
        "source_type",
        "gauge_length",
    )

    @converts_to("unidas.BaseDAS")
    def to_base(self, section) -> BaseDAS:
        """Convert dascore patch to base representation."""
        # TODO figure out how to get units attached, or are they assumed?
        dims = ("distance", "time")  # TODO is dim order always consistent?
        start_time = section.start_time.utc().to_datetime()
        end_time = section.end_time.utc().to_datetime()

        time_coord = EvenlySampledCoordinate(
            tie_values=(start_time, end_time),
            tie_indices=(0, section.data.shape[1] - 1),
            step=section.dt,
            dims=("time",),
        )
        distance_coord = EvenlySampledCoordinate(
            tie_values=(section.start_distance, section.end_distance),
            tie_indices=(0, section.data.shape[0] - 1),
            step=section.dx,
            dims=("distance",),
        )
        coords = {
            "distance": distance_coord,
            "time": time_coord,
        }
        attrs = extract_attrs(section, self._section_attrs)
        # Need to transpose array so the dimensions correspond to dims.
        return BaseDAS(data=section.data, dims=dims, coords=coords, attrs=attrs)


class LightGuideConverter(Converter):
    """
    Converter for Lightguide Blasts.
    """

    name = "lightguide.Blast"

    _attrs_to_extract = "unit"

    def _get_coords(self, blast):
        """Get base coordinates from Blast."""
        # Need to convert channel numbers to distance.
        start_distance = blast.start_channel * blast.channel_spacing
        end_distance = blast.end_channel * blast.channel_spacing
        distance = EvenlySampledCoordinate(
            tie_values=(start_distance, end_distance),
            tie_indices=(0, blast.data.shape[0] - 1),
            step=blast.channel_spacing,
            dims=("distance",),
        )
        time = EvenlySampledCoordinate(
            tie_values=(blast.start_time, blast.end_time),
            tie_indices=(0, blast.data.shape[1] - 1),
            step=blast.delta_t,
            dims=("time",),
        )
        return {"distance": distance, "time": time}

    @converts_to("unidas.BaseDAS")
    def to_base(self, blast) -> BaseDAS:
        """Convert dascore patch to base representation."""
        # From the plot on lightguide's readme it appears the dims are
        # (channel, time). We need to check if this is always true.
        dims = ("distance", "time")
        coords = self._get_coords(blast)
        out = BaseDAS(
            data=blast.data,
            dims=dims,
            coords=coords,
            attrs=extract_attrs(blast, self._attrs_to_extract),
        )
        return out


class XDASConverter(Converter):
    name = "xdas.DataArray"

    def _to_base_coords(self, data_array):
        """Convert the xdas coordinates to unidas coordinates."""
        xcoords = optional_import("xdas.core.coordinates")
        coords = data_array.coords
        coords_out = {}
        for name, coord in coords.items():
            dims = (coord.dim,) if isinstance(coord.dim, str) else (name,)
            # Other libraries handle gaps differently. For now, we raise if
            # there are any gaps, which I interpret as more than 2 tie values.
            # Need to double check that this is right.
            if len(coord.tie_values) > 2:
                msg = (
                    "Tie values of xdas coordinates imply gaps, cant convert to "
                    "other formats"
                )
                raise NotImplementedError(msg)
            # It seems the InterpCoordinate is evenly sampled, monotonic.
            if isinstance(coord, xcoords.InterpCoordinate):
                step = xcoords.get_sampling_interval(
                    da=data_array, dim=name, cast=False
                )
                ucoord = EvenlySampledCoordinate(
                    tie_values=coord.tie_values,
                    tie_indices=coord.tie_indices,
                    step=step,
                    dims=dims,
                )
            else:
                ucoord = ArrayCoordinate(
                    data=coord.values,
                    dims=dims,
                )
            coords_out[name] = ucoord
        return coords_out

    @converts_to("unidas.BaseDAS")
    def to_base(self, data_array) -> BaseDAS:
        """Convert dascore patch to base representation."""
        attrs = {} if data_array.attrs is None else data_array.attrs
        out = BaseDAS(
            data=data_array.data,
            dims=data_array.dims,
            coords=self._to_base_coords(data_array),
            attrs=attrs,
        )
        return out


def adapter(to: str):
    """
    A decorator to make the wrapped function able to accept multiple DAS inputs.

    The decorator function must

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
            key = get_object_key(cls)
            input_obj = convert(obj, to)
            out = func(input_obj, *args, **kwargs)
            output_obj = convert(out, key)

            return output_obj

        # Following the convention of pydantic, we attach the raw function
        # in case it needs to be accessed later. Also ensures to keep the
        # original function if it is already wrapped.
        func.func = getattr(func, "raw_function", func)
        _decorator.raw_function = getattr(func, "raw_function", func)
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
    key = get_object_key(obj_class)
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
