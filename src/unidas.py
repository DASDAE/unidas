"""
Unidas: A DAS Compatibility Package.
"""

from __future__ import annotations

import datetime
import importlib
import inspect
import zoneinfo
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, wraps
from types import ModuleType
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import numpy as np

# Explicitly defines unidas' public API.
__all__ = ("adapter", "convert")

# Keep the version hardcoded so vendored copies report their own version
# without requiring installed package metadata.
__version__ = "0.1.2"

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
    # TODO maybe just use __dict__, but this wont trigger properties.
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


@runtime_checkable
class ArrayLike(Protocol):
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

    def to_dict(self, flavor):
        """Serialize a coordinate for the requested destination."""
        if flavor == "dascore":
            return self.to_dascore_coord()
        if flavor == "xdas":
            return self.to_xdas_coord()
        assert flavor == "xarray", f"Unknown coordinate flavor: {flavor}"
        return self.to_xarray_coord()

    def to_dascore_coord(self):
        """Method to convert to DAScore coordinates."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")

    def to_xdas_coord(self):
        """Method to convert to xdas coordinate."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")

    def to_xarray_coord(self):
        """Convert to an xarray coordinate variable."""
        xr = optional_import("xarray")
        attrs = dict(self.attrs)
        if self.units is not None:
            attrs["units"] = self.units
        return xr.Variable(self.dims, self.get_array(), attrs=attrs)

    def get_array(self):
        """Return the coordinate values as an array."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")

    def get_step(self):
        """Return the coordinate step when it is well-defined."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")

    def get_start(self):
        """Return the first coordinate value."""
        raise NotImplementedError(f"Not implemented for {self.__class__}")


@dataclass
class EvenlySampledCoordinate(Coordinate):
    """
    A coordinate which is evenly sampled, sorted, and contiguous.

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
    attrs
        Coordinate metadata other than units.
    """

    step: Any
    tie_values: Sequence
    tie_indices: Sequence[int]
    units: Any = None
    dims: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self):
        """Return the coordinate shape without expanding its values."""
        return (len(self),)

    def get_array(self):
        """Expand sampled coordinates, keeping datetime arithmetic exact."""
        if len(self.tie_values) > 2:
            raise NotImplementedError("Cannot expand coordinates with gaps.")
        start, step = self.get_start(), self.step
        if isinstance(start, datetime.datetime):
            start = np.datetime64(to_stripped_utc(time_to_datetime(start)))
        if len(self) == 1:
            return np.asarray([start])
        if isinstance(start, np.datetime64 | np.timedelta64) and not isinstance(
            step, np.timedelta64
        ):
            seconds = (
                step.total_seconds() if isinstance(step, datetime.timedelta) else step
            )
            step = np.timedelta64(round(seconds * 1_000_000_000), "ns")
        return start + np.arange(len(self), dtype=np.int64) * step

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
        xdas = optional_import("xdas")
        # Currently, xdas expects a number or numpy datatime, need to convert
        # python datetimes to numpy.
        tie_values = self.tie_values
        # Tie values currently have to be either datetimes or floats
        if isinstance(self.tie_values[0], datetime.datetime):
            tie_values = [np.datetime64(to_stripped_utc(x)) for x in tie_values]
        dim = self.dims[0] if len(self.dims) == 1 else None
        # xdas requires strictly increasing tie indices, which an axis holding a
        # single sample cannot provide, so it is represented as a dense one.
        if len(self) == 1:
            return xdas.DenseCoordinate(data=np.atleast_1d(tie_values[0]), dim=dim)
        data = {"tie_indices": self.tie_indices, "tie_values": tie_values}
        out = xdas.InterpCoordinate(data=data, dim=dim)
        return out

    def __len__(self):
        return self.tie_indices[-1] - self.tie_indices[0] + 1

    def get_step(self):
        """Return the coordinate step."""
        return self.step

    def get_start(self):
        """Return the first coordinate value."""
        return self.tie_values[0]


@dataclass
class ArrayCoordinate(Coordinate):
    """
    A coordinate represented by an explicit array of values.

    The array may contain scalar, one-dimensional, or multidimensional values.

    Parameters
    ----------
    data
        An array of coordinate values.
    units
        The units of the coordinate.
    dims
        The dimensions with which the coordinate is associated.
    attrs
        Coordinate metadata other than units.
    """

    data: ArrayLike
    units: Any = None
    dims: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self):
        """Return the coordinate shape."""
        return np.shape(self.data)

    def get_array(self):
        """Return the coordinate values without copying or computing them."""
        return self.data

    def to_dascore_coord(self):
        """Convert to a dascore coordinate."""
        dc_core = optional_import("dascore.core")
        return dc_core.get_coord(data=self.data, units=self.units)

    def to_xdas_coord(self):
        """Convert to an XDAS coordinate."""
        xdas = optional_import("xdas")
        if len(self.shape) > 1:
            raise ValueError("XDAS does not support multidimensional coordinates.")
        if not self.shape:
            return xdas.ScalarCoordinate(data=self.data)
        dim = self.dims[0] if len(self.dims) == 1 else None
        return xdas.DenseCoordinate(data=self.data, dim=dim)

    def get_step(self):
        """Return the coordinate step when it is evenly sampled."""
        data = np.asarray(self.data)
        if data.ndim != 1 or not data.size or data.dtype.kind not in "iufmM":
            raise ValueError("Sampling requires a nonempty numeric or time axis.")
        if not np.all(np.isfinite(data)):
            raise ValueError("Sampling requires finite coordinate values.")
        if len(self) == 1:
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
        msg = "Array coordinates must be evenly sampled."
        raise ValueError(msg)

    def get_start(self):
        """Return the first coordinate value."""
        if len(self.shape) != 1 or not self.shape[0]:
            raise ValueError("Sampling requires a nonempty one-dimensional axis.")
        return self.data[0]

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

    def _coord_to_dict(self, flavor):
        """Convert the coordinates to a dictionary."""
        out = {}
        for name, coord in self.coords.items():
            try:
                converted = coord.to_dict(flavor=flavor)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{flavor} cannot represent coordinate {name!r}: {exc}"
                ) from exc
            if flavor == "dascore":
                dims = self._get_coord_dims(name, coord)
                out[name] = (dims, converted)
            else:
                out[name] = converted
        return out

    def to_dict(self, flavor: str):
        """
        Convert base das to dict.

        Parameters
        ----------
        flavor
            The target for the output.
        """
        out = dict(self.__dict__)
        if flavor == "dascore":
            out.pop("name")
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
                start, step = coord.get_start(), coord.get_step()
                if time_to_float(step) == 0:
                    raise ValueError("Sampling step must be nonzero.")
            except ValueError as exc:
                raise ValueError(
                    f"{target} cannot represent coordinate {name!r}: {exc}"
                ) from exc
            out[name] = (start, step)
        return out


# ------------------------ Dataformat converters


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
            # TODO: Maybe add a check for DASBase here so that is tried
            # before other potential conversion paths.
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

    @converts_to("xarray.DataArray")
    def to_xarray_dataarray(self, base_das: BaseDAS):
        """Convert to an xarray data array."""
        xr = optional_import("xarray")
        return xr.DataArray(**base_das.to_dict(flavor="xarray"))

    @converts_to("daspy.Section")
    def to_daspy_section(self, base_das: BaseDAS):
        """Convert to a daspy section."""
        daspy = optional_import("daspy")
        dasdt = daspy.DASDateTime
        sampling = base_das.get_sampling("daspy.Section")
        time_start, time_step = sampling["time"]
        distance_start, distance_step = sampling["distance"]
        start_time = time_to_datetime(time_start)
        # Coordinates and data are authoritative when metadata contains stale
        # structural fields such as fs, dx, or start_time.
        kwargs = dict(base_das.attrs)
        kwargs.update(
            data=base_das.transpose("distance", "time").data,
            fs=1 / time_to_float(time_step),
            dx=distance_step,
            start_distance=distance_start,
            start_time=dasdt.from_datetime(start_time),
        )
        return daspy.Section(**kwargs)

    @converts_to("lightguide.Blast")
    def to_lightguide_blast(self, base_das: BaseDAS):
        """Convert to a lightguide blast."""
        lg_blast = optional_import("lightguide.blast")
        sampling = base_das.get_sampling("lightguide.Blast")
        time_start, time_step = sampling["time"]
        distance_start, distance_step = sampling["distance"]
        start_channel = round(distance_start / distance_step)

        out = lg_blast.Blast(
            data=base_das.transpose("distance", "time").data,
            start_time=time_to_datetime(time_start),
            sampling_rate=1 / time_to_float(time_step),
            start_channel=start_channel,
            channel_spacing=distance_step,
        )
        return out


class DASCorePatchConverter(Converter):
    """
    Converter for DASCore's Patch.
    """

    name = "dascore.Patch"
    # PatchAttrs fields which describe the patch's coordinates rather than its
    # acquisition. BaseDAS keeps that information in its own coords and dims,
    # and this copy goes stale as soon as anything changes the extent of an
    # axis, so it must not be dumped into the attrs dict.
    _structural_attrs = frozenset({"coords", "dims"})

    def _to_base_coords(self, coord, dims):
        """Convert a coordinate to base coordinates."""
        # Portable unit strings also keep xarray attrs serializable. DASCore
        # parses these back to quantities, including any unit scale.
        units = str(coord.units) if coord.units is not None else None
        if coord.evenly_sampled:
            tie_inds = (0, len(coord) - 1)
            tie_vals = (coord.start, coord.stop - coord.step)
            return EvenlySampledCoordinate(
                tie_values=tie_vals,
                tie_indices=tie_inds,
                units=units,
                dims=dims,
                step=coord.step,
            )
        else:
            return ArrayCoordinate(data=coord.data, units=units, dims=dims)

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
            "attrs": patch.attrs.model_dump(exclude=self._structural_attrs),
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
        xdas = optional_import("xdas")
        coords = data_array.coords
        coords_out = {}
        for name, coord in coords.items():
            dims = (coord.dim,) if coord.dim is not None else ()
            if isinstance(coord, xdas.InterpCoordinate) and len(coord.tie_values) > 1:
                # Other libraries handle gaps differently. For now, we raise if
                # there are any gaps, which I interpret as more than 2 tie values.
                # Need to double check that this is right.
                if len(coord.tie_values) > 2:
                    msg = (
                        "Tie values of xdas coordinates imply gaps, cant convert to "
                        "other formats"
                    )
                    raise NotImplementedError(msg)
                values = coord.tie_values
                time_like = values.dtype.kind in "mM"
                if not time_like:
                    values = values.astype(float)
                span = int(coord.tie_indices[-1] - coord.tie_indices[0])
                delta = values[-1] - values[0]
                step = delta / span
                # Rounded datetime labels need their native dense values.
                if time_like and step * span != delta:
                    ucoord = ArrayCoordinate(data=coord.values, dims=dims)
                else:
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
        attrs = {} if data_array.attrs is None else dict(data_array.attrs)
        out = BaseDAS(
            data=data_array.data,
            dims=data_array.dims,
            coords=self._to_base_coords(data_array),
            attrs=attrs,
            name=data_array.name,
        )
        return out


class XArrayConverter(Converter):
    """Converter for xarray DataArrays, including non-dimensional coordinates."""

    name = "xarray.DataArray"

    @converts_to("unidas.BaseDAS")
    def to_base(self, data_array) -> BaseDAS:
        """Preserve the DataArray's values, dimensions, and descriptive metadata."""
        coords = {}
        for name, coord in data_array.coords.items():
            attrs = dict(coord.attrs)
            units = attrs.get("units")
            if units is not None:
                attrs.pop("units")
            coords[name] = ArrayCoordinate(
                data=coord.data,
                dims=coord.dims,
                units=units,
                attrs=attrs,
            )
        return BaseDAS(
            data=data_array.data,
            dims=data_array.dims,
            coords=coords,
            attrs=dict(data_array.attrs),
            name=data_array.name,
        )


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
            key = get_class_key(cls)
            conversion_class: Converter = Converter._registry[key]
            input_obj = convert(obj, to)
            func_out = func(input_obj, *args, **kwargs)
            cls_out = obj if inspect.isclass(func_out) else type(func_out)
            # Sometimes a function can return a different type than its input
            # e.g., a dataframe. In this case just return output.
            if get_class_key(cls_out) != to:
                return func_out
            output_obj = convert(func_out, key)
            # Apply class specific logic to compensate for lossy conversion.
            out = conversion_class.post_conversion(input_obj, output_obj)
            return out

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
