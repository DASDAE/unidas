<p align="center">
  <img src="https://raw.githubusercontent.com/dasdae/unidas/main/static/logo.png" alt="unidas logo">
</p>


[![coverage](https://codecov.io/gh/dasdae/unidas/branch/main/graph/badge.svg)](https://codecov.io/gh/dasdae/unidas)
[![PyPI Version](https://img.shields.io/pypi/v/unidas.svg)](https://pypi.python.org/pypi/unidas)
[![Licence](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/license/mit)

A DAS compatibility package.

There is an increasing number of open-source libraries for working with distributed acoustic sensing (DAS) data. Each of these has its own strengths and weaknesses, and often it is desirable to use features from multiple libraries in research workflows. Moreover, creators of DAS packages which perform specific operations (e.g., machine learning for phase picking) currently have to choose a single DAS library to support, or undertake writing conversion codes on their own.

Unidas solves these problems by providing simple ways to interoperate between DAS libraries.  

## Usage

There are two ways to use unidas. First, the `adapter` decorator allows a function to simply declare which library's data structure to use. 

```python
import unidas


@unidas.adapter("daspy.Section")
def daspy_function(sec, **kwargs):
    """A useful daspy function"""
    # Regardless of the actual input type, adapter will convert it to a daspy section
    # then convert it back after the return.
    return sec


import dascore as dc

patch = dc.get_example_patch()
# even though we call a daspy function, the input/output is a dascore patch.
out = daspy_function(patch)
assert isinstance(out, dc.Patch)
```

You can also use `adapter` to wrap un-wrapped functions. 

```python
import dascore as dc
import unidas
from xdas.signal import hilbert

dascore_hilbert = unidas.adapter("xdas.DataArray")(hilbert)
patch = dc.get_example_patch()

patch_hilberto = dascore_hilbert(patch)
```

The `convert` function converts from one library's data structure to another library's data structure.

```python
import daspy
import unidas

# Use lightguide's afk filter with a daspy section. 
sec = daspy.read()
blast = unidas.convert(sec, to="lightguide.Blast")
blast.afk_filter(exponent=0.8)
sec_out = unidas.convert(blast, to='daspy.Section')
```

Xarray DataArrays work with both APIs:

```python
import dascore as dc
import unidas

patch = dc.get_example_patch()
data_array = unidas.convert(patch, to="xarray.DataArray")

@unidas.adapter("xarray.DataArray")
def first_ten_samples(data_array):
    return data_array.isel(time=slice(0, 10))

trimmed_patch = first_ten_samples(patch)
```

## Installation
Unidas requires Python 3.11 or newer. Simply install unidas with pip or mamba:

```bash
pip install unidas 
```

```bash
mamba install unidas
```

By design, unidas has no hard dependencies other than numpy, but an `ImportError` will be raised if the libraries needed to perform a requested conversion are not installed.

To install the supported DAS libraries with unidas:

```bash
pip install "unidas[extras]"
```

Some optional libraries lag new Python releases. The complete `unidas[extras]` set currently targets Python 3.11 and 3.12. On Python 3.13 and newer, the extra installs xarray; install the other optional DAS libraries directly once they publish compatible wheels. Xarray can also be installed separately with `pip install xarray`.

For development and testing:

```bash
pip install "unidas[dev]"
```

Unidas is single file (src/unidas.py) so it can also be vendored (copied directly into your project). If you do this, please consider sharing any improvements so the entire community can benefit. 

## Guidance for package developers
If you are creating/maintaining a library for doing some kind of specialized DAS processing in python, we recommend you do two things:

1. Pick the DAS library you prefer and use it internally. 
2. Apply the `adapter` decorator to your project's API.

Doing so will make your project easily accessible by users of all the libraries supported by unidas. 

For example:

```python
import unidas

@unidas.adapter("daspy.Section")
def fancy_machine_learning_function(sec):
    """Cutting edge machine learning DAS research function."""
    # Here we will use daspy internally, but the function accepts 
    # data structures from other libraries with no additional effort
    # because of the adapter decorator. 
    
    ...  # Fancy stuff goes here.
    
    return sec
```

## Adding support for new libraries to unidas

To add support for a new data structure/library, you need to do two things:

1. Create a subclass of `Converter` which has (at least) a conversion method to unidas' BaseDAS.
2. Add a conversion method to UnidasBaseDASConverter to convert from unidas' BaseDAS back to your data structure.
3. Write a test in test/test_unidas.py (this is important for maintainability).

Feel free to open a discussion if you need help. 

## Supported libraries (in alphabetical order)

- [DASCore](https://github.com/DASDAE/dascore)
- [DASPy](https://github.com/HMZ-03/DASPy)
- [Lightguide](https://github.com/pyrocko/lightguide)
- [Xarray](https://docs.xarray.dev/) (`DataArray` only)
- [Xdas](https://github.com/xdas-dev/xdas)

## Compatibility notes

DASPy sections and Lightguide blasts require `time` and `distance` coordinates, evenly sampled coordinates, and an absolute datetime time coordinate. Objects with relative, numeric, or uneven time/distance coordinates may still convert to other formats, but will raise a `ValueError` when converting to `daspy.Section` or `lightguide.Blast`.

Xarray conversion through unidas' internal representation preserves the array name, data, dimension order, coordinates (including scalar and multidimensional auxiliary coordinates), attributes, and coordinate metadata such as units. On this internal round-trip, dimensions without coordinate labels remain unlabeled. DASCore may replace missing labels with placeholder coordinates. Other formats retain only the structures and metadata they support; unsupported coordinate layouts raise an error identifying the target and coordinate. For example, XDAS supports scalar coordinates but does not support multidimensional coordinates. Some DASCore versions cannot construct scalar or multidimensional coordinates. DASCore coordinate units become portable strings in xarray attributes.

DASPy sampling fields are derived from coordinates and take precedence over conflicting attributes such as `fs` or `dx`. Lightguide represents distance using integer channel indices and rounds the starting distance to the nearest channel when it is not an integer multiple of the spacing.

Xarray `Dataset` objects, storage encoding, and reconstruction of custom indexes are not supported. Data is not explicitly computed on the xarray–internal representation path, but other libraries may require eager arrays.

## Making releases

To publish a release, bump `__version__` in `src/unidas.py`, merge the change to `main`, create a version tag such as `v0.1.0`, then publish a GitHub Release from that tag. Publishing the GitHub Release triggers the PyPI upload workflow.
