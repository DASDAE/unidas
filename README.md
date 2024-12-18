# unidas

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

You can also use `adpater` to wrap un-wrapped functions. 

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

## Installation
Simply install unidas with pip or mamba:

```bash
pip install unidas 
```

```bash
mamba install unidas
```

By design, unidas has no hard dependencies other than numpy, but an `ImportError` will be raised if the libraries needed to perform a requested conversion are not installed.

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
2. Add a conversion method to UnidasBasDASConverter to convert from unidas' BaseDAS back to your data structure.
3. Write a test in test/test_unidas.py (this is important for maintainability).

Feel free to open a discussion if you need help. 

## Supported libraries (in alphabetical order)

- [DASCore](https://github.com/DASDAE/dascore)
- [DASPy](https://github.com/HMZ-03/DASPy)
- [lightguide](https://github.com/pyrocko/lightguide)
- [Xdas](https://github.com/xdas-dev/xdas)
