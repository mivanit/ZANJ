> docs for [`zanj`](https://github.com/mivanit/zanj) v0.5.1


## Contents
[![PyPI](https://img.shields.io/pypi/v/zanj)](https://pypi.org/project/zanj/)
[![Checks](https://github.com/mivanit/zanj/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/zanj/actions/workflows/checks.yml)
[![Coverage](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCI+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImIiIHgyPSIwIiB5Mj0iMTAwJSI+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjYmJiIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPG1hc2sgaWQ9ImEiPgogICAgICAgIDxyZWN0IHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCIgcng9IjMiIGZpbGw9IiNmZmYiLz4KICAgIDwvbWFzaz4KICAgIDxnIG1hc2s9InVybCgjYSkiPgogICAgICAgIDxwYXRoIGZpbGw9IiM1NTUiIGQ9Ik0wIDBoNjN2MjBIMHoiLz4KICAgICAgICA8cGF0aCBmaWxsPSIjNGMxIiBkPSJNNjMgMGgzNnYyMEg2M3oiLz4KICAgICAgICA8cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMCAwaDk5djIwSDB6Ii8+CiAgICA8L2c+CiAgICA8ZyBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iRGVqYVZ1IFNhbnMsVmVyZGFuYSxHZW5ldmEsc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSI+CiAgICAgICAgPHRleHQgeD0iMzEuNSIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPmNvdmVyYWdlPC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjMxLjUiIHk9IjE0Ij5jb3ZlcmFnZTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPjk1JTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTQiPjk1JTwvdGV4dD4KICAgIDwvZz4KPC9zdmc+Cg==)](docs/coverage/coverage.txt)
![code size, bytes](https://img.shields.io/github/languages/code-size/mivanit/zanj)
![PyPI - Downloads](https://img.shields.io/pypi/dm/zanj)

<!-- ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/mivanit/zanj)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/mivanit/zanj) -->
<!-- ![Lines of code](https://img.shields.io/tokei/lines/github.com/mivanit/zanj) -->

# ZANJ

# Overview

The `ZANJ` format is meant to be a way of saving arbitrary objects to disk, in a way that is flexible, allows keeping configuration and data together, and is human readable. It is very loosely inspired by HDF5 and the derived `exdir` format, and the implementation is inspired by `npz` files.

- You can take any `SerializableDataclass` from the [muutils](https://github.com/mivanit/muutils) library and save it to disk -- any large arrays or lists will be stored efficiently as external files in the zip archive, while the basic structure and metadata will be stored in readable JSON files. 
- You can also specify a special `ConfiguredModel`, which inherits from a `torch.nn.Module` which will let you save not just your model weights, but all required configuration information, plus any other metadata (like training logs) in a single file.

This library was originally a module in [muutils](https://github.com/mivanit/muutils/)


# Installation
Available on PyPI as [`zanj`](https://pypi.org/project/zanj/)

```
pip install zanj
```

# Usage

You can find a runnable example of this in [`demo.ipynb`](demo.ipynb)

## Saving a basic object

Any `SerializableDataclass` of basic types can be saved as zanj:

```python
import numpy as np
import pandas as pd
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from zanj import ZANJ

@serializable_dataclass
class BasicZanj(SerializableDataclass):
    a: str
    q: int = 42
    c: list[int] = serializable_field(default_factory=list)

# initialize a zanj reader/writer
zj = ZANJ()

# create an instance
instance: BasicZanj = BasicZanj("hello", 42, [1, 2, 3])
path: str = "tests/junk_data/path_to_save_instance.zanj"
zj.save(instance, path)
recovered: BasicZanj = zj.read(path)
```

ZANJ will intelligently handle nested serializable dataclasses, numpy arrays, pytorch tensors, and pandas dataframes: 

```python
import torch
import pandas as pd

@serializable_dataclass
class Complicated(SerializableDataclass):
    name: str
    arr1: np.ndarray
    arr2: np.ndarray
    iris_data: pd.DataFrame
    brain_data: pd.DataFrame
    container: list[BasicZanj]
    torch_tensor: torch.Tensor
```

For custom classes, you can specify a `serialization_fn` and `loading_fn` to handle the logic of converting to and from a json-serializable format:

```python
@serializable_dataclass
class Complicated(SerializableDataclass):
    name: str
    device: torch.device = serializable_field(
        serialization_fn=lambda self: str(self.device),
        loading_fn=lambda data: torch.device(data["device"]),
    )
```

Note that `loading_fn` takes the dictionary of the whole class -- this is in case you've stored data in multiple fields of the dict which are needed to reconstruct the object.

## Saving Models

First, define a configuration class for your model. This class will hold the parameters for your model and any associated objects (like losses and optimizers). The configuration class should be a subclass of `SerializableDataclass` and use the `serializable_field` function to define fields that need special serialization.

Here's an example that defines a GPT-like model configuration:

```python
from zanj.torchutil import ConfiguredModel, set_config_class

@serializable_dataclass
class MyNNConfig(SerializableDataclass):
    input_dim: int
    hidden_dim: int
    output_dim: int

    # store the activation function by name, reconstruct it by looking it up in torch.nn
    act_fn: torch.nn.Module = serializable_field(
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda x: getattr(torch.nn, x["act_fn"]),
    )

    # same for the loss function
    loss_kwargs: dict = serializable_field(default_factory=dict)
    loss_factory: torch.nn.modules.loss._Loss = serializable_field(
        default_factory=lambda: torch.nn.CrossEntropyLoss,
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda x: getattr(torch.nn, x["loss_factory"]),
    )
    loss = property(lambda self: self.loss_factory(**self.loss_kwargs))
```

Then, define your model class. It should be a subclass of `ConfiguredModel`, and use the `set_config_class` decorator to associate it with your configuration class. The `__init__` method should take a single argument, which is an instance of your configuration class. You must also call the superclass `__init__` method with the configuration instance.

```python
@set_config_class(MyNNConfig)
class MyNN(ConfiguredModel[MyNNConfig]):
    def __init__(self, config: MyNNConfig):
		# call the superclass init!
		# this will store the model in the zanj_model_config field
        super().__init__(config)

		# whatever you want here
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.input_dim, config.hidden_dim),
            config.act_fn(),
            torch.nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

You can now create instances of your model, save them to disk, and load them back into memory:

```python
config = MyNNConfig(
    input_dim=10,
    hidden_dim=20,
    output_dim=2,
    act_fn=torch.nn.ReLU,
    loss_kwargs=dict(reduction="mean"),
)

# create your model from the config, and save
model = MyNN(config)
fname = "tests/junk_data/path_to_save_model.zanj"
ZANJ().save(model, fname)
# load by calling the class method `read()`
loaded_model = MyNN.read(fname)
# zanj will actually infer the type of the object in the file 
# -- and will warn you if you don't have the correct package installed
loaded_another_way = ZANJ().read(fname)
```

## Configuration

When initializing a `ZANJ` object, you can specify some configuration info about saving, such as:

- thresholds for how big an array/table has to be before moving to external file
- compression settings
- error modes
- additional handlers for serialization

```python
# how big an array or list (including pandas DataFrame) can be before moving it from the core JSON file
external_array_threshold: int = ZANJ_GLOBAL_DEFAULTS.external_array_threshold
external_list_threshold: int = ZANJ_GLOBAL_DEFAULTS.external_list_threshold
# compression settings passed to `zipfile` package
compress: bool | int = ZANJ_GLOBAL_DEFAULTS.compress
# for doing very cursed things in your own custom loading or serialization functions
custom_settings: dict[str, Any] | None = ZANJ_GLOBAL_DEFAULTS.custom_settings
# specify additional serialization handlers
handlers_pre: MonoTuple[SerializerHandler] = tuple()
handlers_default: MonoTuple[SerializerHandler] = DEFAULT_SERIALIZER_HANDLERS_ZANJ,
```

# Implementation

The on-disk format is a file `<filename>.zanj` is a zip file containing:

- `__zanj_meta__.json`: a file containing zanj-specific metadata including:
	- system information
	- installed packages
	- information about external files
- `__zanj__.json`: a file containing user-specified data
	- when an element is too big, it can be moved to an external file
		- `.npy` for numpy arrays or torch tensors
		- `.jsonl` for pandas dataframes or large sequences
	- list of external files stored in `__zanj_meta__.json`
	- "$ref" key, specified in `_REF_KEY` in muutils, will have value pointing to external file
	- `_FORMAT_KEY` key will detail an external format type


# Comparison to other formats



| Format                  | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16 |
| ----------------------- | ---- | --------- | ------------ | ------------------ | -------------- | ----------- | -------- |
| pickle (PyTorch)        | ❌   | ❌        | ❌           | ✅                 | ❌             | ✅          | ✅       |
| H5 (Tensorflow)         | ✅   | ❌        | ✅           | ✅                 | ~              | ~           | ❌       |
| HDF5                    | ✅   | ?         | ✅           | ✅                 | ~              | ✅          | ❌       |
| SavedModel (Tensorflow) | ✅   | ❌        | ❌           | ✅                 | ✅             | ❌          | ✅       |
| MsgPack (flax)          | ✅   | ✅        | ❌           | ✅                 | ❌             | ❌          | ✅       |
| Protobuf (ONNX)         | ✅   | ❌        | ❌           | ❌                 | ❌             | ❌          | ✅       |
| Cap'n'Proto             | ✅   | ✅        | ~            | ✅                 | ✅             | ~           | ❌       |
| Numpy (npy,npz)         | ✅   | ?         | ?            | ❌                 | ✅             | ❌          | ❌       |
| SafeTensors             | ✅   | ✅        | ✅           | ✅                 | ✅             | ❌          | ✅       |
| exdir                   | ✅   | ?         | ?            | ?                  | ?              | ✅          | ❌       |
| ZANJ                    | ✅   | ❌        | ❌*          | ✅                 | ✅             | ✅          | ❌*      |


- Safe: Can I use a file randomly downloaded and expect not to run arbitrary code ?
- Zero-copy: Does reading the file require more memory than the original file ?
- Lazy loading: Can I inspect the file without loading everything ? And loading only some tensors in it without scanning the whole file (distributed setting) ?
- Layout control: Lazy loading, is not necessarily enough since if the information about tensors is spread out in your file, then even if the information is lazily accessible you might have to access most of your file to read the available tensors (incurring many DISK -> RAM copies). Controlling the layout to keep fast access to single tensors is important.
- No file size limit: Is there a limit to the file size ?
- Flexibility: Can I save custom code in the format and be able to use it later with zero extra code ? (~ means we can store more than pure tensors, but no custom code)
- Bfloat16: Does the format support native bfloat16 (meaning no weird workarounds are necessary)? This is becoming increasingly important in the ML world.

`*` denotes this feature may be coming at a future date :)

(This table was stolen from [safetensors](https://github.com/huggingface/safetensors/blob/main/README.md))



## Submodules

- [`externals`](#externals)
- [`loading`](#loading)
- [`serializing`](#serializing)
- [`torchutil`](#torchutil)
- [`zanj`](#zanj)

## API Documentation

 - [`register_loader_handler`](#register_loader_handler)
 - [`ZANJ`](#ZANJ)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py)

# `zanj` { #zanj }

[![PyPI](https://img.shields.io/pypi/v/zanj)](https://pypi.org/project/zanj/)
[![Checks](https://github.com/mivanit/zanj/actions/workflows/checks.yml/badge.svg)](https://github.com/mivanit/zanj/actions/workflows/checks.yml)
[![Coverage](data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCI+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9ImIiIHgyPSIwIiB5Mj0iMTAwJSI+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIwIiBzdG9wLWNvbG9yPSIjYmJiIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICAgICAgPHN0b3Agb2Zmc2V0PSIxIiBzdG9wLW9wYWNpdHk9Ii4xIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPG1hc2sgaWQ9ImEiPgogICAgICAgIDxyZWN0IHdpZHRoPSI5OSIgaGVpZ2h0PSIyMCIgcng9IjMiIGZpbGw9IiNmZmYiLz4KICAgIDwvbWFzaz4KICAgIDxnIG1hc2s9InVybCgjYSkiPgogICAgICAgIDxwYXRoIGZpbGw9IiM1NTUiIGQ9Ik0wIDBoNjN2MjBIMHoiLz4KICAgICAgICA8cGF0aCBmaWxsPSIjNGMxIiBkPSJNNjMgMGgzNnYyMEg2M3oiLz4KICAgICAgICA8cGF0aCBmaWxsPSJ1cmwoI2IpIiBkPSJNMCAwaDk5djIwSDB6Ii8+CiAgICA8L2c+CiAgICA8ZyBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBmb250LWZhbWlseT0iRGVqYVZ1IFNhbnMsVmVyZGFuYSxHZW5ldmEsc2Fucy1zZXJpZiIgZm9udC1zaXplPSIxMSI+CiAgICAgICAgPHRleHQgeD0iMzEuNSIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPmNvdmVyYWdlPC90ZXh0PgogICAgICAgIDx0ZXh0IHg9IjMxLjUiIHk9IjE0Ij5jb3ZlcmFnZTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTUiIGZpbGw9IiMwMTAxMDEiIGZpbGwtb3BhY2l0eT0iLjMiPjk1JTwvdGV4dD4KICAgICAgICA8dGV4dCB4PSI4MCIgeT0iMTQiPjk1JTwvdGV4dD4KICAgIDwvZz4KPC9zdmc+Cg==)](docs/coverage/coverage.txt)
![code size, bytes](https://img.shields.io/github/languages/code-size/mivanit/zanj)
![PyPI - Downloads](https://img.shields.io/pypi/dm/zanj)

<!-- ![GitHub commit activity](https://img.shields.io/github/commit-activity/t/mivanit/zanj)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/mivanit/zanj) -->
<!-- ![Lines of code](https://img.shields.io/tokei/lines/github.com/mivanit/zanj) -->

### ZANJ

### Overview

The `ZANJ` format is meant to be a way of saving arbitrary objects to disk, in a way that is flexible, allows keeping configuration and data together, and is human readable. It is very loosely inspired by HDF5 and the derived `exdir` format, and the implementation is inspired by `npz` files.

- You can take any `SerializableDataclass` from the [muutils](https://github.com/mivanit/muutils) library and save it to disk -- any large arrays or lists will be stored efficiently as external files in the zip archive, while the basic structure and metadata will be stored in readable JSON files. 
- You can also specify a special `ConfiguredModel`, which inherits from a `torch.nn.Module` which will let you save not just your model weights, but all required configuration information, plus any other metadata (like training logs) in a single file.

This library was originally a module in [muutils](https://github.com/mivanit/muutils/)


### Installation
Available on PyPI as [`zanj`](https://pypi.org/project/zanj/)

```
pip install zanj
```

### Usage

You can find a runnable example of this in [`demo.ipynb`](demo.ipynb)

#### Saving a basic object

Any `SerializableDataclass` of basic types can be saved as zanj:

```python
import numpy as np
import pandas as pd
from muutils.json_serialize import SerializableDataclass, serializable_dataclass, serializable_field
from zanj import ZANJ

@serializable_dataclass
class BasicZanj(SerializableDataclass):
    a: str
    q: int = 42
    c: list[int] = serializable_field(default_factory=list)

### initialize a zanj reader/writer
zj = ZANJ()

### create an instance
instance: BasicZanj = BasicZanj("hello", 42, [1, 2, 3])
path: str = "tests/junk_data/path_to_save_instance<a href="zanj/zanj.html">zanj.zanj</a>"
zj.save(instance, path)
recovered: BasicZanj = zj.read(path)
```

ZANJ will intelligently handle nested serializable dataclasses, numpy arrays, pytorch tensors, and pandas dataframes: 

```python
import torch
import pandas as pd

@serializable_dataclass
class Complicated(SerializableDataclass):
    name: str
    arr1: np.ndarray
    arr2: np.ndarray
    iris_data: pd.DataFrame
    brain_data: pd.DataFrame
    container: list[BasicZanj]
    torch_tensor: torch.Tensor
```

For custom classes, you can specify a `serialization_fn` and `loading_fn` to handle the logic of converting to and from a json-serializable format:

```python
@serializable_dataclass
class Complicated(SerializableDataclass):
    name: str
    device: torch.device = serializable_field(
        serialization_fn=lambda self: str(self.device),
        loading_fn=lambda data: torch.device(data["device"]),
    )
```

Note that `loading_fn` takes the dictionary of the whole class -- this is in case you've stored data in multiple fields of the dict which are needed to reconstruct the object.

#### Saving Models

First, define a configuration class for your model. This class will hold the parameters for your model and any associated objects (like losses and optimizers). The configuration class should be a subclass of `SerializableDataclass` and use the `serializable_field` function to define fields that need special serialization.

Here's an example that defines a GPT-like model configuration:

```python
from <a href="zanj/torchutil.html">zanj.torchutil</a> import ConfiguredModel, set_config_class

@serializable_dataclass
class MyNNConfig(SerializableDataclass):
    input_dim: int
    hidden_dim: int
    output_dim: int

    # store the activation function by name, reconstruct it by looking it up in torch.nn
    act_fn: torch.nn.Module = serializable_field(
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda x: getattr(torch.nn, x["act_fn"]),
    )

    # same for the loss function
    loss_kwargs: dict = serializable_field(default_factory=dict)
    loss_factory: torch.nn.modules.loss._Loss = serializable_field(
        default_factory=lambda: torch.nn.CrossEntropyLoss,
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda x: getattr(torch.nn, x["loss_factory"]),
    )
    loss = property(lambda self: self.loss_factory(**self.loss_kwargs))
```

Then, define your model class. It should be a subclass of `ConfiguredModel`, and use the `set_config_class` decorator to associate it with your configuration class. The `__init__` method should take a single argument, which is an instance of your configuration class. You must also call the superclass `__init__` method with the configuration instance.

```python
@set_config_class(MyNNConfig)
class MyNN(ConfiguredModel[MyNNConfig]):
    def __init__(self, config: MyNNConfig):
		# call the superclass init!
		# this will store the model in the zanj_model_config field
        super().__init__(config)

		# whatever you want here
        self.net = torch.nn.Sequential(
            torch.nn.Linear(config.input_dim, config.hidden_dim),
            config.act_fn(),
            torch.nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x):
        return self.net(x)
```

You can now create instances of your model, save them to disk, and load them back into memory:

```python
config = MyNNConfig(
    input_dim=10,
    hidden_dim=20,
    output_dim=2,
    act_fn=torch.nn.ReLU,
    loss_kwargs=dict(reduction="mean"),
)

### create your model from the config, and save
model = MyNN(config)
fname = "tests/junk_data/path_to_save_model<a href="zanj/zanj.html">zanj.zanj</a>"
ZANJ().save(model, fname)
### load by calling the class method `read()`
loaded_model = MyNN.read(fname)
### zanj will actually infer the type of the object in the file 
### -- and will warn you if you don't have the correct package installed
loaded_another_way = ZANJ().read(fname)
```

#### Configuration

When initializing a `ZANJ` object, you can specify some configuration info about saving, such as:

- thresholds for how big an array/table has to be before moving to external file
- compression settings
- error modes
- additional handlers for serialization

```python
### how big an array or list (including pandas DataFrame) can be before moving it from the core JSON file
external_array_threshold: int = ZANJ_GLOBAL_DEFAULTS.external_array_threshold
external_list_threshold: int = ZANJ_GLOBAL_DEFAULTS.external_list_threshold
### compression settings passed to `zipfile` package
compress: bool | int = ZANJ_GLOBAL_DEFAULTS.compress
### for doing very cursed things in your own custom loading or serialization functions
custom_settings: dict[str, Any] | None = ZANJ_GLOBAL_DEFAULTS.custom_settings
### specify additional serialization handlers
handlers_pre: MonoTuple[SerializerHandler] = tuple()
handlers_default: MonoTuple[SerializerHandler] = DEFAULT_SERIALIZER_HANDLERS_ZANJ,
```

### Implementation

The on-disk format is a file `<filename><a href="zanj/zanj.html">zanj.zanj</a>` is a zip file containing:

- `__zanj_meta__.json`: a file containing zanj-specific metadata including:
	- system information
	- installed packages
	- information about external files
- `__zanj__.json`: a file containing user-specified data
	- when an element is too big, it can be moved to an external file
		- `.npy` for numpy arrays or torch tensors
		- `.jsonl` for pandas dataframes or large sequences
	- list of external files stored in `__zanj_meta__.json`
	- "$ref" key, specified in `_REF_KEY` in muutils, will have value pointing to external file
	- `_FORMAT_KEY` key will detail an external format type


### Comparison to other formats



| Format                  | Safe | Zero-copy | Lazy loading | No file size limit | Layout control | Flexibility | Bfloat16 |
| ----------------------- | ---- | --------- | ------------ | ------------------ | -------------- | ----------- | -------- |
| pickle (PyTorch)        | ❌   | ❌        | ❌           | ✅                 | ❌             | ✅          | ✅       |
| H5 (Tensorflow)         | ✅   | ❌        | ✅           | ✅                 | ~              | ~           | ❌       |
| HDF5                    | ✅   | ?         | ✅           | ✅                 | ~              | ✅          | ❌       |
| SavedModel (Tensorflow) | ✅   | ❌        | ❌           | ✅                 | ✅             | ❌          | ✅       |
| MsgPack (flax)          | ✅   | ✅        | ❌           | ✅                 | ❌             | ❌          | ✅       |
| Protobuf (ONNX)         | ✅   | ❌        | ❌           | ❌                 | ❌             | ❌          | ✅       |
| Cap'n'Proto             | ✅   | ✅        | ~            | ✅                 | ✅             | ~           | ❌       |
| Numpy (npy,npz)         | ✅   | ?         | ?            | ❌                 | ✅             | ❌          | ❌       |
| SafeTensors             | ✅   | ✅        | ✅           | ✅                 | ✅             | ❌          | ✅       |
| exdir                   | ✅   | ?         | ?            | ?                  | ?              | ✅          | ❌       |
| ZANJ                    | ✅   | ❌        | ❌*          | ✅                 | ✅             | ✅          | ❌*      |


- Safe: Can I use a file randomly downloaded and expect not to run arbitrary code ?
- Zero-copy: Does reading the file require more memory than the original file ?
- Lazy loading: Can I inspect the file without loading everything ? And loading only some tensors in it without scanning the whole file (distributed setting) ?
- Layout control: Lazy loading, is not necessarily enough since if the information about tensors is spread out in your file, then even if the information is lazily accessible you might have to access most of your file to read the available tensors (incurring many DISK -> RAM copies). Controlling the layout to keep fast access to single tensors is important.
- No file size limit: Is there a limit to the file size ?
- Flexibility: Can I save custom code in the format and be able to use it later with zero extra code ? (~ means we can store more than pure tensors, but no custom code)
- Bfloat16: Does the format support native bfloat16 (meaning no weird workarounds are necessary)? This is becoming increasingly important in the ML world.

`*` denotes this feature may be coming at a future date :)

(This table was stolen from [safetensors](https://github.com/huggingface/safetensors/blob/main/README.md))



[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L0-L18)



### `def register_loader_handler` { #register_loader_handler }
```python
(handler: zanj.loading.LoaderHandler)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L251-L255)


register a custom loader handler


### `class ZANJ(muutils.json_serialize.json_serialize.JsonSerializer):` { #ZANJ }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L66-L247)


Zip up: Arrays in Numpy, JSON for everything else

given an arbitrary object, throw into a zip file, with arrays stored in .npy files, and everything else stored in a json file

(basically npz file with json)

- numpy (or pytorch) arrays are stored in paths according to their name and structure in the object
- everything else about the object is stored in a json file `zanj.json` in the root of the archive, via `muutils.json_serialize.JsonSerializer`
- metadata about ZANJ configuration, and optionally packages and versions, is stored in a `__zanj_meta__.json` file in the root of the archive

create a ZANJ-class via `z_cls = ZANJ().create(obj)`, and save/read instances of the object via `z_cls.save(obj, path)`, `z_cls.load(path)`. be sure to pass an **instance** of the object, to make sure that the attributes of the class can be correctly recognized


### `ZANJ` { #ZANJ.__init__ }
```python
(
    error_mode: muutils.errormode.ErrorMode = ErrorMode.Except,
    internal_array_mode: Literal['list', 'array_list_meta', 'array_hex_meta', 'array_b64_meta', 'external', 'zero_dim'] = 'array_list_meta',
    external_array_threshold: int = 256,
    external_list_threshold: int = 256,
    compress: bool | int = True,
    custom_settings: dict[str, typing.Any] | None = None,
    handlers_pre: None = (),
    handlers_default: None = (ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='numpy.ndarray:external', desc='external numpy array', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='torch.Tensor:external', desc='external torch tensor', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='list:external', desc='external list', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='tuple:external', desc='external tuple', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='pandas.DataFrame:external', desc='external pandas DataFrame', source_pckg='zanj'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='base types', desc='base types (bool, int, float, str, None)'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='dictionaries', desc='dictionaries'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='(list, tuple) -> list', desc='lists and tuples as lists'), SerializerHandler(check=<function <lambda>>, serialize_func=<function _serialize_override_serialize_func>, uid='.serialize override', desc='objects with .serialize method'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='namedtuple -> dict', desc='namedtuples as dicts'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='dataclass -> dict', desc='dataclasses as dicts'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='path -> str', desc='Path objects as posix strings'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='obj -> str(obj)', desc='directly serialize objects in `SERIALIZE_DIRECT_AS_STR` to strings'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='numpy.ndarray', desc='numpy arrays'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='torch.Tensor', desc='pytorch tensors'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='pandas.DataFrame', desc='pandas DataFrames'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='(set, list, tuple, Iterable) -> list', desc='sets, lists, tuples, and Iterables as lists'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='fallback', desc='fallback handler -- serialize object attributes and special functions as strings'))
)
```

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L81-L116)




- `external_array_threshold: int `




- `external_list_threshold: int `




- `custom_settings: dict `




- `compress `




### `def externals_info` { #ZANJ.externals_info }
```python
(self) -> dict[str, dict[str, str | int | list[int]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L118-L141)


return information about the current externals


### `def meta` { #ZANJ.meta }
```python
(
    self
) -> Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L143-L164)


return the metadata of the ZANJ archive


### `def save` { #ZANJ.save }
```python
(self, obj: Any, file_path: str | pathlib._local.Path) -> str
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L166-L219)


save the object to a ZANJ archive. returns the path to the archive


### `def read` { #ZANJ.read }
```python
(self, file_path: Union[str, pathlib._local.Path]) -> Any
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1__init__.py#L221-L247)


load the object from a ZANJ archive
### TODO: load only some part of the zanj file by passing an ObjectPath


### Inherited Members                                

- [`array_mode`](#ZANJ.array_mode)
- [`error_mode`](#ZANJ.error_mode)
- [`write_only_format`](#ZANJ.write_only_format)
- [`handlers`](#ZANJ.handlers)
- [`json_serialize`](#ZANJ.json_serialize)
- [`hashify`](#ZANJ.hashify)




> docs for [`zanj`](https://github.com/mivanit/zanj) v0.5.1


## Contents
for storing/retrieving an item externally in a ZANJ archive


## API Documentation

 - [`ZANJ_MAIN`](#ZANJ_MAIN)
 - [`ZANJ_META`](#ZANJ_META)
 - [`ExternalItemType`](#ExternalItemType)
 - [`ExternalItemType_vals`](#ExternalItemType_vals)
 - [`ExternalItem`](#ExternalItem)
 - [`load_jsonl`](#load_jsonl)
 - [`load_npy`](#load_npy)
 - [`EXTERNAL_LOAD_FUNCS`](#EXTERNAL_LOAD_FUNCS)
 - [`GET_EXTERNAL_LOAD_FUNC`](#GET_EXTERNAL_LOAD_FUNC)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1externals.py)

# `zanj.externals` { #zanj.externals }

for storing/retrieving an item externally in a ZANJ archive

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1externals.py#L0-L51)



- `ZANJ_MAIN: str = '__zanj__.json'`




- `ZANJ_META: str = '__zanj_meta__.json'`




- `ExternalItemType = typing.Literal['jsonl', 'npy']`




- `ExternalItemType_vals = ('jsonl', 'npy')`




### `class ExternalItem(typing.NamedTuple):` { #ExternalItem }


ExternalItem(item_type, data, path)


### `ExternalItem` { #ExternalItem.__init__ }
```python
(
    item_type: Literal['jsonl', 'npy'],
    data: Any,
    path: tuple[typing.Union[str, int], ...]
)
```


Create new instance of ExternalItem(item_type, data, path)


- `item_type: Literal['jsonl', 'npy'] `


Alias for field number 0


- `data: Any `


Alias for field number 1


- `path: tuple[typing.Union[str, int], ...] `


Alias for field number 2


### Inherited Members                                

- [`index`](#ExternalItem.index)
- [`count`](#ExternalItem.count)


### `def load_jsonl` { #load_jsonl }
```python
(
    zanj: "'LoadedZANJ'",
    fp: IO[bytes]
) -> list[typing.Union[bool, int, float, str, NoneType, typing.List[typing.Union[bool, int, float, str, NoneType, typing.List[typing.Any], typing.Dict[str, typing.Any]]], typing.Dict[str, typing.Union[bool, int, float, str, NoneType, typing.List[typing.Any], typing.Dict[str, typing.Any]]]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1externals.py#L32-L33)




### `def load_npy` { #load_npy }
```python
(zanj: "'LoadedZANJ'", fp: IO[bytes]) -> numpy.ndarray
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1externals.py#L36-L37)




- `EXTERNAL_LOAD_FUNCS: dict[typing.Literal['jsonl', 'npy'], typing.Callable[[zanj.zanj.ZANJ, typing.IO[bytes]], typing.Any]] = {'jsonl': <function load_jsonl>, 'npy': <function load_npy>}`




### `def GET_EXTERNAL_LOAD_FUNC` { #GET_EXTERNAL_LOAD_FUNC }
```python
(item_type: str) -> Callable[[zanj.zanj.ZANJ, IO[bytes]], Any]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1externals.py#L46-L52)






> docs for [`zanj`](https://github.com/mivanit/zanj) v0.5.1




## API Documentation

 - [`LoaderHandler`](#LoaderHandler)
 - [`LOADER_MAP_LOCK`](#LOADER_MAP_LOCK)
 - [`LOADER_MAP`](#LOADER_MAP)
 - [`register_loader_handler`](#register_loader_handler)
 - [`get_item_loader`](#get_item_loader)
 - [`load_item_recursive`](#load_item_recursive)
 - [`LoadedZANJ`](#LoadedZANJ)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py)

# `zanj.loading` { #zanj.loading }


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L0-L449)



### `class LoaderHandler:` { #LoaderHandler }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L82-L141)


handler for loading an object from a json file or a ZANJ archive


### `LoaderHandler` { #LoaderHandler.__init__ }
```python
(
    check: Callable[[Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]], tuple[Union[str, int], ...], Any], bool],
    load: Callable[[Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]], tuple[Union[str, int], ...], Any], Any],
    uid: str,
    source_pckg: str,
    priority: int = 0,
    desc: str = '(no description)'
)
```




- `check: Callable[[Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]], tuple[Union[str, int], ...], Any], bool] `




- `load: Callable[[Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]], tuple[Union[str, int], ...], Any], Any] `




- `uid: str `




- `source_pckg: str `




- `priority: int = 0`




- `desc: str = '(no description)'`




### `def serialize` { #LoaderHandler.serialize }
```python
(
    self
) -> Dict[str, Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L102-L120)


serialize the handler info


### `def from_formattedclass` { #LoaderHandler.from_formattedclass }
```python
(cls, fc: type, priority: int = 0)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L122-L141)


create a loader from a class with `serialize`, `load` methods and `__muutils_format__` attribute


- `LOADER_MAP_LOCK = <unlocked _thread.lock object>`




- `LOADER_MAP: dict[str, zanj.loading.LoaderHandler] = {'numpy.ndarray': LoaderHandler(check=<function <lambda>>, load=<function <lambda>>, uid='numpy.ndarray', source_pckg='zanj', priority=0, desc='numpy.ndarray loader'), 'torch.Tensor': LoaderHandler(check=<function <lambda>>, load=<function _torch_loaderhandler_load>, uid='torch.Tensor', source_pckg='zanj', priority=0, desc='torch.Tensor loader'), 'pandas.DataFrame': LoaderHandler(check=<function <lambda>>, load=<function <lambda>>, uid='pandas.DataFrame', source_pckg='zanj', priority=0, desc='pandas.DataFrame loader'), 'list': LoaderHandler(check=<function <lambda>>, load=<function <lambda>>, uid='list', source_pckg='zanj', priority=0, desc='list loader, for externals'), 'tuple': LoaderHandler(check=<function <lambda>>, load=<function <lambda>>, uid='tuple', source_pckg='zanj', priority=0, desc='tuple loader, for externals')}`




### `def register_loader_handler` { #register_loader_handler }
```python
(handler: zanj.loading.LoaderHandler)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L251-L255)


register a custom loader handler


### `def get_item_loader` { #get_item_loader }
```python
(
    json_item: Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]],
    path: tuple[typing.Union[str, int], ...],
    zanj: typing.Any | None = None,
    error_mode: muutils.errormode.ErrorMode = ErrorMode.Warn
) -> zanj.loading.LoaderHandler | None
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L258-L283)


get the loader for a json item


### `def load_item_recursive` { #load_item_recursive }
```python
(
    json_item: Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]],
    path: tuple[typing.Union[str, int], ...],
    zanj: typing.Any | None = None,
    error_mode: muutils.errormode.ErrorMode = ErrorMode.Warn,
    allow_not_loading: bool = True
) -> Any
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L286-L364)




### `class LoadedZANJ:` { #LoadedZANJ }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L406-L450)


for loading a zanj file


### `LoadedZANJ` { #LoadedZANJ.__init__ }
```python
(path: str | pathlib._local.Path, zanj: Any)
```

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L409-L438)




### `def populate_externals` { #LoadedZANJ.populate_externals }
```python
(self) -> None
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1loading.py#L440-L450)


put all external items into the main json data




> docs for [`zanj`](https://github.com/mivanit/zanj) v0.5.1




## API Documentation

 - [`KW_ONLY_KWARGS`](#KW_ONLY_KWARGS)
 - [`jsonl_metadata`](#jsonl_metadata)
 - [`store_npy`](#store_npy)
 - [`store_jsonl`](#store_jsonl)
 - [`EXTERNAL_STORE_FUNCS`](#EXTERNAL_STORE_FUNCS)
 - [`ZANJSerializerHandler`](#ZANJSerializerHandler)
 - [`zanj_external_serialize`](#zanj_external_serialize)
 - [`DEFAULT_SERIALIZER_HANDLERS_ZANJ`](#DEFAULT_SERIALIZER_HANDLERS_ZANJ)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py)

# `zanj.serializing` { #zanj.serializing }


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py#L0-L250)



- `KW_ONLY_KWARGS: dict = {'kw_only': True}`




### `def jsonl_metadata` { #jsonl_metadata }
```python
(
    data: list[typing.Dict[str, typing.Union[bool, int, float, str, NoneType, typing.List[typing.Union[bool, int, float, str, NoneType, typing.List[typing.Any], typing.Dict[str, typing.Any]]], typing.Dict[str, typing.Union[bool, int, float, str, NoneType, typing.List[typing.Any], typing.Dict[str, typing.Any]]]]]]
) -> dict
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py#L33-L49)


metadata about a jsonl object


### `def store_npy` { #store_npy }
```python
(self: Any, fp: IO[bytes], data: numpy.ndarray) -> None
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py#L52-L58)


store numpy array to given file as .npy


### `def store_jsonl` { #store_jsonl }
```python
(
    self: Any,
    fp: IO[bytes],
    data: Sequence[Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]]
) -> None
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py#L61-L66)


store sequence to given file as .jsonl


- `EXTERNAL_STORE_FUNCS: dict[typing.Literal['jsonl', 'npy'], typing.Callable[[typing.Any, typing.IO[bytes], typing.Any], NoneType]] = {'npy': <function store_npy>, 'jsonl': <function store_jsonl>}`




### `class ZANJSerializerHandler(muutils.json_serialize.json_serialize.SerializerHandler):` { #ZANJSerializerHandler }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py#L77-L90)


a handler for ZANJ serialization


### `ZANJSerializerHandler` { #ZANJSerializerHandler.__init__ }
```python
(
    uid: str,
    desc: str,
    *,
    check: Callable[[Any, Any, tuple[Union[str, int], ...]], bool],
    serialize_func: Callable[[Any, Any, tuple[Union[str, int], ...]], Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]],
    source_pckg: str
)
```




- `source_pckg: str `




- `check: Callable[[Any, Any, tuple[Union[str, int], ...]], bool] `




- `serialize_func: Callable[[Any, Any, tuple[Union[str, int], ...]], Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]] `




### Inherited Members                                

- [`uid`](#ZANJSerializerHandler.uid)
- [`desc`](#ZANJSerializerHandler.desc)
- [`serialize`](#ZANJSerializerHandler.serialize)


### `def zanj_external_serialize` { #zanj_external_serialize }
```python
(
    jser: Any,
    data: Any,
    path: tuple[typing.Union[str, int], ...],
    item_type: Literal['jsonl', 'npy'],
    _format: str
) -> Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1serializing.py#L93-L171)


stores a numpy array or jsonl externally in a ZANJ object

### Parameters:
 - `jser: ZANJ`
 - `data: Any`
 - `path: ObjectPath`
 - `item_type: ExternalItemType`

### Returns:
 - `JSONitem`
   json data with reference

### Modifies:
 - modifies `jser._externals`


- `DEFAULT_SERIALIZER_HANDLERS_ZANJ: None = (ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='numpy.ndarray:external', desc='external numpy array', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='torch.Tensor:external', desc='external torch tensor', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='list:external', desc='external list', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='tuple:external', desc='external tuple', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='pandas.DataFrame:external', desc='external pandas DataFrame', source_pckg='zanj'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='base types', desc='base types (bool, int, float, str, None)'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='dictionaries', desc='dictionaries'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='(list, tuple) -> list', desc='lists and tuples as lists'), SerializerHandler(check=<function <lambda>>, serialize_func=<function _serialize_override_serialize_func>, uid='.serialize override', desc='objects with .serialize method'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='namedtuple -> dict', desc='namedtuples as dicts'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='dataclass -> dict', desc='dataclasses as dicts'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='path -> str', desc='Path objects as posix strings'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='obj -> str(obj)', desc='directly serialize objects in `SERIALIZE_DIRECT_AS_STR` to strings'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='numpy.ndarray', desc='numpy arrays'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='torch.Tensor', desc='pytorch tensors'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='pandas.DataFrame', desc='pandas DataFrames'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='(set, list, tuple, Iterable) -> list', desc='sets, lists, tuples, and Iterables as lists'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='fallback', desc='fallback handler -- serialize object attributes and special functions as strings'))`






> docs for [`zanj`](https://github.com/mivanit/zanj) v0.5.1


## Contents
torch utilities for zanj -- in particular the `ConfiguredModel` base class

note that this requires torch


## API Documentation

 - [`KWArgs`](#KWArgs)
 - [`num_params`](#num_params)
 - [`get_module_device`](#get_module_device)
 - [`ConfiguredModel`](#ConfiguredModel)
 - [`set_config_class`](#set_config_class)
 - [`ConfigMismatchException`](#ConfigMismatchException)
 - [`assert_model_cfg_equality`](#assert_model_cfg_equality)
 - [`assert_model_exact_equality`](#assert_model_exact_equality)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py)

# `zanj.torchutil` { #zanj.torchutil }

torch utilities for zanj -- in particular the `ConfiguredModel` base class

note that this requires torch

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L0-L293)



- `KWArgs = typing.Any`




### `def num_params` { #num_params }
```python
(m: torch.nn.modules.module.Module, only_trainable: bool = True)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L32-L48)


return total number of parameters in a model

- only counting shared parameters once
- if `only_trainable` is False, will include parameters with `requires_grad = False`

https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model


### `def get_module_device` { #get_module_device }
```python
(
    m: torch.nn.modules.module.Module
) -> tuple[bool, torch.device | dict[str, torch.device]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L51-L67)


get the current devices


### `class ConfiguredModel(torch.nn.modules.module.Module, typing.Generic[~T_config]):` { #ConfiguredModel }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L73-L219)


a model that has a configuration, for saving with ZANJ

```python
@set_config_class(YourConfig)
class YourModule(ConfiguredModel[YourConfig]):
    def __init__(self, cfg: YourConfig):
        super().__init__(cfg)
```

`__init__()` must initialize the model from a config object only, and call
`super().__init__(zanj_model_config)`

If you are inheriting from another class + ConfiguredModel,
ConfiguredModel must be the first class in the inheritance list


- `zanj_config_class `

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L96-L96)




- `zanj_model_config: ~T_config `




- `training_records: dict | None `




### `def serialize` { #ConfiguredModel.serialize }
```python
(
    self,
    path: tuple[typing.Union[str, int], ...] = (),
    zanj: zanj.zanj.ZANJ | None = None
) -> dict[str, typing.Any]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L110-L130)




### `def save` { #ConfiguredModel.save }
```python
(self, file_path: str, zanj: zanj.zanj.ZANJ | None = None)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L132-L135)




### `def load` { #ConfiguredModel.load }
```python
(
    cls,
    obj: dict[str, typing.Any],
    path: tuple[typing.Union[str, int], ...],
    zanj: zanj.zanj.ZANJ | None = None
) -> zanj.torchutil.ConfiguredModel
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L146-L183)


load a model from a serialized object


### `def read` { #ConfiguredModel.read }
```python
(
    cls,
    file_path: str,
    zanj: zanj.zanj.ZANJ | None = None
) -> zanj.torchutil.ConfiguredModel
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L185-L193)


read a model from a file


### `def load_file` { #ConfiguredModel.load_file }
```python
(
    cls,
    file_path: str,
    zanj: zanj.zanj.ZANJ | None = None
) -> zanj.torchutil.ConfiguredModel
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L195-L201)


read a model from a file


### `def get_handler` { #ConfiguredModel.get_handler }
```python
(cls) -> zanj.loading.LoaderHandler
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L203-L216)




### `def num_params` { #ConfiguredModel.num_params }
```python
(self) -> int
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L218-L219)




### Inherited Members                                

- [`Module`](#ConfiguredModel.__init__)
- [`dump_patches`](#ConfiguredModel.dump_patches)
- [`training`](#ConfiguredModel.training)
- [`call_super_init`](#ConfiguredModel.call_super_init)
- [`forward`](#ConfiguredModel.forward)
- [`register_buffer`](#ConfiguredModel.register_buffer)
- [`register_parameter`](#ConfiguredModel.register_parameter)
- [`add_module`](#ConfiguredModel.add_module)
- [`register_module`](#ConfiguredModel.register_module)
- [`get_submodule`](#ConfiguredModel.get_submodule)
- [`set_submodule`](#ConfiguredModel.set_submodule)
- [`get_parameter`](#ConfiguredModel.get_parameter)
- [`get_buffer`](#ConfiguredModel.get_buffer)
- [`get_extra_state`](#ConfiguredModel.get_extra_state)
- [`set_extra_state`](#ConfiguredModel.set_extra_state)
- [`apply`](#ConfiguredModel.apply)
- [`cuda`](#ConfiguredModel.cuda)
- [`ipu`](#ConfiguredModel.ipu)
- [`xpu`](#ConfiguredModel.xpu)
- [`mtia`](#ConfiguredModel.mtia)
- [`cpu`](#ConfiguredModel.cpu)
- [`type`](#ConfiguredModel.type)
- [`float`](#ConfiguredModel.float)
- [`double`](#ConfiguredModel.double)
- [`half`](#ConfiguredModel.half)
- [`bfloat16`](#ConfiguredModel.bfloat16)
- [`to_empty`](#ConfiguredModel.to_empty)
- [`to`](#ConfiguredModel.to)
- [`register_full_backward_pre_hook`](#ConfiguredModel.register_full_backward_pre_hook)
- [`register_backward_hook`](#ConfiguredModel.register_backward_hook)
- [`register_full_backward_hook`](#ConfiguredModel.register_full_backward_hook)
- [`register_forward_pre_hook`](#ConfiguredModel.register_forward_pre_hook)
- [`register_forward_hook`](#ConfiguredModel.register_forward_hook)
- [`register_state_dict_post_hook`](#ConfiguredModel.register_state_dict_post_hook)
- [`register_state_dict_pre_hook`](#ConfiguredModel.register_state_dict_pre_hook)
- [`state_dict`](#ConfiguredModel.state_dict)
- [`register_load_state_dict_pre_hook`](#ConfiguredModel.register_load_state_dict_pre_hook)
- [`register_load_state_dict_post_hook`](#ConfiguredModel.register_load_state_dict_post_hook)
- [`load_state_dict`](#ConfiguredModel.load_state_dict)
- [`parameters`](#ConfiguredModel.parameters)
- [`named_parameters`](#ConfiguredModel.named_parameters)
- [`buffers`](#ConfiguredModel.buffers)
- [`named_buffers`](#ConfiguredModel.named_buffers)
- [`children`](#ConfiguredModel.children)
- [`named_children`](#ConfiguredModel.named_children)
- [`modules`](#ConfiguredModel.modules)
- [`named_modules`](#ConfiguredModel.named_modules)
- [`train`](#ConfiguredModel.train)
- [`eval`](#ConfiguredModel.eval)
- [`requires_grad_`](#ConfiguredModel.requires_grad_)
- [`zero_grad`](#ConfiguredModel.zero_grad)
- [`share_memory`](#ConfiguredModel.share_memory)
- [`extra_repr`](#ConfiguredModel.extra_repr)
- [`compile`](#ConfiguredModel.compile)


### `def set_config_class` { #set_config_class }
```python
(
    config_class: Type[muutils.json_serialize.serializable_dataclass.SerializableDataclass]
) -> Callable[[Type[zanj.torchutil.ConfiguredModel]], Type[zanj.torchutil.ConfiguredModel]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L222-L238)




### `class ConfigMismatchException(builtins.ValueError):` { #ConfigMismatchException }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L241-L247)


Inappropriate argument value (of correct type).


### `ConfigMismatchException` { #ConfigMismatchException.__init__ }
```python
(msg: str, diff)
```

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L242-L244)




- `diff `




### Inherited Members                                

- [`with_traceback`](#ConfigMismatchException.with_traceback)
- [`add_note`](#ConfigMismatchException.add_note)
- [`args`](#ConfigMismatchException.args)


### `def assert_model_cfg_equality` { #assert_model_cfg_equality }
```python
(
    model_a: zanj.torchutil.ConfiguredModel,
    model_b: zanj.torchutil.ConfiguredModel
)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L250-L271)


check both models are correct instances and have the same config

Raises:
    ConfigMismatchException: if the configs don't match, e.diff will contain the diff


### `def assert_model_exact_equality` { #assert_model_exact_equality }
```python
(
    model_a: zanj.torchutil.ConfiguredModel,
    model_b: zanj.torchutil.ConfiguredModel
)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1torchutil.py#L274-L294)


check the models are exactly equal, including state dict contents




> docs for [`zanj`](https://github.com/mivanit/zanj) v0.5.1


## Contents
an HDF5/exdir file alternative, which uses json for attributes, allows serialization of arbitrary data

for large arrays, the output is a .tar.gz file with most data in a json file, but with sufficiently large arrays stored in binary .npy files


"ZANJ" is an acronym that the AI tool [Elicit](https://elicit.org) came up with for me. not to be confused with:

- https://en.wikipedia.org/wiki/Zanj
- https://www.plutojournals.com/zanj/


## API Documentation

 - [`ZANJitem`](#ZANJitem)
 - [`ZANJ_GLOBAL_DEFAULTS`](#ZANJ_GLOBAL_DEFAULTS)
 - [`ZANJ`](#ZANJ)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py)

# `zanj.zanj` { #zanj.zanj }

an HDF5/exdir file alternative, which uses json for attributes, allows serialization of arbitrary data

for large arrays, the output is a .tar.gz file with most data in a json file, but with sufficiently large arrays stored in binary .npy files


"ZANJ" is an acronym that the AI tool [Elicit](https://elicit.org) came up with for me. not to be confused with:

- https://en.wikipedia.org/wiki/Zanj
- https://www.plutojournals.com/zanj/

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L0-L249)



- `ZANJitem = typing.Union[bool, int, float, str, NoneType, typing.List[typing.Union[bool, int, float, str, NoneType, typing.List[typing.Any], typing.Dict[str, typing.Any]]], typing.Dict[str, typing.Union[bool, int, float, str, NoneType, typing.List[typing.Any], typing.Dict[str, typing.Any]]], numpy.ndarray, ForwardRef('pd.DataFrame')]`




- `ZANJ_GLOBAL_DEFAULTS: zanj.zanj._ZANJ_GLOBAL_DEFAULTS_CLASS = _ZANJ_GLOBAL_DEFAULTS_CLASS(error_mode=ErrorMode.Except, internal_array_mode='array_list_meta', external_array_threshold=256, external_list_threshold=256, compress=True, custom_settings=None)`




### `class ZANJ(muutils.json_serialize.json_serialize.JsonSerializer):` { #ZANJ }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L66-L247)


Zip up: Arrays in Numpy, JSON for everything else

given an arbitrary object, throw into a zip file, with arrays stored in .npy files, and everything else stored in a json file

(basically npz file with json)

- numpy (or pytorch) arrays are stored in paths according to their name and structure in the object
- everything else about the object is stored in a json file `zanj.json` in the root of the archive, via `muutils.json_serialize.JsonSerializer`
- metadata about ZANJ configuration, and optionally packages and versions, is stored in a `__zanj_meta__.json` file in the root of the archive

create a ZANJ-class via `z_cls = ZANJ().create(obj)`, and save/read instances of the object via `z_cls.save(obj, path)`, `z_cls.load(path)`. be sure to pass an **instance** of the object, to make sure that the attributes of the class can be correctly recognized


### `ZANJ` { #ZANJ.__init__ }
```python
(
    error_mode: muutils.errormode.ErrorMode = ErrorMode.Except,
    internal_array_mode: Literal['list', 'array_list_meta', 'array_hex_meta', 'array_b64_meta', 'external', 'zero_dim'] = 'array_list_meta',
    external_array_threshold: int = 256,
    external_list_threshold: int = 256,
    compress: bool | int = True,
    custom_settings: dict[str, typing.Any] | None = None,
    handlers_pre: None = (),
    handlers_default: None = (ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='numpy.ndarray:external', desc='external numpy array', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='torch.Tensor:external', desc='external torch tensor', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='list:external', desc='external list', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='tuple:external', desc='external tuple', source_pckg='zanj'), ZANJSerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='pandas.DataFrame:external', desc='external pandas DataFrame', source_pckg='zanj'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='base types', desc='base types (bool, int, float, str, None)'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='dictionaries', desc='dictionaries'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='(list, tuple) -> list', desc='lists and tuples as lists'), SerializerHandler(check=<function <lambda>>, serialize_func=<function _serialize_override_serialize_func>, uid='.serialize override', desc='objects with .serialize method'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='namedtuple -> dict', desc='namedtuples as dicts'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='dataclass -> dict', desc='dataclasses as dicts'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='path -> str', desc='Path objects as posix strings'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='obj -> str(obj)', desc='directly serialize objects in `SERIALIZE_DIRECT_AS_STR` to strings'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='numpy.ndarray', desc='numpy arrays'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='torch.Tensor', desc='pytorch tensors'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='pandas.DataFrame', desc='pandas DataFrames'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='(set, list, tuple, Iterable) -> list', desc='sets, lists, tuples, and Iterables as lists'), SerializerHandler(check=<function <lambda>>, serialize_func=<function <lambda>>, uid='fallback', desc='fallback handler -- serialize object attributes and special functions as strings'))
)
```

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L81-L116)




- `external_array_threshold: int `




- `external_list_threshold: int `




- `custom_settings: dict `




- `compress `




### `def externals_info` { #ZANJ.externals_info }
```python
(self) -> dict[str, dict[str, str | int | list[int]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L118-L141)


return information about the current externals


### `def meta` { #ZANJ.meta }
```python
(
    self
) -> Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L143-L164)


return the metadata of the ZANJ archive


### `def save` { #ZANJ.save }
```python
(self, obj: Any, file_path: str | pathlib._local.Path) -> str
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L166-L219)


save the object to a ZANJ archive. returns the path to the archive


### `def read` { #ZANJ.read }
```python
(self, file_path: Union[str, pathlib._local.Path]) -> Any
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.5.1zanj.py#L221-L247)


load the object from a ZANJ archive
### TODO: load only some part of the zanj file by passing an ObjectPath


### Inherited Members                                

- [`array_mode`](#ZANJ.array_mode)
- [`error_mode`](#ZANJ.error_mode)
- [`write_only_format`](#ZANJ.write_only_format)
- [`handlers`](#ZANJ.handlers)
- [`json_serialize`](#ZANJ.json_serialize)
- [`hashify`](#ZANJ.hashify)



