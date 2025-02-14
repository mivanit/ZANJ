> docs for [`zanj`](https://github.com/mivanit/zanj) v0.4.0




## API Documentation

 - [`register_loader_handler`](#register_loader_handler)
 - [`ZANJ`](#ZANJ)




[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py)

# `zanj` { #zanj }


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L0-L5)



### `def register_loader_handler` { #register_loader_handler }
```python
(handler: zanj.loading.LoaderHandler)
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L234-L238)


register a custom loader handler


### `class ZANJ(muutils.json_serialize.json_serialize.JsonSerializer):` { #ZANJ }

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L66-L247)


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

[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L81-L116)




- `external_array_threshold: int `




- `external_list_threshold: int `




- `custom_settings: dict `




- `compress `




### `def externals_info` { #ZANJ.externals_info }
```python
(self) -> dict[str, dict[str, str | int | list[int]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L118-L141)


return information about the current externals


### `def meta` { #ZANJ.meta }
```python
(
    self
) -> Union[bool, int, float, str, NoneType, List[Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]], Dict[str, Union[bool, int, float, str, NoneType, List[Any], Dict[str, Any]]]]
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L143-L164)


return the metadata of the ZANJ archive


### `def save` { #ZANJ.save }
```python
(self, obj: Any, file_path: str | pathlib.Path) -> str
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L166-L219)


save the object to a ZANJ archive. returns the path to the archive


### `def read` { #ZANJ.read }
```python
(self, file_path: Union[str, pathlib.Path]) -> Any
```


[View Source on GitHub](https://github.com/mivanit/zanj/blob/0.4.0__init__.py#L221-L247)


load the object from a ZANJ archive
### TODO: load only some part of the zanj file by passing an ObjectPath


### Inherited Members                                

- [`array_mode`](#ZANJ.array_mode)
- [`error_mode`](#ZANJ.error_mode)
- [`write_only_format`](#ZANJ.write_only_format)
- [`handlers`](#ZANJ.handlers)
- [`json_serialize`](#ZANJ.json_serialize)
- [`hashify`](#ZANJ.hashify)



