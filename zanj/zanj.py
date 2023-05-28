"""
an HDF5/exdir file alternative, which uses json for attributes, allows serialization of arbitrary data

for large arrays, the output is a .tar.gz file with most data in a json file, but with sufficiently large arrays stored in binary .npy files


"ZANJ" is an acronym that the AI tool [Elicit](https://elicit.org) came up with for me. not to be confused with:

- https://en.wikipedia.org/wiki/Zanj
- https://www.plutojournals.com/zanj/

"""

import json
import os
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd  # type: ignore[import]
from muutils.json_serialize.array import ArrayMode, arr_metadata
from muutils.json_serialize.json_serialize import (
    JsonSerializer,
    SerializerHandler,
    json_serialize,
)
from muutils.json_serialize.util import ErrorMode, JSONitem, MonoTuple
from muutils.sysinfo import SysInfo

from zanj.externals import ZANJ_MAIN, ZANJ_META, ExternalItem, _ZANJ_pre
from zanj.loading import LOADER_MAP, LoadedZANJ, load_item_recursive
from zanj.serializing import DEFAULT_SERIALIZER_HANDLERS_ZANJ, EXTERNAL_STORE_FUNCS

# pylint: disable=protected-access, unused-import, dangerous-default-value, line-too-long

ZANJitem = Union[
    JSONitem,
    np.ndarray,
    pd.DataFrame,
]


@dataclass(kw_only=True)
class _ZANJ_GLOBAL_DEFAULTS_CLASS:
    error_mode: ErrorMode = "except"
    internal_array_mode: ArrayMode = "array_list_meta"
    external_array_threshold: int = 256
    external_list_threshold: int = 256
    compress: bool | int = True
    custom_settings: dict[str, Any] | None = None


ZANJ_GLOBAL_DEFAULTS: _ZANJ_GLOBAL_DEFAULTS_CLASS = _ZANJ_GLOBAL_DEFAULTS_CLASS()


class ZANJ(JsonSerializer):
    """Zip up: Arrays in Numpy, JSON for everything else

    given an arbitrary object, throw into a zip file, with arrays stored in .npy files, and everything else stored in a json file

    (basically npz file with json)

    - numpy (or pytorch) arrays are stored in paths according to their name and structure in the object
    - everything else about the object is stored in a json file `zanj.json` in the root of the archive, via `muutils.json_serialize.JsonSerializer`
    - metadata about ZANJ configuration, and optionally packages and versions, is stored in a `__zanj_meta__.json` file in the root of the archive

    create a ZANJ-class via `z_cls = ZANJ().create(obj)`, and save/read instances of the object via `z_cls.save(obj, path)`, `z_cls.load(path)`. be sure to pass an **instance** of the object, to make sure that the attributes of the class can be correctly recognized

    """

    def __init__(
        self,
        error_mode: ErrorMode = ZANJ_GLOBAL_DEFAULTS.error_mode,
        internal_array_mode: ArrayMode = ZANJ_GLOBAL_DEFAULTS.internal_array_mode,
        external_array_threshold: int = ZANJ_GLOBAL_DEFAULTS.external_array_threshold,
        external_list_threshold: int = ZANJ_GLOBAL_DEFAULTS.external_list_threshold,
        compress: bool | int = ZANJ_GLOBAL_DEFAULTS.compress,
        custom_settings: dict[str, Any] | None = ZANJ_GLOBAL_DEFAULTS.custom_settings,
        handlers_pre: MonoTuple[SerializerHandler] = tuple(),
        handlers_default: MonoTuple[
            SerializerHandler
        ] = DEFAULT_SERIALIZER_HANDLERS_ZANJ,
    ) -> None:
        super().__init__(
            array_mode=internal_array_mode,
            error_mode=error_mode,
            handlers_pre=handlers_pre,
            handlers_default=handlers_default,
        )

        self.external_array_threshold: int = external_array_threshold
        self.external_list_threshold: int = external_list_threshold
        self.custom_settings: dict = (
            custom_settings if custom_settings is not None else dict()
        )

        # process compression to int if bool given
        self.compress = compress
        if isinstance(compress, bool):
            if compress:
                self.compress = zipfile.ZIP_DEFLATED
            else:
                self.compress = zipfile.ZIP_STORED

        # create the externals, leave it empty
        self._externals: dict[str, ExternalItem] = dict()

    def externals_info(self) -> dict[str, dict[str, str | int | list[int]]]:
        """return information about the current externals"""
        output: dict[str, dict] = dict()

        key: str
        item: ExternalItem
        for key, item in self._externals.items():
            data = item.data
            output[key] = {
                "item_type": item.item_type,
                "path": item.path,
                "type(data)": str(type(data)),
                "len(data)": len(data),
            }

            if item.item_type == "ndarray":
                output[key].update(arr_metadata(data))
            elif item.item_type.startswith("jsonl"):
                output[key]["data[0]"] = data[0]

        return {
            key: val
            for key, val in sorted(output.items(), key=lambda x: len(x[1]["path"]))
        }

    def meta(self) -> JSONitem:
        """return the metadata of the ZANJ archive"""

        serialization_handlers = {h.uid: h.serialize() for h in self.handlers}
        load_handlers = {h.uid: h.serialize() for h in LOADER_MAP.values()}

        return dict(
            # configuration of this ZANJ instance
            zanj_cfg=dict(
                error_mode=str(self.error_mode),
                array_mode=str(self.array_mode),
                external_array_threshold=self.external_array_threshold,
                external_list_threshold=self.external_list_threshold,
                compress=self.compress,
                serialization_handlers=serialization_handlers,
                load_handlers=load_handlers,
            ),
            # system info (python, pip packages, torch & cuda, platform info, git info)
            sysinfo=json_serialize(SysInfo.get_all(include=("python", "pytorch"))),
            externals_info=self.externals_info(),
            timestamp=time.time(),
        )

    def save(self, obj: Any, file_path: str | Path) -> str:
        """save the object to a ZANJ archive. returns the path to the archive"""

        # adjust extension
        file_path = str(file_path)
        if not file_path.endswith(".zanj"):
            file_path += ".zanj"

        # make directory
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # clear the externals!
        self._externals = dict()

        # serialize the object -- this will populate self._externals
        # TODO: calling self.json_serialize again here might be slow
        json_data: JSONitem = self.json_serialize(self.json_serialize(obj))

        # open the zip file
        zipf: zipfile.ZipFile = zipfile.ZipFile(
            file=file_path, mode="w", compression=self.compress
        )

        # store base json data and metadata
        zipf.writestr(
            ZANJ_META,
            json.dumps(
                self.json_serialize(self.meta()),
                indent="\t",
            ),
        )
        zipf.writestr(
            ZANJ_MAIN,
            json.dumps(
                json_data,
                indent="\t",
            ),
        )

        # store externals
        for key, (ext_type, ext_data, ext_path) in self._externals.items():
            # why force zip64? numpy.savez does it
            with zipf.open(key, "w", force_zip64=True) as fp:
                EXTERNAL_STORE_FUNCS[ext_type](self, fp, ext_data)

        zipf.close()

        # clear the externals, again
        self._externals = dict()

        return file_path

    def read(
        self,
        file_path: Union[str, Path],
    ) -> Any:
        """load the object from a ZANJ archive
        # TODO: load only some part of the zanj file by passing an ObjectPath
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"file not found: {file_path}")
        if not file_path.is_file():
            raise FileNotFoundError(f"not a file: {file_path}")

        loaded_zanj: LoadedZANJ = LoadedZANJ(
            path=file_path,
            zanj=self,
        )

        loaded_zanj.populate_externals()

        return load_item_recursive(
            loaded_zanj._json_data,
            path=tuple(),
            zanj=self,
            error_mode=self.error_mode,
            # lh_map=loader_handlers,
        )


_ZANJ_pre = ZANJ  # type: ignore
