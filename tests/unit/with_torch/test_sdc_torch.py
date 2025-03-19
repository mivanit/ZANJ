from __future__ import annotations

import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import]
import torch  # type: ignore[import-not-found]
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)

from zanj import ZANJ

np.random.seed(0)

TEST_DATA_PATH: Path = Path("tests/junk_data")

SUPPORTS_KW_ONLY: bool = bool(sys.version_info >= (3, 10))


@serializable_dataclass
class BasicZanjTorch(SerializableDataclass):
    a: str
    q: int = 42
    c: typing.List[int] = serializable_field(default_factory=list)


@serializable_dataclass
class NestedTorch(SerializableDataclass):
    name: str
    basic: BasicZanjTorch
    val: float


@serializable_dataclass
class sdc_with_torch_tensor(SerializableDataclass):
    name: str
    tensor1: torch.Tensor
    tensor2: torch.Tensor


def test_sdc_tensor_small():
    instance = sdc_with_torch_tensor("small tensors", torch.rand(8), torch.rand(16))

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_tensor_small.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


def test_sdc_tensor():
    instance = sdc_with_torch_tensor(
        "bigger tensors", torch.rand(128, 128), torch.rand(256, 256)
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_tensor.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


@serializable_dataclass(kw_only=SUPPORTS_KW_ONLY)
class sdc_complicated(SerializableDataclass):
    name: str
    arr1: np.ndarray
    arr2: np.ndarray
    iris_data: pd.DataFrame
    brain_data: pd.DataFrame
    container: typing.List[NestedTorch]

    tensor: torch.Tensor

    def __eq__(self, value):
        return super().__eq__(value)


def test_sdc_complicated():
    instance = sdc_complicated(
        name="complicated data",
        arr1=np.random.rand(128, 128),
        arr2=np.random.rand(256, 256),
        iris_data=pd.read_csv("tests/input_data/iris.csv"),
        brain_data=pd.read_csv("tests/input_data/brain_networks.csv"),
        container=[
            NestedTorch(
                f"n-{n}",
                BasicZanjTorch(f"n-{n}_b", n * 10 + 1, [n + 1, n + 2, n + 10]),
                n * np.pi,
            )
            for n in range(10)
        ],
        tensor=torch.rand(512, 512),
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_complicated.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered
