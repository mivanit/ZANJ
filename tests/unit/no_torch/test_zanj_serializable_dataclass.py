from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import]
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
class BasicZanj(SerializableDataclass):
    a: str
    q: int = 42
    c: typing.List[int] = serializable_field(default_factory=list)


def test_Basic():
    instance = BasicZanj("hello", 42, [1, 2, 3])

    z = ZANJ()
    path = TEST_DATA_PATH / "test_BasicZanj.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


@serializable_dataclass
class Nested(SerializableDataclass):
    name: str
    basic: BasicZanj
    val: float


def test_Nested():
    instance = Nested("hello", BasicZanj("hello", 42, [1, 2, 3]), 3.14)

    z = ZANJ()
    path = TEST_DATA_PATH / "test_Nested.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


@serializable_dataclass
class Nested_with_container(SerializableDataclass):
    name: str
    basic: BasicZanj
    val: float
    container: typing.List[Nested] = serializable_field(default_factory=list)


def test_Nested_with_container():
    instance = Nested_with_container(
        "hello",
        basic=BasicZanj("hello", 42, [1, 2, 3]),
        val=3.14,
        container=[
            Nested("n1", BasicZanj("n1_b", 123, [4, 5, 7]), 2.71),
            Nested("n2", BasicZanj("n2_b", 456, [7, 8, 9]), 6.28),
        ],
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_Nested_with_container.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


@serializable_dataclass
class sdc_with_np_array(SerializableDataclass):
    name: str
    arr1: np.ndarray
    arr2: np.ndarray


def test_sdc_with_np_array_small():
    instance = sdc_with_np_array("small arrays", np.random.rand(10), np.random.rand(20))

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_with_np_array.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


def test_sdc_with_np_array():
    instance = sdc_with_np_array(
        "bigger arrays", np.random.rand(128, 128), np.random.rand(256, 256)
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_with_np_array.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


@serializable_dataclass
class sdc_with_df(SerializableDataclass):
    name: str
    iris_data: pd.DataFrame
    brain_data: pd.DataFrame


def test_sdc_with_df():
    instance = sdc_with_df(
        "downloaded_data",
        iris_data=pd.read_csv("tests/input_data/iris.csv"),
        brain_data=pd.read_csv("tests/input_data/brain_networks.csv"),
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_with_df.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered


@serializable_dataclass
class sdc_container_explicit(SerializableDataclass):
    name: str
    container: typing.List[Nested] = serializable_field(
        default_factory=list,
        # as jsonl string, for whatever reason
        serialization_fn=lambda c: "\n".join([json.dumps(n.serialize()) for n in c]),
        loading_fn=lambda data: [
            Nested.load(json.loads(n)) for n in data["container"].split("\n")
        ],
        # TODO: explicitly specifying the following does not work, since it gets automatically converted before we call load in `loading_fn`:
        # serialization_fn=lambda c: [n.serialize() for n in c],
        # loading_fn=lambda data: [Nested.load(n) for n in data["container"]],
    )


def test_sdc_container_explicit():
    instance = sdc_container_explicit(
        "container explicit",
        container=[
            Nested(
                f"n-{n}",
                BasicZanj(f"n-{n}_b", n * 10 + 1, [n + 1, n + 2, n + 10]),
                n * np.pi,
            )
            for n in range(10)
        ],
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_sdc_container_explicit.zanj"
    z.save(instance, path)
    recovered = z.read(path)
    assert instance == recovered
