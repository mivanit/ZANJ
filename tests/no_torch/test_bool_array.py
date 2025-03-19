from pathlib import Path

import numpy as np

from muutils.json_serialize import SerializableDataclass, serializable_dataclass

from zanj import ZANJ

TEST_DATA_PATH: Path = Path("tests/junk_data")


@serializable_dataclass
class MyClass_list(SerializableDataclass):
    name: str
    arr_1: list
    arr_2: list


def test_list_bool_array():
    fname: Path = TEST_DATA_PATH / "test_list_bool_array.zanj"
    c: MyClass_list = MyClass_list(
        name="test",
        arr_1=[True, False, True],
        arr_2=[True, False, True],
    )

    z = ZANJ()

    z.save(c, fname)

    c2: MyClass_list = z.read(fname)

    assert c == c2


@serializable_dataclass
class MyClass_np(SerializableDataclass):
    name: str
    arr_1: np.ndarray
    arr_2: np.ndarray


def test_np_bool_array():
    fname: Path = TEST_DATA_PATH / "test_np_bool_array.zanj"
    c: MyClass_np = MyClass_np(
        name="test",
        arr_1=np.array([True, False, True]),
        arr_2=np.array([True, False, True]),
    )

    z = ZANJ()

    z.save(c, fname)

    c2: MyClass_np = z.read(fname)

    assert c2.arr_1.dtype == np.bool_
    assert c2.arr_2.dtype == np.bool_

    assert c == c2
