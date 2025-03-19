from pathlib import Path

import torch
from muutils.json_serialize import SerializableDataclass, serializable_dataclass

from zanj import ZANJ

TEST_DATA_PATH: Path = Path("tests/junk_data")


@serializable_dataclass
class MyClass_torch(SerializableDataclass):
    name: str
    arr_1: torch.Tensor
    arr_2: torch.Tensor


def test_torch_bool_array():
    fname: Path = TEST_DATA_PATH / "test_torch_bool_array.zanj"
    c: MyClass_torch = MyClass_torch(
        name="test",
        arr_1=torch.tensor([True, False, True]),
        arr_2=torch.tensor([True, False, True]),
    )

    z = ZANJ()

    z.save(c, fname)

    c2: MyClass_torch = z.read(fname)

    assert c2.arr_1.dtype == torch.bool
    assert c2.arr_2.dtype == torch.bool

    assert c == c2
