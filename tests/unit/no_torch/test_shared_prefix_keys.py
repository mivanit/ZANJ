from pathlib import Path
import numpy as np

import pytest

from zanj import ZANJ

_TEMP_PATH: Path = Path("tests/.temp/")


# NOTE: as of 2025-11-06 15:32 (v0.5.1), the first test (longer key first) fails, while the second test passes. wtf?

@pytest.mark.parametrize(
    ("keys", "name"),
    [
        (["layer.1.weight", "layer.1"], "longer_key_first"),
        (["layer.1", "layer.1.weight"], "shorter_key_first"),
    ],
)
def test_shared_prefix_keys(keys: list[str], name: str):
    fname: Path = _TEMP_PATH / f"shared_prefix_keys-{name}.zanj"

    #
    data = {
        key: np.random.rand(10, 10)
        for key in keys
    }

    ZANJ(external_array_threshold=0).save(data, fname)

    print("saved successfully")
    loaded = ZANJ().read(fname)
    assert set(data.keys()) == set(loaded.keys())
    for key in data.keys():
        print(f"{key = }")
        print(f"{type(data[key]) = }")
        print(f"{data[key] = }")
        print(f"{type(loaded[key]) = }")
        print(f"{loaded[key] = }")
        np.testing.assert_array_equal(data[key], loaded[key])
