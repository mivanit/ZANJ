from pathlib import Path
import numpy as np

from zanj import ZANJ

_TEMP_PATH: Path = Path("tests/.temp/")


# NOTE: as of 2025-11-06 15:32 (v0.5.1), the first test (longer key first) fails, while the second test passes. wtf?


def test_shared_prefix_keys():
    fname: Path = _TEMP_PATH / "shared_prefix_keys.zanj"

    #
    data = {
        "layer.1.weight": np.random.rand(10, 10),
        "layer.1": np.random.rand(10, 10),
    }

    ZANJ(external_array_threshold=0).save(data, fname)

    loaded = ZANJ().read(fname)

    assert set(data.keys()) == set(loaded.keys())
    for key in data.keys():
        np.testing.assert_array_equal(data[key], loaded[key])


def test_shared_prefix_keys_reverse():
    fname: Path = _TEMP_PATH / "shared_prefix_keys_reverse.zanj"

    data = {
        "layer.1": np.random.rand(10, 10),
        "layer.1.weight": np.random.rand(10, 10),
    }

    ZANJ(external_array_threshold=0).save(data, fname)

    loaded = ZANJ().read(fname)

    assert set(data.keys()) == set(loaded.keys())
    for key in data.keys():
        np.testing.assert_array_equal(data[key], loaded[key])
