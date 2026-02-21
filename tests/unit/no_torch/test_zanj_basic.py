from __future__ import annotations

import json
import typing
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore

from zanj import ZANJ

np.random.seed(0)


TEST_DATA_PATH: Path = Path("tests/junk_data")


def array_meta(x: typing.Any) -> dict:
    if isinstance(x, np.ndarray):
        return dict(
            shape=list(x.shape),
            dtype=str(x.dtype),
            contents=str(x),
        )
    else:
        return dict(
            type=type(x).__name__,
            contents=str(x),
        )


def test_numpy():
    data = dict(
        name="testing zanj",
        some_array=np.random.rand(128, 128),
        some_other_array=np.random.rand(16, 64),
        small_array=np.random.rand(4, 4),
    )
    fname: Path = TEST_DATA_PATH / "test_numpy.zanj"
    z: ZANJ = ZANJ()
    z.save(data, fname)
    recovered_data = z.read(fname)

    print(f"{list(data.keys()) = }")
    print(f"{list(recovered_data.keys()) = }")
    original_vals: dict = {k: array_meta(v) for k, v in data.items()}
    print(json.dumps(original_vals, indent=2))
    recovered_vals: dict = {k: array_meta(v) for k, v in recovered_data.items()}
    print(json.dumps(recovered_vals, indent=2))

    assert sorted(list(data.keys())) == sorted(list(recovered_data.keys()))
    # assert all([type(data[k]) == type(recovered_data[k]) for k in data.keys()])

    assert all(
        [
            data["name"] == recovered_data["name"],
            np.allclose(data["some_array"], recovered_data["some_array"]),
            np.allclose(data["some_other_array"], recovered_data["some_other_array"]),
            np.allclose(data["small_array"], recovered_data["small_array"]),
        ]
    ), f"assert failed:\n{data = }\n{recovered_data = }"


def test_jsonl():
    data = dict(
        name="testing zanj jsonl",
        iris_data=pd.read_csv("tests/input_data/iris.csv"),
        brain_data=pd.read_csv("tests/input_data/brain_networks.csv"),
        some_array=np.random.rand(128, 128),
    )
    fname: Path = TEST_DATA_PATH / "test_jsonl.zanj"
    z: ZANJ = ZANJ()
    z.save(data, fname)
    recovered_data = z.read(fname)

    assert sorted(list(data.keys())) == sorted(list(recovered_data.keys()))
    # assert all([type(data[k]) == type(recovered_data[k]) for k in data.keys()])

    assert all(
        [
            data["name"] == recovered_data["name"],
            np.allclose(data["some_array"], recovered_data["some_array"]),
            data["iris_data"].equals(recovered_data["iris_data"]),
            data["brain_data"].equals(recovered_data["brain_data"]),
        ]
    )


def test_polars_dataframe():
    import polars as pl

    # basic dataframe with various types
    data = dict(
        name="testing zanj polars",
        df=pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
                "c": [1.1, 2.2, 3.3],
            }
        ),
        some_array=np.random.rand(128, 128),
    )
    fname: Path = TEST_DATA_PATH / "test_polars.zanj"
    z: ZANJ = ZANJ()
    z.save(data, fname)
    recovered_data = z.read(fname)

    assert sorted(list(data.keys())) == sorted(list(recovered_data.keys()))

    assert all(
        [
            data["name"] == recovered_data["name"],
            np.allclose(data["some_array"], recovered_data["some_array"]),
            data["df"].equals(recovered_data["df"]),
        ]
    )


def test_polars_dataframe_empty():
    """Test empty polars DataFrame serialization"""
    import polars as pl

    data = dict(
        name="testing empty polars df",
        empty_df=pl.DataFrame({"a": [], "b": [], "c": []}),
    )
    fname: Path = TEST_DATA_PATH / "test_polars_empty.zanj"
    z: ZANJ = ZANJ()
    z.save(data, fname)
    recovered_data = z.read(fname)

    assert data["name"] == recovered_data["name"]
    assert recovered_data["empty_df"].shape == (0, 3)
    assert recovered_data["empty_df"].columns == ["a", "b", "c"]


def test_polars_dataframe_large():
    """Test larger polars DataFrame to ensure external storage works"""
    import polars as pl

    # create a larger dataframe
    n_rows = 1000
    data = dict(
        name="testing large polars df",
        large_df=pl.DataFrame(
            {
                "int_col": list(range(n_rows)),
                "float_col": [float(i) * 0.1 for i in range(n_rows)],
                "str_col": [f"row_{i}" for i in range(n_rows)],
                "bool_col": [i % 2 == 0 for i in range(n_rows)],
            }
        ),
    )
    fname: Path = TEST_DATA_PATH / "test_polars_large.zanj"
    z: ZANJ = ZANJ()
    z.save(data, fname)
    recovered_data = z.read(fname)

    assert data["name"] == recovered_data["name"]
    assert data["large_df"].equals(recovered_data["large_df"])


def test_polars_with_nulls():
    """Test polars DataFrame with null values"""
    import polars as pl

    data = dict(
        name="testing polars with nulls",
        df_with_nulls=pl.DataFrame(
            {
                "a": [1, None, 3],
                "b": ["x", "y", None],
                "c": [1.1, None, 3.3],
            }
        ),
    )
    fname: Path = TEST_DATA_PATH / "test_polars_nulls.zanj"
    z: ZANJ = ZANJ()
    z.save(data, fname)
    recovered_data = z.read(fname)

    assert data["name"] == recovered_data["name"]
    assert data["df_with_nulls"].equals(recovered_data["df_with_nulls"])
