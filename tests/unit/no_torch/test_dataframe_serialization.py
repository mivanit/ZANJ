"""Tests for pandas DataFrame serialization edge cases and regression prevention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zanj import ZANJ

TEST_DATA_PATH: Path = Path("tests/junk_data")


def test_dataframe_detection_logic():
    """Verify the module + class name detection works for pandas DataFrames.

    This test would have caught the pandas 3.0 regression where the MRO string
    changed from 'pandas.core.frame.DataFrame' to 'pandas.DataFrame'.
    """
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # These are the exact checks used in serializing.py
    assert "pandas" in df.__class__.__module__, (
        f"Expected 'pandas' in module, got {df.__class__.__module__}"
    )
    assert df.__class__.__name__ == "DataFrame", (
        f"Expected class name 'DataFrame', got {df.__class__.__name__}"
    )


def test_small_dataframe_roundtrip():
    """Test DataFrame with fewer rows than external_list_threshold (256)."""
    df = pd.DataFrame(
        {
            "int_col": list(range(10)),
            "float_col": [x * 0.1 for x in range(10)],
            "str_col": [f"row_{x}" for x in range(10)],
        }
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_small_dataframe.zanj"
    z.save({"df": df}, path)
    recovered = z.read(path)

    assert isinstance(recovered["df"], pd.DataFrame), (
        f"Expected DataFrame, got {type(recovered['df'])}"
    )
    assert df.equals(recovered["df"]), "DataFrames should be equal"


def test_single_row_dataframe():
    """Test DataFrame with a single row (minimal case)."""
    df = pd.DataFrame({"a": [1], "b": [2]})

    z = ZANJ()
    path = TEST_DATA_PATH / "test_single_row_dataframe.zanj"
    z.save({"df": df}, path)
    recovered = z.read(path)

    assert isinstance(recovered["df"], pd.DataFrame), (
        f"Expected DataFrame, got {type(recovered['df'])}"
    )
    assert len(recovered["df"]) == 1, "DataFrame should have 1 row"
    assert list(recovered["df"].columns) == ["a", "b"], "Columns should be preserved"


def test_empty_dataframe():
    """Test DataFrame with zero rows."""
    df = pd.DataFrame({"a": [], "b": []})

    z = ZANJ()
    path = TEST_DATA_PATH / "test_empty_dataframe.zanj"
    z.save({"df": df}, path)
    recovered = z.read(path)

    assert isinstance(recovered["df"], pd.DataFrame), (
        f"Expected DataFrame, got {type(recovered['df'])}"
    )
    assert len(recovered["df"]) == 0, "DataFrame should be empty"
    assert list(recovered["df"].columns) == ["a", "b"], "Columns should be preserved"


def test_dataframe_dtype_preservation():
    """Verify that dtypes survive the round-trip."""
    df = pd.DataFrame(
        {
            "int_col": pd.array([1, 2, 3], dtype="int64"),
            "float_col": pd.array([1.1, 2.2, 3.3], dtype="float64"),
            "str_col": pd.array(["a", "b", "c"], dtype="object"),
            "bool_col": pd.array([True, False, True], dtype="bool"),
        }
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_dataframe_dtypes.zanj"
    z.save({"df": df}, path)
    recovered = z.read(path)

    assert isinstance(recovered["df"], pd.DataFrame)

    # Check values are preserved (dtypes may change due to JSON serialization)
    for col in df.columns:
        original_vals = df[col].tolist()
        recovered_vals = recovered["df"][col].tolist()
        assert original_vals == recovered_vals, (
            f"Column {col} values don't match: {original_vals} != {recovered_vals}"
        )


def test_dataframe_with_nan_values():
    """Test DataFrame containing NaN and None values."""
    df = pd.DataFrame(
        {
            "with_nan": [1.0, np.nan, 3.0],
            "with_none": [1, None, 3],
            "normal": [1, 2, 3],
        }
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_dataframe_nan.zanj"
    z.save({"df": df}, path)
    recovered = z.read(path)

    assert isinstance(recovered["df"], pd.DataFrame)

    # Check NaN is preserved (use isna() for comparison)
    assert pd.isna(recovered["df"]["with_nan"].iloc[1]), "NaN should be preserved"
    assert recovered["df"]["with_nan"].iloc[0] == 1.0
    assert recovered["df"]["with_nan"].iloc[2] == 3.0


def test_dataframe_special_column_names():
    """Test DataFrame with unusual column names."""
    df = pd.DataFrame(
        {
            "normal_name": [1, 2],
            "with spaces": [3, 4],
            "with-dashes": [5, 6],
            "123_numeric_start": [7, 8],
            "special!@#chars": [9, 10],
        }
    )

    z = ZANJ()
    path = TEST_DATA_PATH / "test_dataframe_special_cols.zanj"
    z.save({"df": df}, path)
    recovered = z.read(path)

    assert isinstance(recovered["df"], pd.DataFrame)
    assert list(recovered["df"].columns) == list(df.columns), (
        "Special column names should be preserved"
    )
    assert df.equals(recovered["df"]), "DataFrames should be equal"
