from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import pytest
from muutils.errormode import ErrorMode
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)
from muutils.json_serialize.util import _FORMAT_KEY, _REF_KEY

from zanj import ZANJ
from zanj.loading import LoadedZANJ, load_item_recursive

TEST_DATA_PATH: Path = Path("tests/junk_data")


def test_load_item_recursive_basic():
    """Test basic functionality of load_item_recursive"""
    # Simple JSON data
    json_data = {
        "name": "test",
        "value": 42,
        "list": [1, 2, 3],
        "nested": {"a": 1, "b": 2},
    }

    # Load with default parameters
    result = load_item_recursive(json_data, tuple(), None)

    # Check the result
    assert result == json_data
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["list"] == [1, 2, 3]
    assert result["nested"] == {"a": 1, "b": 2}


def test_load_item_recursive_numpy_array():
    """Test loading a numpy array"""
    # Create a JSON representation of a numpy array
    array_data = np.random.rand(5, 5)
    json_data = {
        _FORMAT_KEY: "numpy.ndarray",
        "dtype": str(array_data.dtype),
        "shape": list(array_data.shape),
        "data": array_data.tolist(),
    }

    # Load with default parameters
    result = load_item_recursive(json_data, tuple(), None)

    # Check the result
    assert isinstance(result, np.ndarray)
    assert result.shape == tuple(json_data["shape"])
    assert result.dtype == np.dtype(json_data["dtype"])
    assert np.allclose(result, array_data)


def test_load_item_recursive_serializable_dataclass():
    """Test loading a SerializableDataclass"""

    @serializable_dataclass
    class TestClass(SerializableDataclass):
        name: str
        value: int
        data: typing.List[int] = serializable_field(default_factory=list)

    # Create an instance and serialize it
    instance = TestClass("test", 42, [1, 2, 3])
    serialized = instance.serialize()

    # Load with default parameters
    result = load_item_recursive(serialized, tuple(), None)

    # Check the result
    assert isinstance(result, TestClass)
    assert result.name == "test"
    assert result.value == 42
    assert result.data == [1, 2, 3]


def test_load_item_recursive_nested_container():
    """Test loading with nested containers"""
    # Create a complex nested structure
    json_data = {
        "name": "test",
        "arrays": [
            {
                _FORMAT_KEY: "numpy.ndarray",
                "dtype": "float64",
                "shape": [3, 3],
                "data": np.random.rand(3, 3).tolist(),
            },
            {
                _FORMAT_KEY: "numpy.ndarray",
                "dtype": "float64",
                "shape": [2, 2],
                "data": np.random.rand(2, 2).tolist(),
            },
        ],
        "nested": {
            "dict_with_array": {
                _FORMAT_KEY: "numpy.ndarray",
                "dtype": "float64",
                "shape": [4, 4],
                "data": np.random.rand(4, 4).tolist(),
            }
        },
    }

    # Load with default parameters
    result = load_item_recursive(json_data, tuple(), None)

    # Check the result
    assert result["name"] == "test"
    assert len(result["arrays"]) == 2
    assert isinstance(result["arrays"][0], np.ndarray)
    assert isinstance(result["arrays"][1], np.ndarray)
    assert result["arrays"][0].shape == (3, 3)
    assert result["arrays"][1].shape == (2, 2)
    assert isinstance(result["nested"]["dict_with_array"], np.ndarray)
    assert result["nested"]["dict_with_array"].shape == (4, 4)


def test_load_item_recursive_unknown_format():
    """Test loading with an unknown format key"""
    # Create JSON data with an unknown format
    json_data = {_FORMAT_KEY: "unknown.format", "data": [1, 2, 3]}

    # Load with default parameters (should return the JSON as is)
    result = load_item_recursive(json_data, tuple(), None, allow_not_loading=True)

    # Check the result
    assert result == json_data

    # Test with allow_not_loading=False (should raise an error)
    with pytest.raises(ValueError):
        load_item_recursive(json_data, tuple(), None, allow_not_loading=False)


def test_load_item_recursive_with_external_reference():
    """Test loading an item with an external reference"""
    # Create a ZANJ object and save some data to create externals
    z = ZANJ(external_array_threshold=10)
    data = {"large_array": np.random.rand(20, 20)}
    path = TEST_DATA_PATH / "test_load_item_recursive_external.zanj"
    z.save(data, path)

    # Load the ZANJ file
    loaded_zanj = LoadedZANJ(path, z)

    # Find an external item reference in the JSON data
    for ext_path, ext_item, item, _ in loaded_zanj._each_item_in_externals(
        loaded_zanj._externals, loaded_zanj._json_data
    ):
        # Test that the reference is correctly loaded
        assert _REF_KEY in item
        assert item[_REF_KEY] == ext_path

        # Now populate the externals
        loaded_zanj.populate_externals()

        # Check that the data field was added
        assert "data" in item
        assert isinstance(item["data"], np.ndarray)
        assert item["data"].shape == (20, 20)

        # Only need to test one example
        break


def test_load_item_recursive_error_modes():
    """Test different error modes"""
    # Create JSON data with an unknown format
    json_data = {_FORMAT_KEY: "unknown.format", "data": [1, 2, 3]}

    # Test WARN mode (should not raise, just return the data)
    result = load_item_recursive(
        json_data, tuple(), None, error_mode=ErrorMode.WARN, allow_not_loading=False
    )
    assert result == json_data

    # Test IGNORE mode (should not raise, just return the data)
    result = load_item_recursive(
        json_data, tuple(), None, error_mode=ErrorMode.IGNORE, allow_not_loading=False
    )
    assert result == json_data

    # Test EXCEPT mode (should raise)
    with pytest.raises(ValueError):
        load_item_recursive(
            json_data,
            tuple(),
            None,
            error_mode=ErrorMode.EXCEPT,
            allow_not_loading=False,
        )
