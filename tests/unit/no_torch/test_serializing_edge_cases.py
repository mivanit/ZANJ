"""Edge case tests for zanj/serializing.py to improve coverage."""

from __future__ import annotations

import numpy as np
import pytest

from zanj import ZANJ
from zanj.serializing import zanj_external_serialize


class TestZanjExternalSerialize:
    """Tests for zanj_external_serialize function edge cases."""

    def test_duplicate_external_path(self):
        """Lines 124-127: Duplicate external path should raise ValueError."""
        z = ZANJ()
        array1 = np.random.rand(10, 10)
        array2 = np.random.rand(5, 5)

        # First call with path ('data',) should succeed
        zanj_external_serialize(
            z, array1, path=("data",), item_type="npy", _format="numpy.ndarray:external"
        )

        # Second call with same path should raise ValueError
        with pytest.raises(ValueError, match="already exists"):
            zanj_external_serialize(
                z,
                array2,
                path=("data",),
                item_type="npy",
                _format="numpy.ndarray:external",
            )

    def test_path_prefix_conflict_child_first(self):
        """Lines 137-142: Path prefix conflict where new path is parent of existing."""
        z = ZANJ()
        array1 = np.random.rand(10, 10)
        array2 = np.random.rand(5, 5)

        # First: add child path 'layer/1/weight'
        zanj_external_serialize(
            z,
            array1,
            path=("layer", "1", "weight"),
            item_type="npy",
            _format="numpy.ndarray:external",
        )

        # Second: try to add parent path 'layer/1' - should conflict
        with pytest.raises(ValueError, match="is a prefix of another path"):
            zanj_external_serialize(
                z,
                array2,
                path=("layer", "1"),
                item_type="npy",
                _format="numpy.ndarray:external",
            )

    def test_path_prefix_conflict_parent_first(self):
        """Lines 134-136: Path prefix conflict where new path is child of existing."""
        z = ZANJ()
        array1 = np.random.rand(10, 10)
        array2 = np.random.rand(5, 5)

        # First: add parent path 'layer/1'
        zanj_external_serialize(
            z,
            array1,
            path=("layer", "1"),
            item_type="npy",
            _format="numpy.ndarray:external",
        )

        # Second: try to add child path 'layer/1/weight' - should conflict
        with pytest.raises(ValueError, match="is a prefix of another path"):
            zanj_external_serialize(
                z,
                array2,
                path=("layer", "1", "weight"),
                item_type="npy",
                _format="numpy.ndarray:external",
            )

    def test_invalid_npy_data_type(self):
        """Line 160: Invalid data type for NPY serialization should raise TypeError."""
        z = ZANJ()

        # Pass a string instead of array - should fail
        with pytest.raises(TypeError, match="expected numpy.ndarray"):
            zanj_external_serialize(
                z,
                data="not an array",
                path=("test",),
                item_type="npy",
                _format="numpy.ndarray:external",
            )

    def test_invalid_jsonl_data_type(self):
        """Line 184: Invalid data type for JSONL serialization should raise TypeError."""
        z = ZANJ()

        # Create a custom class that is not iterable/sequence/dataframe
        class NotSerializableAsJsonl:
            pass

        obj = NotSerializableAsJsonl()

        with pytest.raises(TypeError, match="expected list or pandas.DataFrame"):
            zanj_external_serialize(
                z,
                data=obj,
                path=("test",),
                item_type="jsonl",
                _format="list:external",
            )

    def test_valid_npy_serialization(self):
        """Verify valid NPY serialization works correctly."""
        z = ZANJ()
        array = np.random.rand(10, 10)

        result = zanj_external_serialize(
            z,
            array,
            path=("valid_array",),
            item_type="npy",
            _format="numpy.ndarray:external",
        )

        assert "__muutils_format__" in result
        assert result["__muutils_format__"] == "numpy.ndarray:external"
        assert "valid_array.npy" in z._externals

    def test_valid_jsonl_list_serialization(self):
        """Verify valid JSONL list serialization works correctly."""
        z = ZANJ()
        data = [{"a": 1}, {"b": 2}, {"c": 3}]

        result = zanj_external_serialize(
            z,
            data,
            path=("valid_list",),
            item_type="jsonl",
            _format="list:external",
        )

        assert "__muutils_format__" in result
        assert result["__muutils_format__"] == "list:external"
        assert "valid_list.jsonl" in z._externals

    def test_non_overlapping_paths_allowed(self):
        """Verify that non-overlapping paths with similar prefixes are allowed."""
        z = ZANJ()
        array1 = np.random.rand(10, 10)
        array2 = np.random.rand(5, 5)

        # These should NOT conflict: 'layer.1' and 'layer.1.weight' as strings
        # but 'layer/1' and 'layer/10' should be fine
        zanj_external_serialize(
            z,
            array1,
            path=("layer", "1"),
            item_type="npy",
            _format="numpy.ndarray:external",
        )

        # 'layer/10' is NOT a prefix of 'layer/1' or vice versa
        zanj_external_serialize(
            z,
            array2,
            path=("layer", "10"),
            item_type="npy",
            _format="numpy.ndarray:external",
        )

        assert "layer/1.npy" in z._externals
        assert "layer/10.npy" in z._externals
