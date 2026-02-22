"""Edge case tests for zanj/loading.py to improve coverage."""

from __future__ import annotations

import pytest

from zanj.consts import _FORMAT_KEY
from zanj.loading import (
    LoaderHandler,
    _populate_externals_error_checking,
    get_item_loader,
    load_item_recursive,
)


class TestPopulateExternalsErrorChecking:
    """Tests for _populate_externals_error_checking function."""

    def test_external_item_missing_data_field(self):
        """Line 68: External item with format key but no data field should raise KeyError."""
        malformed_item = {
            _FORMAT_KEY: "list:external",
            "some_field": "value",
            # missing "data" field
        }
        with pytest.raises(KeyError, match="expected an external item"):
            _populate_externals_error_checking("key", malformed_item)

    def test_sequence_with_non_int_key(self):
        """Line 75-76: Accessing a sequence with a non-int key should raise TypeError."""
        sequence = [1, 2, 3]
        with pytest.raises(TypeError, match="expected int"):
            _populate_externals_error_checking("string_key", sequence)

    def test_sequence_with_out_of_range_key(self):
        """Line 77-78: Accessing a sequence with an out-of-range index should raise IndexError."""
        sequence = [1, 2, 3]
        with pytest.raises(IndexError, match="index out of range"):
            _populate_externals_error_checking(100, sequence)

    def test_mapping_with_non_str_key(self):
        """Line 82-83: Accessing a mapping with a non-str key should raise TypeError."""
        mapping = {"a": 1, "b": 2}
        with pytest.raises(TypeError, match="expected str"):
            _populate_externals_error_checking(123, mapping)

    def test_mapping_with_missing_key(self):
        """Line 84-85: Accessing a mapping with a missing key should raise KeyError."""
        mapping = {"a": 1, "b": 2}
        with pytest.raises(KeyError, match="key not in dict"):
            _populate_externals_error_checking("missing_key", mapping)

    def test_invalid_item_type(self):
        """Line 88-89: Passing an invalid item type should raise TypeError."""
        invalid_item = 42  # int is neither sequence nor mapping
        with pytest.raises(TypeError, match="expected dict or list"):
            _populate_externals_error_checking("key", invalid_item)

    def test_valid_sequence_access(self):
        """Valid sequence access should not raise and return False."""
        sequence = [1, 2, 3]
        result = _populate_externals_error_checking(1, sequence)
        assert result is False

    def test_valid_mapping_access(self):
        """Valid mapping access should not raise and return False."""
        mapping = {"a": 1, "b": 2}
        result = _populate_externals_error_checking("a", mapping)
        assert result is False

    def test_external_item_with_data_returns_true(self):
        """External item with data field should return True."""
        external_item = {
            _FORMAT_KEY: "list:external",
            "data": [1, 2, 3],
        }
        result = _populate_externals_error_checking("key", external_item)
        assert result is True


class TestLoaderHandlerFromFormattedClass:
    """Tests for LoaderHandler.from_formattedclass method."""

    def test_missing_serialize_method(self):
        """Line 137: Class missing serialize method should raise AssertionError."""

        class MissingSerialize:
            __muutils_format__ = "test"

            @classmethod
            def load(cls, json_item, path=None, z=None):
                pass

        with pytest.raises(AssertionError):
            LoaderHandler.from_formattedclass(MissingSerialize)

    def test_missing_load_method(self):
        """Line 139: Class missing load method should raise AssertionError."""

        class MissingLoad:
            __muutils_format__ = "test"

            def serialize(self):
                pass

        with pytest.raises(AssertionError):
            LoaderHandler.from_formattedclass(MissingLoad)

    def test_missing_format_key(self):
        """Line 141: Class missing __muutils_format__ should raise AssertionError."""

        class MissingFormat:
            def serialize(self):
                pass

            @classmethod
            def load(cls, json_item, path=None, z=None):
                pass

        with pytest.raises(AssertionError):
            LoaderHandler.from_formattedclass(MissingFormat)

    def test_non_string_format_key(self):
        """Line 142: Class with non-string __muutils_format__ should raise AssertionError."""

        class NonStringFormat:
            __muutils_format__ = 12345  # should be string

            def serialize(self):
                pass

            @classmethod
            def load(cls, json_item, path=None, z=None):
                pass

        with pytest.raises(AssertionError):
            LoaderHandler.from_formattedclass(NonStringFormat)

    def test_valid_formatted_class(self):
        """Valid class should create a LoaderHandler successfully."""

        class ValidClass:
            __muutils_format__ = "test.ValidClass"

            def serialize(self):
                return {}

            @classmethod
            def load(cls, json_item, path=None, z=None):
                return cls()

        handler = LoaderHandler.from_formattedclass(ValidClass)
        assert handler.uid == "test.ValidClass"
        assert "ValidClass" in handler.desc


class TestGetItemLoader:
    """Tests for get_item_loader function."""

    def test_non_string_format_key(self):
        """Line 306-309: Item with non-string __muutils_format__ should raise TypeError."""
        malformed_item = {
            _FORMAT_KEY: 12345,  # should be str
            "data": [1, 2, 3],
        }
        with pytest.raises(TypeError, match="invalid __muutils_format__ type"):
            get_item_loader(malformed_item, ("test",))


class TestLoadItemRecursive:
    """Tests for load_item_recursive function."""

    def test_unloadable_type_strict_mode(self):
        """Line 395-398: Unloadable type with allow_not_loading=False should raise ValueError."""

        # Create a custom object that isn't handled by any loader
        class CustomUnloadable:
            pass

        item = CustomUnloadable()
        with pytest.raises(ValueError, match="unknown type"):
            load_item_recursive(item, ("test",), None, allow_not_loading=False)

    def test_unloadable_type_permissive_mode(self):
        """Line 395-396: Unloadable type with allow_not_loading=True should return as-is."""

        class CustomUnloadable:
            pass

        item = CustomUnloadable()
        result = load_item_recursive(item, ("test",), None, allow_not_loading=True)
        assert result is item
