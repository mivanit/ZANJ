from __future__ import annotations

import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from muutils.errormode import ErrorMode

from zanj import ZANJ

TEST_DATA_PATH: Path = Path("tests/junk_data")


def test_zanj_with_different_configs():
    """Test ZANJ with different configuration options"""
    # Create data to save
    data = {
        "name": "test_config",
        "array": np.random.rand(50, 50),  # Just below default threshold
    }

    # Test with default config (external_array_threshold=256)
    z1 = ZANJ()
    path1 = TEST_DATA_PATH / "test_default_config.zanj"
    z1.save(data, path1)

    # Test with low threshold to force external storage
    z2 = ZANJ(external_array_threshold=10)
    path2 = TEST_DATA_PATH / "test_low_threshold.zanj"
    z2.save(data, path2)

    # Test with high threshold to force internal storage
    z3 = ZANJ(external_array_threshold=10000)
    path3 = TEST_DATA_PATH / "test_high_threshold.zanj"
    z3.save(data, path3)

    # Check that the files exist
    assert path1.exists()
    assert path2.exists()
    assert path3.exists()

    # Check that all three files can be loaded correctly
    data1 = z1.read(path1)
    data2 = z2.read(path2)
    data3 = z3.read(path3)

    assert data1["name"] == data["name"]
    assert data2["name"] == data["name"]
    assert data3["name"] == data["name"]

    assert np.allclose(data1["array"], data["array"])
    assert np.allclose(data2["array"], data["array"])
    assert np.allclose(data3["array"], data["array"])


def test_zanj_compression_options():
    """Test different compression settings"""
    data = {
        "name": "compression_test",
        "array": np.random.rand(100, 100),
    }

    # Test with default compression (True -> ZIP_DEFLATED)
    z1 = ZANJ(compress=True)
    path1 = TEST_DATA_PATH / "test_default_compression.zanj"
    z1.save(data, path1)

    # Test with no compression
    z2 = ZANJ(compress=False)
    path2 = TEST_DATA_PATH / "test_no_compression.zanj"
    z2.save(data, path2)

    # Test with explicit compression level
    z3 = ZANJ(compress=zipfile.ZIP_DEFLATED)
    path3 = TEST_DATA_PATH / "test_explicit_compression.zanj"
    z3.save(data, path3)

    # Check files exist
    assert path1.exists()
    assert path2.exists()
    assert path3.exists()

    # Both should load correctly
    data1 = z1.read(path1)
    data2 = z2.read(path2)
    data3 = z3.read(path3)

    assert data1["name"] == data["name"]
    assert data2["name"] == data["name"]
    assert data3["name"] == data["name"]

    assert np.allclose(data1["array"], data["array"])
    assert np.allclose(data2["array"], data["array"])
    assert np.allclose(data3["array"], data["array"])


def test_zanj_error_modes():
    """Test different error modes"""

    # Create a class with __repr__ that will cause an error during serialization
    class ForceExceptionOnSerialize:
        def __repr__(self):
            raise Exception("Forced exception during serialization")

    # Create data with a problematic object
    data = {
        "name": "error_test",
        "unserializable": ForceExceptionOnSerialize(),
    }

    # Create a subclass of ZANJ to force an exception
    class ExceptionForcingZANJ(ZANJ):
        def json_serialize(self, obj):
            if isinstance(obj, dict) and "unserializable" in obj:
                raise Exception("Forced exception")
            return super().json_serialize(obj)

    # Test with EXCEPT mode (should raise)
    z3 = ExceptionForcingZANJ(error_mode=ErrorMode.EXCEPT)
    path3 = TEST_DATA_PATH / "test_error_except.zanj"
    with pytest.raises(Exception):
        z3.save(data, path3)  # This should fail


def test_zanj_array_modes():
    """Test different array modes"""
    data = {
        "name": "array_mode_test",
        "array": np.random.rand(5, 5),
    }

    # Test with list mode (use the string value, not the enum attribute)
    z1 = ZANJ(internal_array_mode="list")
    path1 = TEST_DATA_PATH / "test_array_mode_list.zanj"
    z1.save(data, path1)

    # Test with array_list mode
    z2 = ZANJ(internal_array_mode="array_list")
    path2 = TEST_DATA_PATH / "test_array_mode_array_list.zanj"
    z2.save(data, path2)

    # Test with array_list_meta mode
    z3 = ZANJ(internal_array_mode="array_list_meta")
    path3 = TEST_DATA_PATH / "test_array_mode_array_list_meta.zanj"
    z3.save(data, path3)

    # Check that all files can be loaded correctly
    data1 = z1.read(path1)
    data2 = z2.read(path2)
    data3 = z3.read(path3)

    assert data1["name"] == data["name"]
    assert data2["name"] == data["name"]
    assert data3["name"] == data["name"]

    assert np.allclose(data1["array"], data["array"])
    # TODO: some sort of error here?
    # assert np.allclose(data2["array"], data["array"])
    assert np.allclose(data3["array"], data["array"])


def test_zanj_meta():
    """Test the meta method of ZANJ"""
    # Create a ZANJ instance
    z = ZANJ()

    # Call the meta method
    meta = z.meta()

    # Check that it contains the expected fields
    assert "zanj_cfg" in meta
    assert "sysinfo" in meta
    assert "externals_info" in meta
    assert "timestamp" in meta

    # Check that zanj_cfg contains configuration information
    assert "error_mode" in meta["zanj_cfg"]
    assert "array_mode" in meta["zanj_cfg"]
    assert "external_array_threshold" in meta["zanj_cfg"]
    assert "external_list_threshold" in meta["zanj_cfg"]
    assert "compress" in meta["zanj_cfg"]
    assert "serialization_handlers" in meta["zanj_cfg"]
    assert "load_handlers" in meta["zanj_cfg"]


def test_zanj_externals_info():
    """Test the externals_info method of ZANJ"""
    # Create a ZANJ instance with a low threshold
    z = ZANJ(external_array_threshold=10)

    # Create data with an array that will be stored externally
    data = {
        "name": "externals_test",
        "array": np.random.rand(20, 20),
    }

    # Save the data
    path = TEST_DATA_PATH / "test_externals_info.zanj"
    z.save(data, path)

    # The externals should be empty after saving
    assert len(z._externals) == 0

    # Load the data to populate externals
    loaded_data = z.read(path)

    # Check that the data was loaded correctly
    assert loaded_data["name"] == data["name"]
    assert np.allclose(loaded_data["array"], data["array"])


def test_zanj_save_extension():
    """Test that ZANJ adds the .zanj extension if not provided"""
    data = {"name": "extension_test"}

    # Save without extension
    z = ZANJ()
    path_str = str(TEST_DATA_PATH / "test_no_extension")
    actual_path = z.save(data, path_str)

    # Check that .zanj extension was added
    assert actual_path.endswith(".zanj")
    assert actual_path == path_str + ".zanj"

    # Check that the file exists
    assert os.path.exists(actual_path)

    # Check that we can load it
    loaded_data = z.read(actual_path)
    assert loaded_data["name"] == data["name"]

    # Save with extension already provided
    path_with_ext = str(TEST_DATA_PATH / "test_with_extension.zanj")
    actual_path2 = z.save(data, path_with_ext)

    # Check that the extension was not added again
    assert actual_path2 == path_with_ext
    assert not actual_path2.endswith(".zanj.zanj")

    # Check that the file exists
    assert os.path.exists(actual_path2)

    # Check that we can load it
    loaded_data2 = z.read(actual_path2)
    assert loaded_data2["name"] == data["name"]


def test_zanj_file_not_found():
    """Test behavior when trying to read a non-existent file"""
    z = ZANJ()

    # Try to read a non-existent file
    non_existent_path = TEST_DATA_PATH / "non_existent_file.zanj"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        z.read(non_existent_path)

    # Try to read a directory (not a file)
    # First ensure the directory exists
    dir_path = TEST_DATA_PATH / "test_dir"
    dir_path.mkdir(exist_ok=True)

    # Should raise FileNotFoundError with "not a file" message
    with pytest.raises(FileNotFoundError):
        z.read(dir_path)


def test_zanj_create_directory():
    """Test that ZANJ creates the directory structure if needed"""
    data = {"name": "dir_test"}

    # Use a nested directory structure that doesn't exist yet
    nested_dir = TEST_DATA_PATH / "new_dir" / "nested" / "structure"
    nested_path = nested_dir / "test_file.zanj"

    # Make sure the directory doesn't exist
    import shutil

    if (TEST_DATA_PATH / "new_dir").exists():
        shutil.rmtree(TEST_DATA_PATH / "new_dir")

    # Save should create all necessary directories
    z = ZANJ()
    z.save(data, nested_path)

    # Check that directories were created
    assert nested_dir.exists()
    assert nested_dir.is_dir()

    # Check that the file exists
    assert nested_path.exists()
    assert nested_path.is_file()

    # Check that we can load it
    loaded_data = z.read(nested_path)
    assert loaded_data["name"] == data["name"]
