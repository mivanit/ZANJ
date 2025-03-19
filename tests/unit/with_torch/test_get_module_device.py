from __future__ import annotations

import pytest
import torch  # type: ignore[import-not-found]

from zanj.torchutil import get_module_device


def test_get_module_device_single_device():
    # Create a model and move it to a device
    model = torch.nn.Linear(10, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Run the function
    is_single, device_or_dict = get_module_device(model)

    # Assert that all parameters are on the same device and that device is returned
    assert is_single
    assert device_or_dict == device


def test_get_module_device_multiple_devices():
    # Create a model with parameters on different devices
    if torch.cuda.device_count() < 1:
        pytest.skip("This test requires at least one CUDA device")

    with torch.no_grad():
        model = torch.nn.Linear(10, 2)
        print(f"{model = }")
        model.weight = torch.nn.Parameter(model.weight.to("meta"))
        model.bias = torch.nn.Parameter(model.bias.to("cpu"))

    print(f"{model = }")
    print(f"{model.weight = }")
    print(f"{model.bias = }")

    # Run the function
    is_single, device_or_dict = get_module_device(model)

    print(f"{is_single = }, {device_or_dict = }")

    # Assert that not all parameters are on the same device and a dict is returned
    assert not is_single
    assert isinstance(device_or_dict, dict)

    # Check that the dict maps the correct devices
    assert device_or_dict["weight"] == torch.device("meta")
    assert device_or_dict["bias"] == torch.device("cpu")


def test_get_module_device_no_parameters():
    # Create a model with no parameters
    model = torch.nn.Sequential()

    # Run the function
    is_single, device_or_dict = get_module_device(model)

    # Assert that an empty dict is returned
    assert not is_single
    assert device_or_dict == {}
