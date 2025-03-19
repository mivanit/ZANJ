from __future__ import annotations

import typing
from pathlib import Path

import pytest
import torch
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)

from zanj import ZANJ
from zanj.torchutil import (
    ConfiguredModel,
    assert_model_exact_equality,
    get_module_device,
    num_params,
    set_config_class,
)

TEST_DATA_PATH: Path = Path("tests/junk_data")


def test_num_params():
    """Test the num_params function with various models"""
    # Simple model with trainable parameters
    model1 = torch.nn.Sequential(
        torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 1)
    )

    # Expected number of trainable parameters:
    # Linear1: 10*20 + 20 = 220
    # Linear2: 20*1 + 1 = 21
    # Total: 241
    assert num_params(model1) == 241

    # Model with some non-trainable parameters
    model2 = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.BatchNorm1d(20, track_running_stats=True),
        torch.nn.Linear(20, 1),
    )

    # Freeze the batch norm layer
    for param in model2[1].parameters():
        param.requires_grad = False

    # Count only trainable parameters
    trainable_params = num_params(model2, only_trainable=True)

    # Count all parameters
    all_params = num_params(model2, only_trainable=False)

    # Batch norm has 2*20 = 40 parameters (weight and bias)
    # So difference should be 40
    assert all_params - trainable_params == 40


def test_get_module_device_empty():
    """Test get_module_device with a module that has no parameters"""

    # Create a module with no parameters
    class EmptyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x

    empty_module = EmptyModule()

    # Test the function
    is_single, device_dict = get_module_device(empty_module)

    # Should return False and an empty dict
    assert is_single is False
    assert device_dict == {}


def test_load_state_dict_wrapper():
    """Test the _load_state_dict_wrapper method of ConfiguredModel"""

    @serializable_dataclass
    class SimpleConfig(SerializableDataclass):
        size: int

    @set_config_class(SimpleConfig)
    class SimpleModel(ConfiguredModel[SimpleConfig]):
        def __init__(self, config: SimpleConfig):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.size, config.size)

        def forward(self, x):
            return self.linear(x)

        # Override _load_state_dict_wrapper to test custom behavior
        def _load_state_dict_wrapper(self, state_dict, **kwargs):
            # This should be called instead of the standard load_state_dict
            self.custom_wrapper_called = True
            # Still need to actually load the state dict
            return super()._load_state_dict_wrapper(state_dict)

    # Create a model, save it, and load it
    model = SimpleModel(SimpleConfig(10))
    path = TEST_DATA_PATH / "test_load_state_dict_wrapper.zanj"
    ZANJ().save(model, path)

    # Create a new model instance
    model2 = SimpleModel(SimpleConfig(10))
    model2.custom_wrapper_called = False

    # Use read to load the model, which should call _load_state_dict_wrapper
    loaded_model = model2.read(path)

    # Check that our custom wrapper was called
    assert hasattr(loaded_model, "custom_wrapper_called")
    assert loaded_model.custom_wrapper_called is True


def test_deprecated_load_file():
    """Test that the deprecated load_file method works and issues a warning"""

    @serializable_dataclass
    class SimpleConfig(SerializableDataclass):
        size: int

    @set_config_class(SimpleConfig)
    class SimpleModel(ConfiguredModel[SimpleConfig]):
        def __init__(self, config: SimpleConfig):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.size, config.size)

        def forward(self, x):
            return self.linear(x)

    # Create a model and save it
    model = SimpleModel(SimpleConfig(10))
    path = TEST_DATA_PATH / "test_deprecated_load_file.zanj"
    ZANJ().save(model, path)

    # Use the deprecated method with a warning check
    with pytest.warns(DeprecationWarning):
        loaded_model = SimpleModel.load_file(path)

    # Check that the model was loaded correctly
    assert_model_exact_equality(model, loaded_model)


def test_configmodel_training_records():
    """Test that training_records are properly saved and loaded"""

    @serializable_dataclass
    class SimpleConfig(SerializableDataclass):
        size: int

    @set_config_class(SimpleConfig)
    class SimpleModel(ConfiguredModel[SimpleConfig]):
        def __init__(self, config: SimpleConfig):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.size, config.size)
            self.training_records = None  # Initialize to None

        def forward(self, x):
            return self.linear(x)

    # Create a model and set some training records
    model = SimpleModel(SimpleConfig(10))
    model.training_records = {
        "loss": [1.0, 0.9, 0.8, 0.7],
        "accuracy": [0.6, 0.7, 0.8, 0.9],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005],
    }

    # Save the model
    path = TEST_DATA_PATH / "test_training_records.zanj"
    ZANJ().save(model, path)

    # Load the model
    loaded_model = SimpleModel.read(path)

    # Check that training records were loaded correctly
    assert loaded_model.training_records == model.training_records
    assert loaded_model.training_records["loss"] == [1.0, 0.9, 0.8, 0.7]
    assert loaded_model.training_records["accuracy"] == [0.6, 0.7, 0.8, 0.9]
    assert loaded_model.training_records["learning_rate"] == [
        0.01,
        0.005,
        0.001,
        0.0005,
    ]


def test_configmodel_with_custom_settings():
    """Test ConfiguredModel with custom settings for _load_state_dict_wrapper"""

    @serializable_dataclass
    class SimpleConfig(SerializableDataclass):
        size: int

    @set_config_class(SimpleConfig)
    class CustomStateModel(ConfiguredModel[SimpleConfig]):
        def __init__(self, config: SimpleConfig):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.size, config.size)
            self.custom_param_received = None

        def forward(self, x):
            return self.linear(x)

        def _load_state_dict_wrapper(self, state_dict, **kwargs):
            # Store the custom param if provided
            self.custom_param_received = kwargs.get("custom_param", None)
            return super(CustomStateModel, self)._load_state_dict_wrapper(state_dict)

    # Create a model
    model = CustomStateModel(SimpleConfig(10))
    path = TEST_DATA_PATH / "test_custom_settings.zanj"
    ZANJ().save(model, path)

    # Create a ZANJ with custom settings
    z = ZANJ(
        custom_settings={"_load_state_dict_wrapper": {"custom_param": "test_value"}}
    )

    # Load the model with the custom ZANJ
    loaded_model = z.read(path)

    # Check that the custom param was received
    assert loaded_model.custom_param_received == "test_value"


def test_complex_nested_structure():
    """Test saving and loading a complex nested structure with tensors"""

    # Define classes with proper forward references
    @serializable_dataclass
    class NestedConfig(SerializableDataclass):
        tensor_list: typing.List[torch.Tensor] = serializable_field(
            default_factory=list
        )
        tensor_dict: typing.Dict[
            str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]
        ] = serializable_field(default_factory=dict)

    @serializable_dataclass
    class ComplexConfig(SerializableDataclass):
        nested: NestedConfig
        size: int

    @set_config_class(ComplexConfig)
    class ComplexModel(ConfiguredModel[ComplexConfig]):
        def __init__(self, config: ComplexConfig):
            super().__init__(config)
            self.linear = torch.nn.Linear(config.size, config.size)

        def forward(self, x):
            return self.linear(x)

    # Create a complex nested configuration
    nested_config = NestedConfig(
        tensor_list=[torch.randn(5, 5), torch.randn(3, 3), torch.randn(7, 7)],
        tensor_dict={
            "a": torch.randn(10, 10),
            "b": torch.randn(8, 8),
            "c": {"c1": torch.randn(4, 4), "c2": torch.randn(6, 6)},
        },
    )

    complex_config = ComplexConfig(nested=nested_config, size=20)

    # Create a model with this config
    model = ComplexModel(complex_config)
    path = TEST_DATA_PATH / "test_complex_nested.zanj"

    # Save and load
    ZANJ().save(model, path)
    loaded_model = ComplexModel.read(path)

    # Check the model config size parameter
    assert loaded_model.zanj_model_config.size == model.zanj_model_config.size

    # Check that the nested config was loaded
    assert isinstance(loaded_model.zanj_model_config.nested, NestedConfig)

    # Check the tensor list lengths
    assert len(loaded_model.zanj_model_config.nested.tensor_list) == len(
        model.zanj_model_config.nested.tensor_list
    )

    # Check the nested tensors
    for i, tensor in enumerate(model.zanj_model_config.nested.tensor_list):
        loaded_tensor = loaded_model.zanj_model_config.nested.tensor_list[i]
        assert torch.allclose(tensor, loaded_tensor)

    # Check tensor dict keys
    assert set(loaded_model.zanj_model_config.nested.tensor_dict.keys()) == set(
        model.zanj_model_config.nested.tensor_dict.keys()
    )

    # Check the tensors in the dictionary
    for key in ["a", "b"]:
        tensor = model.zanj_model_config.nested.tensor_dict[key]
        loaded_tensor = loaded_model.zanj_model_config.nested.tensor_dict[key]
        assert torch.allclose(tensor, loaded_tensor)

    # Check the nested dictionary
    nested_dict = model.zanj_model_config.nested.tensor_dict["c"]
    loaded_nested_dict = loaded_model.zanj_model_config.nested.tensor_dict["c"]
    assert set(nested_dict.keys()) == set(loaded_nested_dict.keys())

    for key in nested_dict.keys():
        assert torch.allclose(nested_dict[key], loaded_nested_dict[key])
