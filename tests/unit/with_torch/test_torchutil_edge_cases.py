"""Edge case tests for zanj/torchutil.py to improve coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from muutils.json_serialize import SerializableDataclass, serializable_dataclass

from zanj import ZANJ
from zanj.torchutil import (
    ConfiguredModel,
    assert_model_exact_equality,
    num_params,
    set_config_class,
)


TEST_DATA_PATH: Path = Path("tests/junk_data")


@serializable_dataclass
class EdgeCaseTestConfig(SerializableDataclass):
    """Simple config for testing."""

    hidden_size: int
    num_layers: int


@set_config_class(EdgeCaseTestConfig)
class EdgeCaseTestModel(ConfiguredModel[EdgeCaseTestConfig]):
    """Simple model for testing."""

    def __init__(self, cfg: EdgeCaseTestConfig):
        super().__init__(cfg)
        self.linear = torch.nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def forward(self, x):
        return self.linear(x)


class TestConfiguredModelValidation:
    """Tests for ConfiguredModel validation edge cases."""

    def test_missing_config_class_decorator(self):
        """Line 102: Model without @set_config_class decorator should raise NotImplementedError."""

        class UnconfiguredModel(ConfiguredModel):
            def __init__(self, cfg):
                super().__init__(cfg)

        with pytest.raises(NotImplementedError, match="need to set"):
            UnconfiguredModel(EdgeCaseTestConfig(32, 2))

    def test_wrong_config_type(self):
        """Line 104: Passing wrong config type should raise TypeError."""

        class WrongConfig(SerializableDataclass):
            other_field: str = "test"

        with pytest.raises(TypeError, match="must be an instance of"):
            EdgeCaseTestModel(WrongConfig())

    def test_config_not_serializable_dataclass(self):
        """Using config that isn't a dict should raise TypeError."""
        with pytest.raises(TypeError):
            EdgeCaseTestModel({"hidden_size": 32})  # type: ignore


class TestSetConfigClass:
    """Tests for set_config_class decorator."""

    def test_invalid_config_class_type(self):
        """Line 227: Passing non-SerializableDataclass should raise TypeError."""

        class NotSerializable:
            pass

        with pytest.raises(
            TypeError, match="must be a subclass of SerializableDataclass"
        ):

            @set_config_class(NotSerializable)  # type: ignore
            class BadModel(ConfiguredModel):
                pass


class TestConfiguredModelSaveLoad:
    """Tests for save/load with default ZANJ."""

    def test_save_with_default_zanj(self, tmp_path):
        """Lines 134-136: Save without explicit ZANJ should create default."""
        cfg = EdgeCaseTestConfig(16, 1)
        model = EdgeCaseTestModel(cfg)

        file_path = str(tmp_path / "model_default_zanj.zanj")
        model.save(file_path)  # No zanj argument - uses default

        assert Path(file_path).exists()

        # Verify we can load it back
        loaded = EdgeCaseTestModel.read(file_path)
        assert loaded.zanj_model_config.hidden_size == 16

    def test_load_with_default_zanj(self, tmp_path):
        """Line 154: Load without explicit ZANJ should create default."""
        cfg = EdgeCaseTestConfig(16, 1)
        model = EdgeCaseTestModel(cfg)

        # Save first
        file_path = str(tmp_path / "model_for_load_test.zanj")
        z = ZANJ()
        z.save(model.serialize(), file_path)

        # Load with default ZANJ (via read method)
        loaded = EdgeCaseTestModel.read(file_path)  # No zanj argument - uses default
        assert loaded.zanj_model_config.hidden_size == 16

    def test_serialize_with_default_zanj(self):
        """Line 114-115: Serialize without explicit ZANJ should create default."""
        cfg = EdgeCaseTestConfig(16, 1)
        model = EdgeCaseTestModel(cfg)

        # Call serialize without zanj argument
        serialized = model.serialize()  # No zanj argument - uses default

        assert "zanj_model_config" in serialized
        assert "state_dict" in serialized
        assert "__muutils_format__" in serialized


class TestNumParamsWrapper:
    """Tests for num_params instance method."""

    def test_num_params_instance_method(self):
        """Line 220: Test the instance method wrapper for num_params."""
        cfg = EdgeCaseTestConfig(16, 1)
        model = EdgeCaseTestModel(cfg)

        # Call instance method
        instance_result = model.num_params()

        # Call module-level function
        function_result = num_params(model)

        assert instance_result == function_result
        assert instance_result > 0


class TestAssertModelExactEquality:
    """Tests for assert_model_exact_equality function."""

    def test_state_dict_mismatch(self):
        """Lines 289-290: State dict value mismatch should fail assertion."""
        cfg = EdgeCaseTestConfig(16, 1)
        model_a = EdgeCaseTestModel(cfg)
        model_b = EdgeCaseTestModel(cfg)

        # Modify model_b's weights to not match model_a
        with torch.no_grad():
            for param in model_b.parameters():
                param.add_(1.0)  # Add 1 to all parameters

        with pytest.raises(AssertionError, match="state dict elements don't match"):
            assert_model_exact_equality(model_a, model_b)

    def test_equal_models_pass(self):
        """Equal models should pass the assertion."""
        cfg = EdgeCaseTestConfig(16, 1)
        model_a = EdgeCaseTestModel(cfg)
        model_b = EdgeCaseTestModel(cfg)

        # Copy state dict from a to b to make them equal
        model_b.load_state_dict(model_a.state_dict())

        # Should not raise
        assert_model_exact_equality(model_a, model_b)

    def test_state_dict_keys_mismatch(self):
        """Different state dict keys should fail assertion."""

        class DifferentModel(ConfiguredModel[EdgeCaseTestConfig]):
            _config_class = EdgeCaseTestConfig

            def __init__(self, cfg: EdgeCaseTestConfig):
                super().__init__(cfg)
                self.different_linear = torch.nn.Linear(
                    cfg.hidden_size, cfg.hidden_size
                )

            def forward(self, x):
                return self.different_linear(x)

        cfg = EdgeCaseTestConfig(16, 1)
        model_a = EdgeCaseTestModel(cfg)
        model_b = DifferentModel(cfg)

        with pytest.raises(AssertionError, match="state dict keys don't match"):
            assert_model_exact_equality(model_a, model_b)


class TestConfiguredModelReadRoundTrip:
    """Tests for ConfiguredModel read/write round trip."""

    def test_full_round_trip(self, tmp_path):
        """Test complete save and read cycle."""
        cfg = EdgeCaseTestConfig(32, 2)
        model = EdgeCaseTestModel(cfg)

        # Set some training records
        model.training_records = {"epochs": 10, "loss": 0.5}

        file_path = str(tmp_path / "round_trip_test.zanj")
        model.save(file_path)

        loaded = EdgeCaseTestModel.read(file_path)

        assert loaded.zanj_model_config.hidden_size == 32
        assert loaded.zanj_model_config.num_layers == 2
        assert loaded.training_records == {"epochs": 10, "loss": 0.5}

        # Check weights are equal
        assert_model_exact_equality(model, loaded)
