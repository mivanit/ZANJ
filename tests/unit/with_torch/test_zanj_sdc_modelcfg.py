from __future__ import annotations

import json
import sys
import typing
from pathlib import Path

import numpy as np
import torch  # type: ignore[import-not-found]
from muutils.json_serialize import (
    SerializableDataclass,
    serializable_dataclass,
    serializable_field,
)

from zanj import ZANJ

np.random.seed(0)

# pylint: disable=missing-function-docstring,missing-class-docstring

TEST_DATA_PATH: Path = Path("tests/junk_data")


SUPPORTS_KW_ONLY: bool = bool(sys.version_info >= (3, 10))


@serializable_dataclass
class MyModelCfg(SerializableDataclass):
    name: str
    num_layers: int
    hidden_size: int
    dropout: float


@serializable_dataclass(kw_only=SUPPORTS_KW_ONLY)
class TrainCfg(SerializableDataclass):
    name: str
    weight_decay: float
    optimizer: typing.Type[torch.optim.Optimizer] = serializable_field(
        default_factory=lambda: torch.optim.Adam,
        serialization_fn=lambda x: x.__name__,
        loading_fn=lambda data: getattr(torch.optim, data["optimizer"]),
    )
    optimizer_kwargs: typing.Dict[str, typing.Any] = serializable_field(  # type: ignore
        default_factory=lambda: dict(lr=0.000001)
    )


class CustomCfg:
    def __init__(self, x: int, y: str):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def serialize(self):
        return {"x": self.x, "y": self.y}

    @classmethod
    def load(cls, data):
        return cls(
            **{
                "x": data["x"],
                "y": data["y"],
            }
        )


@serializable_dataclass(kw_only=SUPPORTS_KW_ONLY)
class BasicCfgHolder(SerializableDataclass):
    model: MyModelCfg
    optimizer: TrainCfg
    custom: typing.Optional[CustomCfg] = serializable_field(
        default=None,
        serialization_fn=lambda x: x.serialize(),
        loading_fn=lambda data: CustomCfg.load(data["custom"]),
    )


instance_basic: BasicCfgHolder = BasicCfgHolder(  # type: ignore
    model=MyModelCfg("lstm", 3, 128, 0.1),  # type: ignore
    optimizer=TrainCfg(  # type: ignore
        name="adamw",
        weight_decay=0.2,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=0.0001),
    ),
    custom=CustomCfg(42, "forty-two"),
)


def test_config_holder():
    instance_stored = instance_basic.serialize()
    with open(TEST_DATA_PATH / "test_config_holder.json", "w") as f:
        json.dump(instance_stored, f, indent="\t")
    with open(TEST_DATA_PATH / "test_config_holder.json", "r") as f:
        instance_stored_read = json.load(f)
    recovered = BasicCfgHolder.load(instance_stored_read)
    assert isinstance(recovered.model, MyModelCfg)
    assert isinstance(recovered.optimizer, TrainCfg)
    assert isinstance(recovered.custom, CustomCfg)
    assert recovered.custom.x == 42
    assert instance_basic == recovered


def test_config_holder_zanj():
    z = ZANJ()
    path = TEST_DATA_PATH / "test_config_holder.zanj"
    z.save(instance_basic, path)
    recovered = z.read(path)
    assert isinstance(recovered.model, MyModelCfg)
    assert isinstance(recovered.optimizer, TrainCfg)
    assert isinstance(recovered.custom, CustomCfg)
    assert recovered.custom.x == 42
    assert instance_basic == recovered


@serializable_dataclass(kw_only=SUPPORTS_KW_ONLY)
class BaseGPTConfig(SerializableDataclass):
    name: str
    act_fn: str
    d_model: int
    d_head: int
    n_layers: int


@serializable_dataclass(kw_only=SUPPORTS_KW_ONLY)
class AdvCfgHolder(SerializableDataclass):
    model_cfg: BaseGPTConfig
    name: str = serializable_field(default="default")
    tokenizer: typing.Optional[CustomCfg] = serializable_field(
        default=None,
        serialization_fn=lambda x: repr(x) if x is not None else None,
        loading_fn=lambda data: (
            None if data["tokenizer"] is None else NotImplementedError
        ),
    )


instance_adv: AdvCfgHolder = AdvCfgHolder(  # type: ignore
    model_cfg=BaseGPTConfig(  # type: ignore
        name="gpt2",
        act_fn="gelu",
        d_model=128,
        d_head=64,
        n_layers=3,
    ),
    tokenizer=None,
)


def test_adv_config_holder():
    instance_stored = instance_adv.serialize()
    with open(TEST_DATA_PATH / "test_adv_config_holder.json", "w") as f:
        json.dump(instance_stored, f, indent="\t")
    recovered = AdvCfgHolder.load(instance_stored)
    assert isinstance(recovered.model_cfg, BaseGPTConfig)
    assert instance_adv == recovered


def test_adv_config_holder_zanj():
    z = ZANJ()
    path = TEST_DATA_PATH / "test_adv_config_holder.zanj"
    z.save(instance_adv, path)
    recovered = z.read(path)
    assert isinstance(recovered.model_cfg, BaseGPTConfig)
    assert instance_adv == recovered
