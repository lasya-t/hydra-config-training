"""Typed config. Hydra checks overrides against these dataclasses, so a misspelt key or a wrong type fails
before any training starts."""
from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class ModelConf:
    name: str = MISSING
    params: dict = field(default_factory=dict)


@dataclass
class SplitConf:
    kind: str = MISSING           # random | time
    test_size: float = 0.25        # random split: share of rows held out for validation
    test_months: int = 3           # time split: validate on the last N development months


@dataclass
class DataConf:
    rows: int = 24_000
    future_months: int = 6         # the last N months are never used in training or validation
    features: list = field(default_factory=lambda: ["tenure_months", "support_calls", "data_gb",
                                                     "plan_age_years", "monthly_bill", "autopay"])


@dataclass
class Config:
    model: ModelConf = MISSING
    split: SplitConf = MISSING
    data: DataConf = field(default_factory=DataConf)
    seed: int = 0


def register() -> None:
    cs = ConfigStore.instance()
    cs.store(name="schema", node=Config)
    cs.store(group="model", name="schema_model", node=ModelConf)
    cs.store(group="split", name="schema_split", node=SplitConf)
