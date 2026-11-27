import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf.errors import ConfigKeyError, ValidationError

from trainer.collect import collect, summary
from trainer.data import load
from trainer.train import config_hash, run, split

CONF = str(Path(__file__).resolve().parents[1] / "src" / "trainer" / "conf")
SRC = str(Path(__file__).resolve().parents[1] / "src")


def cfg(*overrides):
    with initialize_config_dir(config_dir=CONF, version_base=None):
        return compose("config", overrides=["data.rows=4000", *overrides])


def test_same_config_gives_same_result():
    assert run(cfg("model=gbm")) == run(cfg("model=gbm"))


def test_hash_changes_with_any_setting():
    assert config_hash(cfg("model=gbm")) != config_hash(cfg("model=gbm", "model.params.learning_rate=0.1"))


def test_misspelt_key_is_rejected():
    with pytest.raises((ConfigCompositionException, ConfigKeyError)):
        cfg("split.test_mnths=3")


def test_wrong_type_is_rejected():
    with pytest.raises((ConfigCompositionException, ValidationError)):
        cfg("split.test_months=three")


def test_time_split_validates_on_the_latest_months():
    c = cfg("split=time")
    df = load(4000)
    dev = df[df.month < df.month.max() - c.data.future_months + 1]
    train, val = split(dev, c)
    assert train.month.max() < val.month.min()
    assert val.month.nunique() == c.split.test_months


def test_sweep_writes_one_metrics_file_per_run(tmp_path):
    env = {**os.environ, "PYTHONPATH": SRC}
    subprocess.run([sys.executable, "-m", "trainer.train", "-m", "model=logreg,gbm", "split=random,time",
                    "data.rows=3000", f"hydra.sweep.dir={tmp_path}"], check=True, env=env, capture_output=True)
    d = collect(tmp_path)
    assert len(d) == 4 and set(d.model) == {"logreg", "gbm"}
    assert len(summary(d)) == 4
    assert json.loads(next(tmp_path.glob("**/metrics.json")).read_text())["config_hash"]
