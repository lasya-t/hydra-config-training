"""python -m trainer.train model=gbm split=random seed=1      (one run)
   python -m trainer.train -m model=logreg,gbm split=random,time seed=0,1,2   (a sweep)"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from trainer.data import load
from trainer.schema import register

register()


def build_model(cfg: DictConfig):
    p = OmegaConf.to_container(cfg.model.params)
    if cfg.model.name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(**p, random_state=cfg.seed))
    if cfg.model.name == "random_forest":
        return RandomForestClassifier(**p, random_state=cfg.seed, n_jobs=1)
    if cfg.model.name == "gbm":
        return HistGradientBoostingClassifier(**p, random_state=cfg.seed)
    raise ValueError(f"unknown model {cfg.model.name}")


def split(df, cfg: DictConfig):
    if cfg.split.kind == "random":
        return train_test_split(df, test_size=cfg.split.test_size, random_state=cfg.seed, stratify=df.churned)
    if cfg.split.kind == "time":
        cut = df.month.max() - cfg.split.test_months + 1
        return df[df.month < cut], df[df.month >= cut]
    raise ValueError(f"unknown split {cfg.split.kind}")


def config_hash(cfg: DictConfig) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True).encode()).hexdigest()[:12]


def run(cfg: DictConfig) -> dict:
    """Train on the development months, validate with the configured split, and then check the estimate
    against the future months that nothing has touched."""
    df = load(n=cfg.data.rows, seed=cfg.seed)
    first_future = df.month.max() - cfg.data.future_months + 1
    dev, future = df[df.month < first_future], df[df.month >= first_future]
    train, val = split(dev, cfg)
    feats = list(cfg.data.features)
    model = build_model(cfg).fit(train[feats], train.churned)
    val_auc = roc_auc_score(val.churned, model.predict_proba(val[feats])[:, 1])
    p_future = model.predict_proba(future[feats])[:, 1]
    future_auc = roc_auc_score(future.churned, p_future)
    # what ships: the same settings refitted on every development month, validation months included
    final = build_model(cfg).fit(dev[feats], dev.churned)
    refit_auc = roc_auc_score(future.churned, final.predict_proba(future[feats])[:, 1])
    return {"model": cfg.model.name, "split": cfg.split.kind, "seed": int(cfg.seed),
            "train_rows": len(train), "validation_rows": len(val),
            "validation_auc": round(float(val_auc), 4), "future_auc": round(float(future_auc), 4),
            "estimate_error": round(float(val_auc - future_auc), 4),
            "future_pr_auc": round(float(average_precision_score(future.churned, p_future)), 4),
            "future_auc_refit_on_all_dev": round(float(refit_auc), 4),
            "config_hash": config_hash(cfg)}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> float:
    metrics = run(cfg)
    out = Path(HydraConfig.get().runtime.output_dir)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics))
    return metrics["validation_auc"]


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
