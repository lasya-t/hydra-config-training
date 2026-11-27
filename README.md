# hydra-config-training

A churn-model training script driven by [Hydra](https://hydra.cc): config groups for the model and the
validation split, typed configs that reject typos, one command for a whole sweep, and every run writing
its own folder with the exact config and a `metrics.json`. The experiment it runs asks a practical
question: does the way you validate predict how the model will do on months it has never seen?

## The data

Two years of monthly churn records (24,000 rows). In month 12 a competitor launches, and over the next six
months data-heavy customers on older plans start leaving more. The last 6 months are the **future**:
never used for training or validation, only for checking the estimate at the end. The 18 months before
are for development.

## How the config works

```
src/trainer/conf/
  config.yaml            defaults: model=logreg, split=time; output folders named by model, split and seed
  model/logreg.yaml  model/random_forest.yaml  model/gbm.yaml
  split/random.yaml      validate on a random 25% of development rows
  split/time.yaml        validate on the last 3 development months
src/trainer/schema.py    dataclasses registered with Hydra's ConfigStore
```

```
PYTHONPATH=src python -m trainer.train                                  # one run with the defaults
PYTHONPATH=src python -m trainer.train model=gbm model.params.learning_rate=0.1
PYTHONPATH=src python -m trainer.train -m model=logreg,random_forest,gbm split=random,time seed=0,1,2
PYTHONPATH=src python -m trainer.collect                                # table from multirun/
```

Because the config is typed, `split.test_mnths=3` fails with an unknown-key error and `split.test_months=three`
with a type error, before anything trains. Without a schema, the first would silently add a key nobody reads
and the run would use the default. Each run records a hash of its resolved config, so a number in a report
can be traced to the exact settings.

Each run: train on the development months minus validation, score validation (the *estimate*), score the
future (the *truth*), then refit the same settings on all development months and score the future again
(what would ship).

## Results

The 18-run sweep (3 models × 2 splits × 3 seeds, about 30 seconds). Means over seeds; each seed is a fresh
dataset:

| model | split | validation AUC | future AUC (same model) | estimate error | sd | future AUC after refit |
|---|---|---|---|---|---|---|
| logreg | random | 0.707 | 0.731 | -0.024 | 0.024 | 0.732 |
| logreg | time | 0.699 | 0.708 | **-0.008** | 0.008 | 0.732 |
| random_forest | random | 0.695 | 0.721 | -0.026 | 0.023 | 0.725 |
| random_forest | time | 0.688 | 0.700 | -0.012 | 0.017 | 0.725 |
| gbm | random | 0.689 | 0.712 | -0.023 | 0.020 | 0.716 |
| gbm | time | 0.683 | 0.691 | **-0.008** | 0.010 | 0.716 |

- **Validating on the latest months estimates the future two to three times better** (error about -0.01
  against -0.025) and more consistently across seeds.
- **The random split was pessimistic here, not optimistic.** The usual warning is that random splits
  flatter a model. In this data the future is *easier* than the past: the competitor effect, once it has
  settled, is a strong signal, and a random validation set is mostly made of the earlier months where it
was weaker or absent. The lesson that
  holds either way: a random split measures the past, not what's coming.
- **Both splits picked the same model** (logistic regression) on every seed, and it was the best on the
  future each time. The true churn process here is logistic, which flatters linear models; with fewer than
  20 runs, don't read much into the ranking.
- **Refit before you ship.** The time split's own model scored 0.708 on the future, because it never trained
  on the three validation months, the ones closest to the future. Refitting the same settings on all
  development months gives 0.732. The validation months are for choosing, not for leaving out.

## Layout

```
src/trainer/data.py      the dataset
src/trainer/schema.py    typed config
src/trainer/conf/        config groups
src/trainer/train.py     one run: split, train, validate, future, refit; writes metrics.json
src/trainer/collect.py   sweep table and model picks
tests/                   reproducibility, typo and type rejection, split logic, a small real sweep
```

## Run

```
pip install -r requirements.txt
PYTHONPATH=src python -m trainer.train -m model=logreg,random_forest,gbm split=random,time seed=0,1,2
PYTHONPATH=src python -m trainer.collect
pytest -q
```

## Limits

One synthetic dataset with one kind of change over time. Hydra's sweeper here is the basic grid; the Optuna
sweeper plugin would turn the same config into a search.
