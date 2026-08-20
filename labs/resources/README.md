# Lab resources — ForgeSight

Everything the C539 labs need. All data is **synthetic and deterministic**
(seed 42), so every learner gets identical numbers and it is safe to paste into
an AI coding assistant.

Copy the two CSVs into your `torch-vibe/data/` folder in **Lab 2**.

| File | Used by | What it is |
| --- | --- | --- |
| `machines.csv` | Labs 3–11, 20 | 4000 rows of machine telemetry — 6 sensor features plus two targets |
| `vibration_series.csv` | Labs 18–20 | 3000 hourly vibration readings for time-series forecasting |
| `make_images.py` | Labs 12–16 | Generates the 1200-image surface-defect set into `data/defects/` |
| `train_wear_buggy.py` | Lab 6 | A script that runs fine and is wrong in five places — the review exercise |
| `make_data.py` | — | Regenerates both CSVs from scratch |

## `machines.csv`

4000 rows of telemetry from six CNC machines.

**Features (6):** `spindle_speed`, `feed_rate`, `coolant_temp`,
`vibration_rms`, `spindle_load`, `ambient_temp`
**Identifier (dropped before modelling):** `machine_id`

| Target | Task | Baseline to beat |
| --- | --- | --- |
| `tool_wear_um` | Regression (Lab 8) | Predicting the mean → **MAE ≈ 13.6 µm** |
| `qc_class` | 4-class classification (Lab 9) | Predicting the majority class → **≈ 57.6% accuracy** |

`qc_class`: `0` ok · `1` scratch · `2` dent · `3` burr — deliberately
imbalanced (ok is 57.6%) so accuracy alone flatters a model that never predicts
a defect. `burr` is genuinely hard and is the class most models miss, which is
the point of the confusion matrix in Lab 20.

With a 70/15/15 split the shapes are `X_train (2800, 6)`, `X_val (600, 6)`,
`X_test (600, 6)`. `vibration_rms` is **column index 3** once `machine_id` is
dropped (Lab 4 relies on this).

## `vibration_series.csv`

3000 hourly readings (`timestamp`, `vibration_rms`) starting 2026-01-01.
Contains a slow wear trend, a daily cycle, a weekly maintenance dip, three tool-change
resets, and AR(1) noise — smooth enough that **persistence is a strong baseline**.

| Baseline | Value |
| --- | --- |
| Persistence (next = last) on the test segment | **MAE ≈ 0.0284** |

There is real headroom above persistence (a linear autoregression on a 48-step
lookback reaches ≈ 0.0251, an ~12% improvement), so a well-tuned LSTM can beat
it — but only modestly. A learner reporting a *dramatic* improvement has almost
certainly leaked the future into the past, which is exactly what Lab 18 checks.

Chronological 70/15/15 split → index ranges `0–2099`, `2100–2549`, `2550–2999`.

## `make_images.py`

Run from your **`torch-vibe/` project root** in Lab 12:

```bash
python make_images.py
```

Generates 1200 greyscale 64×64 PNGs of machined surfaces:

```
data/defects/train/{dent,ok,scratch}/   300 each
data/defects/val/{dent,ok,scratch}/     100 each
```

`ImageFolder` assigns indices **alphabetically**, so
`class_to_idx = {'dent': 0, 'ok': 1, 'scratch': 2}` — not the order you would
naturally write. Lab 13 depends on you noticing.

`scratch` is high-contrast and directional (easy); `dent` is soft, rounded and
low-contrast (hard, and the class the CNN confuses with `ok`). Baseline to beat
is the majority class at **33.3%**.

## `train_wear_buggy.py`

The Lab 6 review exercise. It **runs to completion, prints a falling loss and
reports believable-looking numbers** while containing five real defects, none of
which raises an error:

1. Feature standardisation fitted **before** the train/test split — leakage.
2. `optimizer.zero_grad()` missing from the regression loop — gradients accumulate.
3. `nn.Softmax` on the classifier output **and** `nn.CrossEntropyLoss` — softmax applied twice.
4. Evaluation without `model.eval()` / `torch.no_grad()` — dropout left active.
5. Regression target of shape `(N,)` against a prediction of `(N, 1)` — `MSELoss` silently broadcasts.

The result: both models end up at roughly their baselines while looking
plausible. Defects 3 and 5 are the ones that actually stop learning; defect 1 is
the one AI assistants are least likely to catch.

> The file's docstring names all five defects. **Trainers: do not hand learners
> the file with that block visible before they have attempted the review
> themselves** — strip it, or have them run the script before opening it.

## Regenerating

```bash
python make_data.py     # rewrites machines.csv and vibration_series.csv
python make_images.py   # rewrites data/defects/ (run from torch-vibe/)
```

`make_data.py` needs only **numpy**; `make_images.py` needs **numpy** and
**Pillow**. Neither needs PyTorch, so both run before the Lab 2 install.
