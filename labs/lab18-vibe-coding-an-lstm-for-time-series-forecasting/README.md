# Lab 18 — Vibe Coding an LSTM for Time Series Forecasting

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 04:** Vibe Coding Recurrent Networks for Sequence Data  
> **Learning outcome:** Window and split a time series correctly and train an LSTM that beats a persistence baseline.

## Goal

Forecast ForgeSight's vibration sensor. This lab contains two traps that no error message will ever reveal. First, sequence data must be split by TIME — a random split lets the model train on data from after the period it is tested on. Second, the scaler must be fitted on the training window only. You establish the persistence baseline before modelling, prompt for the windowing and the LSTM with both constraints stated explicitly, then verify the split and the scaler yourself before trusting any score. A forecaster that cannot beat 'tomorrow equals today' has not learned the series.

## Why this lab matters

Time-series leakage is the most convincing bug in machine learning — it produces beautiful plots and excellent metrics that vanish entirely in production. Learning to check the split and the scaler before reading the score is the habit that protects every forecast you build afterwards.

## What you'll build

**An LSTM vibration forecaster with a time-ordered split and train-only scaling, measured against a persistence baseline**

**Tools:** PyTorch nn.LSTM, pandas, engine.py, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `windowing.py`
- `lstm_forecast.py`
- `reports/forecast.png`

## Before you start

- Lab 17 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Look at the series before modelling it, and establish the persistence baseline.

Persistence is a genuinely strong baseline for smooth physical signals. Vibration does not jump randomly hour to hour, so 'the same as last hour' is often quite accurate. Beating it requires the model to learn actual structure — trend, cycles, drift — rather than just tracking the level.

**PROMPT** — paste this into your AI coding assistant:

```text
Create baseline_series.py for data/vibration_series.csv, which has a timestamp column and a vibration_rms column of roughly 3000 hourly readings.
OPERATIONS: (1) load it, print the row count, the date range and basic statistics; (2) plot the full series to reports/series_overview.png; (3) compute the PERSISTENCE baseline on the last 20% of the series - predicting that each value equals the previous value - and print its MAE and RMSE; (4) print the standard deviation of the series for comparison.
Add a comment explaining why persistence is the correct baseline for a forecasting task and what it means if a model cannot beat it.
```

### 2. Run it, open the plot, and write the persistence MAE down. This is the number your LSTM must beat.

Write the persistence MAE in review_notes.md next to your Lab 8 and Lab 14 numbers. If your LSTM's MAE is higher than this, the honest report is that the model adds nothing, however good the plot looks.

**COMMAND** — run this in your terminal:

```bash
python baseline_series.py
```

### 3. Prompt for the windowing, with the time-split constraint stated as the primary requirement.

Requirement 4 is subtle and often missed: if you window the full series first and then split, windows near the boundary contain values from both sides, so training windows include validation timesteps. Split first, then window each segment independently.

**PROMPT** — paste this into your AI coding assistant:

```text
Create windowing.py that prepares data/vibration_series.csv for an LSTM.
TASK: turn one long univariate series into supervised (lookback -> horizon) training pairs.
OPERATIONS: build make_windows(series, lookback=48, horizon=1) returning X of shape (N, lookback, 1) and y of shape (N, 1). Then build get_series_loaders(lookback=48, batch_size=64) that:
1. Loads the series in TIME ORDER - never sorted or shuffled before splitting
2. Splits chronologically into the first 70% train, next 15% validation, final 15% test - by INDEX, not randomly
3. Fits the scaler using the TRAINING SEGMENT ONLY, computing mean and std from the training values, then applies those same values to validation and test
4. Windows each segment separately so no window ever spans a split boundary
5. Returns loaders plus the scaler statistics so predictions can be converted back to the original units
CONSTRAINTS: this is critical - do NOT use train_test_split with shuffle, and do NOT fit the scaler on the full series. Shuffling is allowed WITHIN the training loader only. Print the index ranges of the three splits and the training mean and std when run as a script.
EXPECTED OUTPUT: printed split ranges proving they are contiguous and non-overlapping, plus X and y shapes for each split.
```

### 4. Verify the two traps yourself. Check the printed split ranges are contiguous and in ascending order, and find the exact line where the scaler statistics are computed.

Read the printed ranges: they must look like 0-2099, 2100-2549, 2550-2999, contiguous and ascending. If the indices are scattered, a shuffle happened. And find the scaler line — it must compute from the training segment. If it uses the whole series, the future has leaked into the past.

**COMMAND** — run this in your terminal:

```bash
python windowing.py
```

### 5. Prompt for the model and training, reusing the engine again.

`output[:, -1, :]` takes the hidden state after the model has seen all 48 timesteps, which is what you want for a forecast. `output[:, 0, :]` is the state after ONE timestep — same shape, and the model would be forecasting from a single reading while appearing to work perfectly.

**PROMPT** — paste this into your AI coding assistant:

```text
Create lstm_forecast.py.
SHAPES: batches are X of (64, 48, 1) and y of (64, 1).
OPERATIONS: add class VibrationLSTM(nn.Module) to models.py with nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2) followed by nn.Linear(64, 1). In forward, take the LAST timestep of the LSTM output with output[:, -1, :] and pass it to the Linear layer - add a comment on why this is the right slice and what output[:, 0, :] would give instead.
Train with fit from engine.py: nn.MSELoss, Adam lr=1e-3, 60 epochs, an MAE metric computed in the ORIGINAL units by inverting the scaling.
CONSTRAINTS: batch_first=True must be explicit. Print the output shape of the first forward pass to prove the axis order.
EXPECTED OUTPUT: per-epoch losses, final test MAE and RMSE in original units, printed next to the persistence baseline MAE, and a forecast-versus-actual plot of the test segment saved to reports/forecast.png.
```

### 6. Train it and compare the MAE against your persistence baseline.

A meaningful improvement over persistence is a genuine success here. A small improvement is a common and honest outcome on a smooth series. If your MAE is dramatically better than persistence, be suspicious before being pleased — re-check the split and the scaler first.

**COMMAND** — run this in your terminal:

```bash
python lstm_forecast.py
```

### 7. Open reports/forecast.png and look at the shape of the prediction, not just the metric.

Look for a specific failure mode: a forecast that is simply the previous value shifted right by one step. That is the model learning persistence and nothing more. It looks excellent on a plot and is worth spotting — compare the curve's turning points against the actual series to see whether it anticipates or merely follows.

### 8. Prove to yourself that the time split mattered, by deliberately doing it wrong.

Expect the leaky pipeline to report a noticeably better MAE than the correct one. That number is the one you would have proudly presented, and it is unobtainable in production because you cannot scale today's reading using next month's statistics. This is the time-series version of the Lab 6 lesson.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a function leaky_comparison() to lstm_forecast.py that repeats the entire pipeline with two deliberate errors: a RANDOM shuffled split instead of a chronological one, and a scaler fitted on the FULL series before splitting. Train the identical model for the same 60 epochs and print the resulting test MAE next to the correct pipeline's MAE and the persistence baseline. Add a comment stating which number you would have reported if you had never checked the split, and why it is not achievable in production.
```

## Verification — Test it

windowing.py prints three contiguous ascending index ranges and the scaler statistics computed from the training segment only. lstm_forecast.py prints the first forward pass shape as torch.Size([64, 1]), reports a test MAE in original units alongside the persistence baseline, and saves reports/forecast.png. leaky_comparison prints a better-but-unachievable MAE from the shuffled, fully-scaled pipeline.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The model's MAE is worse than persistence | Common and often honest. Try a longer lookback, more epochs, or a lower learning rate — and if it still cannot beat persistence, report that finding rather than hiding it. |
| Loss falls to almost zero immediately | Leakage, or the target is included in the input window. Confirm y comes from the timestep AFTER the last input timestep, not from within the window. |
| RuntimeError: input must have 3 dimensions, got 2 | Add the feature axis with `.unsqueeze(-1)` so the shape becomes (N, lookback, 1). |
| Predictions are all the same value | The model collapsed to the mean. Check the scaler statistics are sensible, lower the learning rate, and confirm the last-timestep slice is correct. |
| The forecast plot is a flat line | You may be plotting scaled values while the actuals are in original units. Invert the scaling on both before plotting. |

## Going further (optional)

- Change the horizon to 6 steps ahead and see how much the error grows relative to persistence.
- Swap the LSTM for a GRU with the same hidden size and compare accuracy and training time.
- Add the coolant temperature as a second input feature and see whether a multivariate model helps.

---

[← Lab 17: Overview of RNNs, LSTM and GRU](../lab17-overview-of-rnns-lstm-and-gru/README.md) · [All labs](../README.md) · [Lab 19: Tuning Sequence Models with Follow-Up Prompts →](../lab19-tuning-sequence-models-with-follow-up-prompts/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
