# Lab 19 — Tuning Sequence Models with Follow-Up Prompts

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 04:** Vibe Coding Recurrent Networks for Sequence Data  
> **Learning outcome:** Improve a sequence model through disciplined follow-up prompts and record what each change actually bought.

## Goal

Tuning is where vibe coding either pays off or descends into random prompting. You run a structured search over the choices that matter for a sequence model — lookback length, hidden size, layer count, cell type and learning rate — changing one thing at a time and recording every result in a table. The discipline being trained is the follow-up prompt: naming a specific symptom and a specific change rather than asking the assistant to 'make it better'. You finish with a tuned model, a results table that justifies it, and an explicit note of which changes made no difference at all.

## Why this lab matters

An assistant will happily change five things at once when asked to improve a model, leaving you unable to explain any result. One-factor-at-a-time tuning with a recorded table is slower per step and far faster overall, and it is the only version you can defend to anyone.

## What you'll build

**A tuning results table across lookback, hidden size, layers, cell type and learning rate, plus a justified best model**

**Tools:** PyTorch, engine.py, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `tune_lstm.py`
- `reports/tuning_results.csv`
- `prompts.md`

## Before you start

- Lab 18 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. First, see what an undisciplined prompt produces, so the contrast is concrete.

Expect the assistant to return a model with a different hidden size, an extra layer, a new learning rate, bidirectionality and possibly attention — all at once. If it improves, you cannot say why; if it gets worse, you cannot say why either. This is the failure mode the rest of the lab prevents.

**PROMPT** — paste this into your AI coding assistant:

```text
My LSTM forecaster gets a test MAE only slightly better than a persistence baseline. Make it better.
```

### 2. Read what came back and count how many things it changed simultaneously. Do not run it — this is evidence, not code.

Count the changes. Four or five is typical. That is not tuning, it is replacement — and it is exactly what 'make it better' asks for. The prompt was the problem, not the assistant.

### 3. Now the disciplined version. Prompt for a structured one-factor-at-a-time search.

One factor at a time is not the most sample-efficient search that exists, but it is the one that produces an explanation. Keeping the persistence baseline as a constant column is what stops a table of tiny differences from looking like progress when nothing beats the baseline.

**PROMPT** — paste this into your AI coding assistant:

```text
Create tune_lstm.py running a one-factor-at-a-time search over the vibration forecasting model.
BASELINE CONFIG: lookback=48, hidden_size=64, num_layers=2, cell=LSTM, lr=1e-3, 40 epochs.
RUNS: starting from that baseline, vary ONE factor at a time:
- lookback in [12, 24, 48, 96]
- hidden_size in [16, 32, 64, 128]
- num_layers in [1, 2, 3]
- cell in [LSTM, GRU]
- lr in [1e-2, 1e-3, 1e-4]
For each run record: the factor changed, its value, best validation MAE in original units, test MAE in original units, epochs to best, and wall-clock seconds.
CONSTRAINTS: reset manual seed 42 before every run and build a fresh model each time. Identical data and batch size across all runs. Include the persistence baseline MAE as a constant column so every row can be read against it.
EXPECTED OUTPUT: a printed table grouped by factor, saved to reports/tuning_results.csv, and a figure with one subplot per factor showing test MAE against the factor's value.
```

### 4. Run the search. It is a few dozen short training runs — expect several minutes on CPU.

Fresh model and reset seed per run is the clause assistants most often drop. If run 2 continues from run 1's weights the whole table is meaningless — check the generated code for a model built inside the loop, not outside it.

**COMMAND** — run this in your terminal:

```bash
python tune_lstm.py
```

### 5. Read the table by factor and answer: which factor produced the largest spread in test MAE, and which produced almost none?

Typical findings: lookback and learning rate produce the largest spread; hidden size shows diminishing returns past a point; num_layers beyond 2 usually adds time without accuracy; LSTM and GRU come out very close, with GRU faster. Knowing which knobs do nothing is as valuable as knowing which ones work.

### 6. Practise the disciplined follow-up on whatever your table actually showed. Adapt this to your own result.

Notice the structure of this follow-up: it names the file, states the observed pattern with numbers, offers an interpretation, specifies the exact change, and explicitly fences everything else with 'change nothing else'. That fence is what keeps your tuning table comparable across rounds.

**PROMPT** — paste this into your AI coding assistant:

```text
In tune_lstm.py the lookback sweep shows test MAE improving from 12 to 48 timesteps but getting worse at 96, which suggests the longer window adds noise rather than signal. Add two intermediate values, 64 and 80, to the lookback sweep only. Change nothing else - same seed, same architecture, same number of epochs - and print the extended lookback results as a single sorted table so I can see where the minimum actually sits.
```

### 7. Train the winning configuration properly and check it against the baseline one final time.

The winning configuration deserves a longer, properly early-stopped run — the sweep used 40 epochs to keep the search fast, which usually undertrains. Recording the full configuration in the checkpoint means the result is reproducible rather than remembered.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a function train_best() to tune_lstm.py that takes the best configuration from reports/tuning_results.csv, trains it for 120 epochs with early stopping on validation loss, reports test MAE and RMSE in original units next to the persistence baseline, saves the model with save_checkpoint to models/vibration_lstm_v1.pt recording the full winning configuration in the metrics dict, and writes models/vibration_model_card.md stating the task, the lookback, the baseline comparison and the fact that the split was chronological.
```

### 8. Record in prompts.md the three follow-up prompts that worked and, just as importantly, which tuned factors turned out not to matter.

Writing down what did NOT help is the part everyone skips and the part most worth having. Next time you tune a sequence model you will start from lookback and learning rate instead of adding layers.

## Verification — Test it

reports/tuning_results.csv contains one row per configuration with the factor changed, both MAE columns and the persistence baseline as a constant column. You can name the factor with the largest spread and the one with the smallest. models/vibration_lstm_v1.pt holds the winning configuration and beats the persistence baseline, and prompts.md records three working follow-up prompts.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Every configuration gives an identical MAE | The model is not being rebuilt per run, or the factor is not actually reaching the constructor. Print the configuration at the start of each run to confirm. |
| The sweep takes too long | Cut epochs to 25 for the search only, then train the winner properly. Note the reduction so the table stays honest. |
| The best configuration performs worse when retrained longer | Overfitting at higher epoch counts. Use the early-stopped best weights, which is what train_best is for. |
| lookback=96 raises a shape or memory error | Longer windows mean fewer samples and more memory per batch. Reduce the batch size for that run and note it in the table. |
| GRU and LSTM results are identical to several decimal places | Suspicious — confirm the cell type is actually being switched rather than the string only being recorded. |

## Going further (optional)

- Add a bidirectional LSTM to the cell sweep and consider carefully whether it is even legitimate for forecasting.
- Replace the grid with a small random search over the same ranges and compare how quickly each finds the best configuration.
- Add a 2-factor interaction check on the two most influential factors and see whether their effects are independent.

---

[← Lab 18: Vibe Coding an LSTM for Time Series Forecasting](../lab18-vibe-coding-an-lstm-for-time-series-forecasting/README.md) · [All labs](../README.md) · [Lab 20: Evaluating and Visualizing Model Performance →](../lab20-evaluating-and-visualizing-model-performance/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
