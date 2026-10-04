# Lab 10 — Generating Training Loops, Optimizers and Metrics from Prompts

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 02:** Vibe Coding Neural Networks  
> **Learning outcome:** Refactor training into one reusable fit function and compare optimizers and learning rates with it.

## Goal

You have now written the same training loop twice. Refactor it once, properly, and never write it again. You prompt for engine.py containing a single fit() that takes a model, loaders, a loss, an optimizer and a metric function, returns a history dictionary, and works unchanged for both the regression and the classification task. With that in place, experimentation becomes cheap: you run a small sweep comparing SGD against Adam across three learning rates and produce a results table that shows learning rate matters more than the choice of optimizer.

## Why this lab matters

Copy-pasted training loops are where bugs breed — the fifth copy is the one missing zero_grad(). One reviewed, tested loop used everywhere is both better engineering and the thing that makes the sweep in this lab possible at all.

## What you'll build

**A reusable engine.py fit/evaluate pair driving both models, plus a sweep table comparing optimizers and learning rates**

**Tools:** PyTorch, torch.optim, DataLoader, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `engine.py`
- `sweep.py`
- `reports/sweep.csv`

## Before you start

- Lab 9 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the reusable engine. The hard requirement is that it must work for BOTH tasks without modification.

Point 2 is a real subtlety most generated loops get wrong: averaging the per-batch means is only correct when every batch is the same size, and the last batch usually is not. Weighting by batch size is the correct reduction. Point 6 is early stopping's useful half — keeping the best weights rather than the last.

**PROMPT** — paste this into your AI coding assistant:

```text
Create engine.py with two functions that work unchanged for both a regression and a classification task.
fit(model, train_loader, val_loader, loss_fn, optimizer, epochs, metric_fn=None, device='cpu') should:
1. Loop over epochs; for each batch call optimizer.zero_grad() BEFORE the forward pass, compute the loss, call backward, then step
2. Accumulate the training loss weighted by batch size, not a plain mean of batch means
3. After each epoch run a validation pass under model.eval() and torch.no_grad(), then return the model to train mode
4. Apply metric_fn(preds, targets) on the validation set if one is given
5. Return a history dict with keys train_loss, val_loss and val_metric, each a list of length epochs
6. Track and restore the state_dict from the epoch with the best validation loss, and report which epoch that was
evaluate(model, loader, loss_fn, metric_fn, device) should run one pass under eval/no_grad and return the loss and metric.
CONSTRAINTS: no printing inside fit except an optional every-N-epochs line controlled by a verbose argument; no task-specific logic - the caller supplies loss_fn and metric_fn.
EXPECTED OUTPUT: engine.py importable by both train_wear.py and train_qc.py.
```

### 2. Review the generated fit() against your Lab 6 checklist before you trust it — this function is about to run every experiment for the rest of the course.

Run your checklist: is zero_grad() before backward()? Is the validation pass wrapped in eval() and no_grad()? Does it return to train() afterwards — a loop that forgets this leaves dropout off for the rest of training. Is anything task-specific hiding in there, like an argmax that only makes sense for classification?

### 3. Rewire both existing training scripts to use it, and confirm the results still match what you got in Labs 8 and 9.

A shared engine is only safe if it is genuinely general. If fit() contains an argmax or a .float() cast that only suits one task, it will silently corrupt the other. Making both scripts use it is the test.

**PROMPT** — paste this into your AI coding assistant:

```text
Refactor train_wear.py and train_qc.py to use fit and evaluate from engine.py instead of their own inline training loops. Keep the same seeds, architectures, hyperparameters and reporting so the results are directly comparable. train_wear.py passes nn.MSELoss and an MAE metric function; train_qc.py passes nn.CrossEntropyLoss and an accuracy metric function. Delete the now-duplicated loop code from both files.
```

### 4. Re-run both and check the final numbers are essentially unchanged from the previous labs. A refactor that changes results is a refactor that introduced a bug.

'Essentially unchanged' means within normal run-to-run variation, not bit-identical — the batching order may differ slightly. If the numbers move a lot, diff the old loop against fit() and find what changed. This is a genuine regression test, and it is why you kept the seeds fixed.

**COMMAND** — run this in your terminal:

```bash
python train_wear.py
python train_qc.py
```

### 5. Now use the engine for what it was built for. Prompt for a sweep across optimizers and learning rates.

Re-instantiating the model each run is the clause that makes the comparison valid. Reusing a trained model across configurations means each run starts from the previous one's weights and the table is nonsense. Assistants get this wrong regularly — check it explicitly in the generated code.

**PROMPT** — paste this into your AI coding assistant:

```text
Create sweep.py using fit from engine.py.
TASK: compare optimizers and learning rates on the QC classification task.
OPERATIONS: for each combination of optimizer in [SGD with momentum 0.9, Adam] and learning rate in [1e-2, 1e-3, 1e-4], train a FRESH QCNet for 60 epochs with manual seed 42 reset before each run, and record the best validation loss, the validation accuracy at that epoch, the epoch it occurred, and the wall-clock seconds.
CONSTRAINTS: identical data, batch size and seed across all 6 runs - only the optimizer and learning rate change. Re-instantiate the model each run so no weights carry over.
EXPECTED OUTPUT: a printed table sorted by best validation loss, saved to reports/sweep.csv, plus a single figure overlaying all 6 validation loss curves with a legend.
```

### 6. Run the sweep and read the table. Answer: does the optimizer or the learning rate account for more of the spread in results?

The usual finding: SGD at 1e-4 barely moves, Adam at 1e-2 is unstable, and the middle settings work. The spread across learning rates within one optimizer is typically much larger than the spread across optimizers at a sensible learning rate. Learning rate is the hyperparameter worth your attention.

**COMMAND** — run this in your terminal:

```bash
python sweep.py
```

### 7. Record the winning configuration and your answer in prompts.md, and adopt that configuration for the rest of the course.

You will reuse fit() in Labs 13 to 19 without modification. That is the payoff: from here on, a new model is a new nn.Module plus a metric function, not another hand-written loop with another chance to forget zero_grad().

## Verification — Test it

engine.py's fit runs both train_wear.py and train_qc.py unchanged and reproduces the Labs 8 and 9 results. sweep.py produces reports/sweep.csv with 6 rows, a sorted printed table, and an overlay figure of 6 validation curves. You can state which factor - optimizer or learning rate - drove more of the difference.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| fit works for regression but crashes for classification | Task-specific logic leaked into the engine, usually a target reshape. Move it into the caller or the metric function. |
| All 6 sweep runs give identical results | The model is not being re-instantiated, or the seed reset is missing. Print the initial loss of each run — identical first losses with different optimizers means the model carried over. |
| The best epoch is always the last one | Validation loss is still improving, so train longer. If it is improving on every configuration, 60 epochs is too few for this comparison. |
| Adam at 1e-2 produces nan | That is a legitimate result — record it as diverged rather than dropping the row. It is evidence for the learning-rate conclusion. |
| The refactored scripts give noticeably different results | Check shuffle and seed placement. The DataLoader must be constructed the same way, and the seed set before model instantiation, not after. |

## Going further (optional)

- Add AdamW and a cosine learning-rate schedule to the sweep and see whether either beats the winner.
- Extend fit with gradient clipping and confirm it rescues the diverging Adam-at-1e-2 run.
- Add a patience argument that stops training early when validation loss has not improved for N epochs, and check it picks the same best epoch.

---

[← Lab 9: Vibe Coding a Classification Model with Softmax and Cross Entropy](../lab09-vibe-coding-a-classification-model-with-softmax-and-cross-entropy/README.md) · [All labs](../README.md) · [Lab 11: Saving, Loading and Iterating on Models →](../lab11-saving-loading-and-iterating-on-models/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
