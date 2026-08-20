# Lab 8 — Vibe Coding a Regression Model in PyTorch

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 02:** Vibe Coding Neural Networks  
> **Learning outcome:** Build, train and honestly evaluate a regression network that predicts a continuous value.

## Goal

Build ForgeSight's first real model: a network predicting tool wear in microns from six sensor readings. You prompt for an nn.Module with a deliberate architecture, an MSELoss, and a training loop that reports train and validation loss each epoch. Crucially you establish a BASELINE first — predicting the training mean for every row — because a regression score means nothing on its own. You then check the prediction and target shapes explicitly, which is where the (N,) versus (N,1) trap from Lab 4 comes back to bite anyone who skipped it.

## Why this lab matters

This is the first lab where a silent bug costs you a real result rather than a printed shape. Getting the baseline, the shapes and the loss right here establishes the pattern that every remaining model in the course reuses.

## What you'll build

**A trained tool-wear regression network reporting MAE and RMSE that beat the mean baseline**

**Tools:** PyTorch, nn.Module, MSELoss, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `models.py`
- `train_wear.py`
- `reports/wear_curve.png`

## Before you start

- Lab 7 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Compute the baseline FIRST, before any model exists. You cannot judge a regression score without it.

A mean-predictor is the honest floor for regression. Its RMSE is essentially the target's standard deviation, because that is what standard deviation measures. If your network cannot beat this, it has learned nothing, no matter how impressive the loss curve looks.

**PROMPT** — paste this into your AI coding assistant:

```text
Create baseline_wear.py. Using load_forgesight from load_data.py, predict the MEAN of y_wear_train for every row in the validation set, and print the resulting MAE and RMSE in microns. Print the standard deviation of y_wear_train alongside them. Add a comment explaining why RMSE for a mean-predictor is close to the target's standard deviation.
```

### 2. Run it and write the baseline MAE down. Every number you produce for the rest of this lab is judged against it.

Expect a baseline MAE in the region of tens of microns. Write the exact number in review_notes.md. Learners who skip this step routinely celebrate an MAE that is worse than predicting the average.

**COMMAND** — run this in your terminal:

```bash
python baseline_wear.py
```

### 3. Prompt for the model and the training script. Note that the prompt names the shapes, the loss and the exact reporting format.

Two clauses are doing heavy lifting. 'NO activation on the output' — a ReLU there would clamp every prediction to be non-negative, which sounds harmless for wear but silently caps the model; a sigmoid there would squash all predictions into (0,1) and make the task impossible. 'Print both shapes once before the first loss call' — this is the Lab 4 broadcast trap, made visible.

**PROMPT** — paste this into your AI coding assistant:

```text
Create models.py and train_wear.py for a PyTorch regression task.
SHAPES: X_train is float32 (2800, 6), y_wear_train is float32 (2800,). Features are already standardised from load_data.py.
TASK: predict tool_wear_um, a continuous value in microns.
OPERATIONS: in models.py define class WearNet(nn.Module) with Linear(6,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1) and NO activation on the output. In train_wear.py use nn.MSELoss and torch.optim.Adam at lr=1e-3, batch size 64 via TensorDataset and DataLoader, and train for 100 epochs.
CONSTRAINTS: reshape the target to (N,1) so it matches the prediction shape exactly - print both shapes once before the first loss call to prove they match. Set manual seed 42. Evaluate under model.eval() and torch.no_grad(). Do not apply any activation to the output layer.
EXPECTED OUTPUT: per-epoch train and validation loss every 10 epochs; final validation MAE and RMSE in microns; a saved plot of both loss curves at reports/wear_curve.png.
```

### 4. Read the generated code and check three things before running: the output layer has no activation, the target is reshaped, and eval() plus no_grad() wrap the validation pass.

If the printed shapes are torch.Size([64, 1]) and torch.Size([64, 1]), you are safe. If you see torch.Size([64]) for the target, stop and fix it — the loss will broadcast to (64, 64) and every number after that is meaningless, without any error being raised.

### 5. Run the training. Watch that both losses fall and that the printed shapes match.

Training 100 epochs on 2800 rows takes well under a minute on CPU. If the loss is not falling at all, check the learning rate first; if it explodes to nan, the target was probably not standardised or the learning rate is far too high.

**COMMAND** — run this in your terminal:

```bash
python train_wear.py
```

### 6. Compare your final validation MAE against the baseline MAE you wrote down. State the improvement as a percentage.

A good result here is a validation MAE meaningfully below the baseline. If the improvement is under a few percent, the features may genuinely not carry much signal about wear — which is a legitimate finding to report, not a failure to hide. Say so out loud; that honesty is the professional habit being trained.

### 7. Open reports/wear_curve.png and read the two curves. Note whether validation loss is still falling at epoch 100 or has flattened.

If validation loss is still falling, the model is underfitted and more epochs would help. If it has flattened while training loss keeps dropping, you are watching the beginning of overfitting — the exact pattern you will diagnose properly in Lab 14.

## Verification — Test it

baseline_wear.py prints a baseline MAE and RMSE. train_wear.py prints matching prediction and target shapes of torch.Size([N, 1]), trains with both losses falling, and reports a final validation MAE clearly BELOW the baseline MAE. reports/wear_curve.png shows both curves labelled.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The loss is enormous and never falls | The target is unscaled and in the hundreds. Either standardise y as well (and invert it for reporting) or lower the learning rate — check what the assistant did to y. |
| Loss becomes nan after a few epochs | Learning rate too high, or a nan in the input. Print X_train.isnan().any() and drop the lr to 1e-4. |
| Validation MAE is worse than the baseline | Usually underfitting or a too-small learning rate. Train longer, or check the output layer really has no activation squashing the range. |
| RuntimeError about the size of tensor a and tensor b | Prediction and target shapes disagree. Reshape the target with .view(-1, 1) — this is the Lab 4 trap. |
| The model predicts almost the same value for every row | It has collapsed to the mean, which means it is not learning. Check the features really are standardised and the learning rate is not far too small. |

## Going further (optional)

- Ask the assistant to add a parity plot (predicted versus actual) and read where the model is worst.
- Swap Adam for SGD with momentum at the same learning rate and note how differently the curve behaves.
- Widen the first layer to 128 units and see whether validation MAE improves or the gap to training loss widens.

---

[← Lab 7: Neural Network Architectures, Activation and Loss Functions](../lab07-neural-network-architectures-activation-and-loss-functions/README.md) · [All labs](../README.md) · [Lab 9: Vibe Coding a Classification Model with Softmax and Cross Entropy →](../lab09-vibe-coding-a-classification-model-with-softmax-and-cross-entropy/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
