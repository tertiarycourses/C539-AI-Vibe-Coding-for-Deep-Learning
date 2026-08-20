# Lab 11 — Saving, Loading and Iterating on Models

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 02:** Vibe Coding Neural Networks  
> **Learning outcome:** Save a model's state_dict with its configuration, reload it into a fresh instance and prove the predictions are identical.

## Goal

A model that only exists in memory is not a deliverable. You prompt for a checkpoint helper that saves the state_dict together with everything needed to rebuild it — architecture arguments, the standardisation statistics, the class names, the metrics and the library versions — then reload it into a freshly constructed model in a NEW process and assert the predictions match to machine precision. That assertion is the difference between believing your model saved and knowing it did. You finish by writing the first ForgeSight model card.

## Why this lab matters

Saving only the weights is the most common way to produce an unusable artifact. Without the architecture arguments the file will not load; without the standardisation statistics the model gets raw inputs it was never trained on and returns confident nonsense. Both failures happen after the project looks finished.

## What you'll build

**A checkpoint helper saving weights plus config and statistics, a proven identical-prediction reload, and a model card**

**Tools:** PyTorch, torch.save / torch.load, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `checkpoint.py`
- `models/qc_net_v1.pt`
- `reload_check.py`
- `models/qc_model_card.md`

## Before you start

- Lab 10 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the checkpoint helper. The key insight is that the file must contain everything needed to rebuild, not just the weights.

Pickling the whole model object embeds your file paths and class definitions, so it breaks when the code moves or the class is renamed. A state_dict is just tensors, and the config tells you how to rebuild the shell they go into. This is why the official PyTorch recommendation is state_dict plus code.

**PROMPT** — paste this into your AI coding assistant:

```text
Create checkpoint.py with save_checkpoint and load_checkpoint functions.
save_checkpoint(path, model, config, feat_mean, feat_std, metrics, class_names=None) must save a single dict containing: the model's state_dict; the config dict of constructor arguments needed to rebuild the class; the model class NAME as a string; feat_mean and feat_std tensors; the metrics dict; the class names; the torch version; and an ISO timestamp.
load_checkpoint(path, model_class) must construct a fresh instance from the saved config, load the state_dict into it, call model.eval(), and return the model plus the full metadata dict.
CONSTRAINTS: save the state_dict, never the pickled model object - explain why in a comment. Create the models/ directory if missing. Use weights_only=False on load only where required by your torch version, and note why in a comment.
EXPECTED OUTPUT: both functions importable, with a __main__ block that round-trips a small dummy model as a self-test.
```

### 2. Wire it into the QC training script so a completed training run always leaves a loadable artifact behind.

feat_mean and feat_std are part of the model, not part of the training script. If they are not in the checkpoint, whoever loads this model six months from now has no way to preprocess input correctly, and nothing will tell them they got it wrong.

**PROMPT** — paste this into your AI coding assistant:

```text
Update train_qc.py to call save_checkpoint at the end of training. Save to models/qc_net_v1.pt with the config needed to rebuild QCNet, the feat_mean and feat_std returned by load_forgesight, the final validation accuracy and per-class recall as metrics, and class_names=['ok','scratch','dent','burr'].
```

### 3. Run training so the checkpoint is written.

Check the file actually appeared and note its size — a few hundred kilobytes for this network. If it is suspiciously large, the whole model object was pickled rather than the state_dict.

**COMMAND** — run this in your terminal:

```bash
python train_qc.py
```

### 4. Now prove the reload works, from a SEPARATE script that never sees the trained object in memory.

The separate-script constraint is the real test. Loading in the same process as training can pass by accident because the trained object is still in memory. A fresh process proves the file alone is enough.

**PROMPT** — paste this into your AI coding assistant:

```text
Create reload_check.py that proves the checkpoint round-trips exactly.
OPERATIONS: (1) load models/qc_net_v1.pt with load_checkpoint, rebuilding QCNet from the saved config; (2) take the first 16 rows of the validation set; (3) run them through the reloaded model under torch.no_grad(); (4) print the predicted class indices and the saved metrics; (5) assert the reloaded model's predictions are identical to the predictions the training script produced for the same rows, which you should also save alongside the checkpoint for this purpose; (6) print the saved timestamp, torch version and class names.
CONSTRAINTS: this script must NOT import or re-run training - it may only read the checkpoint file.
```

### 5. Run it and confirm the assertion passes. This is the moment you know the artifact is real.

Identical means exactly identical — same architecture, same weights, eval mode, no dropout randomness. If predictions differ, the usual causes are the model left in train mode, or the config rebuilding a different architecture from the one that was trained.

**COMMAND** — run this in your terminal:

```bash
python reload_check.py
```

### 6. Deliberately break it to see the failure mode you are guarding against.

This is the failure you are being inoculated against. Raw sensor values are orders of magnitude away from the standardised range the network trained on, so the predictions are garbage — but they are still valid class indices with confident probabilities. Nothing crashes. Nothing warns.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a commented-out demonstration to reload_check.py: reload the model but skip applying feat_mean and feat_std to the input rows, so the model receives raw unstandardised sensor values. Print the predictions from both paths side by side and a comment noting that no error is raised - the model simply predicts confidently and wrongly.
```

### 7. Write the model card. Do it in your own words — this is the document that travels with the model.

A model card states: what it predicts, what data trained it, how it scored (with the baseline for comparison), the known weaknesses from your confusion matrix, and explicitly where it must NOT be used — for example, on a machine type absent from the training data. Keep it to one page.

## Verification — Test it

models/qc_net_v1.pt exists and contains the state_dict, config, feat_mean, feat_std, metrics, class names, torch version and timestamp. reload_check.py runs in a fresh process, rebuilds QCNet from the checkpoint alone, and its assertion of identical predictions passes. models/qc_model_card.md states the metric, the baseline and at least one place the model must not be used.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| RuntimeError: Error(s) in loading state_dict - size mismatch | The config rebuilt a different architecture. Print the saved config and compare each layer size against QCNet's constructor. |
| UnpicklingError or a weights_only warning on torch.load | Newer PyTorch defaults to weights_only=True. Pass weights_only=False for your own trusted checkpoint, and note in a comment that this is only safe for files you produced. |
| Predictions differ slightly between runs | The model was left in train mode, so dropout is active. Confirm load_checkpoint calls model.eval() and the inference is inside no_grad(). |
| FileNotFoundError on models/qc_net_v1.pt | Training did not reach the save call, or models/ does not exist. Add os.makedirs('models', exist_ok=True) inside save_checkpoint. |
| The checkpoint file is tens of megabytes | The optimizer state or the whole model object is being saved. For this network the state_dict alone should be small. |

## Going further (optional)

- Add the git commit hash to the checkpoint metadata so a model can be traced back to the exact code that made it.
- Save a second version with a different architecture as qc_net_v2.pt and write a script that compares the two checkpoints' metrics.
- Extend save_checkpoint to also store the optimizer state so training can be resumed, and prove it by resuming for 10 more epochs.

---

[← Lab 10: Generating Training Loops, Optimizers and Metrics from Prompts](../lab10-generating-training-loops-optimizers-and-metrics-from-prompts/README.md) · [All labs](../README.md) · [Lab 12: Overview of CNNs: Convolution, Pooling and Padding →](../lab12-overview-of-cnns-convolution-pooling-and-padding/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
