# Lab 15 — Data Augmentation and Regularization via Prompts

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 03:** Vibe Coding Convolutional Neural Networks  
> **Learning outcome:** Apply augmentation, dropout, weight decay and early stopping, and measure how much each narrows the overfitting gap.

## Goal

Fix what you measured. You add the four standard remedies one at a time — augmentation on the training set only, dropout in the classifier head, weight decay in the optimizer, and early stopping via the best-epoch tracking already in engine.fit() — and after each change you record the validation accuracy and the train/validation gap. The critical trap is deliberate: augmentation must never be applied to validation, and an assistant asked for 'augmented loaders' will very often apply it to both, which silently makes your validation score noisy and pessimistic.

## Why this lab matters

Applying all four fixes at once tells you nothing about which one mattered. Adding them one at a time, with the gap measured after each, is how you learn what actually helps for a given problem — and it is how you would justify the choices to a colleague.

## What you'll build

**A regularized training pipeline plus an ablation table showing the measured contribution of each remedy**

**Tools:** PyTorch, torchvision.transforms, engine.py, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `datasets.py`
- `models.py`
- `train_cnn_v2.py`
- `reports/ablation.csv`

## Before you start

- Lab 14 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Add augmented loaders, with an explicit instruction about which split gets augmented.

This is the trap. An assistant asked for 'augmented data loaders' will frequently build one transform and use it for both splits. Augmented validation is not just wrong, it is invisibly wrong: your score changes every run and is systematically pessimistic, so you will conclude your fixes did not work.

**PROMPT** — paste this into your AI coding assistant:

```text
Update datasets.py with a new function get_augmented_loaders(batch_size=32, data_dir='data/defects').
TASK: return training and validation loaders where ONLY the training set is augmented.
OPERATIONS: the TRAINING transform is RandomHorizontalFlip(0.5), RandomRotation(10), RandomResizedCrop(64, scale=(0.8, 1.0)), then Grayscale(1), ToTensor(), Normalize([0.5],[0.5]). The VALIDATION transform is ONLY Grayscale(1), ToTensor(), Normalize([0.5],[0.5]) with NO random operations whatsoever.
CONSTRAINTS: this is the critical requirement - validation must be deterministic, because a randomly augmented validation set gives a different score every run and cannot be compared across experiments. Keep the original get_defect_loaders unchanged so the two can be compared.
EXPECTED OUTPUT: a __main__ block that prints the two transform pipelines side by side so the difference is visible, and saves a grid of 8 augmented versions of the SAME training image to reports/augmented_samples.png.
```

### 2. Verify the trap did not catch you. Print both pipelines and confirm no random transform appears in the validation list.

Read the two printed pipelines carefully. The validation list must contain exactly three entries — Grayscale, ToTensor, Normalize. If RandomHorizontalFlip or RandomResizedCrop appears there, fix it before running anything else.

**COMMAND** — run this in your terminal:

```bash
python datasets.py
```

### 3. Open reports/augmented_samples.png. Check the augmentations are plausible — a rotation so extreme that a scratch becomes unrecognisable teaches the model nothing.

Augmentation must preserve the label. A horizontal flip of a scratch is still a scratch, so that is safe. If you were classifying digits, a flip would change a 2 into something that is not a 2 — the transformation has to make sense for YOUR data, which is a judgement the assistant cannot make for you.

### 4. Add dropout to the model, in the right place.

Dropout belongs in the dense head because convolutional layers already share weights heavily and are naturally regularized; heavy Dropout2d early in a CNN tends to hurt. And the eval() note matters: dropout left active at evaluation makes your validation score randomly worse, which is the mirror image of the augmented-validation bug.

**PROMPT** — paste this into your AI coding assistant:

```text
Add class DefectCNNv2(nn.Module) to models.py, identical to DefectCNN but with nn.Dropout(p=0.3) inserted after the ReLU that follows Linear(4096, 128), and nn.Dropout2d(p=0.1) after the final MaxPool.
Add a comment explaining why dropout goes in the classifier head rather than between early convolutional layers, and a second comment noting that model.eval() disables it automatically - which is why evaluation must never run in train mode.
Keep the output as raw logits with no softmax.
```

### 5. Run the ablation. This is the actual experiment — five configurations, one change at a time.

One change per run is what makes this an ablation rather than a guess. If two things change between rows you cannot attribute the difference, and the table becomes decoration.

**PROMPT** — paste this into your AI coding assistant:

```text
Create train_cnn_v2.py running an ablation over regularization, using fit from engine.py.
RUNS - each 40 epochs, fresh model, manual seed 42 reset before each:
1. baseline: DefectCNN, plain loaders, Adam lr=1e-3, weight_decay=0
2. plus augmentation: DefectCNN, augmented loaders
3. plus dropout: DefectCNNv2, augmented loaders
4. plus weight decay: DefectCNNv2, augmented loaders, weight_decay=1e-4
5. plus early stopping: as run 4 but reporting the metrics from the BEST validation epoch rather than the last
For each run record: best validation loss and its epoch, final validation accuracy, best validation accuracy, final training accuracy, and the train-validation accuracy gap in percentage points.
EXPECTED OUTPUT: a printed table in run order, saved to reports/ablation.csv, plus a figure overlaying the five validation loss curves with a legend.
```

### 6. Run the ablation and read the gap column down the table. Which single change reduced the gap most?

Typical finding: augmentation contributes the largest single reduction in the gap, dropout adds a modest further improvement, weight decay a small one, and early stopping does not change the model at all — it changes which epoch's weights you keep. That last distinction is worth stating out loud.

**COMMAND** — run this in your terminal:

```bash
python train_cnn_v2.py
```

### 7. Save the best configuration as the new ForgeSight defect model.

The regularization field in the metrics is exactly the kind of provenance that makes a checkpoint trustworthy six months later. A model card that records what was tried, not just what was chosen, is far more useful to the next person.

**PROMPT** — paste this into your AI coding assistant:

```text
Update train_cnn_v2.py to retrain the winning configuration and save it with save_checkpoint to models/defect_cnn_v2.pt, including in the metrics dict the validation accuracy, the best epoch, and a regularization field listing which techniques were used. Update models/defect_model_card.md with the new score, the baseline for comparison, and one sentence on what changed since v1.
```

### 8. Compare v2's accuracy and gap against the Lab 14 numbers you wrote in review_notes.md.

A successful outcome is a smaller train/validation gap AND a validation accuracy at least as good as before. If the gap shrank because training accuracy collapsed, you over-regularized — reduce the dropout or soften the augmentation.

## Verification — Test it

datasets.py prints a validation pipeline containing no random transforms, and reports/augmented_samples.png shows eight plausible variants of one training image. reports/ablation.csv holds five rows with a train-validation gap column that narrows down the table. models/defect_cnn_v2.pt is saved with a regularization field, and v2's gap is smaller than the Lab 14 gap.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Validation accuracy changes every run | Augmentation is being applied to validation. Print the validation transform and remove every random operation. |
| Augmentation makes results clearly worse | The transforms are too aggressive for 64x64 images. Reduce RandomRotation to 5 degrees and raise the RandomResizedCrop lower scale bound to 0.9. |
| Dropout produces a training accuracy lower than validation accuracy | That is normal and expected — dropout is active during training and off during evaluation. It is not a bug. |
| Training is now much slower | Augmentation happens on the CPU per image. Reduce the image count or the batch size; on Windows keep num_workers=0. |
| Weight decay makes no measurable difference | 1e-4 is mild. Try 1e-3 and note the effect, but expect weight decay to matter less than augmentation on image data. |

## Going further (optional)

- Add ColorJitter on brightness and contrast and see whether it helps on greyscale inspection images or not.
- Implement mixup and compare it against the standard augmentation set.
- Plot the ablation as a bar chart of the gap per configuration and check the visual story matches the table.

---

[← Lab 14: Diagnosing Overfitting with AI Assistance](../lab14-diagnosing-overfitting-with-ai-assistance/README.md) · [All labs](../README.md) · [Lab 16: Transfer Learning with Pre-Trained Models →](../lab16-transfer-learning-with-pre-trained-models/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
