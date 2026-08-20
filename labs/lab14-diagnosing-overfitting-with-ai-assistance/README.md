# Lab 14 — Diagnosing Overfitting with AI Assistance

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 03:** Vibe Coding Convolutional Neural Networks  
> **Learning outcome:** Recognise overfitting from training and validation curves and locate the epoch where generalisation stops improving.

## Goal

Your Lab 13 model almost certainly overfits — now measure it rather than guess. You train the same CNN deliberately hard and plot training against validation loss and accuracy on shared axes, then identify the exact epoch where validation stops improving while training keeps falling. To make the pattern unmistakable you also train on a deliberately tiny subset until it reaches near-perfect training accuracy and useless validation accuracy. You finish by asking the assistant to diagnose the curves from a description alone, and you check its reasoning against what you can see.

## Why this lab matters

Overfitting is invisible if you only watch training loss, which is the number that prints most often and always looks encouraging. Learning to read the GAP between two curves, and to spot the epoch where they diverge, is what tells you when to stop and what to fix.

## What you'll build

**An overfitting diagnosis with annotated curves, a memorised-subset demonstration, and a written diagnosis you verified yourself**

**Tools:** PyTorch, matplotlib, engine.py, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `diagnose_overfit.py`
- `reports/overfit_curves.png`
- `reports/memorise.png`
- `review_notes.md`

## Before you start

- Lab 13 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Train long enough for the problem to appear. Prompt for a diagnostic run with no regularization at all.

Sixty epochs with no regularization is not how you would train a production model — it is how you make a phenomenon visible. Marking the minimum-validation-loss epoch on the plot is what turns a vague 'it overfits' into a specific, actionable number.

**PROMPT** — paste this into your AI coding assistant:

```text
Create diagnose_overfit.py that trains DefectCNN for 60 epochs using fit from engine.py with NO augmentation, NO dropout and NO weight decay - deliberately unregularized.
OPERATIONS: record train loss, validation loss, train accuracy and validation accuracy every epoch. Save a 2-panel figure to reports/overfit_curves.png - left panel both losses, right panel both accuracies, epochs on the x axis, with a vertical dashed line marking the epoch of MINIMUM validation loss and that epoch number in the legend. Print: the best validation loss and its epoch; the final training loss; the final validation loss; the final gap between training and validation accuracy in percentage points.
CONSTRAINTS: manual seed 42, batch size 32, Adam lr=1e-3. Do not stop early - run all 60 epochs so the divergence is visible.
```

### 2. Run it. This takes several minutes on CPU — while it runs, write down what you expect the two curves to do.

Expect validation loss to bottom out somewhere in the first third of training and then climb, while training loss keeps falling towards zero. The rising validation loss is the model becoming more confident about the training images specifically, which is the definition of overfitting.

**COMMAND** — run this in your terminal:

```bash
python diagnose_overfit.py
```

### 3. Open reports/overfit_curves.png and answer three questions before reading any explanation: at which epoch does validation loss bottom out, what does training loss do after that, and how wide is the final accuracy gap?

The single most important reading: validation loss can rise while validation ACCURACY is still roughly flat. Loss is sensitive to confidence, accuracy only to the argmax. Loss turns first, which is why it is the better early-stopping signal.

### 4. Now make the effect undeniable. Prompt for a memorisation demonstration on a tiny subset.

Thirty images cannot possibly represent the variation in the full set, so the network simply memorises them. This is the same mechanism as the main run, just fast and obvious. It is also a genuinely useful debugging trick: a model that CANNOT reach high accuracy on 30 images has a bug, not a data problem.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a function memorise_demo() to diagnose_overfit.py.
TASK: show overfitting in its purest form by training on far too little data.
OPERATIONS: take only 30 training images (10 per class) using torch.utils.data.Subset, keep the FULL validation set, and train a fresh DefectCNN for 100 epochs with the same settings. Plot training and validation accuracy to reports/memorise.png and print both final accuracies.
CONSTRAINTS: same seed and architecture as the main run - only the amount of training data changes.
EXPECTED OUTPUT: training accuracy approaching 100% while validation accuracy stays near the majority-class baseline.
```

### 5. Run it and compare the two numbers. This is memorisation with nothing learned.

Expect training accuracy near 100% and validation accuracy near the baseline. Nothing generalised. Keep reports/memorise.png — it is the clearest single picture of overfitting you will produce today.

**COMMAND** — run this in your terminal:

```bash
python diagnose_overfit.py
```

### 6. Test the assistant's diagnostic reasoning — and then test the assistant.

The numbers in this prompt are illustrative; substitute your own if you prefer. What you are testing is whether the assistant distinguishes fixes that address the CAUSE — more data, augmentation, a smaller model — from those that only limit the damage, like early stopping. Early stopping does not make the model generalise better; it stops you shipping the worse version.

**PROMPT** — paste this into your AI coding assistant:

```text
I trained a CNN image classifier for 60 epochs with no regularization. Training loss fell steadily from 1.05 to 0.04 and training accuracy reached 99%. Validation loss fell until epoch 14, reaching 0.52, then rose steadily to 0.95 by epoch 60, while validation accuracy peaked at 78% around epoch 14 and drifted down to 71%.
Diagnose what is happening, name the epoch I should have stopped at, and rank the following fixes by how much improvement you would expect for THIS symptom, with a one-line reason each: more training data, data augmentation, dropout, weight decay, early stopping, a smaller model, a lower learning rate.
Be explicit about which of these address the cause and which only limit the damage.
```

### 7. Compare the assistant's ranking against your own curves. Does its recommended stopping epoch match the dashed line in your figure?

Assistants are generally strong at this diagnosis, which is worth noticing: they are good at pattern-matching a described symptom and weaker at spotting the same problem inside code they just wrote. Use them accordingly — describe symptoms to them, do not ask them to audit themselves.

### 8. Write the diagnosis into review_notes.md in your own words, with the specific numbers from YOUR run.

Write your own numbers: best validation loss and its epoch, the final train/validation accuracy gap, and the two fixes you intend to apply in Lab 15. You will compare against these exact figures next lab.

## Verification — Test it

reports/overfit_curves.png shows training loss falling while validation loss turns upward, with a dashed line at the minimum-validation-loss epoch and that epoch printed. reports/memorise.png shows near-100% training accuracy against near-baseline validation accuracy on 30 images. review_notes.md records your best epoch, your accuracy gap and the fixes you will apply next.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Validation loss never rises within 60 epochs | The model may be too small or the task too easy. Note it honestly, then use memorise_demo to show the effect instead — it is guaranteed to appear there. |
| The curves are too noisy to read | Batch-to-batch noise. Plot a rolling mean over 3 epochs alongside the raw curve, or raise the batch size. |
| The memorisation demo does not reach high training accuracy | Train longer or raise the learning rate. If 30 images still cannot be memorised in 100 epochs, there is a bug in the pipeline — a genuinely useful signal. |
| Training takes too long on CPU | Reduce to 40 epochs and note the change, or shrink the images to 32x32 in the transform. The pattern still appears. |
| Subset produces a class-imbalanced sample | Select indices per class explicitly rather than taking the first 30 rows, which would be all one class in a sorted ImageFolder. |

## Going further (optional)

- Add a third panel plotting the train-validation gap directly, and see whether its shape is easier to read.
- Re-run with half the training data and confirm the divergence epoch arrives earlier.
- Track the mean absolute weight of the first Linear layer per epoch and see whether it grows as overfitting sets in.

---

[← Lab 13: Vibe Coding a CNN Image Classifier](../lab13-vibe-coding-a-cnn-image-classifier/README.md) · [All labs](../README.md) · [Lab 15: Data Augmentation and Regularization via Prompts →](../lab15-data-augmentation-and-regularization-via-prompts/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
