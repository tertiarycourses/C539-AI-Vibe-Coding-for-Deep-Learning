# Lab 9 — Vibe Coding a Classification Model with Softmax and Cross Entropy

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 02:** Vibe Coding Neural Networks  
> **Learning outcome:** Build a multi-class classifier that outputs raw logits and pair it correctly with cross entropy loss.

## Goal

Predict the ForgeSight quality-control class — ok, scratch, dent or burr — from the same six sensors. This lab is deliberately a trap. You first ask the assistant for the model in a way that invites the classic mistake, and there is a strong chance it hands you a network with nn.Softmax on the output next to nn.CrossEntropyLoss. You catch it, prove numerically that it is wrong, and fix it. You then evaluate properly against a majority-class baseline with a confusion matrix, because on imbalanced classes accuracy alone will flatter a model that never predicts the rare defect at all.

## Why this lab matters

The softmax-before-cross-entropy bug is the most common defect in AI-generated PyTorch, and it never raises an error. Catching it once, deliberately, with the numbers in front of you, is worth more than being told about it ten times.

## What you'll build

**A QC classifier outputting raw logits, evaluated with accuracy, per-class recall and a confusion matrix against the majority baseline**

**Tools:** PyTorch, nn.CrossEntropyLoss, scikit-learn metrics, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `models.py`
- `train_qc.py`
- `reports/qc_confusion.png`

## Before you start

- Lab 8 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Establish the majority-class baseline first, and look at how imbalanced the classes actually are.

With four classes and a realistic factory distribution, most readings are 'ok'. A majority-class predictor therefore scores far above 25% while being completely useless — it never flags a single defect. This is why accuracy alone cannot be the reported metric and why per-class recall matters.

**PROMPT** — paste this into your AI coding assistant:

```text
Create baseline_qc.py. Using load_forgesight, print the count and percentage of each of the 4 qc_class values in the training split. Then predict the single most frequent class for every validation row and print the resulting accuracy. Print the per-class recall for that baseline as well, and add a comment on what recall is for the classes it never predicts.
```

### 2. Run it. Note the majority-class accuracy — this is the number your model must beat, and it is higher than most people expect.

Write the baseline accuracy down. If your trained model ends up close to it, the model is probably just predicting 'ok' for everything — check the confusion matrix rather than celebrating the accuracy.

**COMMAND** — run this in your terminal:

```bash
python baseline_qc.py
```

### 3. Now prompt for the classifier, deliberately WITHOUT specifying the output activation. You are setting a trap for the assistant.

The prompt names the layers but deliberately says nothing about the output activation. This is exactly how these prompts get written in real life, and it is why the bug is so common. Assistants frequently add a softmax because it 'looks like' what a classifier should do.

**PROMPT** — paste this into your AI coding assistant:

```text
Add class QCNet(nn.Module) to models.py and create train_qc.py.
SHAPES: X_train is float32 (2800, 6); y_qc_train is int64 (2800,) with 4 classes.
TASK: classify each machine reading into one of 4 quality-control classes.
OPERATIONS: QCNet should be Linear(6,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,4). Train with nn.CrossEntropyLoss and Adam at lr=1e-3, batch size 64, 100 epochs.
CONSTRAINTS: manual seed 42, evaluate under model.eval() and torch.no_grad().
EXPECTED OUTPUT: per-epoch train and validation loss, final validation accuracy, per-class precision and recall, and a confusion matrix saved to reports/qc_confusion.png.
```

### 4. STOP before running. Inspect the generated QCNet output layer. Is there an nn.Softmax, nn.LogSoftmax or F.softmax anywhere after the final Linear?

Look at the last layer of the Sequential or the end of forward(). Any of nn.Softmax(dim=1), F.softmax(x, dim=1) or nn.LogSoftmax before returning is the bug. If your assistant returned raw logits, it got it right — still run the proof script, because you need to see the numbers.

### 5. If a softmax is present, prove it is wrong before removing it rather than just deleting it.

Expect the double-softmax loss to be noticeably different and its gradients an order of magnitude smaller. Smaller gradients mean smaller updates mean slower learning — the model still trains, the loss still falls, and it simply ends up worse. Nothing warns you. This is the whole lesson.

**PROMPT** — paste this into your AI coding assistant:

```text
Write a short script proof_softmax.py that takes one batch of logits of shape (8, 4) with manual seed 42 and computes nn.CrossEntropyLoss twice: once on the raw logits, and once on torch.softmax(logits, dim=1). Print both loss values and both gradients with respect to the logits, and print the ratio of the gradient magnitudes. Add a comment explaining that CrossEntropyLoss applies log_softmax internally, so the second version applies softmax twice and produces much smaller gradients.
```

### 6. Fix the model with a targeted follow-up, then train it.

Note the last clause: softmax is not banned, it is relocated. You still want probabilities when reporting a confidence to a user — you just compute them at reporting time, outside the loss path.

**PROMPT** — paste this into your AI coding assistant:

```text
In models.py, QCNet applies softmax to its output, but nn.CrossEntropyLoss already applies log_softmax internally, so the network is applying it twice and its gradients are being flattened. Remove the softmax layer so QCNet returns raw logits from the final Linear layer. Keep everything else unchanged. Where a probability is needed for reporting, apply torch.softmax at that point instead.
```

### 7. Train the corrected model and compare accuracy and per-class recall against the majority baseline.

A corrected model should beat the majority baseline on accuracy AND find a meaningful share of at least the common defect classes. If accuracy is high but recall on the rare classes is near zero, say so — that is an honest and important result about class imbalance.

**COMMAND** — run this in your terminal:

```bash
python train_qc.py
```

### 8. Open reports/qc_confusion.png and find which class the model confuses most. Note whether it predicts the rarest class at all.

The confusion matrix is the real report. Off-diagonal mass tells you which defects look alike to the model. A completely empty row means the model never predicts that class at all, which no accuracy number would have told you.

## Verification — Test it

baseline_qc.py prints class counts and the majority-class accuracy. proof_softmax.py prints two different loss values and shows the double-softmax gradients are much smaller. train_qc.py's QCNet returns raw logits with no softmax layer, and reports a validation accuracy above the majority baseline with a confusion matrix saved to reports/qc_confusion.png.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| RuntimeError: expected scalar type Long but found Float | y_qc is float32. CrossEntropyLoss needs int64 class indices — cast with .long(). This is the Lab 3 dtype note. |
| Accuracy exactly equals the majority baseline | The model is predicting one class for everything. Check the softmax was actually removed, and try a slightly higher learning rate or more epochs. |
| The target has shape (N, 1) and CrossEntropyLoss complains | Unlike MSELoss, CrossEntropyLoss wants a 1-D target of class indices. Use .squeeze() or .view(-1) — the opposite of what Lab 8 needed. |
| Loss starts near 1.386 and never moves | 1.386 is ln(4), the loss of a uniform guess over 4 classes. The model is not learning — check for the double softmax, then the learning rate. |
| The confusion matrix plot is unreadable | Ask for class names on both axes and counts annotated in each cell. A confusion matrix without labels is not a report. |

## Going further (optional)

- Add class weights to CrossEntropyLoss to penalise missing the rare defects and see how the confusion matrix changes.
- Print the model's softmax probabilities for ten misclassified rows and see whether it was confidently wrong or nearly right.
- Compare macro-averaged F1 against accuracy and decide which you would report to the plant manager.

---

[← Lab 8: Vibe Coding a Regression Model in PyTorch](../lab08-vibe-coding-a-regression-model-in-pytorch/README.md) · [All labs](../README.md) · [Lab 10: Generating Training Loops, Optimizers and Metrics from Prompts →](../lab10-generating-training-loops-optimizers-and-metrics-from-prompts/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
