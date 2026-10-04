# Lab 6 — Reviewing and Debugging AI-Generated PyTorch Code

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 01:** AI Vibe Coding for PyTorch Fundamentals  
> **Learning outcome:** Review AI-generated PyTorch against a fixed checklist and correct each defect with a targeted follow-up prompt.

## Goal

Everything so far has been about producing code. This lab is about refusing to trust it. You are given train_wear_buggy.py — a script that runs to completion, prints a decreasing loss and reports believable-looking numbers, while containing five real defects. Nothing crashes. You review it against a written checklist, predict what each defect does to the result, then fix them one at a time with targeted follow-up prompts, re-running after each fix so you can see which defect was costing what. This is the single most transferable skill in the course.

## Why this lab matters

Deep learning code fails silently more often than it crashes. An assistant will produce a script that trains, prints falling numbers and returns confident predictions while doing something subtly wrong. The reviewer's checklist you build here is what you will actually use at work, long after you have forgotten the specific syntax.

## What you'll build

**A corrected train_wear.py, a review_notes.md recording all five defects, and a reusable AI PyTorch review checklist**

**Tools:** PyTorch, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `train_wear_buggy.py`
- `train_wear.py`
- `review_notes.md`

## Before you start

- Lab 5 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Copy the buggy script from the course resources into your workspace and run it BEFORE reading it. Note the final loss and score it reports.

Write the numbers down — the reported MAE in microns and the QC accuracy. The entire lesson lands only if you have the 'before' figures in front of you when you see the 'after'. Note also that the script never prints a baseline, which is itself the sixth defect nobody asked you to look for.

**COMMAND** — run this in your terminal:

```bash
python train_wear_buggy.py
```

### 2. Now read it against the checklist without running anything. Write your suspicions into review_notes.md before you ask the assistant anything.

The checklist, which becomes your permanent one: (1) Is any scaling or fitting done before the split? (2) Is optimizer.zero_grad() present and before backward()? (3) Does the output layer apply an activation that the loss also applies? (4) Is evaluation inside model.eval() and torch.no_grad()? (5) Do prediction and target shapes match exactly? Most learners find three of the five unaided — that is a good score.

### 3. Ask the assistant to review it — but constrain the review so it reports rather than rewrites.

The 'do NOT rewrite' constraint matters. An unconstrained assistant returns a fixed file, you accept it, and you learn nothing about what was wrong. Forcing it to report line-by-line keeps you as the reviewer and makes the assistant explain itself. Ranking by distortion teaches you which bugs actually matter.

**PROMPT** — paste this into your AI coding assistant:

```text
Review the PyTorch script train_wear_buggy.py for correctness defects. Do NOT rewrite the file.
For each defect, report exactly three things: (1) the line, quoted verbatim; (2) what it does to the RESULT - be specific about whether it makes the score too high, too low, or meaningless; (3) the minimal one-line fix.
Check specifically for: standardisation or scaling fitted before the train/test split; a missing or misplaced optimizer.zero_grad(); a softmax or sigmoid applied before a loss function that already applies it; evaluation performed without model.eval() or torch.no_grad(); a target tensor whose shape does not match the prediction shape.
Rank the defects by how much they distort the reported result.
```

### 4. Compare the assistant's list against your own. Anything you found that it missed is worth more than anything it found that you missed — record both in review_notes.md.

Assistants are good at the shape and zero_grad defects and noticeably weaker at leakage, because leakage is about the ORDER of operations rather than any single wrong line. That asymmetry is exactly why the human review step survives.

### 5. Fix the defects ONE AT A TIME, re-running after each. Start with the one the assistant ranked most distorting.

One at a time, re-running each time, is the discipline. Fix all five at once and you have no idea which one was responsible for the inflated score — and in a real project that is the question you will be asked.

**PROMPT** — paste this into your AI coding assistant:

```text
In train_wear_buggy.py, the features are standardised using the mean and standard deviation of the full dataset before the train/test split, which leaks test information into training. Recompute the mean and std from the training split only and apply those same values to the test split. Change only that block and save the result as train_wear.py.
```

### 6. Continue with the remaining defects, using the same specific-symptom style. After each fix, record the new loss and score in review_notes.md.

The five defects and their effects: leakage before the split inflates the score; the missing zero_grad makes updates too large and training unstable; the extra softmax before CrossEntropyLoss flattens the gradients so the model learns slowly or not at all; evaluating without eval()/no_grad() leaves dropout on and wastes memory, giving a noisy and pessimistic score; the (N,) versus (N,1) target mismatch silently broadcasts and produces an entirely wrong loss.

### 7. Compare the original reported numbers against the corrected ones, and write one sentence in review_notes.md explaining why the buggy script's numbers looked believable while both models were actually stuck at their baselines.

The answer that matters: the numbers were believable, not impressive. A tool-wear MAE in the low tens of microns sounds like a working model until you compare it with simply predicting the average — which is all the broadcast loss actually trained it to do. The QC accuracy looks respectable for the same reason: it is roughly the share of 'ok' parts, so the model is predicting 'ok' for everything. Write that in your own words. A number is only meaningful next to its baseline, which is why every lab from here on computes the baseline FIRST.

## Verification — Test it

review_notes.md lists all five defects with the line quoted, the effect on the result and the fix. train_wear.py runs clean, and its corrected tool-wear MAE is clearly BETTER than the buggy script's — which was no better than predicting the average — while the corrected QC accuracy rises above the share of 'ok' parts the buggy version was merely echoing. You can state in one sentence why the buggy numbers looked believable, and your reusable checklist is written down in prompts.md.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| train_wear_buggy.py will not run at all | It is meant to run. Check data/machines.csv exists and that you are in torch-vibe/ — the planted defects are all silent, so a crash means a path problem. |
| The assistant rewrites the file despite the instruction | Re-prompt with 'Report only. Do not output a corrected file. List each defect as line / effect / one-line fix.' Some assistants need the prohibition twice. |
| The corrected numbers moved less than expected | Compare against the BASELINE, not against the buggy run. The buggy regression was trained by a broadcast loss towards the batch mean, so its MAE was baseline-level all along. |
| You cannot find the fifth defect | Print pred.shape and target.shape immediately before the loss call. The mismatch is invisible in the source and obvious in the output. |
| After fixing zero_grad the loss becomes unstable | You may have placed it after backward(). It must come before the forward pass, or immediately before backward — never between backward and step. |

## Going further (optional)

- Plant a sixth defect of your own in a copy, hand it to a partner, and see whether their checklist catches it.
- Ask the assistant to convert your five-point checklist into a pytest file that fails on the buggy script and passes on the fixed one.
- Run the same review prompt in a second assistant and compare which defects each one ranked as most distorting.

---

[← Lab 5: Computation Graphs and Autograd with AI Assistance](../lab05-computation-graphs-and-autograd-with-ai-assistance/README.md) · [All labs](../README.md) · [Lab 7: Neural Network Architectures, Activation and Loss Functions →](../lab07-neural-network-architectures-activation-and-loss-functions/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
