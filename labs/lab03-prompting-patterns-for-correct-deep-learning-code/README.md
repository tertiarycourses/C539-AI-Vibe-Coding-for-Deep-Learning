# Lab 3 — Prompting Patterns for Correct Deep Learning Code

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 01:** AI Vibe Coding for PyTorch Fundamentals  
> **Learning outcome:** Apply a repeatable five-part prompting pattern that produces correct, runnable deep learning code.

## Goal

Turn yesterday's lucky prompt into a method. You formalise the five-part pattern — shapes, task, layers and operations, constraints, expected output — and then test it under pressure on a genuinely shape-sensitive task: loading the ForgeSight telemetry and reshaping it into batches. You run a controlled A/B: the same task prompted vaguely and prompted with the pattern, and you diff the two results. You then practise the follow-up prompt, which is the skill that actually saves time — feeding back a specific symptom instead of the word 'fix'.

## Why this lab matters

Most time lost to an AI assistant is lost re-prompting after a vague first attempt. A pattern you can apply in thirty seconds removes almost all of that, and the follow-up discipline removes the rest. These two habits are what make the remaining eighteen labs fast.

## What you'll build

**A prompts.md pattern library with the five-part template and a worked A/B comparison, plus a working load_data.py**

**Tools:** PyTorch, pandas, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `prompts.md`
- `load_data.py`

## Before you start

- Lab 2 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Write the five-part pattern into prompts.md as a reusable template you will fill in for every later lab.

The template to write down: **SHAPES** (what data exists, what shape, what dtype) · **TASK** (what the code must achieve) · **OPERATIONS** (the specific layers, transforms or steps) · **CONSTRAINTS** (what it must NOT do, seeds, libraries allowed) · **EXPECTED OUTPUT** (the function signature and what it prints). You will fill in all five for every substantial prompt from here on.

### 2. Prompt the task VAGUELY first, and keep the result in a scratch file. You are building evidence, not code.

Expect the vague version to invent column names, guess at the target, split randomly with no seed, and very likely standardise before splitting. Every one of those is a real defect and none of them raises an error. Save it as scratch_vague.py — you are going to diff it in a moment.

**PROMPT** — paste this into your AI coding assistant:

```text
Load the machine data and get it ready for a PyTorch model.
```

### 3. Now prompt the SAME task using the five-part pattern. Every clause maps to one part of the template.

This is a long prompt and that is the point. It takes about ninety seconds to write and it replaces four rounds of correction. Notice CONSTRAINTS is where the real expertise sits: 'do not fit on the full dataset' is the sentence that prevents the leakage bug this entire lab is built around.

**PROMPT** — paste this into your AI coding assistant:

```text
Create load_data.py for a PyTorch project.
SHAPES: data/machines.csv has 4000 rows and these columns - machine_id, spindle_speed, feed_rate, coolant_temp, vibration_rms, spindle_load, ambient_temp, tool_wear_um, qc_class.
TASK: load the CSV and return PyTorch tensors ready for supervised learning.
OPERATIONS: drop machine_id; use the 6 sensor columns as features X; return tool_wear_um as a float32 regression target and qc_class as an int64 classification target; split 70/15/15 into train/val/test with a fixed seed of 42.
CONSTRAINTS: standardise the features using the TRAINING split's mean and std only, then apply those same statistics to val and test - do not fit on the full dataset. Return the mean and std so they can be reused later.
EXPECTED OUTPUT: a function load_forgesight(path) returning a dict with keys X_train, y_wear_train, y_qc_train and the same for val and test, plus feat_mean and feat_std. When run as a script it prints the shape and dtype of every tensor.
```

### 4. Read the generated code and check ONE thing specifically: where is the mean and standard deviation computed? Find that line before you run anything.

Look for `X_train.mean(0)` or a StandardScaler fitted after the split — correct. If you find the mean computed on the full X before the split, the assistant leaked, even though you told it not to. Assistants get this wrong often enough that checking it is a permanent habit, not a one-off.

### 5. Run it and confirm every shape and dtype is what the prompt asked for.

Expected: X_train around torch.Size([2800, 6]) float32, X_val and X_test around torch.Size([600, 6]), y_wear_* float32 of matching length, y_qc_* int64. If y_qc is float32 the assistant missed the dtype — CrossEntropyLoss in Lab 9 will reject it, so fix it now with a follow-up.

**COMMAND** — run this in your terminal:

```bash
python load_data.py
```

### 6. Practise the follow-up prompt. Feed back a precise symptom rather than asking it to 'fix it', using this template with whatever your script actually got wrong.

Compare this to typing 'fix it'. You named the file, the symptom, the cause, the required change and the scope ('change only the standardisation block'). That last clause is what stops the assistant helpfully rewriting your whole file and losing the parts that already worked.

**PROMPT** — paste this into your AI coding assistant:

```text
In load_data.py the features are standardised using the mean and std of the whole dataset before the split, which leaks validation and test information into training. Recompute mean and std from X_train only, then apply those same values to val and test. Change only the standardisation block.
```

### 7. Append both prompts and the corrected result to prompts.md, with a note naming the specific defect the pattern caught.

Your prompts.md is now genuinely useful — the template plus one worked example of the pattern catching a silent bug. Keep appending to it. By Lab 21 it is the most portable thing you take home.

## Verification — Test it

`python load_data.py` prints X_train torch.Size([2800, 6]) float32, y_qc_train int64, and matching val/test shapes. You can point at the exact line where mean and std are computed and confirm it uses X_train only. prompts.md holds the five-part template, both A/B prompts and the follow-up prompt.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| FileNotFoundError on data/machines.csv | You are running from the wrong folder, or the CSV was not copied in Lab 2. Run from torch-vibe/ and confirm `data/machines.csv` exists. |
| y_qc is float32 and CrossEntropyLoss later complains | Follow-up prompt: 'qc_class must be int64 class indices for nn.CrossEntropyLoss, not float. Cast it with .long() and change nothing else.' |
| The splits do not add up to 4000 rows | Rounding in the split. Ask for the split sizes to be printed and for the remainder to go to the training set. |
| The assistant used sklearn's StandardScaler when you wanted plain torch | Either is fine here, but if you want consistency add 'use torch operations only, no sklearn' to CONSTRAINTS and re-prompt. |
| Standardisation still happens before the split after your follow-up | Be more specific about location: quote the offending line back to the assistant verbatim and say 'move this to after the split and compute from X_train'. |

## Going further (optional)

- Diff scratch_vague.py against load_data.py and count the concrete defects the pattern prevented — most learners find four or five.
- Add a sixth part to your template, VERIFICATION ('print the shapes and assert the training mean is near zero'), and test whether it improves the first attempt.
- Re-run the pattern prompt in a different assistant and compare which constraints each one respected.

---

[← Lab 2: Setting Up Cursor, GitHub Copilot and Claude for PyTorch](../lab02-setting-up-cursor-github-copilot-and-claude-for-pytorch/README.md) · [All labs](../README.md) · [Lab 4: Vibe Coding PyTorch Tensor Operations →](../lab04-vibe-coding-pytorch-tensor-operations/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
