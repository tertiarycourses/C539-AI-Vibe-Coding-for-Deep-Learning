# Lab 4 — Vibe Coding PyTorch Tensor Operations

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 01:** AI Vibe Coding for PyTorch Fundamentals  
> **Learning outcome:** Create, reshape, broadcast, index and move tensors, and read a shape error well enough to know which axis is wrong.

## Goal

Tensors are where almost every PyTorch bug actually lives, so you meet them deliberately. You prompt the assistant for a tensor workout script that runs over the real ForgeSight telemetry: dtype and device, reshape versus view, broadcasting, reduction along a chosen axis, boolean masking and matrix multiplication. Each section prints a shape, and each shape you predict first. You then trigger two shape errors on purpose and practise reading the message — because the error text tells you exactly which axis disagreed, if you know how to read it.

## Why this lab matters

A model is just tensors moving through layers. Learners who can predict shapes debug a network in minutes; learners who cannot spend the afternoon guessing. Deliberately causing shape errors now, on six lines of code, is how you learn to read them later inside a training loop.

## What you'll build

**A tensor_ops.py workout over the real telemetry, plus a shape_errors.py that demonstrates and explains two classic shape failures**

**Tools:** PyTorch, pandas, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `tensor_ops.py`
- `shape_errors.py`

## Before you start

- Lab 3 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the tensor workout. Note that every section is required to PRINT a shape — you cannot verify what you cannot see.

Section 3 is subtle and worth reading twice: `.view()` requires contiguous memory and fails after a transpose, `.reshape()` silently copies instead. If the assistant does not actually produce a non-contiguous tensor (usually by transposing first), the demo is fake — follow up and say so.

**PROMPT** — paste this into your AI coding assistant:

```text
Create tensor_ops.py using the load_forgesight function from load_data.py.
SHAPES: X_train is a float32 tensor of shape (2800, 6); y_wear_train is float32 of shape (2800,).
TASK: demonstrate the core tensor operations on this real data, printing the shape and dtype after every step.
OPERATIONS: (1) print shape, dtype, device and number of elements of X_train; (2) reshape y_wear_train to a column vector (2800, 1) and explain in a comment why a regression target usually needs this; (3) show the difference between .view() and .reshape() on a non-contiguous tensor; (4) compute the per-feature mean with dim=0 and the per-sample mean with dim=1 and print both shapes; (5) broadcast-subtract the per-feature mean from X_train and print the resulting shape; (6) use a boolean mask to select the rows where vibration_rms (column index 3) is above its mean, and print how many rows survived; (7) matrix-multiply X_train by a random weight tensor of shape (6, 1) and print the output shape.
CONSTRAINTS: torch only, manual seed 42, no training and no gradients.
EXPECTED OUTPUT: labelled print lines for every step above.
```

### 2. Before running, write down your predicted answer to three questions: what shape does dim=0 mean give, what shape does dim=1 give, and what shape comes out of the matmul?

The answers: `dim=0` reduces ACROSS rows and gives one number per feature, so torch.Size([6]); `dim=1` reduces across features and gives one number per sample, torch.Size([2800]); the matmul gives torch.Size([2800, 1]). The rule worth memorising is that the dim you name is the one that disappears.

### 3. Run it and check your three predictions against the output.

If a prediction was wrong, that is the most valuable line of output on your screen today. Note which one in prompts.md — dim=0 versus dim=1 is the single most common confusion in the room, and it is the reason a per-feature normalisation sometimes silently normalises per-sample instead.

**COMMAND** — run this in your terminal:

```bash
python tensor_ops.py
```

### 4. Now break it deliberately. Prompt for a script that causes two classic shape errors and explains each one.

Error 2 is the whole point of this lab. A prediction of shape (32, 1) against a target of shape (32,) broadcasts to a (32, 32) matrix of pairwise differences, and MSELoss cheerfully averages all 1024 of them. It does not raise. Your training loop will run, your loss will decrease, and your model will be wrong.

**PROMPT** — paste this into your AI coding assistant:

```text
Create shape_errors.py that deliberately triggers two common PyTorch shape errors and catches each one.
ERROR 1: matrix-multiply a tensor of shape (2800, 6) by a tensor of shape (1, 6) so the inner dimensions do not match.
ERROR 2: compute MSELoss between a prediction of shape (32, 1) and a target of shape (32,), which does NOT raise but silently broadcasts to a (32, 32) result and returns a wrong loss.
For each: wrap it in try/except, print the full error message for error 1, and for error 2 print the shape of the broadcast result and the wrong loss value next to the correct loss when the target is reshaped to (32, 1).
Add a comment under each explaining which axis disagreed and the one-line fix.
```

### 5. Run it and read Error 1's message carefully. Identify in the text exactly which two numbers had to match and did not.

The message names the mismatched dimensions explicitly — something like 'mat1 and mat2 shapes cannot be multiplied (2800x6 and 1x6)'. The inner numbers, 6 and 1, are the ones that must agree. Once you know to read the two shapes in that sentence, most shape errors take about five seconds to diagnose.

**COMMAND** — run this in your terminal:

```bash
python shape_errors.py
```

### 6. Study Error 2 — it is the dangerous one, because nothing raises. Confirm the two loss numbers differ and note which one is correct.

Expect the broadcast loss to be substantially larger and completely meaningless. The fix is `target.view(-1, 1)` — or `pred.squeeze()`. Add a line to prompts.md: 'always print pred.shape and target.shape before the first loss call'. You will thank yourself in Lab 8.

## Verification — Test it

tensor_ops.py prints torch.Size([6]) for the dim=0 mean, torch.Size([2800]) for the dim=1 mean and torch.Size([2800, 1]) for the matmul, and your written predictions match. shape_errors.py prints a caught error naming the (2800x6 and 1x6) mismatch, and shows a (32, 32) broadcast result whose loss value visibly differs from the correct (32, 1) loss.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| ImportError: cannot import name 'load_forgesight' | Lab 3's load_data.py is missing or the function has a different name. Open it and use the actual name, or re-run the Lab 3 prompt. |
| `.view()` works where the lab said it should fail | The tensor was still contiguous. Follow up: 'transpose the tensor first so it is non-contiguous, then show that .view() raises and .reshape() succeeds.' |
| Error 2 raises instead of broadcasting | Your PyTorch version warns rather than broadcasting silently. Read the warning — it is telling you the same thing. Keep both loss numbers in the output. |
| The boolean mask returns zero rows | The column index is wrong. vibration_rms is index 3 only if machine_id was dropped in Lab 3 — print the column order and correct the index. |
| RuntimeError about expected scalar type Double but found Float | The CSV loaded as float64. Cast with `.float()` when building the tensor — pandas defaults to float64, torch wants float32. |

## Going further (optional)

- Ask the assistant to add a timing comparison between a Python loop over rows and the vectorised matmul, and note the ratio.
- Add a third deliberate error: indexing with a float tensor instead of a long tensor, and read that message.
- Rewrite the broadcasting section to use `keepdim=True` and describe in a comment exactly what changes about the resulting shape.

---

[← Lab 3: Prompting Patterns for Correct Deep Learning Code](../lab03-prompting-patterns-for-correct-deep-learning-code/README.md) · [All labs](../README.md) · [Lab 5: Computation Graphs and Autograd with AI Assistance →](../lab05-computation-graphs-and-autograd-with-ai-assistance/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
