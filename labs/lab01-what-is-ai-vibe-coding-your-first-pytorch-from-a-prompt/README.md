# Lab 1 — What Is AI Vibe Coding — Your First PyTorch from a Prompt

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 01:** AI Vibe Coding for PyTorch Fundamentals  
> **Learning outcome:** Explain what AI vibe coding is and generate, read, run and verify your first PyTorch script from a plain-language prompt.

## Goal

Meet the vibe coding loop before you install anything. You open Google Colab, where PyTorch is already available, and write a PROMPT instead of a line of code. The assistant returns a script that builds a tensor of ForgeSight machine readings, prints its shape and dtype, and computes a column mean. You read that code and predict every number it will print BEFORE you run it. Then you repeat the exercise with a deliberately vague prompt and compare what comes back. The gap between the two answers is the whole lesson: the assistant is only as precise as your framing.

## Why this lab matters

This lab exists to separate vibe coding from guessing. Both prompts produce code that runs; only one produces code that does what you meant. Everything else in the course depends on you noticing that difference, and it is much easier to notice on a six-line script than on a training loop.

## What you'll build

**A working hello_torch.py generated entirely from prompts, plus the first two entries in your prompts.md pattern library**

**Tools:** Google Colab, PyTorch, Claude / ChatGPT / Copilot Chat

**Files you end up with:**

- `hello_torch.py`
- `prompts.md`

## Before you start

- A web browser and a Google account — Lab 1 runs entirely in Google Colab, with nothing to install.
- An AI coding assistant available: GitHub Copilot, Cursor, or Claude in a browser tab.

## Steps

### 1. Open Google Colab in your browser and start a new notebook. Nothing is installed today — Colab already has PyTorch, so the environment cannot be the thing that goes wrong.

Go to https://colab.research.google.com and choose File > New notebook. You do not need a paid plan and you do not need a GPU — every lab in this course is sized to run on CPU. If Colab is blocked on your corporate network, tell the trainer now and pair with someone who can reach it; Lab 2 moves everything local.

### 2. Confirm PyTorch is present and note the version. Run this in the first Colab cell.

You should see a version like 2.x.x. `torch.cuda.is_available()` printing False is completely normal and expected — it simply means you are on CPU, which is what this course assumes throughout.

**PROMPT** — paste this into your AI coding assistant:

```text
import torch; print(torch.__version__); print(torch.cuda.is_available())
```

### 3. Write the VAGUE prompt first, in your AI assistant, and paste the result into a Colab cell. Do not fix anything yet — run it exactly as returned.

Expect something generic and probably useless: a random tensor, no shapes printed, maybe numpy instead of torch. That is the point. Keep the output on screen so you can compare it in a moment. Resist the urge to improve the prompt — you will do that next, deliberately.

**PROMPT** — paste this into your AI coding assistant:

```text
Write some PyTorch code that works with machine sensor data.
```

### 4. Now write the SPECIFIC prompt. Notice it names the shape, the dtype, the operation and the exact expected output.

Count what this prompt pins down: the shape (6, 3), the dtype (float32), the meaning of each column, five numbered outputs, the seed, and one explicit prohibition. That is the five-part pattern you will formalise in Lab 3 — shapes, task, layers/ops, constraints, expected output.

**PROMPT** — paste this into your AI coding assistant:

```text
Write a short PyTorch script called hello_torch.py for a factory dataset.
It should:
1. Create a tensor of shape (6, 3) of float32 values representing 6 machine readings of (spindle_speed, coolant_temp, vibration_rms)
2. Print the tensor's shape, dtype and device on separate labelled lines
3. Print the mean of each of the 3 columns, labelled with the column name
4. Print the single reading with the highest vibration_rms
5. Use a fixed manual seed of 42 so the numbers are reproducible
Do not use pandas or numpy - torch only.
```

### 5. Read the generated code line by line BEFORE running it. Write down, on paper, the shape it will print and how many numbers the column-mean line will produce.

This is the habit the whole course rests on. Ask yourself: what does `.mean(dim=0)` return here, a scalar or three numbers? If you are not sure, that uncertainty is exactly what running the code is about to resolve — but predict first, then check. Being wrong here is useful; being wrong silently in Lab 18 is not.

### 6. Run the specific version in Colab and compare the output against your written prediction.

If the printed shape is `torch.Size([6, 3])` and you see three column means, the assistant read your prompt correctly. If it printed one number for the mean it used `.mean()` instead of `.mean(dim=0)` — a one-word correction, and a good first taste of refining rather than rewriting.

### 7. Start your prompt pattern library. Create prompts.md and paste BOTH prompts into it with a one-line note on what the specific one named that the vague one did not.

A prompts.md that grows all course is worth more than any single script you write today. You will add the formal pattern in Lab 3 and keep appending the prompts that worked through to Lab 21.

## Verification — Test it

Your specific prompt produced a script that prints torch.Size([6, 3]), a float32 dtype, cpu as the device, THREE column means (not one), and one highest-vibration reading. Your written prediction of the shape matches what actually printed, and prompts.md contains both prompts with a note on the difference.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The assistant returns numpy code instead of torch | Your prompt did not forbid it. Re-prompt with 'Use torch only, no numpy and no pandas' — being explicit about what NOT to use is part of the pattern. |
| `.mean()` returns a single number instead of three | The assistant omitted the dim argument. Reply with the exact symptom: 'The column mean printed one scalar; I need one mean per column, so use dim=0.' |
| Colab says the runtime disconnected | Reconnect with Runtime > Reconnect and re-run the cells from the top. Colab drops idle runtimes; nothing is lost because your prompts are saved in prompts.md. |
| torch.cuda.is_available() prints False | That is correct and expected on a free CPU runtime. Every lab in this course runs on CPU by design. |

## Going further (optional)

- Ask the assistant to rewrite hello_torch.py using only tensor operations with no Python loops, then compare readability.
- Give the same specific prompt to a second assistant (Claude vs Copilot vs ChatGPT) and diff the two scripts — note which one checked the dtype without being asked.
- Add a sixth requirement to the prompt: 'print the reading index, not just the values' and see how small a change the assistant makes.

---

[All labs](../README.md) · [Lab 2: Setting Up Cursor, GitHub Copilot and Claude for PyTorch →](../lab02-setting-up-cursor-github-copilot-and-claude-for-pytorch/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
