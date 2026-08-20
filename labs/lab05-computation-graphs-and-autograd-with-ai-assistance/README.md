# Lab 5 — Computation Graphs and Autograd with AI Assistance

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 01:** AI Vibe Coding for PyTorch Fundamentals  
> **Learning outcome:** Explain what autograd records, verify a gradient by hand, and show what detach and no_grad actually change.

## Goal

Autograd is the machinery that makes every later lab possible, so you verify it rather than trust it. You prompt for a script that builds a tiny expression on a requires_grad tensor, calls backward(), and prints the gradient next to the value you derive analytically on paper — they must match to several decimal places. You then investigate the three ways gradients silently stop flowing: detach(), a no_grad() block, and a tensor created without requires_grad. Finally you watch gradients ACCUMULATE across two backward calls, which is exactly the bug a missing zero_grad() causes in a training loop.

## Why this lab matters

Two of the most common silent failures in the course are caused by autograd: a detached tensor that stops the model learning, and a missing zero_grad() that corrupts every update. Both produce running code and plausible-looking numbers. Seeing them isolated here means you will recognise them inside a training loop tomorrow.

## What you'll build

**An autograd_lab.py proving a hand-derived gradient, plus a demonstration of detach, no_grad and gradient accumulation**

**Tools:** PyTorch autograd, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `autograd_lab.py`

## Before you start

- Lab 4 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Derive the gradient on paper FIRST. For y = 3x^2 + 2x + 1 at x = 4, write down dy/dx by hand before you write any prompt.

dy/dx = 6x + 2, so at x = 4 the gradient is 26. Do this on paper before you run anything. The habit of having an expected number before you look at the output is the difference between verifying and hoping.

### 2. Prompt for the verification script. You are asking the assistant to prove your arithmetic, not to teach you calculus.

Part 4 is the one that matters most for the rest of the course, so make sure the assistant actually implements it as described — two backward() calls on a freshly rebuilt expression, without zeroing. Some assistants 'helpfully' insert the zero_grad and destroy the demonstration. If yours does, follow up and insist the first version omits it.

**PROMPT** — paste this into your AI coding assistant:

```text
Create autograd_lab.py demonstrating PyTorch autograd.
PART 1 - verify a gradient: create x = torch.tensor(4.0, requires_grad=True), compute y = 3*x**2 + 2*x + 1, call y.backward(), and print x.grad. Also print the analytic value 6*x + 2 computed manually, and assert the two agree to 5 decimal places.
PART 2 - inspect the graph: print y.requires_grad, y.grad_fn and x.is_leaf, with a comment explaining what each one tells you.
PART 3 - three ways gradients stop: (a) repeat Part 1 but call .detach() on x before the expression, (b) repeat it inside a torch.no_grad() block, (c) repeat it with requires_grad left as False. For each, print whether grad_fn exists and whether x.grad is None, and add a comment on when you would WANT each behaviour.
PART 4 - accumulation: with a fresh x, call backward() twice on the same expression without zeroing, printing x.grad after each call. Then show that x.grad.zero_() between calls gives the correct value both times.
CONSTRAINTS: torch only, no training loop, no nn.Module. Label every printed line.
```

### 3. Read the code, then run Part 1 and compare the printed gradient against your paper answer.

x.grad must print 26.0 (or 26.000000...). If the assert fires, read which side is wrong — usually the assistant transcribed the analytic derivative incorrectly, which is itself a nice illustration that the assistant is not automatically right about mathematics.

**COMMAND** — run this in your terminal:

```bash
python autograd_lab.py
```

### 4. Study Part 2's output. Note which tensor has a grad_fn and which is a leaf — this is the shape of the computation graph in miniature.

`y.grad_fn` shows something like `<AddBackward0>` — that is the last operation recorded in the graph, and it is the thread PyTorch pulls to walk backwards. `x.is_leaf` is True because you created x directly; y is not a leaf because it was computed. Only leaf tensors with requires_grad get a populated .grad.

### 5. Work through Part 3 and answer for yourself: which of the three would you actually use at evaluation time, and which one is a bug when it appears in a model?

no_grad() is what you want at evaluation and inference — it is correct and it saves memory. detach() is what you want when deliberately cutting a graph, for example to stop a gradient flowing into a target. requires_grad=False on something that should be learning is almost always a bug.

### 6. Study Part 4 closely. Note the exact value printed after the second backward() call and compare it to the first.

Expect 26.0 then 52.0. Nothing raises, nothing warns — the gradient simply doubled. In a training loop this means every batch after the first applies an update built from the sum of all previous gradients, so the model moves too far in a stale direction. This is why zero_grad() comes first in the five-line loop.

### 7. Add the accumulation finding to prompts.md as a checklist item for reviewing any AI-generated training loop.

Write it as a review question you will apply to generated code: 'Is optimizer.zero_grad() present, and is it before the backward call rather than after?' You will use this exact check in Lab 6 and Lab 10.

## Verification — Test it

Part 1 prints x.grad as 26.0 and the assertion against your hand-derived 6x+2 passes. Part 2 shows y has a grad_fn while x is a leaf. Part 3 shows grad_fn absent and x.grad None in all three cases. Part 4 prints 26.0 after the first backward and 52.0 after the second, then 26.0 both times once zero_() is called.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| RuntimeError: Trying to backward through the graph a second time | The graph is freed after backward(). Either rebuild the expression before the second call, or pass retain_graph=True — the lab wants the expression rebuilt. |
| x.grad is None in Part 1 | requires_grad was not set, or you called backward on a non-scalar. Confirm x was created with requires_grad=True and y is a single number. |
| The assistant inserted zero_grad into Part 4 and both values print 26.0 | It removed the demonstration. Follow up: 'Part 4 must show accumulation. Remove the zeroing from the first pair of backward calls and only add it in the second pair.' |
| The assert in Part 1 fails | Read both printed numbers. If the analytic value is wrong, the assistant mis-derived it — correct the formula to 6*x+2 yourself. |
| grad can be implicitly created only for scalar outputs | y is a tensor, not a scalar. Reduce it with .sum() or .mean() before calling backward, or pass an explicit gradient argument. |

## Going further (optional)

- Extend Part 1 to a two-variable function and verify both partial derivatives by hand.
- Ask the assistant to draw the computation graph as ASCII art from the grad_fn chain, then check it against the expression.
- Time a forward pass inside and outside no_grad() on a large tensor and note the memory and speed difference.

---

[← Lab 4: Vibe Coding PyTorch Tensor Operations](../lab04-vibe-coding-pytorch-tensor-operations/README.md) · [All labs](../README.md) · [Lab 6: Reviewing and Debugging AI-Generated PyTorch Code →](../lab06-reviewing-and-debugging-ai-generated-pytorch-code/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
