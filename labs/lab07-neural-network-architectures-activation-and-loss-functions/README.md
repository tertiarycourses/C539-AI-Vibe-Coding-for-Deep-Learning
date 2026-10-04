# Lab 7 — Neural Network Architectures, Activation and Loss Functions

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 02:** Vibe Coding Neural Networks  
> **Learning outcome:** Choose hidden and output activations and match the loss function to the output layer.

## Goal

Before you train anything, settle the three decisions that determine whether training can work at all: the architecture, the activation and the loss. You prompt for a script that plots ReLU, sigmoid and tanh next to their gradients, then proves the claim that a stack of linear layers without activations collapses into a single linear layer — by showing two networks produce identical outputs. You finish by building a decision table linking each task type to its output layer and its loss, which becomes the reference you use for the next fourteen labs.

## Why this lab matters

Almost every 'my model will not learn' problem traces back to one of these three choices. Learners who can state why CrossEntropyLoss wants raw logits, and why a sigmoid deep in a stack stalls learning, diagnose in seconds what otherwise costs an afternoon.

## What you'll build

**An activations.py with plotted activations and a linear-collapse proof, plus a losses_cheatsheet.md decision table**

**Tools:** PyTorch, matplotlib, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `activations.py`
- `losses_cheatsheet.md`
- `reports/activations.png`

## Before you start

- Lab 6 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the activation comparison. Ask for the gradients as well as the functions — the gradient is what explains the behaviour.

The saturated-fraction numbers are the quantitative version of what the plot shows. Sigmoid and tanh flatten at both ends, so their gradients approach zero over a large part of the range; ReLU has gradient exactly 1 for every positive input. That is why ReLU became the default hidden activation.

**PROMPT** — paste this into your AI coding assistant:

```text
Create activations.py for a PyTorch course.
TASK: compare the three common activation functions and show why the choice matters.
OPERATIONS: (1) create x = torch.linspace(-6, 6, 200); (2) compute ReLU, sigmoid and tanh of x using torch.nn.functional; (3) compute the gradient of each with respect to x using autograd; (4) plot a 2-row figure - top row the three activations, bottom row their three gradients, sharing the x axis - and save it to reports/activations.png; (5) print, for each activation, the fraction of the input range where its gradient is smaller than 0.01.
CONSTRAINTS: torch and matplotlib only, no seaborn, label every axis and subplot.
EXPECTED OUTPUT: the saved figure plus three printed 'saturated fraction' lines.
```

### 2. Run it, open the figure, and answer one question from the bottom row: which activation keeps a usable gradient across the widest input range?

The answer is ReLU. Sigmoid's gradient peaks at 0.25 and decays fast, so in a deep stack the gradients multiply down towards nothing — the vanishing gradient problem you will meet again with plain RNNs in Lab 17. Note that ReLU's flat negative half is a real cost (dead units), just a smaller one.

**COMMAND** — run this in your terminal:

```bash
python activations.py
```

### 3. Now prove the linear-collapse claim. This is the reason activations exist at all.

Composing the layers means multiplying the weight matrices in the right order and carrying the biases through: W = W3 @ W2 @ W1 and b = W3 @ (W2 @ b1 + b2) + b3, give or take the transpose convention PyTorch uses. If the assistant gets the order wrong the assertion fails — that is a real bug worth making it fix rather than accepting a loosened tolerance.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a section to activations.py called linear_collapse().
TASK: prove that a stack of linear layers with no activation is equivalent to a single linear layer.
OPERATIONS: (1) build net_a = nn.Sequential(nn.Linear(6, 32), nn.Linear(32, 16), nn.Linear(16, 1)) with manual seed 42; (2) analytically collapse its three weight matrices and biases into ONE equivalent nn.Linear(6, 1) by composing them; (3) run the same random input batch of shape (8, 6) through both and print the two outputs side by side plus their maximum absolute difference; (4) assert the difference is below 1e-4; (5) repeat with nn.ReLU() inserted between the layers and print the maximum absolute difference now.
EXPECTED OUTPUT: near-zero difference without activations, clearly non-zero difference with ReLU.
```

### 4. Run it and confirm the assertion passes. Read the second difference — that number is what the non-linearity is buying you.

A difference below 1e-4 without activations means three layers did exactly as much as one: 561 parameters achieving what 7 could. With ReLU the difference is large, because the network can now bend. If the assistant 'fixes' a failing assert by raising the tolerance to 1.0, reject it and make it fix the algebra.

**COMMAND** — run this in your terminal:

```bash
python activations.py
```

### 5. Build the decision table by hand in losses_cheatsheet.md. Fill it in yourself before checking it with the assistant.

Fill in three rows: task type, final layer, output activation, loss function. Do it from memory. The row people get wrong is multi-class — write down what you believe before you check, so you find out whether you actually knew it.

### 6. Check your table against the assistant, and specifically interrogate the classification row.

The critical answer: nn.CrossEntropyLoss applies log-softmax internally and nn.BCEWithLogitsLoss applies sigmoid internally. Adding your own softmax before CrossEntropyLoss applies it twice, which flattens the distribution, shrinks the gradients and makes the model learn slowly or not at all — while still running and still printing a falling loss. This is the single most common AI-generated PyTorch bug.

**PROMPT** — paste this into your AI coding assistant:

```text
I have written a table mapping task type to output layer and PyTorch loss function:
- Regression (one continuous value) -> Linear(h, 1), no activation -> nn.MSELoss
- Binary classification -> Linear(h, 1), no activation -> nn.BCEWithLogitsLoss
- Multi-class classification (4 classes) -> Linear(h, 4), no activation -> nn.CrossEntropyLoss
For each row, confirm or correct it, and explain in one sentence what happens numerically if someone adds a softmax or sigmoid to the output layer before these losses. Be specific about which losses already apply that function internally.
```

### 7. Record the answer in losses_cheatsheet.md in your own words, especially the softmax warning.

Write the warning in your own words, not copied. Something like: 'CrossEntropyLoss wants raw logits. If I see nn.Softmax in a model's output layer next to CrossEntropyLoss, one of them must go.' You will apply this exact check in Lab 9, where the assistant will very likely make the mistake for you.

## Verification — Test it

reports/activations.png shows three activations over three gradients, and the printed saturated fractions are near zero for ReLU and clearly non-zero for sigmoid and tanh. linear_collapse() asserts a difference below 1e-4 without activations and prints a visibly larger difference with ReLU. losses_cheatsheet.md has all three rows and states which losses apply softmax or sigmoid internally.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The gradient plot is empty or all zeros | Gradients were computed without requires_grad on x. Follow up: 'x must be created with requires_grad=True, and use autograd.grad or backward on the sum to get elementwise gradients.' |
| The linear_collapse assertion fails with a large difference | The weight composition order or a transpose is wrong. Ask the assistant to print the shapes of each weight matrix and re-derive the composition — do not raise the tolerance. |
| matplotlib opens a window and blocks the script | Add `matplotlib.use('Agg')` before importing pyplot, or call plt.savefig without plt.show. Every plot in this course is saved to a file, not shown. |
| reports/ does not exist | Create it, or ask for `os.makedirs('reports', exist_ok=True)` before saving. Lab 2 created this folder — check you are running from torch-vibe/. |
| The assistant insists softmax before CrossEntropyLoss is fine | It is not. Ask it to compute the loss both ways on the same logits and print the two numbers — the demonstration settles it. |

## Going further (optional)

- Add LeakyReLU and GELU to the comparison and see how they handle the negative half of the range.
- Extend linear_collapse to five layers and confirm the equivalence still holds exactly.
- Compute the loss with and without an extra softmax on the same batch of logits, and record how much smaller the gradients become.

---

[← Lab 6: Reviewing and Debugging AI-Generated PyTorch Code](../lab06-reviewing-and-debugging-ai-generated-pytorch-code/README.md) · [All labs](../README.md) · [Lab 8: Vibe Coding a Regression Model in PyTorch →](../lab08-vibe-coding-a-regression-model-in-pytorch/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
