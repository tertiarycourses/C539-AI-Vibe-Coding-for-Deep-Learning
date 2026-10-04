# Lab 17 — Overview of RNNs, LSTM and GRU

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 04:** Vibe Coding Recurrent Networks for Sequence Data  
> **Learning outcome:** Compare RNN, LSTM and GRU cells by their shapes, gates and parameter counts, and demonstrate why plain RNNs forget.

## Goal

Sequence models introduce a new axis and a new class of silent bug, so you explore the cells before you forecast with them. You prompt for a script that runs the same input batch through nn.RNN, nn.LSTM and nn.GRU, printing every output and hidden-state shape, the parameter count of each, and what LSTM returns that the others do not. You then demonstrate the vanishing-gradient problem directly: propagate a gradient back through a long sequence in an RNN and in an LSTM and compare the magnitudes at the first timestep. Finally you meet batch_first, the axis-order flag that silently trains models on nonsense.

## Why this lab matters

Sequence bugs are the hardest to see because they never change a shape you would notice — reading the wrong axis out of an LSTM produces a tensor of exactly the right size and completely wrong contents. Understanding what each cell returns, in what order, is the only defence.

## What you'll build

**An rnn_cells.py comparing the three cells' shapes, gates and parameter counts, plus a measured vanishing-gradient demonstration**

**Tools:** PyTorch nn.RNN / nn.LSTM / nn.GRU, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `rnn_cells.py`
- `reports/gradient_flow.png`

## Before you start

- Lab 16 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the three-cell comparison. Insist on printing every returned object's shape, because that is where the confusion lives.

The shapes: `output` is (16, 50, 32) — the hidden state at EVERY timestep — while `h_n` is (1, 16, 32), the final hidden state only, with the leading 1 being num_layers. For forecasting you usually want the last timestep of `output`, which is `output[:, -1, :]`, or equivalently the squeezed `h_n`. Taking `output[:, 0, :]` by mistake gives you the state after ONE timestep — a right-shaped, useless tensor.

**PROMPT** — paste this into your AI coding assistant:

```text
Create rnn_cells.py comparing PyTorch's three recurrent cells.
SHAPES: the input is a batch of 16 sequences, each 50 timesteps long with 1 feature per step, so shape (16, 50, 1) with batch_first=True.
TASK: show exactly what each cell returns and how many parameters it has.
OPERATIONS: build nn.RNN, nn.LSTM and nn.GRU each with input_size=1, hidden_size=32, num_layers=1, batch_first=True. For each: run the batch through it; print the shape of EVERY returned object, naming them (output, and h_n, and for LSTM also c_n); print the total parameter count; and print a one-line description of its gates - RNN has none, GRU has reset and update, LSTM has forget, input and output.
Then print a comparison table of the three parameter counts and add a comment explaining why LSTM has roughly four times the parameters of RNN and GRU roughly three times.
CONSTRAINTS: manual seed 42, no training. Label every printed line clearly.
```

### 2. Before running, predict two shapes: what shape is `output` and what shape is `h_n`? They are different, and confusing them is the classic bug.

Parameter counts follow directly from the gates. An RNN has one weight set; a GRU has three (reset, update, candidate); an LSTM has four (forget, input, output, candidate). That is the whole explanation for the roughly 1:3:4 ratio you will see printed.

### 3. Run it and check your predictions.

If your prediction of `h_n` missed the leading num_layers axis, note it. That leading 1 is the reason so much sequence code contains a `.squeeze(0)` whose purpose nobody remembers.

**COMMAND** — run this in your terminal:

```bash
python rnn_cells.py
```

### 4. Now demonstrate the batch_first trap, which is the reason sequence models silently learn nothing.

batch_first=False is the PyTorch DEFAULT, which is why this trap is so common: an assistant that omits the flag gives you a model expecting (seq, batch, feature) while your data is (batch, seq, feature). Nothing errors as long as the numbers happen to be compatible, and the model trains on transposed nonsense.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a function batch_first_trap() to rnn_cells.py.
OPERATIONS: take the same (16, 50, 1) input. Run it through an nn.LSTM built with batch_first=True and one built with batch_first=False, WITHOUT transposing the input for the second. Print the output shape from each and the shape of h_n from each.
Add a comment explaining what the second model actually did: it interpreted the batch axis as the time axis, so it treated 16 timesteps of a 50-sequence batch. Note that this raises NO error and produces a plausibly shaped tensor, which is why it is so dangerous.
Then show the correct fix - transposing the input to (50, 16, 1) for the batch_first=False model - and confirm the output shapes now correspond.
```

### 5. Run it and study what happened. Confirm for yourself that nothing raised an error.

The lesson to carry forward: always pass batch_first explicitly, and always print the output shape of the first forward pass. Add that to your review checklist alongside the zero_grad and softmax checks.

**COMMAND** — run this in your terminal:

```bash
python rnn_cells.py
```

### 6. Measure the vanishing gradient rather than just describing it.

Summing the last timestep and backpropagating to the input is a direct measurement of how far influence reaches back through time. The log y axis is necessary because the decay is exponential — on a linear axis the RNN's early timesteps are indistinguishable from zero.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a function vanishing_gradient() to rnn_cells.py.
TASK: measure how gradient magnitude decays back through time in an RNN versus an LSTM.
OPERATIONS: create an input of shape (1, 100, 1) with requires_grad=True. Pass it through a single-layer nn.RNN(1, 16, batch_first=True), take the LAST timestep of the output, sum it, and call backward(). Record the absolute gradient magnitude at the input for each of the 100 timesteps. Repeat with an nn.LSTM of the same size.
Plot both gradient-magnitude curves against timestep on a LOG y axis and save to reports/gradient_flow.png. Print the ratio of gradient magnitude at timestep 0 to timestep 99 for each cell.
CONSTRAINTS: same seed and hidden size for both so the comparison is fair.
```

### 7. Run it, open reports/gradient_flow.png, and read the two ratios. This is the vanishing gradient, measured.

Expect the RNN's gradient at timestep 0 to be orders of magnitude smaller than at timestep 99, while the LSTM's decays far more gently. That gap is precisely what the cell state and the forget gate buy you, and it is why LSTM and GRU replaced plain RNNs for anything longer than a few dozen steps.

**COMMAND** — run this in your terminal:

```bash
python rnn_cells.py
```

### 8. Write your conclusion in prompts.md: which cell you will use in Lab 18, and the one-sentence reason.

The expected conclusion is LSTM or GRU, because the vibration series in Lab 18 uses a lookback well beyond the range where a plain RNN retains gradient. State it in your own words with the ratio you measured.

## Verification — Test it

rnn_cells.py prints output as torch.Size([16, 50, 32]) and h_n as torch.Size([1, 16, 32]) for all three cells, plus c_n for the LSTM only, with parameter counts in roughly a 1:3:4 ratio. batch_first_trap shows a wrongly shaped-but-unraised result and its fix. reports/gradient_flow.png shows the RNN gradient decaying far faster than the LSTM's on a log axis.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| The gradient plot is empty or entirely flat | The input was not created with requires_grad=True, or the gradient was read from the wrong tensor. Grab the gradient from the input tensor, per timestep. |
| RuntimeError about the input having an unexpected number of dimensions | The (batch, seq, feature) 3-D shape is required. A 2-D input of (batch, seq) needs an explicit `.unsqueeze(-1)` to add the feature axis. |
| The LSTM returns two objects and the code unpacks three | nn.LSTM returns `output, (h_n, c_n)` — the second is a tuple. Unpack as `out, (h, c) = lstm(x)`. |
| Parameter counts do not show the expected ratio | Check all three were built with the same hidden_size and num_layers. Bias terms shift the ratio slightly, which is normal. |
| Gradient values are all nan | The sum over a long RNN can explode rather than vanish. Reduce the sequence to 50 steps or lower the hidden size, and note that exploding gradients are the other half of the same problem. |

## Going further (optional)

- Add a 2-layer LSTM and see how the h_n leading axis changes.
- Repeat the vanishing-gradient measurement with a GRU and place it between the RNN and the LSTM.
- Add gradient clipping to the RNN case and observe what it fixes and what it does not.

---

[← Lab 16: Transfer Learning with Pre-Trained Models](../lab16-transfer-learning-with-pre-trained-models/README.md) · [All labs](../README.md) · [Lab 18: Vibe Coding an LSTM for Time Series Forecasting →](../lab18-vibe-coding-an-lstm-for-time-series-forecasting/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
