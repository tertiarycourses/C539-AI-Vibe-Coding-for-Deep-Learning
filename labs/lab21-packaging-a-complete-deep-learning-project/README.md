# Lab 21 — Packaging a Complete Deep Learning Project

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 04:** Vibe Coding Recurrent Networks for Sequence Data  
> **Learning outcome:** Package models, code, dependencies and documentation into a project another person can run.

## Goal

Turn twenty labs of scripts into something you could hand to a colleague and walk away from. You restructure the workspace into a clean layout, pin the dependencies, write a predict.py command-line tool that loads any of the three saved models and scores new input, add a smoke test that proves the prediction path works end to end, and write the README and model cards. The real test is the last step: a partner clones your project into a fresh folder and runs it from the README alone, without asking you a single question.

## Why this lab matters

A project that only runs on your laptop, in your head, in the order you happen to remember, is not a deliverable. Packaging is what converts two days of learning into something with a life beyond the classroom — and the handover test is the only honest way to know you have done it.

## What you'll build

**A packaged, documented, tested ForgeSight project with a working CLI that another person can run unaided**

**Tools:** Python packaging, pytest, git, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `README.md`
- `requirements.txt`
- `predict.py`
- `tests/test_predict.py`
- `.gitignore`

## Before you start

- Lab 20 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Restructure the workspace. Ask for a plan first rather than letting the assistant move files immediately.

Asking for the plan before the action is the pattern to carry away from this lab. A restructure that moves thirty files and rewrites the imports is exactly the kind of change you want to review as a diagram first — reverting it afterwards is far more work than reading a tree.

**PROMPT** — paste this into your AI coding assistant:

```text
Review my torch-vibe project folder and propose a clean package structure for handover. Do NOT move anything yet - give me the proposed layout as a tree with a one-line purpose for each folder and each top-level file.
The project contains: data loading for tabular and image and series data; three trained models (tool-wear regression, defect CNN, vibration LSTM); a shared training engine; a checkpoint helper; evaluation scripts; saved checkpoints; and generated reports.
Constraints: source code separated from data, models and reports; every model loadable without running any training script; no absolute paths anywhere; data and model binaries excluded from version control but their absence explained in the README.
```

### 2. Review the proposed tree and adjust it before anything moves. You are the architect here, not the assistant.

A reasonable layout: src/ for data, models, engine and checkpoint code; scripts/ for the training entry points; tests/ for the smoke tests; data/, models/ and reports/ for artifacts; and README.md, requirements.txt and .gitignore at the root. Push back if the proposal buries the entry points.

### 3. Apply the restructure and fix the imports it breaks.

Absolute paths are the most common reason a project fails on someone else's machine, and they are invisible on yours. `pathlib.Path(__file__).resolve().parent.parent` gives you a project root you can build every other path from.

**PROMPT** — paste this into your AI coding assistant:

```text
Apply the agreed structure. Move the files, update every import to match, and confirm each of the three training scripts and the evaluation script still run from the project root. Replace any absolute path with a path relative to the project root resolved via pathlib. Report every file you moved and every import you changed.
```

### 4. Write the prediction CLI — the thing that makes the project usable by someone who will never read your training code.

This is the deliverable. Everything before this lab was for you; predict.py is for someone else. The clause about reading preprocessing statistics from the checkpoint is the direct payoff of Lab 11 — without it, this tool would feed raw values to a model trained on standardised ones and print confident nonsense, exactly as you demonstrated then.

**PROMPT** — paste this into your AI coding assistant:

```text
Create predict.py, a command-line tool for the packaged project.
USAGE: `python predict.py --model wear --input data/sample_reading.csv`, `python predict.py --model defect --input path/to/image.png`, `python predict.py --model vibration --input data/recent_series.csv`.
OPERATIONS: for each model - load the checkpoint, apply the SAME preprocessing the model was trained with by reading the saved statistics from the checkpoint, run inference under torch.no_grad() and model.eval(), and print a human-readable result. For the defect model print the predicted class name and its probability; for wear print the predicted microns; for vibration print the next predicted value and the persistence value alongside it for comparison.
CONSTRAINTS: validate the input file exists and has the expected columns or format, and fail with a clear message naming what is wrong rather than a traceback. Never re-fit any scaler - always use the statistics stored in the checkpoint. Include --help text for every argument.
EXPECTED OUTPUT: three working commands, each printing a labelled prediction.
```

### 5. Test all three commands yourself with the sample inputs.

Test the error paths too, not just the happy ones. Point it at a file that does not exist and at a CSV with a missing column, and check the message names the actual problem. A tool that tracebacks at a colleague has not been handed over, it has been abandoned.

**COMMAND** — run this in your terminal:

```bash
python predict.py --model defect --input data/defects/val/scratch/scratch_001.png
python predict.py --model wear --input data/sample_reading.csv
python predict.py --model vibration --input data/vibration_series.csv
```

### 6. Add the smoke test that proves the prediction path works, so a future change that breaks it fails loudly.

Test 5 is the one people leave out and the one that catches the most embarrassing failure: a model that loads, runs, returns the right shape and predicts the identical value for every input. All the other tests pass. Only varying the input reveals it.

**PROMPT** — paste this into your AI coding assistant:

```text
Create tests/test_predict.py using pytest.
TESTS: (1) each of the three checkpoints loads and returns a model in eval mode; (2) each model produces an output of the expected shape for a single synthetic input; (3) the defect model's softmax probabilities sum to 1 and the predicted class is a valid index; (4) predict.py exits with a clear non-zero status and a readable message for a missing input file; (5) the wear model's prediction changes when the input changes, which catches a model that always returns the same value.
CONSTRAINTS: tests must run without retraining anything and must not require internet access. Skip cleanly with a clear message if a checkpoint file is absent.
```

### 7. Run the tests and confirm they pass.

Tests that need retraining or internet are tests nobody runs. Under a second, offline, is the target.

**COMMAND** — run this in your terminal:

```bash
python -m pytest tests/ -v
```

### 8. Write the README and finalise the model cards, then pin the dependencies.

The README's real audience is a capable stranger, which in practice is you in three months. If a step assumes something you happen to know today, it is a bug in the README. The results table with baselines is what stops anyone quoting your accuracy without its context.

**PROMPT** — paste this into your AI coding assistant:

```text
Write README.md for the packaged project covering: what ForgeSight does and the three models it contains; the exact setup commands from a fresh clone including creating the virtual environment and installing requirements; how to obtain or regenerate the data, since data files are not in version control; how to run each of the three predictions with a copyable example command; how to retrain each model; the project structure as a tree; and a results table of each model's headline metric next to its baseline. Also create .gitignore excluding .venv, __pycache__, data/, models/*.pt and reports/. Then regenerate requirements.txt with pinned versions.
```

### 9. Run the handover test, which is the only verification that counts.

The handover test: swap projects with a partner, clone into a fresh folder, and follow their README without speaking. Every question you have to ask is a defect in their documentation, and every question they ask is a defect in yours. This is the most useful fifteen minutes of the two days.

## Verification — Test it

A partner clones your project into a fresh folder, follows README.md alone, creates the environment, installs from requirements.txt and successfully runs all three `python predict.py` commands without asking you anything. `python -m pytest tests/ -v` passes. No absolute path appears anywhere in the source, and each model has a model card stating its metric, its baseline and where it must not be used.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| ModuleNotFoundError after the restructure | Run from the project root, and add __init__.py files to the package folders — or install the project in editable mode with `pip install -e .`. |
| predict.py loads the model but predicts nonsense | The preprocessing statistics from the checkpoint are not being applied. This is the exact failure demonstrated in Lab 11 — print the input tensor before inference and check its scale. |
| pytest cannot find the tests | Run `python -m pytest` from the project root rather than `pytest`, so the current directory is on the path. |
| Your partner cannot get the data | Data is correctly excluded from version control, but the README must say how to regenerate it — point at make_images.py and the resources folder. |
| requirements.txt has hundreds of entries | pip freeze captures everything in the venv. Either keep it, which is safest for reproducibility, or list only the direct dependencies with pinned versions and say which approach you chose. |

## Going further (optional)

- Add a Dockerfile so the project runs identically without any local Python setup.
- Wrap predict.py in a small Streamlit or Gradio interface so a non-programmer can drop in an image and see the prediction.
- Add a GitHub Actions workflow that runs the smoke tests on every push.

---

[← Lab 20: Evaluating and Visualizing Model Performance](../lab20-evaluating-and-visualizing-model-performance/README.md) · [All labs](../README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
