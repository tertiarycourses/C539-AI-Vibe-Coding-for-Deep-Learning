# Lab 2 — Setting Up Cursor, GitHub Copilot and Claude for PyTorch

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 01:** AI Vibe Coding for PyTorch Fundamentals  
> **Learning outcome:** Set up a local Python environment with PyTorch and an AI coding assistant that can see your project files.

## Goal

Build the workspace every remaining lab runs in. You create the torch-vibe/ folder and an isolated virtual environment, install PyTorch, torchvision, pandas and matplotlib, then open the folder in PyCharm, VS Code or Cursor and confirm your AI assistant is actually active and able to read your files. You finish by having the assistant generate check_env.py — a script that prints every version and ends with a single 'environment ready' line, so a broken install is obvious now rather than at 4pm tomorrow in the middle of the transfer-learning lab.

## Why this lab matters

Every later lab assumes this workspace exists and that your assistant can see your files. An assistant that cannot read your project gives generic answers; one that can read it gives answers that match your actual shapes and filenames. Getting this right once is what stops the rest of the course becoming an install session.

## What you'll build

**A torch-vibe/ workspace with a virtual environment, PyTorch installed and verified, and an AI assistant that can see the project**

**Tools:** Python 3.10+, PyTorch, torchvision, PyCharm / VS Code / Cursor, GitHub Copilot / Claude

**Files you end up with:**

- `torch-vibe/`
- `.venv/`
- `requirements.txt`
- `check_env.py`
- `data/`

## Before you start

- Lab 1 completed, so you have seen the vibe coding loop and started `prompts.md`.
- A Windows or Mac laptop you can install software on, with about 3 GB free.

## Steps

### 1. Create the course workspace and an isolated virtual environment so today's installs cannot break your other Python projects.

On a Mac use `source .venv/bin/activate` instead of the Windows activate script. If you prefer Anaconda, `conda create -n torch-vibe python=3.11` then `conda activate torch-vibe` works just as well. The point is isolation — do not install into your system Python. Your prompt should change to `(.venv)` once activated.

**COMMAND** — run this in your terminal:

```bash
mkdir torch-vibe
cd torch-vibe
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install the CPU build of PyTorch plus the libraries every lab uses. This is the largest download of the course.

The `--index-url` flag selects the CPU-only wheels, which are a fraction of the size of the CUDA build and are all this course needs. On a slow connection this still takes several minutes — start it and read the next step while it downloads. If pip is very slow, add `--no-cache-dir`.

**COMMAND** — run this in your terminal:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pandas matplotlib scikit-learn
```

### 3. Create the folder structure the whole ForgeSight project will use, and copy in the course data files.

`data/` holds the ForgeSight CSVs and the images you generate in Lab 12, `models/` holds saved checkpoints from Lab 11 onward, and `reports/` holds the plots you produce in Lab 20. Copy machines.csv and vibration_series.csv from the course labs/resources/ folder into torch-vibe/data/ now.

**COMMAND** — run this in your terminal:

```bash
mkdir data models reports
cd ..
```

### 4. Open the workspace in your editor and confirm your AI assistant is active — you should see inline suggestions or a chat panel that can reference your open files.

In VS Code the Copilot icon in the status bar should not have a slash through it. In Cursor press Ctrl+L for chat. In PyCharm check the AI Assistant tool window. If none is available, keep Claude open in a browser tab and paste prompts there — every prompt in this course works in all four.

**COMMAND** — run this in your terminal:

```bash
code torch-vibe
```

### 5. Prompt the assistant to write the environment check. Notice you are describing the OUTPUT you want, not the imports.

Point 4 is the one learners skip. A bare `import torch` that fails gives a traceback; a named except gives 'torchvision is not installed', which is what you actually need at 9am with twenty people in the room. Point 3 matters because it proves the device string works, rather than just printing it.

**PROMPT** — paste this into your AI coding assistant:

```text
Create a file called check_env.py in this project. It should:
1. Print the Python version, then the installed versions of torch, torchvision, pandas, matplotlib and sklearn, one per labelled line
2. Print whether a CUDA GPU is available, and print which device the course will use ('cuda' if available else 'cpu')
3. Create a random tensor of shape (2, 3), move it to that device, and print its shape and device to prove the device line works
4. Wrap each import in a try/except that names the missing package clearly if it fails
5. Print exactly 'environment ready' as the last line if everything succeeded
```

### 6. Read the generated code, then run it. Every version should print and the last line must read 'environment ready'.

If an import fails, install just that package and re-run rather than reinstalling everything. If torch itself fails to import on Windows, it is almost always a 32-bit Python — check that `python -c "import platform; print(platform.architecture())"` says 64bit.

**COMMAND** — run this in your terminal:

```bash
python check_env.py
```

### 7. Freeze the exact versions you installed, so the project is reproducible on another machine later in Lab 21.

requirements.txt now pins the exact versions that work on your machine. Lab 21 turns this into part of the packaged project; a saved model is only reloadable against the library versions it was written with.

**COMMAND** — run this in your terminal:

```bash
pip freeze > requirements.txt
```

## Verification — Test it

`python check_env.py` prints a version number for Python, torch, torchvision, pandas, matplotlib and sklearn, prints the device as 'cpu' (or 'cuda' if you have a GPU), prints a tensor of shape torch.Size([2, 3]) on that device, and ends with exactly the line 'environment ready'. requirements.txt exists and contains a pinned torch version.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| 'python' is not recognised | Python is not on PATH. Reinstall from python.org with 'Add Python to PATH' ticked, or use the full path. On Mac try `python3` instead of `python`. |
| Activating the venv fails on Windows PowerShell with an execution-policy error | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. It applies to that session only. |
| pip install torch fails or is extremely slow | Use the CPU index URL exactly as shown, and add `--no-cache-dir`. If your network blocks it, fall back to Google Colab for the day — every lab runs there unchanged. |
| The assistant gives generic answers that ignore your filenames | It cannot see your project. In VS Code/Cursor open the folder (not a single file); in Claude, paste the relevant file contents into the conversation. |
| check_env.py prints versions but not 'environment ready' | An import failed inside a try/except. Read which package it named and install that one, then re-run. |

## Going further (optional)

- Ask the assistant to add a check that warns if the installed torch version is older than 2.0, and test it by reading the version string.
- Create a second venv with a different Python version and compare what pip freeze produces — this is the reproducibility problem Lab 21 solves.
- Configure your editor to use the .venv interpreter explicitly, then confirm the assistant's suggested imports resolve without warnings.

---

[← Lab 1: What Is AI Vibe Coding — Your First PyTorch from a Prompt](../lab01-what-is-ai-vibe-coding-your-first-pytorch-from-a-prompt/README.md) · [All labs](../README.md) · [Lab 3: Prompting Patterns for Correct Deep Learning Code →](../lab03-prompting-patterns-for-correct-deep-learning-code/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
