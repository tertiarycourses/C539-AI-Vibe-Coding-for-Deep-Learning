"""
Topic 1 hands-on activities - AI Vibe Coding for PyTorch Fundamentals.

Labs 1-6, mapping 1:1 onto the six published sub-topics. Single source for the
PPT activity/step slides, the Learner Guide sections, the Lesson Plan schedule
rows and the labs/labNN-*/README.md files.

Per-lab keys beyond the engine's required set (used only by build_labs.py):
  why          - why this lab matters, one short paragraph for the README
  files        - the files the learner ends up with in this lab's folder
  notes        - extra detail per step, index-aligned with `steps`
  troubleshoot - [(symptom, fix), ...]
  stretch      - optional extensions for fast finishers
"""

DOMAIN1 = [

 dict(
  num=1, topic=1,
  title="What Is AI Vibe Coding — Your First PyTorch from a Prompt",
  objective="explain what AI vibe coding is and generate, read, run and verify your first PyTorch script from a plain-language prompt",
  desc=("Meet the vibe coding loop before you install anything. You open Google Colab, where PyTorch is "
        "already available, and write a PROMPT instead of a line of code. The assistant returns a script "
        "that builds a tensor of ForgeSight machine readings, prints its shape and dtype, and computes a "
        "column mean. You read that code and predict every number it will print BEFORE you run it. Then "
        "you repeat the exercise with a deliberately vague prompt and compare what comes back. The gap "
        "between the two answers is the whole lesson: the assistant is only as precise as your framing."),
  build="A working hello_torch.py generated entirely from prompts, plus the first two entries in your prompts.md pattern library",
  services="Google Colab, PyTorch, Claude / ChatGPT / Copilot Chat",
  why=("This lab exists to separate vibe coding from guessing. Both prompts produce code that runs; only "
       "one produces code that does what you meant. Everything else in the course depends on you noticing "
       "that difference, and it is much easier to notice on a six-line script than on a training loop."),
  files=["hello_torch.py", "prompts.md"],
  steps=[
   ("Open Google Colab in your browser and start a new notebook. Nothing is installed today — Colab already has PyTorch, so the environment cannot be the thing that goes wrong.",
    ""),
   ("Confirm PyTorch is present and note the version. Run this in the first Colab cell.",
    "import torch; print(torch.__version__); print(torch.cuda.is_available())"),
   ("Write the VAGUE prompt first, in your AI assistant, and paste the result into a Colab cell. Do not fix anything yet — run it exactly as returned.",
    "Write some PyTorch code that works with machine sensor data."),
   ("Now write the SPECIFIC prompt. Notice it names the shape, the dtype, the operation and the exact expected output.",
    "Write a short PyTorch script called hello_torch.py for a factory dataset.\nIt should:\n1. Create a tensor of shape (6, 3) of float32 values representing 6 machine readings of (spindle_speed, coolant_temp, vibration_rms)\n2. Print the tensor's shape, dtype and device on separate labelled lines\n3. Print the mean of each of the 3 columns, labelled with the column name\n4. Print the single reading with the highest vibration_rms\n5. Use a fixed manual seed of 42 so the numbers are reproducible\nDo not use pandas or numpy - torch only."),
   ("Read the generated code line by line BEFORE running it. Write down, on paper, the shape it will print and how many numbers the column-mean line will produce.",
    ""),
   ("Run the specific version in Colab and compare the output against your written prediction.",
    ""),
   ("Start your prompt pattern library. Create prompts.md and paste BOTH prompts into it with a one-line note on what the specific one named that the vague one did not.",
    ""),
  ],
  notes=[
   ("Go to https://colab.research.google.com and choose File > New notebook. You do not need a paid plan and "
    "you do not need a GPU — every lab in this course is sized to run on CPU. If Colab is blocked on your "
    "corporate network, tell the trainer now and pair with someone who can reach it; Lab 2 moves everything local."),
   ("You should see a version like 2.x.x. `torch.cuda.is_available()` printing False is completely normal and "
    "expected — it simply means you are on CPU, which is what this course assumes throughout."),
   ("Expect something generic and probably useless: a random tensor, no shapes printed, maybe numpy instead of "
    "torch. That is the point. Keep the output on screen so you can compare it in a moment. Resist the urge to "
    "improve the prompt — you will do that next, deliberately."),
   ("Count what this prompt pins down: the shape (6, 3), the dtype (float32), the meaning of each column, five "
    "numbered outputs, the seed, and one explicit prohibition. That is the five-part pattern you will formalise "
    "in Lab 3 — shapes, task, layers/ops, constraints, expected output."),
   ("This is the habit the whole course rests on. Ask yourself: what does `.mean(dim=0)` return here, a scalar "
    "or three numbers? If you are not sure, that uncertainty is exactly what running the code is about to "
    "resolve — but predict first, then check. Being wrong here is useful; being wrong silently in Lab 18 is not."),
   ("If the printed shape is `torch.Size([6, 3])` and you see three column means, the assistant read your prompt "
    "correctly. If it printed one number for the mean it used `.mean()` instead of `.mean(dim=0)` — a one-word "
    "correction, and a good first taste of refining rather than rewriting."),
   ("A prompts.md that grows all course is worth more than any single script you write today. You will add the "
    "formal pattern in Lab 3 and keep appending the prompts that worked through to Lab 21."),
  ],
  test=("Your specific prompt produced a script that prints torch.Size([6, 3]), a float32 dtype, cpu as the "
        "device, THREE column means (not one), and one highest-vibration reading. Your written prediction of "
        "the shape matches what actually printed, and prompts.md contains both prompts with a note on the difference."),
  troubleshoot=[
   ("The assistant returns numpy code instead of torch", "Your prompt did not forbid it. Re-prompt with 'Use torch only, no numpy and no pandas' — being explicit about what NOT to use is part of the pattern."),
   ("`.mean()` returns a single number instead of three", "The assistant omitted the dim argument. Reply with the exact symptom: 'The column mean printed one scalar; I need one mean per column, so use dim=0.'"),
   ("Colab says the runtime disconnected", "Reconnect with Runtime > Reconnect and re-run the cells from the top. Colab drops idle runtimes; nothing is lost because your prompts are saved in prompts.md."),
   ("torch.cuda.is_available() prints False", "That is correct and expected on a free CPU runtime. Every lab in this course runs on CPU by design."),
  ],
  stretch=[
   "Ask the assistant to rewrite hello_torch.py using only tensor operations with no Python loops, then compare readability.",
   "Give the same specific prompt to a second assistant (Claude vs Copilot vs ChatGPT) and diff the two scripts — note which one checked the dtype without being asked.",
   "Add a sixth requirement to the prompt: 'print the reading index, not just the values' and see how small a change the assistant makes.",
  ],
 ),

 dict(
  num=2, topic=1,
  title="Setting Up Cursor, GitHub Copilot and Claude for PyTorch",
  objective="set up a local Python environment with PyTorch and an AI coding assistant that can see your project files",
  desc=("Build the workspace every remaining lab runs in. You create the torch-vibe/ folder and an isolated "
        "virtual environment, install PyTorch, torchvision, pandas and matplotlib, then open the folder in "
        "PyCharm, VS Code or Cursor and confirm your AI assistant is actually active and able to read your "
        "files. You finish by having the assistant generate check_env.py — a script that prints every "
        "version and ends with a single 'environment ready' line, so a broken install is obvious now rather "
        "than at 4pm tomorrow in the middle of the transfer-learning lab."),
  build="A torch-vibe/ workspace with a virtual environment, PyTorch installed and verified, and an AI assistant that can see the project",
  services="Python 3.10+, PyTorch, torchvision, PyCharm / VS Code / Cursor, GitHub Copilot / Claude",
  why=("Every later lab assumes this workspace exists and that your assistant can see your files. An assistant "
       "that cannot read your project gives generic answers; one that can read it gives answers that match your "
       "actual shapes and filenames. Getting this right once is what stops the rest of the course becoming an "
       "install session."),
  files=["torch-vibe/", ".venv/", "requirements.txt", "check_env.py", "data/"],
  steps=[
   ("Create the course workspace and an isolated virtual environment so today's installs cannot break your other Python projects.",
    "mkdir torch-vibe\ncd torch-vibe\npython -m venv .venv\n.venv\\Scripts\\activate"),
   ("Install the CPU build of PyTorch plus the libraries every lab uses. This is the largest download of the course.",
    "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\npip install pandas matplotlib scikit-learn"),
   ("Create the folder structure the whole ForgeSight project will use, and copy in the course data files.",
    "mkdir data models reports\ncd .."),
   ("Open the workspace in your editor and confirm your AI assistant is active — you should see inline suggestions or a chat panel that can reference your open files.",
    "code torch-vibe"),
   ("Prompt the assistant to write the environment check. Notice you are describing the OUTPUT you want, not the imports.",
    "Create a file called check_env.py in this project. It should:\n1. Print the Python version, then the installed versions of torch, torchvision, pandas, matplotlib and sklearn, one per labelled line\n2. Print whether a CUDA GPU is available, and print which device the course will use ('cuda' if available else 'cpu')\n3. Create a random tensor of shape (2, 3), move it to that device, and print its shape and device to prove the device line works\n4. Wrap each import in a try/except that names the missing package clearly if it fails\n5. Print exactly 'environment ready' as the last line if everything succeeded"),
   ("Read the generated code, then run it. Every version should print and the last line must read 'environment ready'.",
    "python check_env.py"),
   ("Freeze the exact versions you installed, so the project is reproducible on another machine later in Lab 21.",
    "pip freeze > requirements.txt"),
  ],
  notes=[
   ("On a Mac use `source .venv/bin/activate` instead of the Windows activate script. If you prefer Anaconda, "
    "`conda create -n torch-vibe python=3.11` then `conda activate torch-vibe` works just as well. The point is "
    "isolation — do not install into your system Python. Your prompt should change to `(.venv)` once activated."),
   ("The `--index-url` flag selects the CPU-only wheels, which are a fraction of the size of the CUDA build and "
    "are all this course needs. On a slow connection this still takes several minutes — start it and read the "
    "next step while it downloads. If pip is very slow, add `--no-cache-dir`."),
   ("`data/` holds the ForgeSight CSVs and the images you generate in Lab 12, `models/` holds saved checkpoints "
    "from Lab 11 onward, and `reports/` holds the plots you produce in Lab 20. Copy machines.csv and "
    "vibration_series.csv from the course labs/resources/ folder into torch-vibe/data/ now."),
   ("In VS Code the Copilot icon in the status bar should not have a slash through it. In Cursor press Ctrl+L "
    "for chat. In PyCharm check the AI Assistant tool window. If none is available, keep Claude open in a "
    "browser tab and paste prompts there — every prompt in this course works in all four."),
   ("Point 4 is the one learners skip. A bare `import torch` that fails gives a traceback; a named except gives "
    "'torchvision is not installed', which is what you actually need at 9am with twenty people in the room. "
    "Point 3 matters because it proves the device string works, rather than just printing it."),
   ("If an import fails, install just that package and re-run rather than reinstalling everything. If torch "
    "itself fails to import on Windows, it is almost always a 32-bit Python — check that "
    "`python -c \"import platform; print(platform.architecture())\"` says 64bit."),
   ("requirements.txt now pins the exact versions that work on your machine. Lab 21 turns this into part of the "
    "packaged project; a saved model is only reloadable against the library versions it was written with."),
  ],
  test=("`python check_env.py` prints a version number for Python, torch, torchvision, pandas, matplotlib and "
        "sklearn, prints the device as 'cpu' (or 'cuda' if you have a GPU), prints a tensor of shape "
        "torch.Size([2, 3]) on that device, and ends with exactly the line 'environment ready'. "
        "requirements.txt exists and contains a pinned torch version."),
  troubleshoot=[
   ("'python' is not recognised", "Python is not on PATH. Reinstall from python.org with 'Add Python to PATH' ticked, or use the full path. On Mac try `python3` instead of `python`."),
   ("Activating the venv fails on Windows PowerShell with an execution-policy error", "Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. It applies to that session only."),
   ("pip install torch fails or is extremely slow", "Use the CPU index URL exactly as shown, and add `--no-cache-dir`. If your network blocks it, fall back to Google Colab for the day — every lab runs there unchanged."),
   ("The assistant gives generic answers that ignore your filenames", "It cannot see your project. In VS Code/Cursor open the folder (not a single file); in Claude, paste the relevant file contents into the conversation."),
   ("check_env.py prints versions but not 'environment ready'", "An import failed inside a try/except. Read which package it named and install that one, then re-run."),
  ],
  stretch=[
   "Ask the assistant to add a check that warns if the installed torch version is older than 2.0, and test it by reading the version string.",
   "Create a second venv with a different Python version and compare what pip freeze produces — this is the reproducibility problem Lab 21 solves.",
   "Configure your editor to use the .venv interpreter explicitly, then confirm the assistant's suggested imports resolve without warnings.",
  ],
 ),

 dict(
  num=3, topic=1,
  title="Prompting Patterns for Correct Deep Learning Code",
  objective="apply a repeatable five-part prompting pattern that produces correct, runnable deep learning code",
  desc=("Turn yesterday's lucky prompt into a method. You formalise the five-part pattern — shapes, task, "
        "layers and operations, constraints, expected output — and then test it under pressure on a genuinely "
        "shape-sensitive task: loading the ForgeSight telemetry and reshaping it into batches. You run a "
        "controlled A/B: the same task prompted vaguely and prompted with the pattern, and you diff the two "
        "results. You then practise the follow-up prompt, which is the skill that actually saves time — "
        "feeding back a specific symptom instead of the word 'fix'."),
  build="A prompts.md pattern library with the five-part template and a worked A/B comparison, plus a working load_data.py",
  services="PyTorch, pandas, Cursor / GitHub Copilot / Claude",
  why=("Most time lost to an AI assistant is lost re-prompting after a vague first attempt. A pattern you can "
       "apply in thirty seconds removes almost all of that, and the follow-up discipline removes the rest. "
       "These two habits are what make the remaining eighteen labs fast."),
  files=["prompts.md", "load_data.py"],
  steps=[
   ("Write the five-part pattern into prompts.md as a reusable template you will fill in for every later lab.",
    ""),
   ("Prompt the task VAGUELY first, and keep the result in a scratch file. You are building evidence, not code.",
    "Load the machine data and get it ready for a PyTorch model."),
   ("Now prompt the SAME task using the five-part pattern. Every clause maps to one part of the template.",
    "Create load_data.py for a PyTorch project.\nSHAPES: data/machines.csv has 4000 rows and these columns - machine_id, spindle_speed, feed_rate, coolant_temp, vibration_rms, spindle_load, ambient_temp, tool_wear_um, qc_class.\nTASK: load the CSV and return PyTorch tensors ready for supervised learning.\nOPERATIONS: drop machine_id; use the 6 sensor columns as features X; return tool_wear_um as a float32 regression target and qc_class as an int64 classification target; split 70/15/15 into train/val/test with a fixed seed of 42.\nCONSTRAINTS: standardise the features using the TRAINING split's mean and std only, then apply those same statistics to val and test - do not fit on the full dataset. Return the mean and std so they can be reused later.\nEXPECTED OUTPUT: a function load_forgesight(path) returning a dict with keys X_train, y_wear_train, y_qc_train and the same for val and test, plus feat_mean and feat_std. When run as a script it prints the shape and dtype of every tensor."),
   ("Read the generated code and check ONE thing specifically: where is the mean and standard deviation computed? Find that line before you run anything.",
    ""),
   ("Run it and confirm every shape and dtype is what the prompt asked for.",
    "python load_data.py"),
   ("Practise the follow-up prompt. Feed back a precise symptom rather than asking it to 'fix it', using this template with whatever your script actually got wrong.",
    "In load_data.py the features are standardised using the mean and std of the whole dataset before the split, which leaks validation and test information into training. Recompute mean and std from X_train only, then apply those same values to val and test. Change only the standardisation block."),
   ("Append both prompts and the corrected result to prompts.md, with a note naming the specific defect the pattern caught.",
    ""),
  ],
  notes=[
   ("The template to write down: **SHAPES** (what data exists, what shape, what dtype) · **TASK** (what the "
    "code must achieve) · **OPERATIONS** (the specific layers, transforms or steps) · **CONSTRAINTS** (what it "
    "must NOT do, seeds, libraries allowed) · **EXPECTED OUTPUT** (the function signature and what it prints). "
    "You will fill in all five for every substantial prompt from here on."),
   ("Expect the vague version to invent column names, guess at the target, split randomly with no seed, and "
    "very likely standardise before splitting. Every one of those is a real defect and none of them raises an "
    "error. Save it as scratch_vague.py — you are going to diff it in a moment."),
   ("This is a long prompt and that is the point. It takes about ninety seconds to write and it replaces four "
    "rounds of correction. Notice CONSTRAINTS is where the real expertise sits: 'do not fit on the full dataset' "
    "is the sentence that prevents the leakage bug this entire lab is built around."),
   ("Look for `X_train.mean(0)` or a StandardScaler fitted after the split — correct. If you find the mean "
    "computed on the full X before the split, the assistant leaked, even though you told it not to. Assistants "
    "get this wrong often enough that checking it is a permanent habit, not a one-off."),
   ("Expected: X_train around torch.Size([2800, 6]) float32, X_val and X_test around torch.Size([600, 6]), "
    "y_wear_* float32 of matching length, y_qc_* int64. If y_qc is float32 the assistant missed the dtype — "
    "CrossEntropyLoss in Lab 9 will reject it, so fix it now with a follow-up."),
   ("Compare this to typing 'fix it'. You named the file, the symptom, the cause, the required change and the "
    "scope ('change only the standardisation block'). That last clause is what stops the assistant helpfully "
    "rewriting your whole file and losing the parts that already worked."),
   ("Your prompts.md is now genuinely useful — the template plus one worked example of the pattern catching a "
    "silent bug. Keep appending to it. By Lab 21 it is the most portable thing you take home."),
  ],
  test=("`python load_data.py` prints X_train torch.Size([2800, 6]) float32, y_qc_train int64, and matching "
        "val/test shapes. You can point at the exact line where mean and std are computed and confirm it uses "
        "X_train only. prompts.md holds the five-part template, both A/B prompts and the follow-up prompt."),
  troubleshoot=[
   ("FileNotFoundError on data/machines.csv", "You are running from the wrong folder, or the CSV was not copied in Lab 2. Run from torch-vibe/ and confirm `data/machines.csv` exists."),
   ("y_qc is float32 and CrossEntropyLoss later complains", "Follow-up prompt: 'qc_class must be int64 class indices for nn.CrossEntropyLoss, not float. Cast it with .long() and change nothing else.'"),
   ("The splits do not add up to 4000 rows", "Rounding in the split. Ask for the split sizes to be printed and for the remainder to go to the training set."),
   ("The assistant used sklearn's StandardScaler when you wanted plain torch", "Either is fine here, but if you want consistency add 'use torch operations only, no sklearn' to CONSTRAINTS and re-prompt."),
   ("Standardisation still happens before the split after your follow-up", "Be more specific about location: quote the offending line back to the assistant verbatim and say 'move this to after the split and compute from X_train'."),
  ],
  stretch=[
   "Diff scratch_vague.py against load_data.py and count the concrete defects the pattern prevented — most learners find four or five.",
   "Add a sixth part to your template, VERIFICATION ('print the shapes and assert the training mean is near zero'), and test whether it improves the first attempt.",
   "Re-run the pattern prompt in a different assistant and compare which constraints each one respected.",
  ],
 ),

 dict(
  num=4, topic=1,
  title="Vibe Coding PyTorch Tensor Operations",
  objective="create, reshape, broadcast, index and move tensors, and read a shape error well enough to know which axis is wrong",
  desc=("Tensors are where almost every PyTorch bug actually lives, so you meet them deliberately. You prompt "
        "the assistant for a tensor workout script that runs over the real ForgeSight telemetry: dtype and "
        "device, reshape versus view, broadcasting, reduction along a chosen axis, boolean masking and matrix "
        "multiplication. Each section prints a shape, and each shape you predict first. You then trigger two "
        "shape errors on purpose and practise reading the message — because the error text tells you exactly "
        "which axis disagreed, if you know how to read it."),
  build="A tensor_ops.py workout over the real telemetry, plus a shape_errors.py that demonstrates and explains two classic shape failures",
  services="PyTorch, pandas, Cursor / GitHub Copilot / Claude",
  why=("A model is just tensors moving through layers. Learners who can predict shapes debug a network in "
       "minutes; learners who cannot spend the afternoon guessing. Deliberately causing shape errors now, on "
       "six lines of code, is how you learn to read them later inside a training loop."),
  files=["tensor_ops.py", "shape_errors.py"],
  steps=[
   ("Prompt for the tensor workout. Note that every section is required to PRINT a shape — you cannot verify what you cannot see.",
    "Create tensor_ops.py using the load_forgesight function from load_data.py.\nSHAPES: X_train is a float32 tensor of shape (2800, 6); y_wear_train is float32 of shape (2800,).\nTASK: demonstrate the core tensor operations on this real data, printing the shape and dtype after every step.\nOPERATIONS: (1) print shape, dtype, device and number of elements of X_train; (2) reshape y_wear_train to a column vector (2800, 1) and explain in a comment why a regression target usually needs this; (3) show the difference between .view() and .reshape() on a non-contiguous tensor; (4) compute the per-feature mean with dim=0 and the per-sample mean with dim=1 and print both shapes; (5) broadcast-subtract the per-feature mean from X_train and print the resulting shape; (6) use a boolean mask to select the rows where vibration_rms (column index 3) is above its mean, and print how many rows survived; (7) matrix-multiply X_train by a random weight tensor of shape (6, 1) and print the output shape.\nCONSTRAINTS: torch only, manual seed 42, no training and no gradients.\nEXPECTED OUTPUT: labelled print lines for every step above."),
   ("Before running, write down your predicted answer to three questions: what shape does dim=0 mean give, what shape does dim=1 give, and what shape comes out of the matmul?",
    ""),
   ("Run it and check your three predictions against the output.",
    "python tensor_ops.py"),
   ("Now break it deliberately. Prompt for a script that causes two classic shape errors and explains each one.",
    "Create shape_errors.py that deliberately triggers two common PyTorch shape errors and catches each one.\nERROR 1: matrix-multiply a tensor of shape (2800, 6) by a tensor of shape (1, 6) so the inner dimensions do not match.\nERROR 2: compute MSELoss between a prediction of shape (32, 1) and a target of shape (32,), which does NOT raise but silently broadcasts to a (32, 32) result and returns a wrong loss.\nFor each: wrap it in try/except, print the full error message for error 1, and for error 2 print the shape of the broadcast result and the wrong loss value next to the correct loss when the target is reshaped to (32, 1).\nAdd a comment under each explaining which axis disagreed and the one-line fix."),
   ("Run it and read Error 1's message carefully. Identify in the text exactly which two numbers had to match and did not.",
    "python shape_errors.py"),
   ("Study Error 2 — it is the dangerous one, because nothing raises. Confirm the two loss numbers differ and note which one is correct.",
    ""),
  ],
  notes=[
   ("Section 3 is subtle and worth reading twice: `.view()` requires contiguous memory and fails after a "
    "transpose, `.reshape()` silently copies instead. If the assistant does not actually produce a "
    "non-contiguous tensor (usually by transposing first), the demo is fake — follow up and say so."),
   ("The answers: `dim=0` reduces ACROSS rows and gives one number per feature, so torch.Size([6]); `dim=1` "
    "reduces across features and gives one number per sample, torch.Size([2800]); the matmul gives "
    "torch.Size([2800, 1]). The rule worth memorising is that the dim you name is the one that disappears."),
   ("If a prediction was wrong, that is the most valuable line of output on your screen today. Note which one "
    "in prompts.md — dim=0 versus dim=1 is the single most common confusion in the room, and it is the reason "
    "a per-feature normalisation sometimes silently normalises per-sample instead."),
   ("Error 2 is the whole point of this lab. A prediction of shape (32, 1) against a target of shape (32,) "
    "broadcasts to a (32, 32) matrix of pairwise differences, and MSELoss cheerfully averages all 1024 of them. "
    "It does not raise. Your training loop will run, your loss will decrease, and your model will be wrong."),
   ("The message names the mismatched dimensions explicitly — something like 'mat1 and mat2 shapes cannot be "
    "multiplied (2800x6 and 1x6)'. The inner numbers, 6 and 1, are the ones that must agree. Once you know to "
    "read the two shapes in that sentence, most shape errors take about five seconds to diagnose."),
   ("Expect the broadcast loss to be substantially larger and completely meaningless. The fix is "
    "`target.view(-1, 1)` — or `pred.squeeze()`. Add a line to prompts.md: 'always print pred.shape and "
    "target.shape before the first loss call'. You will thank yourself in Lab 8."),
  ],
  test=("tensor_ops.py prints torch.Size([6]) for the dim=0 mean, torch.Size([2800]) for the dim=1 mean and "
        "torch.Size([2800, 1]) for the matmul, and your written predictions match. shape_errors.py prints a "
        "caught error naming the (2800x6 and 1x6) mismatch, and shows a (32, 32) broadcast result whose loss "
        "value visibly differs from the correct (32, 1) loss."),
  troubleshoot=[
   ("ImportError: cannot import name 'load_forgesight'", "Lab 3's load_data.py is missing or the function has a different name. Open it and use the actual name, or re-run the Lab 3 prompt."),
   ("`.view()` works where the lab said it should fail", "The tensor was still contiguous. Follow up: 'transpose the tensor first so it is non-contiguous, then show that .view() raises and .reshape() succeeds.'"),
   ("Error 2 raises instead of broadcasting", "Your PyTorch version warns rather than broadcasting silently. Read the warning — it is telling you the same thing. Keep both loss numbers in the output."),
   ("The boolean mask returns zero rows", "The column index is wrong. vibration_rms is index 3 only if machine_id was dropped in Lab 3 — print the column order and correct the index."),
   ("RuntimeError about expected scalar type Double but found Float", "The CSV loaded as float64. Cast with `.float()` when building the tensor — pandas defaults to float64, torch wants float32."),
  ],
  stretch=[
   "Ask the assistant to add a timing comparison between a Python loop over rows and the vectorised matmul, and note the ratio.",
   "Add a third deliberate error: indexing with a float tensor instead of a long tensor, and read that message.",
   "Rewrite the broadcasting section to use `keepdim=True` and describe in a comment exactly what changes about the resulting shape.",
  ],
 ),

 dict(
  num=5, topic=1,
  title="Computation Graphs and Autograd with AI Assistance",
  objective="explain what autograd records, verify a gradient by hand, and show what detach and no_grad actually change",
  desc=("Autograd is the machinery that makes every later lab possible, so you verify it rather than trust it. "
        "You prompt for a script that builds a tiny expression on a requires_grad tensor, calls backward(), and "
        "prints the gradient next to the value you derive analytically on paper — they must match to several "
        "decimal places. You then investigate the three ways gradients silently stop flowing: detach(), a "
        "no_grad() block, and a tensor created without requires_grad. Finally you watch gradients ACCUMULATE "
        "across two backward calls, which is exactly the bug a missing zero_grad() causes in a training loop."),
  build="An autograd_lab.py proving a hand-derived gradient, plus a demonstration of detach, no_grad and gradient accumulation",
  services="PyTorch autograd, Cursor / GitHub Copilot / Claude",
  why=("Two of the most common silent failures in the course are caused by autograd: a detached tensor that "
       "stops the model learning, and a missing zero_grad() that corrupts every update. Both produce running "
       "code and plausible-looking numbers. Seeing them isolated here means you will recognise them inside a "
       "training loop tomorrow."),
  files=["autograd_lab.py"],
  steps=[
   ("Derive the gradient on paper FIRST. For y = 3x^2 + 2x + 1 at x = 4, write down dy/dx by hand before you write any prompt.",
    ""),
   ("Prompt for the verification script. You are asking the assistant to prove your arithmetic, not to teach you calculus.",
    "Create autograd_lab.py demonstrating PyTorch autograd.\nPART 1 - verify a gradient: create x = torch.tensor(4.0, requires_grad=True), compute y = 3*x**2 + 2*x + 1, call y.backward(), and print x.grad. Also print the analytic value 6*x + 2 computed manually, and assert the two agree to 5 decimal places.\nPART 2 - inspect the graph: print y.requires_grad, y.grad_fn and x.is_leaf, with a comment explaining what each one tells you.\nPART 3 - three ways gradients stop: (a) repeat Part 1 but call .detach() on x before the expression, (b) repeat it inside a torch.no_grad() block, (c) repeat it with requires_grad left as False. For each, print whether grad_fn exists and whether x.grad is None, and add a comment on when you would WANT each behaviour.\nPART 4 - accumulation: with a fresh x, call backward() twice on the same expression without zeroing, printing x.grad after each call. Then show that x.grad.zero_() between calls gives the correct value both times.\nCONSTRAINTS: torch only, no training loop, no nn.Module. Label every printed line."),
   ("Read the code, then run Part 1 and compare the printed gradient against your paper answer.",
    "python autograd_lab.py"),
   ("Study Part 2's output. Note which tensor has a grad_fn and which is a leaf — this is the shape of the computation graph in miniature.",
    ""),
   ("Work through Part 3 and answer for yourself: which of the three would you actually use at evaluation time, and which one is a bug when it appears in a model?",
    ""),
   ("Study Part 4 closely. Note the exact value printed after the second backward() call and compare it to the first.",
    ""),
   ("Add the accumulation finding to prompts.md as a checklist item for reviewing any AI-generated training loop.",
    ""),
  ],
  notes=[
   ("dy/dx = 6x + 2, so at x = 4 the gradient is 26. Do this on paper before you run anything. The habit of "
    "having an expected number before you look at the output is the difference between verifying and hoping."),
   ("Part 4 is the one that matters most for the rest of the course, so make sure the assistant actually "
    "implements it as described — two backward() calls on a freshly rebuilt expression, without zeroing. Some "
    "assistants 'helpfully' insert the zero_grad and destroy the demonstration. If yours does, follow up and "
    "insist the first version omits it."),
   ("x.grad must print 26.0 (or 26.000000...). If the assert fires, read which side is wrong — usually the "
    "assistant transcribed the analytic derivative incorrectly, which is itself a nice illustration that the "
    "assistant is not automatically right about mathematics."),
   ("`y.grad_fn` shows something like `<AddBackward0>` — that is the last operation recorded in the graph, and "
    "it is the thread PyTorch pulls to walk backwards. `x.is_leaf` is True because you created x directly; y is "
    "not a leaf because it was computed. Only leaf tensors with requires_grad get a populated .grad."),
   ("no_grad() is what you want at evaluation and inference — it is correct and it saves memory. detach() is "
    "what you want when deliberately cutting a graph, for example to stop a gradient flowing into a target. "
    "requires_grad=False on something that should be learning is almost always a bug."),
   ("Expect 26.0 then 52.0. Nothing raises, nothing warns — the gradient simply doubled. In a training loop "
    "this means every batch after the first applies an update built from the sum of all previous gradients, so "
    "the model moves too far in a stale direction. This is why zero_grad() comes first in the five-line loop."),
   ("Write it as a review question you will apply to generated code: 'Is optimizer.zero_grad() present, and is "
    "it before the backward call rather than after?' You will use this exact check in Lab 6 and Lab 10."),
  ],
  test=("Part 1 prints x.grad as 26.0 and the assertion against your hand-derived 6x+2 passes. Part 2 shows y "
        "has a grad_fn while x is a leaf. Part 3 shows grad_fn absent and x.grad None in all three cases. Part 4 "
        "prints 26.0 after the first backward and 52.0 after the second, then 26.0 both times once zero_() is called."),
  troubleshoot=[
   ("RuntimeError: Trying to backward through the graph a second time", "The graph is freed after backward(). Either rebuild the expression before the second call, or pass retain_graph=True — the lab wants the expression rebuilt."),
   ("x.grad is None in Part 1", "requires_grad was not set, or you called backward on a non-scalar. Confirm x was created with requires_grad=True and y is a single number."),
   ("The assistant inserted zero_grad into Part 4 and both values print 26.0", "It removed the demonstration. Follow up: 'Part 4 must show accumulation. Remove the zeroing from the first pair of backward calls and only add it in the second pair.'"),
   ("The assert in Part 1 fails", "Read both printed numbers. If the analytic value is wrong, the assistant mis-derived it — correct the formula to 6*x+2 yourself."),
   ("grad can be implicitly created only for scalar outputs", "y is a tensor, not a scalar. Reduce it with .sum() or .mean() before calling backward, or pass an explicit gradient argument."),
  ],
  stretch=[
   "Extend Part 1 to a two-variable function and verify both partial derivatives by hand.",
   "Ask the assistant to draw the computation graph as ASCII art from the grad_fn chain, then check it against the expression.",
   "Time a forward pass inside and outside no_grad() on a large tensor and note the memory and speed difference.",
  ],
 ),

 dict(
  num=6, topic=1,
  title="Reviewing and Debugging AI-Generated PyTorch Code",
  objective="review AI-generated PyTorch against a fixed checklist and correct each defect with a targeted follow-up prompt",
  desc=("Everything so far has been about producing code. This lab is about refusing to trust it. You are given "
        "train_wear_buggy.py — a script that runs to completion, prints a decreasing loss and reports "
        "believable-looking numbers, while containing five real defects. Nothing crashes. You review it against a "
        "written checklist, predict what each defect does to the result, then fix them one at a time with "
        "targeted follow-up prompts, re-running after each fix so you can see which defect was costing what. "
        "This is the single most transferable skill in the course."),
  build="A corrected train_wear.py, a review_notes.md recording all five defects, and a reusable AI PyTorch review checklist",
  services="PyTorch, Cursor / GitHub Copilot / Claude",
  why=("Deep learning code fails silently more often than it crashes. An assistant will produce a script that "
       "trains, prints falling numbers and returns confident predictions while doing something subtly wrong. "
       "The reviewer's checklist you build here is what you will actually use at work, long after you have "
       "forgotten the specific syntax."),
  files=["train_wear_buggy.py", "train_wear.py", "review_notes.md"],
  steps=[
   ("Copy the buggy script from the course resources into your workspace and run it BEFORE reading it. Note the final loss and score it reports.",
    "python train_wear_buggy.py"),
   ("Now read it against the checklist without running anything. Write your suspicions into review_notes.md before you ask the assistant anything.",
    ""),
   ("Ask the assistant to review it — but constrain the review so it reports rather than rewrites.",
    "Review the PyTorch script train_wear_buggy.py for correctness defects. Do NOT rewrite the file.\nFor each defect, report exactly three things: (1) the line, quoted verbatim; (2) what it does to the RESULT - be specific about whether it makes the score too high, too low, or meaningless; (3) the minimal one-line fix.\nCheck specifically for: standardisation or scaling fitted before the train/test split; a missing or misplaced optimizer.zero_grad(); a softmax or sigmoid applied before a loss function that already applies it; evaluation performed without model.eval() or torch.no_grad(); a target tensor whose shape does not match the prediction shape.\nRank the defects by how much they distort the reported result."),
   ("Compare the assistant's list against your own. Anything you found that it missed is worth more than anything it found that you missed — record both in review_notes.md.",
    ""),
   ("Fix the defects ONE AT A TIME, re-running after each. Start with the one the assistant ranked most distorting.",
    "In train_wear_buggy.py, the features are standardised using the mean and standard deviation of the full dataset before the train/test split, which leaks test information into training. Recompute the mean and std from the training split only and apply those same values to the test split. Change only that block and save the result as train_wear.py."),
   ("Continue with the remaining defects, using the same specific-symptom style. After each fix, record the new loss and score in review_notes.md.",
    ""),
   ("Compare the original reported numbers against the corrected ones, and write one sentence in review_notes.md explaining why the buggy script's numbers looked believable while both models were actually stuck at their baselines.",
    ""),
  ],
  notes=[
   ("Write the numbers down — the reported MAE in microns and the QC accuracy. The entire lesson lands only if "
    "you have the 'before' figures in front of you when you see the 'after'. Note also that the script never "
    "prints a baseline, which is itself the sixth defect nobody asked you to look for."),
   ("The checklist, which becomes your permanent one: (1) Is any scaling or fitting done before the split? "
    "(2) Is optimizer.zero_grad() present and before backward()? (3) Does the output layer apply an activation "
    "that the loss also applies? (4) Is evaluation inside model.eval() and torch.no_grad()? (5) Do prediction "
    "and target shapes match exactly? Most learners find three of the five unaided — that is a good score."),
   ("The 'do NOT rewrite' constraint matters. An unconstrained assistant returns a fixed file, you accept it, "
    "and you learn nothing about what was wrong. Forcing it to report line-by-line keeps you as the reviewer "
    "and makes the assistant explain itself. Ranking by distortion teaches you which bugs actually matter."),
   ("Assistants are good at the shape and zero_grad defects and noticeably weaker at leakage, because leakage "
    "is about the ORDER of operations rather than any single wrong line. That asymmetry is exactly why the "
    "human review step survives."),
   ("One at a time, re-running each time, is the discipline. Fix all five at once and you have no idea which "
    "one was responsible for the inflated score — and in a real project that is the question you will be asked."),
   ("The five defects and their effects: leakage before the split inflates the score; the missing zero_grad "
    "makes updates too large and training unstable; the extra softmax before CrossEntropyLoss flattens the "
    "gradients so the model learns slowly or not at all; evaluating without eval()/no_grad() leaves dropout on "
    "and wastes memory, giving a noisy and pessimistic score; the (N,) versus (N,1) target mismatch silently "
    "broadcasts and produces an entirely wrong loss."),
   ("The answer that matters: the numbers were believable, not impressive. A tool-wear MAE in the low tens of "
    "microns sounds like a working model until you compare it with simply predicting the average — which is "
    "all the broadcast loss actually trained it to do. The QC accuracy looks respectable for the same reason: "
    "it is roughly the share of 'ok' parts, so the model is predicting 'ok' for everything. Write that in your "
    "own words. A number is only meaningful next to its baseline, which is why every lab from here on computes "
    "the baseline FIRST."),
  ],
  test=("review_notes.md lists all five defects with the line quoted, the effect on the result and the fix. "
        "train_wear.py runs clean, and its corrected tool-wear MAE is clearly BETTER than the buggy script's — "
        "which was no better than predicting the average — while the corrected QC accuracy rises above the "
        "share of 'ok' parts the buggy version was merely echoing. You can state in one sentence why the buggy "
        "numbers looked believable, and your reusable checklist is written down in prompts.md."),
  troubleshoot=[
   ("train_wear_buggy.py will not run at all", "It is meant to run. Check data/machines.csv exists and that you are in torch-vibe/ — the planted defects are all silent, so a crash means a path problem."),
   ("The assistant rewrites the file despite the instruction", "Re-prompt with 'Report only. Do not output a corrected file. List each defect as line / effect / one-line fix.' Some assistants need the prohibition twice."),
   ("The corrected numbers moved less than expected", "Compare against the BASELINE, not against the buggy run. The buggy regression was trained by a broadcast loss towards the batch mean, so its MAE was baseline-level all along."),
   ("You cannot find the fifth defect", "Print pred.shape and target.shape immediately before the loss call. The mismatch is invisible in the source and obvious in the output."),
   ("After fixing zero_grad the loss becomes unstable", "You may have placed it after backward(). It must come before the forward pass, or immediately before backward — never between backward and step."),
  ],
  stretch=[
   "Plant a sixth defect of your own in a copy, hand it to a partner, and see whether their checklist catches it.",
   "Ask the assistant to convert your five-point checklist into a pytest file that fails on the buggy script and passes on the fixed one.",
   "Run the same review prompt in a second assistant and compare which defects each one ranked as most distorting.",
  ],
 ),

]
