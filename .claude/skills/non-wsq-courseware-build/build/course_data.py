"""
SINGLE SOURCE OF TRUTH - C539 AI Vibe Coding with PyTorch Deep Learning (NON-WSQ).

Every artifact (the slide deck PPT, the Lesson Plan LP, the Learner Guide LG +
its Markdown mirror, and the labs/labNN-*/README.md files) is generated from
THIS module plus data_domain1.py ... data_domain4.py, so all four stay 100%
aligned by construction.

COURSE SHAPE
------------
2 days, 7.5 instructional hours per day (15 hours), 4 topics, 21 hands-on labs
mapping 1:1 onto the 21 sub-topics published on the course page:

  Topic 1 - AI Vibe Coding for PyTorch Fundamentals
    * What Is AI Vibe Coding                                         -> Lab 1
    * Setting Up Cursor, GitHub Copilot and Claude for PyTorch       -> Lab 2
    * Prompting Patterns for Correct Deep Learning Code              -> Lab 3
    * Vibe Coding PyTorch Tensor Operations                          -> Lab 4
    * Computation Graphs and Autograd with AI Assistance             -> Lab 5
    * Reviewing and Debugging AI-Generated PyTorch Code              -> Lab 6
  Topic 2 - Vibe Coding Neural Networks
    * Neural Network Architectures, Activation and Loss Functions    -> Lab 7
    * Vibe Coding a Regression Model in PyTorch                      -> Lab 8
    * Vibe Coding a Classification Model with Softmax/Cross Entropy  -> Lab 9
    * Generating Training Loops, Optimizers and Metrics from Prompts -> Lab 10
    * Saving, Loading and Iterating on Models                        -> Lab 11
  Topic 3 - Vibe Coding Convolutional Neural Networks
    * Overview of CNNs: Convolution, Pooling and Padding             -> Lab 12
    * Vibe Coding a CNN Image Classifier                             -> Lab 13
    * Diagnosing Overfitting with AI Assistance                      -> Lab 14
    * Data Augmentation and Regularization via Prompts               -> Lab 15
    * Transfer Learning with Pre-Trained Models                      -> Lab 16
  Topic 4 - Vibe Coding Recurrent Networks for Sequence Data
    * Overview of RNNs, LSTM and GRU                                 -> Lab 17
    * Vibe Coding an LSTM for Time Series Forecasting                -> Lab 18
    * Tuning Sequence Models with Follow-Up Prompts                  -> Lab 19
    * Evaluating and Visualizing Model Performance                   -> Lab 20
    * Packaging a Complete Deep Learning Project                     -> Lab 21

ONE PROJECT, TWENTY-ONE LABS
----------------------------
The labs are not twenty-one disconnected exercises. Every lab advances ONE
project - ForgeSight, a deep learning suite for a precision metal-parts factory
- from an empty folder in Lab 1 to a packaged, documented project in Lab 21.
The factory gives all three data modalities a single home:

  * tabular machine telemetry  -> regression (tool wear) and classification (QC)
  * surface-inspection images  -> CNN defect classifier and transfer learning
  * vibration sensor readings  -> LSTM time-series forecasting

A learner who falls behind can rejoin at any lab boundary because each lab
states the exact files it expects to already exist.

NON-WSQ RULES - never reintroduce any of these:
  * NO assessment of any kind (no WA/SAQ, no PP, no case study, no marking guide).
  * NO SSG / SkillsFuture / WSQ funding, subsidy or eligibility content.
  * NO TRAQOM survey, NO digital attendance, NO 75% attendance rule.
  * NO TGS- course reference - this course carries the plain code C539.

TEACHING STANCE - "vibe coding, not blind coding"
-------------------------------------------------
The AI assistant writes the PyTorch; the learner owns the modelling decisions.
Deep learning punishes blind trust harder than ordinary programming, because a
network with a silently wrong loss function still trains, still prints a
decreasing number, and still returns confident predictions. Every lab therefore
ends in a 'Test it' step that forces the learner to VERIFY the AI-generated
result - a shape, a gradient, a metric, a saved file - and several labs
deliberately lead the assistant into a wrong-but-runnable answer so learners
learn to catch it.
"""

# ------------------------------------------------------------------ metadata
TITLE        = "AI Vibe Coding with PyTorch Deep Learning"
SHORT_TITLE  = "AI Vibe Coding with PyTorch Deep Learning"   # used in output filenames
COURSE_CODE  = "C539"                      # non-WSQ code - never a TGS- ref
VERSION      = "v1.0"
VERSION_DATE = "20 August 2026"
ORG          = "Tertiary Infotech Academy Pte Ltd"
UEN          = "UEN: 201200696W"
TRAINER      = "Dr. Alfred Ang"
DAYS         = 2
MODE         = "Instructor-led, hands-on practical labs (classroom, live online via Zoom, or on-site)"

# Read by the deck's schedule slide so it can never contradict the Lesson Plan.
HOURS_PER_DAY = 7.5
DAILY_TIMING  = "9:30am-6:30pm  ·  1-hour lunch  ·  tea breaks within training time"

TRAINER_CERT     = "ACLP-certified trainer; Python, deep learning and AI-assisted software development."
TRAINER_DELIVERS = "Professional short courses in Python, data science, machine learning and deep learning."

DARK_THEME = False

# ------------------------------------------------------------------ outcomes
LEARNING_OUTCOMES = [
    "LO1: Explain what AI vibe coding is, set up Cursor, GitHub Copilot and Claude as PyTorch coding partners, and apply prompting patterns that produce correct deep learning code.",
    "LO2: Vibe code PyTorch tensor operations, computation graphs and autograd, and review and debug AI-generated PyTorch code rather than trusting it.",
    "LO3: Build regression and classification neural networks in PyTorch from prompts, choosing architectures, activation functions and loss functions deliberately.",
    "LO4: Generate training loops, optimizers and metrics from prompts, and save, load and iterate on trained models.",
    "LO5: Vibe code convolutional neural networks for image classification, diagnose overfitting, and apply data augmentation, regularization and transfer learning.",
    "LO6: Vibe code recurrent networks for sequence data, tune and evaluate them with follow-up prompts, and package a complete deep learning project.",
]
LO_TITLES = [
    "Vibe coding setup",
    "Tensors and autograd",
    "Neural networks",
    "Train and iterate",
    "CNNs for vision",
    "Sequences and shipping",
]

# ------------------------------------------------------------------ topics
TOPICS = [
    dict(num=1, code="01",
         title="AI Vibe Coding for PyTorch Fundamentals",
         subtitle="What is AI vibe coding  ·  Setting up Cursor, GitHub Copilot and Claude for PyTorch  ·  Prompting patterns for correct deep learning code  ·  Tensor operations  ·  Computation graphs and autograd  ·  Reviewing and debugging AI-generated PyTorch code",
         weighting="~29% of course time",
         concepts=[
            "Vibe coding means you describe the OUTCOME in plain language and the AI assistant writes the PyTorch. You keep the architecture, the loss, the shapes and the verification.",
            "Cursor, GitHub Copilot and Claude are AI pair programmers: they complete code inline, explain an unfamiliar torch argument, and refactor a working script on request.",
            "Deep learning code fails silently. A wrong loss still decreases, a wrong axis still returns a number, and a detached graph still trains to nothing, so 'it ran' is never the test.",
            "A good PyTorch prompt names five things: the tensor shapes, the task, the layers, the constraints and the expected output. Vague prompts return code that runs but learns nothing.",
            "A tensor is an n-dimensional array with a dtype, a device and optionally a gradient history. Almost every PyTorch bug is a shape, a dtype or a device bug.",
            "Autograd records every operation on a requires_grad tensor into a computation graph, then replays it backwards. loss.backward() fills .grad; the optimizer consumes it.",
            "Read the generated code before you run it and predict the printed shapes. Predicting wrongly is the fastest way to find the bug the assistant just wrote for you.",
         ]),
    dict(num=2, code="02",
         title="Vibe Coding Neural Networks",
         subtitle="Neural network architectures, activation and loss functions  ·  Vibe coding a regression model  ·  Vibe coding a classification model with softmax and cross entropy  ·  Generating training loops, optimizers and metrics from prompts  ·  Saving, loading and iterating on models",
         weighting="~24% of course time",
         concepts=[
            "A neural network is a stack of linear layers separated by non-linearities. Without the activation function the whole stack collapses into one linear layer.",
            "ReLU is the default hidden activation; sigmoid and tanh saturate and stall learning in deep stacks. The OUTPUT layer's activation is decided by the task, not by taste.",
            "The loss function must match the output layer: MSELoss for regression, CrossEntropyLoss for multi-class. Mismatch them and the model trains happily towards nonsense.",
            "nn.CrossEntropyLoss applies log-softmax internally, so the model must output RAW LOGITS. Adding a softmax layer before it is the single most common PyTorch bug.",
            "The training loop is always the same five lines: zero the gradients, forward, compute loss, backward, step. Forgetting zero_grad() accumulates gradients across batches.",
            "Optimizers differ in how they use the gradient: SGD follows it, Adam adapts a per-parameter learning rate. Learning rate matters more than the choice of optimizer.",
            "Save the state_dict, not the pickled object, and record the architecture alongside it - weights without the class that built them are unloadable.",
         ]),
    dict(num=3, code="03",
         title="Vibe Coding Convolutional Neural Networks",
         subtitle="Overview of CNNs: convolution, pooling and padding  ·  Vibe coding a CNN image classifier  ·  Diagnosing overfitting with AI assistance  ·  Data augmentation and regularization via prompts  ·  Transfer learning with pre-trained models",
         weighting="~24% of course time",
         concepts=[
            "A convolution slides a small learned kernel across the image, so the same feature detector works wherever the feature appears - this is why CNNs beat dense layers on images.",
            "Padding preserves spatial size at the border; stride and pooling shrink it. Getting these wrong is how the flatten-to-linear layer ends up with a shape mismatch.",
            "Pooling downsamples and buys a little translation tolerance. Channels grow as spatial size shrinks: the network trades WHERE for WHAT as it goes deeper.",
            "Overfitting is the model memorising training images. You see it as a widening gap between training and validation loss, not as a bad training number.",
            "Data augmentation - random flips, rotations, crops, colour jitter - manufactures new training views for free. Augment the TRAINING set only, never validation.",
            "Dropout, weight decay and early stopping are the three regularizers you reach for first. Dropout must be switched off at evaluation time by model.eval().",
            "Transfer learning reuses a network already trained on millions of images: freeze the backbone, replace the classifier head, and fine-tune on a few hundred of your own images.",
         ]),
    dict(num=4, code="04",
         title="Vibe Coding Recurrent Networks for Sequence Data",
         subtitle="Overview of RNNs, LSTM and GRU  ·  Vibe coding an LSTM for time series forecasting  ·  Tuning sequence models with follow-up prompts  ·  Evaluating and visualizing model performance  ·  Packaging a complete deep learning project",
         weighting="~23% of course time",
         concepts=[
            "A recurrent network carries a hidden state from one timestep to the next, so order matters. A plain RNN forgets quickly because gradients vanish across long sequences.",
            "An LSTM adds a cell state and three gates - forget, input, output - so it can retain information over hundreds of steps. A GRU merges the gates into a cheaper, faster cell.",
            "Sequence data must be windowed before a model sees it: turn one long series into many (lookback -> horizon) pairs, and split by TIME, never at random.",
            "Scale the series using statistics from the training window only. Fitting the scaler on the whole series leaks the future into the past and inflates every score.",
            "nn.LSTM returns output and (h_n, c_n), and batch_first changes the axis order. Reading the wrong axis is the classic silent sequence bug - it runs and predicts noise.",
            "A forecast is judged against a naive baseline: predicting 'tomorrow equals today' is often strong. A model that cannot beat persistence has not learned the series.",
            "Packaging is what makes the work real: a clear folder structure, pinned dependencies, a README somebody else can follow, and a model card recording where the model must NOT be used.",
         ]),
]

# ------------------------------------------------------------------ day themes (kept short - this text renders in a deck panel)
DAY_THEMES = {
    1: "Fundamentals to first networks - set up your AI pair programmer, vibe code tensors and autograd, then build, train and save regression and classification models.",
    2: "Vision and sequences - vibe code a CNN image classifier, beat overfitting with augmentation and transfer learning, then forecast with an LSTM and package the project.",
}

# ------------------------------------------------------------------ schedule
# 9:30am-6:30pm. 480 scheduled minutes per day excluding lunch, of which 450
# (7.5 hours) are instructional and 30 are tea breaks.
# kind: admin | topic | lab | break | lunch | recap   (NEVER "assess")
def SCHEDULE(lab_titles):
    return {
     1: (DAY_THEMES[1], [
        ("9:30","9:50",20,"admin","Welcome, course introduction, ice-breaker and ground rules"),
        ("9:50","10:25",35,"topic","Topic 1 — "+TOPICS[0]["title"]+" (concepts + trainer demo)"),
        ("10:25","11:15",50,"lab","Hands-on: "+lab_titles([1,2])),
        ("11:15","11:30",15,"break","Tea break"),
        ("11:30","13:00",90,"lab","Hands-on: "+lab_titles([3,4,5])),
        ("13:00","14:00",60,"lunch","Lunch break"),
        ("14:00","14:20",20,"lab","Hands-on: "+lab_titles([6])),
        ("14:20","14:55",35,"topic","Topic 2 — "+TOPICS[1]["title"]+" (concepts + trainer demo)"),
        ("14:55","15:45",50,"lab","Hands-on: "+lab_titles([7,8])),
        ("15:45","16:00",15,"break","Tea break"),
        ("16:00","17:50",110,"lab","Hands-on: "+lab_titles([9,10,11])),
        ("17:50","18:15",25,"topic","Consolidation for Topic 1 and Topic 2: reading, predicting and correcting AI-generated PyTorch — shapes, losses and the training loop"),
        ("18:15","18:30",15,"recap","Day 1 recap and Q&A"),
     ]),
     2: (DAY_THEMES[2], [
        ("9:30","9:40",10,"admin","Day 1 review, outstanding questions and objectives for the day"),
        ("9:40","10:15",35,"topic","Topic 3 — "+TOPICS[2]["title"]+" (concepts + trainer demo)"),
        ("10:15","11:15",60,"lab","Hands-on: "+lab_titles([12,13])),
        ("11:15","11:30",15,"break","Tea break"),
        ("11:30","13:00",90,"lab","Hands-on: "+lab_titles([14,15,16])),
        ("13:00","14:00",60,"lunch","Lunch break"),
        ("14:00","14:35",35,"topic","Topic 4 — "+TOPICS[3]["title"]+" (concepts + trainer demo)"),
        ("14:35","15:45",70,"lab","Hands-on: "+lab_titles([17,18])),
        ("15:45","16:00",15,"break","Tea break"),
        ("16:00","17:50",110,"lab","Hands-on: "+lab_titles([19,20,21])),
        ("17:50","18:15",25,"topic","Project showcase: learners demo the packaged ForgeSight project and walk through the prompts that built each model"),
        ("18:15","18:30",15,"recap","Course recap, Q&A, next steps and close"),
     ]),
    }

# ------------------------------------------------------------------ core concepts section
COURSE_OVERVIEW = dict(
    section_title="Deep Learning and Vibe Coding",
    concepts_title="Key Concepts",
    concepts=[
        ("Deep learning", "Stacked layers that learn their own features from raw data - images, sequences, signals - instead of features you hand-engineer."),
        ("PyTorch", "The define-by-run deep learning framework: tensors, autograd, nn.Module and an explicit training loop you can read."),
        ("Vibe coding", "You describe the outcome in plain English; the AI assistant writes the PyTorch. You review, run, verify and correct it."),
        ("AI pair programmer", "Cursor, GitHub Copilot or Claude inside your editor - completing, explaining, debugging and refactoring code as you work."),
        ("Autograd", "The recorded computation graph that makes loss.backward() possible, and the reason a detached tensor silently stops learning."),
        ("Verification", "The step that separates vibe coding from guessing: you check the shape, the gradient and the metric before you believe the result."),
    ],
    framework_title="The Vibe Coding Loop",
    framework=[
        ("1. Frame", "State the tensor shapes, the task, the layers, the constraints and the expected output in one clear prompt."),
        ("2. Generate", "Let the assistant write the PyTorch. Read it before you run it and predict every shape it will print."),
        ("3. Run", "Execute it on the real data. Look at the actual shapes, losses and predictions, not just the absence of a traceback."),
        ("4. Verify", "Check the result against what you expected. Compare to the baseline. Inspect the samples it gets wrong."),
        ("5. Refine", "Feed the assistant the specific symptom - not 'fix it' - and ask for a targeted change. Repeat."),
        ("6. Keep", "Save the working script with a comment recording WHY the architecture, the loss and the metric are what they are."),
    ],
    statement=dict(
        headline="The AI writes the PyTorch. You own the model.",
        body="An assistant that produces runnable deep learning code is easy. An assistant that produces a model you can defend needs you to know what a right answer looks like - that is what these two days build.",
        kicker="THE POINT OF THIS COURSE"),
    pillars_title="What You'll Build — the ForgeSight Project",
    pillars=[
        ("Day 1 · Fundamentals to first networks", [
            "PyTorch, an editor and an AI assistant, verified",
            "A prompt pattern library that produces correct code",
            "Tensor and autograd utilities you wrote by prompting",
            "A tool-wear regression network",
            "A QC classifier with logits and cross entropy"]),
        ("Day 2 · Vision", [
            "A convolution and pooling shape explorer",
            "A CNN classifier for surface-defect images",
            "An overfitting diagnosis you solve yourself",
            "An augmentation and regularization pipeline",
            "A fine-tuned pre-trained ResNet"]),
        ("Day 2 · Sequences and shipping", [
            "An RNN, LSTM and GRU comparison",
            "An LSTM vibration forecaster beating persistence",
            "A tuned model from follow-up prompts",
            "Evaluation plots and a model card",
            "A packaged project you can hand to a colleague"]),
    ],
    arc_title="How Every Lab Progresses",
    arc=[
        "The trainer demonstrates the concept on the real ForgeSight data so you know what a correct result looks like.",
        "You write the PROMPT yourself — the lab gives you a starting prompt, you adapt it to your own project folder.",
        "The AI assistant generates the PyTorch; you read it and predict every shape before you run it.",
        "You run it, look at the shapes and the losses, and compare them against the expected result stated in the lab.",
        "The 'Test it' step forces a concrete check — a shape, a gradient, a metric, a saved file — not just 'no errors'.",
        "When it is wrong, you feed the assistant the specific symptom and refine, which is the real skill.",
    ],
)

# Optional per-lab screenshots (courseware/assets/screenshots/).
LAB_SHOTS = {}

# ------------------------------------------------------------------ Learner Guide content
LG_INTRO = (
    "This Learner Guide accompanies the 2-day course AI Vibe Coding with PyTorch Deep Learning (C539), "
    "conducted by Tertiary Infotech Academy Pte Ltd. It provides step-by-step instructions for all "
    "21 hands-on labs, organised into the 4 topics that follow the course slides and Lesson Plan. "
    "Across those 21 labs you build one project end to end — ForgeSight, a deep learning suite for a "
    "precision metal-parts factory — from an empty folder to a packaged, documented project, with an "
    "AI coding assistant writing the PyTorch alongside you.")
LG_INTRO2 = (
    "Work through the labs in order: each one reuses the project folder, the data and the prompting "
    "habits established by the labs before it, and each lab states exactly which files it expects to "
    "already exist so you can rejoin if you fall behind. Every lab gives you a starting PROMPT to paste "
    "into your AI assistant (Cursor, GitHub Copilot Chat or Claude) and a 'Test it' step that tells you "
    "exactly what a correct result looks like. Read the generated code before you run it — deep learning "
    "code fails silently, and the point of this course is that you stay in control of the architecture, "
    "the loss and the evaluation rather than trusting the assistant to be right.")

LG_SETUP = dict(
    needs=[
        "A Windows or Mac laptop with at least 8 GB RAM and permission to install software. A GPU is NOT required — every lab is sized to run on CPU.",
        "Python 3.10 or newer — installed directly from python.org, or via Anaconda / Miniconda.",
        "PyCharm, Visual Studio Code, or Cursor if you prefer an AI-first editor.",
        "An AI coding assistant you can use in class: GitHub Copilot, Cursor's built-in assistant, or Claude in a browser tab.",
        "About 3 GB of free disk space — PyTorch, torchvision and the pre-trained ResNet weights are the largest downloads.",
        "A Google account for the Google Colab fallback, in case a local install fails.",
        "The course data from labs/resources/ (machines.csv and vibration_series.csv), plus the defect images you generate in Lab 12.",
    ],
    verify_text=(
        "Confirm your environment before Lab 1. This should print a version number for each library "
        "and finish with the line 'environment ready'. If any import fails, fix it before you continue "
        "— every later lab depends on this."),
    verify_code=("python -c \"import sys, torch, torchvision, pandas, numpy, matplotlib; "
                 "print('python', sys.version.split()[0]); print('torch', torch.__version__); "
                 "print('torchvision', torchvision.__version__); print('environment ready')\""),
    conventions=[
        "Text shown as PROMPT is pasted into your AI coding assistant, not into a terminal.",
        "Text shown as COMMAND is typed into a terminal (PowerShell on Windows, Terminal on Mac).",
        "Placeholders such as <YOUR-NAME> are replaced with your own value.",
        "Every lab works inside a single course workspace folder, torch-vibe/, created in Lab 2.",
        "Random seeds are fixed at 42 throughout, so your numbers should be close to the guide — if they differ wildly, something is wrong.",
        "Every model runs on CPU by default; the device line is written so it uses a GPU automatically if you have one.",
        "Never paste a real API key, password or customer record into an AI assistant; the course data is synthetic for exactly this reason.",
    ],
)

LAB_NOTE = ("Read the AI-generated code before you run it, and use only data and accounts you are "
            "authorised to use. The ForgeSight telemetry, images and vibration series are synthetic "
            "and safe to share with an assistant.")

LG_WRAPUP = dict(
    title="Wrap-Up — What You Can Now Do",
    intro=("Across twenty-one labs you built a complete deep learning project with an AI pair programmer, "
           "from an empty folder to a packaged project covering tabular, image and sequence data. The "
           "torch-vibe/ folder on your laptop is yours to keep, reuse and adapt at work."),
    sections=[
        dict(title="The PyTorch skills",
             bullets=[
                "Create, reshape, broadcast and move tensors, and read a shape error well enough to know which axis is wrong.",
                "Explain what autograd records, why loss.backward() works, and what detach() and no_grad() actually change.",
                "Build a network with nn.Module, choosing hidden activations by default and the output layer by the task.",
                "Match the loss to the output: MSELoss for regression, CrossEntropyLoss over raw logits for classification.",
                "Write the five-line training loop from memory, with zero_grad in the right place and model.eval() at evaluation.",
                "Build a CNN, reason about convolution, padding, stride and pooling, and fix a flatten-to-linear shape mismatch.",
                "Diagnose overfitting from the train/validation curves, then fix it with augmentation, dropout, weight decay and early stopping.",
                "Fine-tune a pre-trained ResNet by freezing the backbone and replacing the classifier head.",
                "Window a time series correctly, split it by time, and forecast with an LSTM that beats a persistence baseline.",
             ]),
        dict(title="The vibe coding skills",
             bullets=[
                "Write a prompt that names the tensor shapes, the task, the layers, the constraints and the expected output.",
                "Read AI-generated PyTorch well enough to predict every printed shape before you run it.",
                "Recognise the classic AI PyTorch bugs: a softmax before CrossEntropyLoss, a missing zero_grad, augmentation applied to validation, a scaler fitted on the whole series, the wrong axis out of nn.LSTM.",
                "Feed a specific symptom back to the assistant instead of asking it to 'fix it'.",
                "Review and refactor a working-but-messy AI draft into modules you would be willing to sign your name to.",
                "Know when to stop prompting and read the PyTorch documentation instead.",
             ]),
        dict(title="Where this goes next",
             bullets=[
                "Point the same pipeline at real data at work — machine telemetry, inspection images, or any sensor series.",
                "Swap the CNN for a modern backbone, or the LSTM for a Transformer, once the workflow is second nature.",
                "Move training onto a GPU or Colab when your dataset outgrows the CPU, changing only the device line.",
                "Add monitoring: track the live prediction distribution and retrain when it drifts away from the training data.",
             ]),
    ],
)

LG_NEXT_STEPS = [
    "First pass: complete every lab yourself, following the steps in this guide.",
    "Second pass: delete your scripts and rebuild Labs 8, 13 and 18 from memory, prompting from scratch.",
    "Point the models at a real dataset in your own organisation and see which of the twenty-one steps breaks first.",
    "Keep a prompts.md file of the prompts that worked — it becomes your personal deep learning toolkit.",
    "Explore the follow-on courses in machine learning, computer vision and NLP at www.tertiarycourses.com.sg.",
]

LG_GLOSSARY = [
    ("Activation function", "The non-linearity between layers - ReLU, sigmoid, tanh. Without it a stack of linear layers collapses into a single linear layer."),
    ("Autograd", "PyTorch's automatic differentiation engine. It records operations on requires_grad tensors into a graph and replays it backwards to compute gradients."),
    ("Backpropagation", "Applying the chain rule backwards through the computation graph to get the gradient of the loss with respect to every parameter."),
    ("batch_first", "The nn.LSTM/nn.GRU flag that selects (batch, seq, feature) instead of (seq, batch, feature). Getting it wrong silently trains on the wrong axis."),
    ("Broadcasting", "NumPy-style automatic expansion of tensor shapes in an elementwise operation. Convenient, and the reason a shape bug can run without error."),
    ("Computation graph", "The dynamic record of operations PyTorch builds on the forward pass, and consumes on the backward pass. Define-by-run means it is rebuilt every iteration."),
    ("Convolution", "Sliding a small learned kernel across an input so the same feature detector applies everywhere. The core operation of a CNN."),
    ("Cross entropy", "The loss for multi-class classification. nn.CrossEntropyLoss applies log-softmax internally, so the model must output raw logits."),
    ("Data augmentation", "Manufacturing new training views by randomly flipping, rotating, cropping or jittering inputs. Applied to the training set only."),
    ("DataLoader", "The PyTorch iterator that batches, shuffles and optionally parallel-loads a Dataset."),
    ("Dropout", "Randomly zeroing a fraction of activations during training to stop co-adaptation. Must be disabled at evaluation with model.eval()."),
    ("Early stopping", "Halting training when validation loss stops improving, and keeping the best weights rather than the last ones."),
    ("Epoch", "One complete pass of the training data through the network."),
    ("Fine-tuning", "Continuing training a pre-trained network on your own data, often after freezing most of its layers."),
    ("GRU", "Gated Recurrent Unit - a recurrent cell with reset and update gates. Cheaper and faster than an LSTM, often just as accurate."),
    ("Learning rate", "The step size the optimizer takes along the gradient. The single hyperparameter most likely to be the reason training fails."),
    ("Logits", "The raw, unnormalised scores a classifier outputs before softmax. What nn.CrossEntropyLoss expects to be given."),
    ("Loss function", "The scalar the model minimises. It must match the output layer: MSELoss for regression, CrossEntropyLoss for multi-class classification."),
    ("LSTM", "Long Short-Term Memory - a recurrent cell with a cell state and forget, input and output gates, able to retain information over long sequences."),
    ("Lookback window", "The number of past timesteps a sequence model sees when predicting the next value or values."),
    ("model.eval() / model.train()", "The mode switch that turns dropout and batch-norm updating off and on. Forgetting it makes evaluation results wrong, not crash."),
    ("nn.Module", "The base class for every PyTorch model. Define layers in __init__ and the computation in forward()."),
    ("no_grad()", "A context manager that stops autograd recording. Used at inference and evaluation to save memory and time."),
    ("Optimizer", "The rule that turns gradients into parameter updates - SGD follows the gradient, Adam adapts a per-parameter learning rate."),
    ("Overfitting", "Learning the training examples rather than the pattern. Recognised by a widening gap between training and validation loss."),
    ("Padding", "Adding a border of zeros before a convolution so the output keeps the input's spatial size."),
    ("Persistence baseline", "The naive forecast 'the next value equals the last value'. A sequence model that cannot beat it has not learned the series."),
    ("Pooling", "Downsampling a feature map by taking the maximum or mean over small windows, shrinking spatial size and adding a little translation tolerance."),
    ("PyTorch", "The define-by-run deep learning framework used throughout this course: tensors, autograd, nn.Module and an explicit training loop."),
    ("Regularization", "Anything that reduces overfitting - dropout, weight decay, augmentation, early stopping."),
    ("RNN", "Recurrent Neural Network - a network that carries a hidden state across timesteps, so order matters. Plain RNNs forget quickly."),
    ("Softmax", "The function that turns logits into probabilities summing to 1. Applied for reporting, not before nn.CrossEntropyLoss."),
    ("state_dict", "The dictionary of a model's learned tensors. The recommended thing to save - weights plus the class that built them, never a pickled object."),
    ("Stride", "How far the convolution kernel moves between positions. A stride above 1 downsamples."),
    ("Tensor", "An n-dimensional array with a dtype, a device and optionally a gradient history. The PyTorch data structure."),
    ("Transfer learning", "Reusing a network trained on a large dataset as the feature extractor for your own smaller problem."),
    ("Vibe coding", "Describing the outcome you want in plain language and letting an AI assistant write the code, while you keep control of the logic and verify the result."),
    ("Weight decay", "L2 regularization applied through the optimizer, penalising large weights and reducing overfitting."),
    ("zero_grad()", "Clearing accumulated gradients before backward(). Forgetting it silently sums gradients across batches and corrupts training."),
]

NEXT_STEPS = dict(title="Continuing Your Journey", items=[
    "Rebuild the ForgeSight project from an empty folder, prompting your assistant from scratch — that is the real test of the skill.",
    "Replace the synthetic telemetry, images and vibration series with data from your own organisation and see which of the twenty-one steps breaks first.",
    "Try the same labs in a different assistant (Cursor vs Copilot vs Claude) and compare the PyTorch each one produces.",
    "Move on to computer vision, natural language processing or MLOps at www.tertiarycourses.com.sg.",
])

THANK_YOU = dict(
    kicker="THANK YOU FOR ATTENDING",
    body="You now have a complete, tested deep learning project and the prompting habits that let you build the next one without us.")

# ------------------------------------------------------------------ ice breaker
ICE_BREAKER = [
    "Your name, organisation and role.",
    "Have you used Python, NumPy or PyTorch before — and have you used an AI coding assistant?",
    "What images, sensor readings or sequences at work would you most like a model for?",
]

# ------------------------------------------------------------------ version history
VERSION_HISTORY = [
    ("1.0", VERSION_DATE, "Initial release. 4 topics, 21 hands-on labs, 2 days / 15 instructional hours.", TRAINER),
]
