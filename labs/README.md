# Labs — AI Vibe Coding with PyTorch Deep Learning (`C539`)

21 hands-on labs across 4 topics. Work through them in order — each lab reuses the workspace, the scripts and the prompting habits of the one before it.

Every lab follows the same shape: a **Goal**, the **Steps** (each with the exact PROMPT to paste into your AI coding assistant, or the COMMAND to run), a **Test it** check that tells you what a correct result looks like, and a **Troubleshooting** table for when it does not.

## Topic 01 — AI Vibe Coding for PyTorch Fundamentals

| Lab | Title | You'll build |
| --- | --- | --- |
| 1 | [What Is AI Vibe Coding — Your First PyTorch from a Prompt](lab01-what-is-ai-vibe-coding-your-first-pytorch-from-a-prompt/README.md) | A working hello_torch.py generated entirely from prompts, plus the first two entries in your prompts.md pattern library |
| 2 | [Setting Up Cursor, GitHub Copilot and Claude for PyTorch](lab02-setting-up-cursor-github-copilot-and-claude-for-pytorch/README.md) | A torch-vibe/ workspace with a virtual environment, PyTorch installed and verified, and an AI assistant that can see the project |
| 3 | [Prompting Patterns for Correct Deep Learning Code](lab03-prompting-patterns-for-correct-deep-learning-code/README.md) | A prompts.md pattern library with the five-part template and a worked A/B comparison, plus a working load_data.py |
| 4 | [Vibe Coding PyTorch Tensor Operations](lab04-vibe-coding-pytorch-tensor-operations/README.md) | A tensor_ops.py workout over the real telemetry, plus a shape_errors.py that demonstrates and explains two classic shape failures |
| 5 | [Computation Graphs and Autograd with AI Assistance](lab05-computation-graphs-and-autograd-with-ai-assistance/README.md) | An autograd_lab.py proving a hand-derived gradient, plus a demonstration of detach, no_grad and gradient accumulation |
| 6 | [Reviewing and Debugging AI-Generated PyTorch Code](lab06-reviewing-and-debugging-ai-generated-pytorch-code/README.md) | A corrected train_wear.py, a review_notes.md recording all five defects, and a reusable AI PyTorch review checklist |

## Topic 02 — Vibe Coding Neural Networks

| Lab | Title | You'll build |
| --- | --- | --- |
| 7 | [Neural Network Architectures, Activation and Loss Functions](lab07-neural-network-architectures-activation-and-loss-functions/README.md) | An activations.py with plotted activations and a linear-collapse proof, plus a losses_cheatsheet.md decision table |
| 8 | [Vibe Coding a Regression Model in PyTorch](lab08-vibe-coding-a-regression-model-in-pytorch/README.md) | A trained tool-wear regression network reporting MAE and RMSE that beat the mean baseline |
| 9 | [Vibe Coding a Classification Model with Softmax and Cross Entropy](lab09-vibe-coding-a-classification-model-with-softmax-and-cross-entropy/README.md) | A QC classifier outputting raw logits, evaluated with accuracy, per-class recall and a confusion matrix against the majority baseline |
| 10 | [Generating Training Loops, Optimizers and Metrics from Prompts](lab10-generating-training-loops-optimizers-and-metrics-from-prompts/README.md) | A reusable engine.py fit/evaluate pair driving both models, plus a sweep table comparing optimizers and learning rates |
| 11 | [Saving, Loading and Iterating on Models](lab11-saving-loading-and-iterating-on-models/README.md) | A checkpoint helper saving weights plus config and statistics, a proven identical-prediction reload, and a model card |

## Topic 03 — Vibe Coding Convolutional Neural Networks

| Lab | Title | You'll build |
| --- | --- | --- |
| 12 | [Overview of CNNs: Convolution, Pooling and Padding](lab12-overview-of-cnns-convolution-pooling-and-padding/README.md) | A generated 1200-image defect dataset plus a conv_explorer.py that predicts and verifies every feature-map shape |
| 13 | [Vibe Coding a CNN Image Classifier](lab13-vibe-coding-a-cnn-image-classifier/README.md) | A trained DefectCNN classifying ok, scratch and dent, evaluated with a confusion matrix and a misclassified-image grid |
| 14 | [Diagnosing Overfitting with AI Assistance](lab14-diagnosing-overfitting-with-ai-assistance/README.md) | An overfitting diagnosis with annotated curves, a memorised-subset demonstration, and a written diagnosis you verified yourself |
| 15 | [Data Augmentation and Regularization via Prompts](lab15-data-augmentation-and-regularization-via-prompts/README.md) | A regularized training pipeline plus an ablation table showing the measured contribution of each remedy |
| 16 | [Transfer Learning with Pre-Trained Models](lab16-transfer-learning-with-pre-trained-models/README.md) | A fine-tuned ResNet-18 defect classifier plus a three-way comparison against your scratch CNN |

## Topic 04 — Vibe Coding Recurrent Networks for Sequence Data

| Lab | Title | You'll build |
| --- | --- | --- |
| 17 | [Overview of RNNs, LSTM and GRU](lab17-overview-of-rnns-lstm-and-gru/README.md) | An rnn_cells.py comparing the three cells' shapes, gates and parameter counts, plus a measured vanishing-gradient demonstration |
| 18 | [Vibe Coding an LSTM for Time Series Forecasting](lab18-vibe-coding-an-lstm-for-time-series-forecasting/README.md) | An LSTM vibration forecaster with a time-ordered split and train-only scaling, measured against a persistence baseline |
| 19 | [Tuning Sequence Models with Follow-Up Prompts](lab19-tuning-sequence-models-with-follow-up-prompts/README.md) | A tuning results table across lookback, hidden size, layers, cell type and learning rate, plus a justified best model |
| 20 | [Evaluating and Visualizing Model Performance](lab20-evaluating-and-visualizing-model-performance/README.md) | A complete evaluation report covering all three models, with the plots and the written findings a stakeholder would need |
| 21 | [Packaging a Complete Deep Learning Project](lab21-packaging-a-complete-deep-learning-project/README.md) | A packaged, documented, tested ForgeSight project with a working CLI that another person can run unaided |

## The ForgeSight project

Every lab advances ONE project — **ForgeSight**, a deep learning suite for a precision metal-parts factory — from an empty folder in Lab 1 to a packaged, documented project in Lab 21. The factory gives all three data modalities a single home:

| Data | Model | Labs |
| --- | --- | --- |
| Tabular machine telemetry | Regression (tool wear) and 4-class QC classification | 3–11 |
| Surface-inspection images | CNN defect classifier and transfer learning | 12–16 |
| Vibration sensor readings | LSTM time-series forecasting | 17–20 |

Each lab states the exact files it expects to already exist, so you can rejoin at any lab boundary if you fall behind.

## Data and resources

All datasets, the image generator and the Lab 6 review script live in [`resources/`](resources/) — see that folder's README for what each file is and the baseline every model has to beat. Everything is synthetic and deterministic (seed 42), so your numbers should match the Learner Guide.

_Tertiary Infotech Academy Pte Ltd · Version v1.0 · 20 August 2026_
