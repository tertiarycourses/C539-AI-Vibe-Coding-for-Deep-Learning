# Lab 13 — Vibe Coding a CNN Image Classifier

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 03:** Vibe Coding Convolutional Neural Networks  
> **Learning outcome:** Build, train and evaluate a convolutional network that classifies images into three defect classes.

## Goal

Turn the shape knowledge into a working classifier. You prompt for an ImageFolder-based data pipeline and a DefectCNN whose first Linear layer takes exactly the flatten size you derived in Lab 12, then train it with the same engine.fit() you built in Lab 10 — no new training loop. You evaluate against a majority-class baseline with a confusion matrix, and you inspect the images the model gets wrong. Because the architecture, the loss and the loop are all things you have already validated, the only new failure surface is the data pipeline and the shape arithmetic.

## Why this lab matters

This is the payoff for Labs 10 and 12: a genuinely different model type, built in one lab, because the training engine is reusable and the shapes are predictable. It is also the model you will spend the next three labs improving.

## What you'll build

**A trained DefectCNN classifying ok, scratch and dent, evaluated with a confusion matrix and a misclassified-image grid**

**Tools:** PyTorch, torchvision ImageFolder, engine.py, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `datasets.py`
- `models.py`
- `train_cnn.py`
- `reports/cnn_confusion.png`

## Before you start

- Lab 12 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the image data pipeline. Keep the transforms minimal for now — augmentation is deliberately Lab 15's job.

ImageFolder assigns class indices alphabetically, so expect dent=0, ok=1, scratch=2 — not the order you would naturally write. Reading a confusion matrix with the wrong mapping in your head is a classic and entirely avoidable mistake. Normalize with mean 0.5 and std 0.5 maps the 0-1 pixel range to roughly -1 to 1, which is a reasonable default for a network with ReLU activations.

**PROMPT** — paste this into your AI coding assistant:

```text
Create datasets.py for the defect images.
SHAPES: data/defects/train and data/defects/val each contain subfolders ok, scratch and dent holding 64x64 greyscale PNGs.
TASK: return DataLoaders for training and validation.
OPERATIONS: build a function get_defect_loaders(batch_size=32, data_dir='data/defects') that uses torchvision.datasets.ImageFolder with a transform pipeline of Grayscale(1) then ToTensor() then Normalize(mean=[0.5], std=[0.5]). Return train_loader (shuffled), val_loader (not shuffled) and the class_to_idx mapping.
CONSTRAINTS: do NOT add any augmentation - that comes in a later lab. Set a fixed generator seed of 42 for the training shuffle. Print the class_to_idx mapping and the number of images in each split when run as a script.
EXPECTED OUTPUT: importable loaders plus a printed summary showing the three classes and their counts.
```

### 2. Run it and confirm the class mapping. Note which integer maps to which class name — you will need it to read the confusion matrix.

If the counts are badly imbalanced, note it now — it changes how you must read the accuracy later. The generated set is roughly balanced, so a majority-class baseline should sit near a third.

**COMMAND** — run this in your terminal:

```bash
python datasets.py
```

### 3. Prompt for the CNN. Give it the flatten size you derived in Lab 12 rather than letting it guess.

64*8*8 = 4096, which is the number you derived in Lab 12: three MaxPool2d(2) layers take 64 to 32 to 16 to 8, with 64 channels at the end. If you let the assistant guess this number it will often be wrong, and the error appears only at the first forward pass.

**PROMPT** — paste this into your AI coding assistant:

```text
Add class DefectCNN(nn.Module) to models.py.
SHAPES: input batches are (N, 1, 64, 64); there are 3 output classes.
ARCHITECTURE: Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(32,64,3,padding=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(64*8*8, 128) -> ReLU -> Linear(128, 3).
CONSTRAINTS: the final layer must output RAW LOGITS with no softmax, because training uses nn.CrossEntropyLoss. Add a comment above the Flatten deriving 64*8*8 from three poolings of a 64x64 input. Include a __main__ block that runs a random (4,1,64,64) tensor through the model and prints the shape after each block to prove the arithmetic.
```

### 4. Before training, run the model's self-test and confirm the printed shapes match your own derivation.

The self-test is the cheapest possible verification — a random tensor and a set of printed shapes, no data loading and no training. If the flatten size is wrong you find out in one second rather than after a five-minute training run.

**COMMAND** — run this in your terminal:

```bash
python models.py
```

### 5. Prompt for the training script, reusing the engine rather than writing another loop.

Reusing engine.fit() is the discipline being tested. If the assistant writes a fresh loop anyway, follow up: 'Use fit from engine.py. Do not write a training loop in this file.' Every loop you avoid writing is a loop that cannot be missing zero_grad().

**PROMPT** — paste this into your AI coding assistant:

```text
Create train_cnn.py that trains DefectCNN using fit and evaluate from engine.py - do NOT write a new training loop.
OPERATIONS: get the loaders from datasets.py; instantiate DefectCNN with manual seed 42; train for 25 epochs with nn.CrossEntropyLoss, Adam at lr=1e-3, and an accuracy metric function; print the majority-class baseline accuracy on the validation set BEFORE training starts; after training report final validation accuracy, per-class precision and recall, and save a labelled confusion matrix to reports/cnn_confusion.png; save a checkpoint to models/defect_cnn_v1.pt using save_checkpoint from checkpoint.py; save the train and validation loss curves to reports/cnn_curve.png.
CONSTRAINTS: evaluation under model.eval() and torch.no_grad() - which fit already handles. Use the class names from class_to_idx on the confusion matrix axes.
```

### 6. Train it. On CPU this takes a few minutes — watch the two loss curves as they print.

Expect training accuracy to climb faster than validation accuracy — that widening gap is overfitting starting, and it is exactly what Lab 14 measures. Do not fix it yet. If validation accuracy is stuck near the baseline, check the logits are raw and the learning rate is sensible.

**COMMAND** — run this in your terminal:

```bash
python train_cnn.py
```

### 7. Compare final accuracy against the majority baseline, then read the confusion matrix to see which two classes the model confuses.

A good result is clearly above the baseline. The confusion the model usually shows is dent against ok, because a faint dent is genuinely low-contrast — which matches what you saw yourself in the sample grid in Lab 12. When your model's mistakes match your own, that is a sign the pipeline is sound.

### 8. Look at what it got wrong, which tells you more than any single number.

Confidently wrong predictions are the interesting ones. If the model is 95% sure an ok part is a dent, look at that image — often there is a real artefact in it, and occasionally you will find a genuinely mislabelled example, which is a normal and important discovery in real datasets.

**PROMPT** — paste this into your AI coding assistant:

```text
Create inspect_errors.py that loads models/defect_cnn_v1.pt, runs the validation set, selects up to 12 misclassified images, and saves a grid to reports/cnn_errors.png where each subplot title shows the true class, the predicted class and the model's confidence in its wrong answer. Sort them by confidence so the most confidently wrong images appear first.
```

## Verification — Test it

datasets.py prints three classes with their counts and the class_to_idx mapping. models.py's self-test prints per-block shapes ending in a flatten of 4096. train_cnn.py reports a validation accuracy clearly above the printed majority-class baseline, saves reports/cnn_confusion.png with named axes, and writes models/defect_cnn_v1.pt. reports/cnn_errors.png shows misclassified images with confidences.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| RuntimeError: mat1 and mat2 shapes cannot be multiplied | The Linear input size is wrong. Run the models.py self-test and read the shape printed just before Flatten — that product is the number the Linear layer needs. |
| RuntimeError: expected input[N, 3, 64, 64] to have 1 channel | The PNGs loaded as RGB. Confirm Grayscale(1) is first in the transform pipeline, before ToTensor. |
| Validation accuracy stays at the baseline | Check the output layer is raw logits with no softmax, then raise the learning rate or train longer. Also confirm the loaders are not returning the same class repeatedly. |
| Training is very slow | Reduce batch size or image count, or set num_workers=0 on Windows — multiprocessing workers often stall there. A few minutes on CPU is normal. |
| The confusion matrix axes are numbers, not class names | Pass the inverted class_to_idx as tick labels. An unlabelled confusion matrix cannot be read reliably. |

## Going further (optional)

- Add a fourth convolutional block and see whether the extra capacity helps or just overfits faster.
- Replace the three MaxPools with stride-2 convolutions and compare parameter count and accuracy.
- Visualise the learned first-layer kernels and compare them with the hand-built edge detectors from Lab 12.

---

[← Lab 12: Overview of CNNs: Convolution, Pooling and Padding](../lab12-overview-of-cnns-convolution-pooling-and-padding/README.md) · [All labs](../README.md) · [Lab 14: Diagnosing Overfitting with AI Assistance →](../lab14-diagnosing-overfitting-with-ai-assistance/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
