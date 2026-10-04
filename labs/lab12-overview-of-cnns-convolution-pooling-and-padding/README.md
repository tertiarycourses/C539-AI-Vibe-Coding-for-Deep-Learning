# Lab 12 — Overview of CNNs: Convolution, Pooling and Padding

> **Course:** AI Vibe Coding for Deep Learning (`C539`) · **Topic 03:** Vibe Coding Convolutional Neural Networks  
> **Learning outcome:** Reason about convolution, padding, stride and pooling well enough to predict a feature map's shape before running it.

## Goal

ForgeSight gains a camera. You generate the surface-inspection image set — 1200 greyscale images of machined parts labelled ok, scratch or dent — then build a shape explorer rather than a model. You apply Conv2d and MaxPool2d with different kernel sizes, paddings and strides, predicting each output shape from the formula before you print it. You visualise what an edge-detecting kernel actually does to a scratch image, and you finish by deriving the flatten size that a classifier's first Linear layer needs — the number that causes more CNN shape errors than anything else.

## Why this lab matters

A CNN is mostly an exercise in shape bookkeeping. The 'shape mismatch at the first Linear layer' error stops more learners than any other, and it is entirely predictable from the output-size formula. Ten minutes with the formula now saves an hour of guessing in the next three labs.

## What you'll build

**A generated 1200-image defect dataset plus a conv_explorer.py that predicts and verifies every feature-map shape**

**Tools:** PyTorch, torchvision, Pillow, matplotlib, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `make_images.py`
- `data/defects/`
- `conv_explorer.py`
- `reports/feature_maps.png`

## Before you start

- Lab 11 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Generate the ForgeSight surface-inspection images. Copy make_images.py from the course resources and run it.

make_images.py synthesises the dataset deterministically with a fixed seed, so everyone in the room gets identical images. It creates data/defects/train/{ok,scratch,dent} and data/defects/val/{ok,scratch,dent}, roughly 1200 images at 64x64 greyscale. It takes under a minute and needs no internet.

**COMMAND** — run this in your terminal:

```bash
python make_images.py
```

### 2. Inspect what you generated before you model it. Never train on a dataset you have not looked at.

Counting per class matters because an imbalanced image set produces the same flattering-accuracy trap you met in Lab 9. ToTensor() also rescales pixel values from 0-255 into 0-1 and moves the channel axis to the front, giving (1, 64, 64) — note both changes, because forgetting the rescale is a classic bug.

**PROMPT** — paste this into your AI coding assistant:

```text
Create peek_images.py that loads the generated dataset from data/defects/ and: (1) prints how many images are in each class folder for both train and val; (2) loads one image and prints its size, mode and the shape it becomes as a tensor via torchvision.transforms.ToTensor(); (3) saves a 3x4 grid figure to reports/sample_defects.png showing four examples of each of the three classes with the class name as the subplot title.
CONSTRAINTS: use PIL and torchvision only. Do not train anything.
```

### 3. Run it, open reports/sample_defects.png, and describe out loud what visually distinguishes a scratch from a dent. If you cannot see it, the model will struggle too.

Scratches are thin, high-contrast, directional lines; dents are softer, rounder, lower-contrast blobs. That difference is precisely what a convolutional kernel is good at picking up, and it is why the edge detector in the last step will light up on scratches.

**COMMAND** — run this in your terminal:

```bash
python peek_images.py
```

### 4. Now the shape work. Write down the output-size formula and predict three shapes on paper before prompting.

The formula is out = floor((W - K + 2P) / S) + 1, where W is the input size, K the kernel size, P the padding and S the stride. Predict (a), (b) and (d) now: with W=64 they come out as 62, 64 and 32. The 'same padding' rule worth memorising is P = (K-1)/2 for odd K, which is why 3 pairs with padding 1 and 5 pairs with padding 2.

### 5. Prompt for the explorer, which forces you to commit to a prediction before it reveals the answer.

The assert is what makes this a lab rather than a demo. If an assertion fails, the formula in the script is wrong, not PyTorch — read which configuration failed and recompute by hand. Configurations (b) and (c) both preserve 64x64; that is 'same' padding.

**PROMPT** — paste this into your AI coding assistant:

```text
Create conv_explorer.py demonstrating how convolution and pooling change tensor shape.
SHAPES: the input is a batch of 8 greyscale images of shape (8, 1, 64, 64).
TASK: show the output shape of each operation and confirm it against the analytic formula.
OPERATIONS: for each of these configurations, print the config, the analytically computed output size using the formula floor((W - K + 2P)/S) + 1, and the ACTUAL shape from running the layer, then assert they agree:
(a) Conv2d(1, 8, kernel_size=3, padding=0, stride=1)
(b) Conv2d(1, 8, kernel_size=3, padding=1, stride=1)
(c) Conv2d(1, 8, kernel_size=5, padding=2, stride=1)
(d) Conv2d(1, 8, kernel_size=3, padding=1, stride=2)
(e) MaxPool2d(kernel_size=2, stride=2) applied after (b)
Then build a small stack Conv(1,8,3,p=1) -> ReLU -> MaxPool(2) -> Conv(8,16,3,p=1) -> ReLU -> MaxPool(2) and print the shape after every single layer, finishing with the number of features a Flatten would produce.
CONSTRAINTS: no training, no gradients, manual seed 42. Print a comment line explaining which configuration preserves spatial size and why.
```

### 6. Run it and check your three paper predictions against the printed shapes.

The stack should print something like (8,8,64,64) -> (8,8,32,32) -> (8,16,32,32) -> (8,16,16,16), giving a flatten of 16*16*16 = 4096 features. Notice the pattern: channels double as spatial size halves. The network is trading WHERE information for WHAT information as it goes deeper.

**COMMAND** — run this in your terminal:

```bash
python conv_explorer.py
```

### 7. Visualise what a convolution actually computes, so the operation stops being abstract.

Setting the kernel weights by hand is the point. A learned CNN discovers kernels like these on its own — seeing a hand-built edge detector respond to a scratch makes the first convolutional layer concrete rather than magical.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a section to conv_explorer.py that takes one scratch image and one ok image from data/defects/, applies three FIXED 3x3 kernels - a horizontal edge detector, a vertical edge detector, and a blur - using F.conv2d with manually set weights rather than learned ones, and saves a figure to reports/feature_maps.png showing the original next to the three responses for both images. Add a comment on which kernel responds most strongly to a scratch and why.
```

### 8. Run it, open the figure, and note which kernel makes the scratch most visible. Record the final flatten size from the stack — you need it in Lab 13.

Write the flatten size in prompts.md. In Lab 13 the first Linear layer must accept exactly this number, and getting it wrong is the single most common CNN error. Now you can derive it instead of guessing.

**COMMAND** — run this in your terminal:

```bash
python conv_explorer.py
```

## Verification — Test it

data/defects/ contains train and val folders with three class subfolders each and roughly 1200 images total. conv_explorer.py prints predicted and actual shapes for all five configurations with every assertion passing, prints the full stack's per-layer shapes ending in a flatten size of 4096, and reports/feature_maps.png shows the edge kernels responding to a scratch.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| make_images.py fails with a Pillow import error | Install it: `pip install pillow`. torchvision usually pulls it in, so this means the venv is not active. |
| An assertion in conv_explorer.py fails | The analytic formula in the script is wrong. Print W, K, P and S for that configuration and recompute by hand — do not delete the assert. |
| The flatten number does not match 4096 | Count the pooling layers. Each MaxPool2d(2) halves both spatial dimensions, so two pools take 64 down to 16, and 16 channels times 16 times 16 is 4096. |
| The feature-map figure is entirely black or white | The response was not normalised for display. Ask for each response to be min-max scaled to 0-1 before imshow, and use cmap='gray'. |
| RuntimeError: expected input to have 1 channel but got 3 | The images loaded as RGB. Convert with `.convert('L')` in the loader, or add transforms.Grayscale(num_output_channels=1). |

## Going further (optional)

- Add a dilated convolution to the explorer and work out how dilation changes the output-size formula.
- Replace MaxPool with AvgPool in the stack and compare what the feature maps look like.
- Compute the receptive field of the final layer in the stack and check it against the size of a typical scratch.

---

[← Lab 11: Saving, Loading and Iterating on Models](../lab11-saving-loading-and-iterating-on-models/README.md) · [All labs](../README.md) · [Lab 13: Vibe Coding a CNN Image Classifier →](../lab13-vibe-coding-a-cnn-image-classifier/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.1 · 4 October 2026_
