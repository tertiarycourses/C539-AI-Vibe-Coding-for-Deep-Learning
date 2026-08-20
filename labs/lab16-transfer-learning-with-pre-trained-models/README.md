# Lab 16 — Transfer Learning with Pre-Trained Models

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 03:** Vibe Coding Convolutional Neural Networks  
> **Learning outcome:** Fine-tune a pre-trained network by freezing its backbone and replacing the classifier head.

## Goal

Stop training from scratch. You load a ResNet-18 pre-trained on ImageNet, adapt it to greyscale 64x64 inspection images, freeze the backbone so only a new three-class head learns, and fine-tune. You then unfreeze the last block at a much lower learning rate and compare all three approaches — scratch CNN, frozen backbone, partial fine-tune — on accuracy, training time and trainable parameter count. The lab makes concrete why transfer learning is the default professional starting point when you have hundreds of images rather than hundreds of thousands.

## Why this lab matters

Almost no production vision model is trained from scratch. Knowing how to freeze, replace a head and choose a fine-tuning learning rate is the difference between needing a hundred thousand labelled images and needing a few hundred.

## What you'll build

**A fine-tuned ResNet-18 defect classifier plus a three-way comparison against your scratch CNN**

**Tools:** PyTorch, torchvision.models, engine.py, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `transfer.py`
- `models/defect_resnet_v1.pt`
- `reports/transfer_comparison.csv`

## Before you start

- Lab 15 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Prompt for the transfer setup, being explicit about the two adaptations a greyscale 64x64 input requires.

There are two legitimate ways to feed greyscale 64x64 images to a ResNet: repeat the single channel three times and resize to 224x224 to match ImageNet exactly, or surgically replace the first conv layer to accept one channel. The first reuses the pre-trained weights fully and is the right default; the second discards the first layer's learned filters. Making the assistant state which it chose forces the decision into the open.

**PROMPT** — paste this into your AI coding assistant:

```text
Create transfer.py that fine-tunes a pre-trained ResNet-18 on the defect images.
TASK: adapt an ImageNet ResNet-18 to 3-class greyscale 64x64 defect classification.
OPERATIONS: (1) load torchvision.models.resnet18 with the default pre-trained weights; (2) adapt the input - the simplest correct approach is a transform pipeline that converts the greyscale image to 3 channels and resizes to 224x224 using the ImageNet normalisation statistics, so state clearly in a comment which approach you used and why; (3) FREEZE every parameter by setting requires_grad=False; (4) replace model.fc with a new nn.Linear(512, 3) whose parameters are trainable; (5) print the total parameter count and the TRAINABLE parameter count so the difference is obvious; (6) train for 12 epochs with Adam lr=1e-3 using fit from engine.py, passing only the trainable parameters to the optimizer.
CONSTRAINTS: the new head outputs raw logits. Use the augmented training transform and the deterministic validation transform from Lab 15, adjusted for 3-channel 224x224 ImageNet input.
EXPECTED OUTPUT: printed parameter counts, per-epoch losses, final validation accuracy, and the elapsed training time.
```

### 2. Before training, check the parameter counts. The trainable number should be a tiny fraction of the total.

ResNet-18 has around 11 million parameters; the new head has about 1,500. Training under 0.02% of the network is why this runs quickly and why it does not overfit 1200 images. If the trainable count is in the millions, the freeze did not take — check requires_grad was set before fc was replaced.

**COMMAND** — run this in your terminal:

```bash
python transfer.py
```

### 3. Note the accuracy and time, then compare against your Lab 15 scratch model trained for far longer.

Expect the frozen backbone to reach comparable or better accuracy than your scratch CNN in a fraction of the epochs, though each epoch is slower because 224x224 inputs are twelve times larger than 64x64. That trade-off is exactly what the comparison table is for.

### 4. Now unfreeze the last block and fine-tune at a much lower learning rate.

The differential learning rate is the professional detail. The pre-trained features took an enormous amount of compute to learn; a 1e-3 update would wreck them in a single epoch. Small learning rate for pre-trained weights, larger for the randomly initialised head, is the rule.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a function partial_finetune() to transfer.py.
OPERATIONS: start from the frozen model trained above, then set requires_grad=True for layer4 and fc only. Use two parameter groups in Adam - layer4 at lr=1e-4 and fc at lr=1e-3 - and train for a further 8 epochs. Print the new trainable parameter count and the final validation accuracy.
Add a comment explaining why the unfrozen backbone layer uses a learning rate roughly ten times smaller than the head: the pre-trained weights are already good, and a large update would destroy the features you are trying to reuse.
CONSTRAINTS: do not re-initialise the head - continue from the weights just trained.
```

### 5. Run the partial fine-tune and record whether the extra 8 epochs improved on the frozen result.

Partial fine-tuning sometimes helps meaningfully and sometimes barely moves the number, particularly when your images look nothing like ImageNet photographs. Synthetic greyscale inspection images are quite far from ImageNet, so a modest gain is a perfectly honest result to report.

**COMMAND** — run this in your terminal:

```bash
python transfer.py
```

### 6. Build the three-way comparison table that answers the practical question.

The 'accuracy per minute' column is the one that changes decisions. In a factory with no GPU, an approach that reaches 88% in four minutes usually beats one that reaches 90% in forty. Say which trade-off you are making rather than just picking the top accuracy.

**PROMPT** — paste this into your AI coding assistant:

```text
Add a comparison to transfer.py that produces reports/transfer_comparison.csv with one row per approach: (1) DefectCNNv2 trained from scratch with the Lab 15 best configuration; (2) ResNet-18 frozen backbone with a new head; (3) ResNet-18 with layer4 unfrozen. Columns: approach, trainable parameters, epochs trained, wall-clock training seconds, best validation accuracy, and validation accuracy per minute of training. Print the table sorted by best validation accuracy and add a one-line printed conclusion naming which approach you would choose for a factory with 1200 labelled images and no GPU.
```

### 7. Save the best transfer model with a checkpoint and a model card entry.

Recording which layers were trainable is essential provenance — 'ResNet-18 fine-tuned' is not reproducible, whereas 'ResNet-18, layer4 and fc trainable, lr 1e-4 and 1e-3, 20 epochs total' is.

**PROMPT** — paste this into your AI coding assistant:

```text
Update transfer.py to save the best-performing transfer model with save_checkpoint to models/defect_resnet_v1.pt, recording in the metrics dict the approach used, which layers were trainable, the validation accuracy and the training time. Append a section to models/defect_model_card.md comparing the scratch CNN and the transfer model, and stating which one you recommend for deployment and why.
```

### 8. Read the comparison table and state your recommendation out loud, with the trade-off you are accepting.

There is no single right answer here, and the trainer will push you to defend yours. Model size, inference speed on the plant's hardware, and the cost of a missed defect all belong in the argument alongside accuracy.

## Verification — Test it

transfer.py prints a trainable parameter count that is a tiny fraction of the ~11M total, trains a frozen-backbone ResNet-18 to a validation accuracy comparable with or better than your scratch CNN, and reports the partial fine-tune result. reports/transfer_comparison.csv holds three rows with parameters, time and accuracy. models/defect_resnet_v1.pt is saved and the model card names a recommendation.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Downloading the pre-trained weights fails | No internet or a blocked host. Ask the trainer for the cached weights file, or set weights=None and note honestly that the run is no longer transfer learning. |
| RuntimeError: expected input[N, 1, 64, 64] to have 3 channels | The transform is not converting to 3 channels. Add Grayscale(num_output_channels=3) and Resize(224) before ToTensor. |
| The trainable parameter count equals the total | requires_grad=False was applied after fc was replaced, or not at all. Freeze first, then replace the head. |
| Training is much slower than the scratch CNN | Expected — 224x224 inputs are far larger. Reduce to Resize(128) and note the accuracy change, or lower the batch size. |
| Accuracy is stuck near the baseline | The optimizer probably received all parameters including frozen ones. Pass only `filter(lambda p: p.requires_grad, model.parameters())`. |

## Going further (optional)

- Try ResNet-34 or MobileNetV3-Small and add them to the comparison table.
- Freeze everything except the final BatchNorm layers and see how that compares to unfreezing layer4.
- Measure single-image inference time for the scratch CNN and the ResNet, and add it to the deployment argument.

---

[← Lab 15: Data Augmentation and Regularization via Prompts](../lab15-data-augmentation-and-regularization-via-prompts/README.md) · [All labs](../README.md) · [Lab 17: Overview of RNNs, LSTM and GRU →](../lab17-overview-of-rnns-lstm-and-gru/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
