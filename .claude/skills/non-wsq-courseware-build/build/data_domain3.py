"""
Topic 3 hands-on activities - Vibe Coding Convolutional Neural Networks.

Labs 12-16, mapping 1:1 onto the five published sub-topics. Single source for the
PPT activity/step slides, the Learner Guide sections, the Lesson Plan schedule
rows and the labs/labNN-*/README.md files.
"""

DOMAIN3 = [

 dict(
  num=12, topic=3,
  title="Overview of CNNs: Convolution, Pooling and Padding",
  objective="reason about convolution, padding, stride and pooling well enough to predict a feature map's shape before running it",
  desc=("ForgeSight gains a camera. You generate the surface-inspection image set — 1200 greyscale images of "
        "machined parts labelled ok, scratch or dent — then build a shape explorer rather than a model. You "
        "apply Conv2d and MaxPool2d with different kernel sizes, paddings and strides, predicting each output "
        "shape from the formula before you print it. You visualise what an edge-detecting kernel actually does "
        "to a scratch image, and you finish by deriving the flatten size that a classifier's first Linear "
        "layer needs — the number that causes more CNN shape errors than anything else."),
  build="A generated 1200-image defect dataset plus a conv_explorer.py that predicts and verifies every feature-map shape",
  services="PyTorch, torchvision, Pillow, matplotlib, Cursor / GitHub Copilot / Claude",
  why=("A CNN is mostly an exercise in shape bookkeeping. The 'shape mismatch at the first Linear layer' error "
       "stops more learners than any other, and it is entirely predictable from the output-size formula. Ten "
       "minutes with the formula now saves an hour of guessing in the next three labs."),
  files=["make_images.py", "data/defects/", "conv_explorer.py", "reports/feature_maps.png"],
  steps=[
   ("Generate the ForgeSight surface-inspection images. Copy make_images.py from the course resources and run it.",
    "python make_images.py"),
   ("Inspect what you generated before you model it. Never train on a dataset you have not looked at.",
    "Create peek_images.py that loads the generated dataset from data/defects/ and: (1) prints how many images are in each class folder for both train and val; (2) loads one image and prints its size, mode and the shape it becomes as a tensor via torchvision.transforms.ToTensor(); (3) saves a 3x4 grid figure to reports/sample_defects.png showing four examples of each of the three classes with the class name as the subplot title.\nCONSTRAINTS: use PIL and torchvision only. Do not train anything."),
   ("Run it, open reports/sample_defects.png, and describe out loud what visually distinguishes a scratch from a dent. If you cannot see it, the model will struggle too.",
    "python peek_images.py"),
   ("Now the shape work. Write down the output-size formula and predict three shapes on paper before prompting.",
    ""),
   ("Prompt for the explorer, which forces you to commit to a prediction before it reveals the answer.",
    "Create conv_explorer.py demonstrating how convolution and pooling change tensor shape.\nSHAPES: the input is a batch of 8 greyscale images of shape (8, 1, 64, 64).\nTASK: show the output shape of each operation and confirm it against the analytic formula.\nOPERATIONS: for each of these configurations, print the config, the analytically computed output size using the formula floor((W - K + 2P)/S) + 1, and the ACTUAL shape from running the layer, then assert they agree:\n(a) Conv2d(1, 8, kernel_size=3, padding=0, stride=1)\n(b) Conv2d(1, 8, kernel_size=3, padding=1, stride=1)\n(c) Conv2d(1, 8, kernel_size=5, padding=2, stride=1)\n(d) Conv2d(1, 8, kernel_size=3, padding=1, stride=2)\n(e) MaxPool2d(kernel_size=2, stride=2) applied after (b)\nThen build a small stack Conv(1,8,3,p=1) -> ReLU -> MaxPool(2) -> Conv(8,16,3,p=1) -> ReLU -> MaxPool(2) and print the shape after every single layer, finishing with the number of features a Flatten would produce.\nCONSTRAINTS: no training, no gradients, manual seed 42. Print a comment line explaining which configuration preserves spatial size and why."),
   ("Run it and check your three paper predictions against the printed shapes.",
    "python conv_explorer.py"),
   ("Visualise what a convolution actually computes, so the operation stops being abstract.",
    "Add a section to conv_explorer.py that takes one scratch image and one ok image from data/defects/, applies three FIXED 3x3 kernels - a horizontal edge detector, a vertical edge detector, and a blur - using F.conv2d with manually set weights rather than learned ones, and saves a figure to reports/feature_maps.png showing the original next to the three responses for both images. Add a comment on which kernel responds most strongly to a scratch and why."),
   ("Run it, open the figure, and note which kernel makes the scratch most visible. Record the final flatten size from the stack — you need it in Lab 13.",
    "python conv_explorer.py"),
  ],
  notes=[
   ("make_images.py synthesises the dataset deterministically with a fixed seed, so everyone in the room gets "
    "identical images. It creates data/defects/train/{ok,scratch,dent} and data/defects/val/{ok,scratch,dent}, "
    "roughly 1200 images at 64x64 greyscale. It takes under a minute and needs no internet."),
   ("Counting per class matters because an imbalanced image set produces the same flattering-accuracy trap you "
    "met in Lab 9. ToTensor() also rescales pixel values from 0-255 into 0-1 and moves the channel axis to the "
    "front, giving (1, 64, 64) — note both changes, because forgetting the rescale is a classic bug."),
   ("Scratches are thin, high-contrast, directional lines; dents are softer, rounder, lower-contrast blobs. "
    "That difference is precisely what a convolutional kernel is good at picking up, and it is why the edge "
    "detector in the last step will light up on scratches."),
   ("The formula is out = floor((W - K + 2P) / S) + 1, where W is the input size, K the kernel size, P the "
    "padding and S the stride. Predict (a), (b) and (d) now: with W=64 they come out as 62, 64 and 32. The "
    "'same padding' rule worth memorising is P = (K-1)/2 for odd K, which is why 3 pairs with padding 1 and 5 "
    "pairs with padding 2."),
   ("The assert is what makes this a lab rather than a demo. If an assertion fails, the formula in the script "
    "is wrong, not PyTorch — read which configuration failed and recompute by hand. Configurations (b) and (c) "
    "both preserve 64x64; that is 'same' padding."),
   ("The stack should print something like (8,8,64,64) -> (8,8,32,32) -> (8,16,32,32) -> (8,16,16,16), giving a "
    "flatten of 16*16*16 = 4096 features. Notice the pattern: channels double as spatial size halves. The "
    "network is trading WHERE information for WHAT information as it goes deeper."),
   ("Setting the kernel weights by hand is the point. A learned CNN discovers kernels like these on its own — "
    "seeing a hand-built edge detector respond to a scratch makes the first convolutional layer concrete "
    "rather than magical."),
   ("Write the flatten size in prompts.md. In Lab 13 the first Linear layer must accept exactly this number, "
    "and getting it wrong is the single most common CNN error. Now you can derive it instead of guessing."),
  ],
  test=("data/defects/ contains train and val folders with three class subfolders each and roughly 1200 images "
        "total. conv_explorer.py prints predicted and actual shapes for all five configurations with every "
        "assertion passing, prints the full stack's per-layer shapes ending in a flatten size of 4096, and "
        "reports/feature_maps.png shows the edge kernels responding to a scratch."),
  troubleshoot=[
   ("make_images.py fails with a Pillow import error", "Install it: `pip install pillow`. torchvision usually pulls it in, so this means the venv is not active."),
   ("An assertion in conv_explorer.py fails", "The analytic formula in the script is wrong. Print W, K, P and S for that configuration and recompute by hand — do not delete the assert."),
   ("The flatten number does not match 4096", "Count the pooling layers. Each MaxPool2d(2) halves both spatial dimensions, so two pools take 64 down to 16, and 16 channels times 16 times 16 is 4096."),
   ("The feature-map figure is entirely black or white", "The response was not normalised for display. Ask for each response to be min-max scaled to 0-1 before imshow, and use cmap='gray'."),
   ("RuntimeError: expected input to have 1 channel but got 3", "The images loaded as RGB. Convert with `.convert('L')` in the loader, or add transforms.Grayscale(num_output_channels=1)."),
  ],
  stretch=[
   "Add a dilated convolution to the explorer and work out how dilation changes the output-size formula.",
   "Replace MaxPool with AvgPool in the stack and compare what the feature maps look like.",
   "Compute the receptive field of the final layer in the stack and check it against the size of a typical scratch.",
  ],
 ),

 dict(
  num=13, topic=3,
  title="Vibe Coding a CNN Image Classifier",
  objective="build, train and evaluate a convolutional network that classifies images into three defect classes",
  desc=("Turn the shape knowledge into a working classifier. You prompt for an ImageFolder-based data pipeline "
        "and a DefectCNN whose first Linear layer takes exactly the flatten size you derived in Lab 12, then "
        "train it with the same engine.fit() you built in Lab 10 — no new training loop. You evaluate against "
        "a majority-class baseline with a confusion matrix, and you inspect the images the model gets wrong. "
        "Because the architecture, the loss and the loop are all things you have already validated, the only "
        "new failure surface is the data pipeline and the shape arithmetic."),
  build="A trained DefectCNN classifying ok, scratch and dent, evaluated with a confusion matrix and a misclassified-image grid",
  services="PyTorch, torchvision ImageFolder, engine.py, Cursor / GitHub Copilot / Claude",
  why=("This is the payoff for Labs 10 and 12: a genuinely different model type, built in one lab, because the "
       "training engine is reusable and the shapes are predictable. It is also the model you will spend the "
       "next three labs improving."),
  files=["datasets.py", "models.py", "train_cnn.py", "reports/cnn_confusion.png"],
  steps=[
   ("Prompt for the image data pipeline. Keep the transforms minimal for now — augmentation is deliberately Lab 15's job.",
    "Create datasets.py for the defect images.\nSHAPES: data/defects/train and data/defects/val each contain subfolders ok, scratch and dent holding 64x64 greyscale PNGs.\nTASK: return DataLoaders for training and validation.\nOPERATIONS: build a function get_defect_loaders(batch_size=32, data_dir='data/defects') that uses torchvision.datasets.ImageFolder with a transform pipeline of Grayscale(1) then ToTensor() then Normalize(mean=[0.5], std=[0.5]). Return train_loader (shuffled), val_loader (not shuffled) and the class_to_idx mapping.\nCONSTRAINTS: do NOT add any augmentation - that comes in a later lab. Set a fixed generator seed of 42 for the training shuffle. Print the class_to_idx mapping and the number of images in each split when run as a script.\nEXPECTED OUTPUT: importable loaders plus a printed summary showing the three classes and their counts."),
   ("Run it and confirm the class mapping. Note which integer maps to which class name — you will need it to read the confusion matrix.",
    "python datasets.py"),
   ("Prompt for the CNN. Give it the flatten size you derived in Lab 12 rather than letting it guess.",
    "Add class DefectCNN(nn.Module) to models.py.\nSHAPES: input batches are (N, 1, 64, 64); there are 3 output classes.\nARCHITECTURE: Conv2d(1,16,3,padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(16,32,3,padding=1) -> ReLU -> MaxPool2d(2) -> Conv2d(32,64,3,padding=1) -> ReLU -> MaxPool2d(2) -> Flatten -> Linear(64*8*8, 128) -> ReLU -> Linear(128, 3).\nCONSTRAINTS: the final layer must output RAW LOGITS with no softmax, because training uses nn.CrossEntropyLoss. Add a comment above the Flatten deriving 64*8*8 from three poolings of a 64x64 input. Include a __main__ block that runs a random (4,1,64,64) tensor through the model and prints the shape after each block to prove the arithmetic."),
   ("Before training, run the model's self-test and confirm the printed shapes match your own derivation.",
    "python models.py"),
   ("Prompt for the training script, reusing the engine rather than writing another loop.",
    "Create train_cnn.py that trains DefectCNN using fit and evaluate from engine.py - do NOT write a new training loop.\nOPERATIONS: get the loaders from datasets.py; instantiate DefectCNN with manual seed 42; train for 25 epochs with nn.CrossEntropyLoss, Adam at lr=1e-3, and an accuracy metric function; print the majority-class baseline accuracy on the validation set BEFORE training starts; after training report final validation accuracy, per-class precision and recall, and save a labelled confusion matrix to reports/cnn_confusion.png; save a checkpoint to models/defect_cnn_v1.pt using save_checkpoint from checkpoint.py; save the train and validation loss curves to reports/cnn_curve.png.\nCONSTRAINTS: evaluation under model.eval() and torch.no_grad() - which fit already handles. Use the class names from class_to_idx on the confusion matrix axes."),
   ("Train it. On CPU this takes a few minutes — watch the two loss curves as they print.",
    "python train_cnn.py"),
   ("Compare final accuracy against the majority baseline, then read the confusion matrix to see which two classes the model confuses.",
    ""),
   ("Look at what it got wrong, which tells you more than any single number.",
    "Create inspect_errors.py that loads models/defect_cnn_v1.pt, runs the validation set, selects up to 12 misclassified images, and saves a grid to reports/cnn_errors.png where each subplot title shows the true class, the predicted class and the model's confidence in its wrong answer. Sort them by confidence so the most confidently wrong images appear first."),
  ],
  notes=[
   ("ImageFolder assigns class indices alphabetically, so expect dent=0, ok=1, scratch=2 — not the order you "
    "would naturally write. Reading a confusion matrix with the wrong mapping in your head is a classic and "
    "entirely avoidable mistake. Normalize with mean 0.5 and std 0.5 maps the 0-1 pixel range to roughly "
    "-1 to 1, which is a reasonable default for a network with ReLU activations."),
   ("If the counts are badly imbalanced, note it now — it changes how you must read the accuracy later. "
    "The generated set is roughly balanced, so a majority-class baseline should sit near a third."),
   ("64*8*8 = 4096, which is the number you derived in Lab 12: three MaxPool2d(2) layers take 64 to 32 to 16 to "
    "8, with 64 channels at the end. If you let the assistant guess this number it will often be wrong, and "
    "the error appears only at the first forward pass."),
   ("The self-test is the cheapest possible verification — a random tensor and a set of printed shapes, no data "
    "loading and no training. If the flatten size is wrong you find out in one second rather than after a "
    "five-minute training run."),
   ("Reusing engine.fit() is the discipline being tested. If the assistant writes a fresh loop anyway, follow "
    "up: 'Use fit from engine.py. Do not write a training loop in this file.' Every loop you avoid writing is "
    "a loop that cannot be missing zero_grad()."),
   ("Expect training accuracy to climb faster than validation accuracy — that widening gap is overfitting "
    "starting, and it is exactly what Lab 14 measures. Do not fix it yet. If validation accuracy is stuck near "
    "the baseline, check the logits are raw and the learning rate is sensible."),
   ("A good result is clearly above the baseline. The confusion the model usually shows is dent against ok, "
    "because a faint dent is genuinely low-contrast — which matches what you saw yourself in the sample grid "
    "in Lab 12. When your model's mistakes match your own, that is a sign the pipeline is sound."),
   ("Confidently wrong predictions are the interesting ones. If the model is 95% sure an ok part is a dent, "
    "look at that image — often there is a real artefact in it, and occasionally you will find a genuinely "
    "mislabelled example, which is a normal and important discovery in real datasets."),
  ],
  test=("datasets.py prints three classes with their counts and the class_to_idx mapping. models.py's self-test "
        "prints per-block shapes ending in a flatten of 4096. train_cnn.py reports a validation accuracy "
        "clearly above the printed majority-class baseline, saves reports/cnn_confusion.png with named axes, "
        "and writes models/defect_cnn_v1.pt. reports/cnn_errors.png shows misclassified images with confidences."),
  troubleshoot=[
   ("RuntimeError: mat1 and mat2 shapes cannot be multiplied", "The Linear input size is wrong. Run the models.py self-test and read the shape printed just before Flatten — that product is the number the Linear layer needs."),
   ("RuntimeError: expected input[N, 3, 64, 64] to have 1 channel", "The PNGs loaded as RGB. Confirm Grayscale(1) is first in the transform pipeline, before ToTensor."),
   ("Validation accuracy stays at the baseline", "Check the output layer is raw logits with no softmax, then raise the learning rate or train longer. Also confirm the loaders are not returning the same class repeatedly."),
   ("Training is very slow", "Reduce batch size or image count, or set num_workers=0 on Windows — multiprocessing workers often stall there. A few minutes on CPU is normal."),
   ("The confusion matrix axes are numbers, not class names", "Pass the inverted class_to_idx as tick labels. An unlabelled confusion matrix cannot be read reliably."),
  ],
  stretch=[
   "Add a fourth convolutional block and see whether the extra capacity helps or just overfits faster.",
   "Replace the three MaxPools with stride-2 convolutions and compare parameter count and accuracy.",
   "Visualise the learned first-layer kernels and compare them with the hand-built edge detectors from Lab 12.",
  ],
 ),

 dict(
  num=14, topic=3,
  title="Diagnosing Overfitting with AI Assistance",
  objective="recognise overfitting from training and validation curves and locate the epoch where generalisation stops improving",
  desc=("Your Lab 13 model almost certainly overfits — now measure it rather than guess. You train the same CNN "
        "deliberately hard and plot training against validation loss and accuracy on shared axes, then identify "
        "the exact epoch where validation stops improving while training keeps falling. To make the pattern "
        "unmistakable you also train on a deliberately tiny subset until it reaches near-perfect training "
        "accuracy and useless validation accuracy. You finish by asking the assistant to diagnose the curves "
        "from a description alone, and you check its reasoning against what you can see."),
  build="An overfitting diagnosis with annotated curves, a memorised-subset demonstration, and a written diagnosis you verified yourself",
  services="PyTorch, matplotlib, engine.py, Cursor / GitHub Copilot / Claude",
  why=("Overfitting is invisible if you only watch training loss, which is the number that prints most often "
       "and always looks encouraging. Learning to read the GAP between two curves, and to spot the epoch where "
       "they diverge, is what tells you when to stop and what to fix."),
  files=["diagnose_overfit.py", "reports/overfit_curves.png", "reports/memorise.png", "review_notes.md"],
  steps=[
   ("Train long enough for the problem to appear. Prompt for a diagnostic run with no regularization at all.",
    "Create diagnose_overfit.py that trains DefectCNN for 60 epochs using fit from engine.py with NO augmentation, NO dropout and NO weight decay - deliberately unregularized.\nOPERATIONS: record train loss, validation loss, train accuracy and validation accuracy every epoch. Save a 2-panel figure to reports/overfit_curves.png - left panel both losses, right panel both accuracies, epochs on the x axis, with a vertical dashed line marking the epoch of MINIMUM validation loss and that epoch number in the legend. Print: the best validation loss and its epoch; the final training loss; the final validation loss; the final gap between training and validation accuracy in percentage points.\nCONSTRAINTS: manual seed 42, batch size 32, Adam lr=1e-3. Do not stop early - run all 60 epochs so the divergence is visible."),
   ("Run it. This takes several minutes on CPU — while it runs, write down what you expect the two curves to do.",
    "python diagnose_overfit.py"),
   ("Open reports/overfit_curves.png and answer three questions before reading any explanation: at which epoch does validation loss bottom out, what does training loss do after that, and how wide is the final accuracy gap?",
    ""),
   ("Now make the effect undeniable. Prompt for a memorisation demonstration on a tiny subset.",
    "Add a function memorise_demo() to diagnose_overfit.py.\nTASK: show overfitting in its purest form by training on far too little data.\nOPERATIONS: take only 30 training images (10 per class) using torch.utils.data.Subset, keep the FULL validation set, and train a fresh DefectCNN for 100 epochs with the same settings. Plot training and validation accuracy to reports/memorise.png and print both final accuracies.\nCONSTRAINTS: same seed and architecture as the main run - only the amount of training data changes.\nEXPECTED OUTPUT: training accuracy approaching 100% while validation accuracy stays near the majority-class baseline."),
   ("Run it and compare the two numbers. This is memorisation with nothing learned.",
    "python diagnose_overfit.py"),
   ("Test the assistant's diagnostic reasoning — and then test the assistant.",
    "I trained a CNN image classifier for 60 epochs with no regularization. Training loss fell steadily from 1.05 to 0.04 and training accuracy reached 99%. Validation loss fell until epoch 14, reaching 0.52, then rose steadily to 0.95 by epoch 60, while validation accuracy peaked at 78% around epoch 14 and drifted down to 71%.\nDiagnose what is happening, name the epoch I should have stopped at, and rank the following fixes by how much improvement you would expect for THIS symptom, with a one-line reason each: more training data, data augmentation, dropout, weight decay, early stopping, a smaller model, a lower learning rate.\nBe explicit about which of these address the cause and which only limit the damage."),
   ("Compare the assistant's ranking against your own curves. Does its recommended stopping epoch match the dashed line in your figure?",
    ""),
   ("Write the diagnosis into review_notes.md in your own words, with the specific numbers from YOUR run.",
    ""),
  ],
  notes=[
   ("Sixty epochs with no regularization is not how you would train a production model — it is how you make a "
    "phenomenon visible. Marking the minimum-validation-loss epoch on the plot is what turns a vague 'it "
    "overfits' into a specific, actionable number."),
   ("Expect validation loss to bottom out somewhere in the first third of training and then climb, while "
    "training loss keeps falling towards zero. The rising validation loss is the model becoming more confident "
    "about the training images specifically, which is the definition of overfitting."),
   ("The single most important reading: validation loss can rise while validation ACCURACY is still roughly "
    "flat. Loss is sensitive to confidence, accuracy only to the argmax. Loss turns first, which is why it is "
    "the better early-stopping signal."),
   ("Thirty images cannot possibly represent the variation in the full set, so the network simply memorises "
    "them. This is the same mechanism as the main run, just fast and obvious. It is also a genuinely useful "
    "debugging trick: a model that CANNOT reach high accuracy on 30 images has a bug, not a data problem."),
   ("Expect training accuracy near 100% and validation accuracy near the baseline. Nothing generalised. Keep "
    "reports/memorise.png — it is the clearest single picture of overfitting you will produce today."),
   ("The numbers in this prompt are illustrative; substitute your own if you prefer. What you are testing is "
    "whether the assistant distinguishes fixes that address the CAUSE — more data, augmentation, a smaller "
    "model — from those that only limit the damage, like early stopping. Early stopping does not make the "
    "model generalise better; it stops you shipping the worse version."),
   ("Assistants are generally strong at this diagnosis, which is worth noticing: they are good at pattern-"
    "matching a described symptom and weaker at spotting the same problem inside code they just wrote. Use "
    "them accordingly — describe symptoms to them, do not ask them to audit themselves."),
   ("Write your own numbers: best validation loss and its epoch, the final train/validation accuracy gap, and "
    "the two fixes you intend to apply in Lab 15. You will compare against these exact figures next lab."),
  ],
  test=("reports/overfit_curves.png shows training loss falling while validation loss turns upward, with a "
        "dashed line at the minimum-validation-loss epoch and that epoch printed. reports/memorise.png shows "
        "near-100% training accuracy against near-baseline validation accuracy on 30 images. review_notes.md "
        "records your best epoch, your accuracy gap and the fixes you will apply next."),
  troubleshoot=[
   ("Validation loss never rises within 60 epochs", "The model may be too small or the task too easy. Note it honestly, then use memorise_demo to show the effect instead — it is guaranteed to appear there."),
   ("The curves are too noisy to read", "Batch-to-batch noise. Plot a rolling mean over 3 epochs alongside the raw curve, or raise the batch size."),
   ("The memorisation demo does not reach high training accuracy", "Train longer or raise the learning rate. If 30 images still cannot be memorised in 100 epochs, there is a bug in the pipeline — a genuinely useful signal."),
   ("Training takes too long on CPU", "Reduce to 40 epochs and note the change, or shrink the images to 32x32 in the transform. The pattern still appears."),
   ("Subset produces a class-imbalanced sample", "Select indices per class explicitly rather than taking the first 30 rows, which would be all one class in a sorted ImageFolder."),
  ],
  stretch=[
   "Add a third panel plotting the train-validation gap directly, and see whether its shape is easier to read.",
   "Re-run with half the training data and confirm the divergence epoch arrives earlier.",
   "Track the mean absolute weight of the first Linear layer per epoch and see whether it grows as overfitting sets in.",
  ],
 ),

 dict(
  num=15, topic=3,
  title="Data Augmentation and Regularization via Prompts",
  objective="apply augmentation, dropout, weight decay and early stopping, and measure how much each narrows the overfitting gap",
  desc=("Fix what you measured. You add the four standard remedies one at a time — augmentation on the training "
        "set only, dropout in the classifier head, weight decay in the optimizer, and early stopping via the "
        "best-epoch tracking already in engine.fit() — and after each change you record the validation accuracy "
        "and the train/validation gap. The critical trap is deliberate: augmentation must never be applied to "
        "validation, and an assistant asked for 'augmented loaders' will very often apply it to both, which "
        "silently makes your validation score noisy and pessimistic."),
  build="A regularized training pipeline plus an ablation table showing the measured contribution of each remedy",
  services="PyTorch, torchvision.transforms, engine.py, Cursor / GitHub Copilot / Claude",
  why=("Applying all four fixes at once tells you nothing about which one mattered. Adding them one at a time, "
       "with the gap measured after each, is how you learn what actually helps for a given problem — and it is "
       "how you would justify the choices to a colleague."),
  files=["datasets.py", "models.py", "train_cnn_v2.py", "reports/ablation.csv"],
  steps=[
   ("Add augmented loaders, with an explicit instruction about which split gets augmented.",
    "Update datasets.py with a new function get_augmented_loaders(batch_size=32, data_dir='data/defects').\nTASK: return training and validation loaders where ONLY the training set is augmented.\nOPERATIONS: the TRAINING transform is RandomHorizontalFlip(0.5), RandomRotation(10), RandomResizedCrop(64, scale=(0.8, 1.0)), then Grayscale(1), ToTensor(), Normalize([0.5],[0.5]). The VALIDATION transform is ONLY Grayscale(1), ToTensor(), Normalize([0.5],[0.5]) with NO random operations whatsoever.\nCONSTRAINTS: this is the critical requirement - validation must be deterministic, because a randomly augmented validation set gives a different score every run and cannot be compared across experiments. Keep the original get_defect_loaders unchanged so the two can be compared.\nEXPECTED OUTPUT: a __main__ block that prints the two transform pipelines side by side so the difference is visible, and saves a grid of 8 augmented versions of the SAME training image to reports/augmented_samples.png."),
   ("Verify the trap did not catch you. Print both pipelines and confirm no random transform appears in the validation list.",
    "python datasets.py"),
   ("Open reports/augmented_samples.png. Check the augmentations are plausible — a rotation so extreme that a scratch becomes unrecognisable teaches the model nothing.",
    ""),
   ("Add dropout to the model, in the right place.",
    "Add class DefectCNNv2(nn.Module) to models.py, identical to DefectCNN but with nn.Dropout(p=0.3) inserted after the ReLU that follows Linear(4096, 128), and nn.Dropout2d(p=0.1) after the final MaxPool.\nAdd a comment explaining why dropout goes in the classifier head rather than between early convolutional layers, and a second comment noting that model.eval() disables it automatically - which is why evaluation must never run in train mode.\nKeep the output as raw logits with no softmax."),
   ("Run the ablation. This is the actual experiment — five configurations, one change at a time.",
    "Create train_cnn_v2.py running an ablation over regularization, using fit from engine.py.\nRUNS - each 40 epochs, fresh model, manual seed 42 reset before each:\n1. baseline: DefectCNN, plain loaders, Adam lr=1e-3, weight_decay=0\n2. plus augmentation: DefectCNN, augmented loaders\n3. plus dropout: DefectCNNv2, augmented loaders\n4. plus weight decay: DefectCNNv2, augmented loaders, weight_decay=1e-4\n5. plus early stopping: as run 4 but reporting the metrics from the BEST validation epoch rather than the last\nFor each run record: best validation loss and its epoch, final validation accuracy, best validation accuracy, final training accuracy, and the train-validation accuracy gap in percentage points.\nEXPECTED OUTPUT: a printed table in run order, saved to reports/ablation.csv, plus a figure overlaying the five validation loss curves with a legend."),
   ("Run the ablation and read the gap column down the table. Which single change reduced the gap most?",
    "python train_cnn_v2.py"),
   ("Save the best configuration as the new ForgeSight defect model.",
    "Update train_cnn_v2.py to retrain the winning configuration and save it with save_checkpoint to models/defect_cnn_v2.pt, including in the metrics dict the validation accuracy, the best epoch, and a regularization field listing which techniques were used. Update models/defect_model_card.md with the new score, the baseline for comparison, and one sentence on what changed since v1."),
   ("Compare v2's accuracy and gap against the Lab 14 numbers you wrote in review_notes.md.",
    ""),
  ],
  notes=[
   ("This is the trap. An assistant asked for 'augmented data loaders' will frequently build one transform and "
    "use it for both splits. Augmented validation is not just wrong, it is invisibly wrong: your score changes "
    "every run and is systematically pessimistic, so you will conclude your fixes did not work."),
   ("Read the two printed pipelines carefully. The validation list must contain exactly three entries — "
    "Grayscale, ToTensor, Normalize. If RandomHorizontalFlip or RandomResizedCrop appears there, fix it before "
    "running anything else."),
   ("Augmentation must preserve the label. A horizontal flip of a scratch is still a scratch, so that is safe. "
    "If you were classifying digits, a flip would change a 2 into something that is not a 2 — the "
    "transformation has to make sense for YOUR data, which is a judgement the assistant cannot make for you."),
   ("Dropout belongs in the dense head because convolutional layers already share weights heavily and are "
    "naturally regularized; heavy Dropout2d early in a CNN tends to hurt. And the eval() note matters: dropout "
    "left active at evaluation makes your validation score randomly worse, which is the mirror image of the "
    "augmented-validation bug."),
   ("One change per run is what makes this an ablation rather than a guess. If two things change between rows "
    "you cannot attribute the difference, and the table becomes decoration."),
   ("Typical finding: augmentation contributes the largest single reduction in the gap, dropout adds a modest "
    "further improvement, weight decay a small one, and early stopping does not change the model at all — it "
    "changes which epoch's weights you keep. That last distinction is worth stating out loud."),
   ("The regularization field in the metrics is exactly the kind of provenance that makes a checkpoint "
    "trustworthy six months later. A model card that records what was tried, not just what was chosen, is far "
    "more useful to the next person."),
   ("A successful outcome is a smaller train/validation gap AND a validation accuracy at least as good as "
    "before. If the gap shrank because training accuracy collapsed, you over-regularized — reduce the dropout "
    "or soften the augmentation."),
  ],
  test=("datasets.py prints a validation pipeline containing no random transforms, and "
        "reports/augmented_samples.png shows eight plausible variants of one training image. "
        "reports/ablation.csv holds five rows with a train-validation gap column that narrows down the table. "
        "models/defect_cnn_v2.pt is saved with a regularization field, and v2's gap is smaller than the Lab 14 gap."),
  troubleshoot=[
   ("Validation accuracy changes every run", "Augmentation is being applied to validation. Print the validation transform and remove every random operation."),
   ("Augmentation makes results clearly worse", "The transforms are too aggressive for 64x64 images. Reduce RandomRotation to 5 degrees and raise the RandomResizedCrop lower scale bound to 0.9."),
   ("Dropout produces a training accuracy lower than validation accuracy", "That is normal and expected — dropout is active during training and off during evaluation. It is not a bug."),
   ("Training is now much slower", "Augmentation happens on the CPU per image. Reduce the image count or the batch size; on Windows keep num_workers=0."),
   ("Weight decay makes no measurable difference", "1e-4 is mild. Try 1e-3 and note the effect, but expect weight decay to matter less than augmentation on image data."),
  ],
  stretch=[
   "Add ColorJitter on brightness and contrast and see whether it helps on greyscale inspection images or not.",
   "Implement mixup and compare it against the standard augmentation set.",
   "Plot the ablation as a bar chart of the gap per configuration and check the visual story matches the table.",
  ],
 ),

 dict(
  num=16, topic=3,
  title="Transfer Learning with Pre-Trained Models",
  objective="fine-tune a pre-trained network by freezing its backbone and replacing the classifier head",
  desc=("Stop training from scratch. You load a ResNet-18 pre-trained on ImageNet, adapt it to greyscale "
        "64x64 inspection images, freeze the backbone so only a new three-class head learns, and fine-tune. "
        "You then unfreeze the last block at a much lower learning rate and compare all three approaches — "
        "scratch CNN, frozen backbone, partial fine-tune — on accuracy, training time and trainable parameter "
        "count. The lab makes concrete why transfer learning is the default professional starting point when "
        "you have hundreds of images rather than hundreds of thousands."),
  build="A fine-tuned ResNet-18 defect classifier plus a three-way comparison against your scratch CNN",
  services="PyTorch, torchvision.models, engine.py, Cursor / GitHub Copilot / Claude",
  why=("Almost no production vision model is trained from scratch. Knowing how to freeze, replace a head and "
       "choose a fine-tuning learning rate is the difference between needing a hundred thousand labelled "
       "images and needing a few hundred."),
  files=["transfer.py", "models/defect_resnet_v1.pt", "reports/transfer_comparison.csv"],
  steps=[
   ("Prompt for the transfer setup, being explicit about the two adaptations a greyscale 64x64 input requires.",
    "Create transfer.py that fine-tunes a pre-trained ResNet-18 on the defect images.\nTASK: adapt an ImageNet ResNet-18 to 3-class greyscale 64x64 defect classification.\nOPERATIONS: (1) load torchvision.models.resnet18 with the default pre-trained weights; (2) adapt the input - the simplest correct approach is a transform pipeline that converts the greyscale image to 3 channels and resizes to 224x224 using the ImageNet normalisation statistics, so state clearly in a comment which approach you used and why; (3) FREEZE every parameter by setting requires_grad=False; (4) replace model.fc with a new nn.Linear(512, 3) whose parameters are trainable; (5) print the total parameter count and the TRAINABLE parameter count so the difference is obvious; (6) train for 12 epochs with Adam lr=1e-3 using fit from engine.py, passing only the trainable parameters to the optimizer.\nCONSTRAINTS: the new head outputs raw logits. Use the augmented training transform and the deterministic validation transform from Lab 15, adjusted for 3-channel 224x224 ImageNet input.\nEXPECTED OUTPUT: printed parameter counts, per-epoch losses, final validation accuracy, and the elapsed training time."),
   ("Before training, check the parameter counts. The trainable number should be a tiny fraction of the total.",
    "python transfer.py"),
   ("Note the accuracy and time, then compare against your Lab 15 scratch model trained for far longer.",
    ""),
   ("Now unfreeze the last block and fine-tune at a much lower learning rate.",
    "Add a function partial_finetune() to transfer.py.\nOPERATIONS: start from the frozen model trained above, then set requires_grad=True for layer4 and fc only. Use two parameter groups in Adam - layer4 at lr=1e-4 and fc at lr=1e-3 - and train for a further 8 epochs. Print the new trainable parameter count and the final validation accuracy.\nAdd a comment explaining why the unfrozen backbone layer uses a learning rate roughly ten times smaller than the head: the pre-trained weights are already good, and a large update would destroy the features you are trying to reuse.\nCONSTRAINTS: do not re-initialise the head - continue from the weights just trained."),
   ("Run the partial fine-tune and record whether the extra 8 epochs improved on the frozen result.",
    "python transfer.py"),
   ("Build the three-way comparison table that answers the practical question.",
    "Add a comparison to transfer.py that produces reports/transfer_comparison.csv with one row per approach: (1) DefectCNNv2 trained from scratch with the Lab 15 best configuration; (2) ResNet-18 frozen backbone with a new head; (3) ResNet-18 with layer4 unfrozen. Columns: approach, trainable parameters, epochs trained, wall-clock training seconds, best validation accuracy, and validation accuracy per minute of training. Print the table sorted by best validation accuracy and add a one-line printed conclusion naming which approach you would choose for a factory with 1200 labelled images and no GPU."),
   ("Save the best transfer model with a checkpoint and a model card entry.",
    "Update transfer.py to save the best-performing transfer model with save_checkpoint to models/defect_resnet_v1.pt, recording in the metrics dict the approach used, which layers were trainable, the validation accuracy and the training time. Append a section to models/defect_model_card.md comparing the scratch CNN and the transfer model, and stating which one you recommend for deployment and why."),
   ("Read the comparison table and state your recommendation out loud, with the trade-off you are accepting.",
    ""),
  ],
  notes=[
   ("There are two legitimate ways to feed greyscale 64x64 images to a ResNet: repeat the single channel three "
    "times and resize to 224x224 to match ImageNet exactly, or surgically replace the first conv layer to "
    "accept one channel. The first reuses the pre-trained weights fully and is the right default; the second "
    "discards the first layer's learned filters. Making the assistant state which it chose forces the decision "
    "into the open."),
   ("ResNet-18 has around 11 million parameters; the new head has about 1,500. Training under 0.02% of the "
    "network is why this runs quickly and why it does not overfit 1200 images. If the trainable count is in "
    "the millions, the freeze did not take — check requires_grad was set before fc was replaced."),
   ("Expect the frozen backbone to reach comparable or better accuracy than your scratch CNN in a fraction of "
    "the epochs, though each epoch is slower because 224x224 inputs are twelve times larger than 64x64. That "
    "trade-off is exactly what the comparison table is for."),
   ("The differential learning rate is the professional detail. The pre-trained features took an enormous "
    "amount of compute to learn; a 1e-3 update would wreck them in a single epoch. Small learning rate for "
    "pre-trained weights, larger for the randomly initialised head, is the rule."),
   ("Partial fine-tuning sometimes helps meaningfully and sometimes barely moves the number, particularly when "
    "your images look nothing like ImageNet photographs. Synthetic greyscale inspection images are quite far "
    "from ImageNet, so a modest gain is a perfectly honest result to report."),
   ("The 'accuracy per minute' column is the one that changes decisions. In a factory with no GPU, an approach "
    "that reaches 88% in four minutes usually beats one that reaches 90% in forty. Say which trade-off you are "
    "making rather than just picking the top accuracy."),
   ("Recording which layers were trainable is essential provenance — 'ResNet-18 fine-tuned' is not reproducible, "
    "whereas 'ResNet-18, layer4 and fc trainable, lr 1e-4 and 1e-3, 20 epochs total' is."),
   ("There is no single right answer here, and the trainer will push you to defend yours. Model size, inference "
    "speed on the plant's hardware, and the cost of a missed defect all belong in the argument alongside accuracy."),
  ],
  test=("transfer.py prints a trainable parameter count that is a tiny fraction of the ~11M total, trains a "
        "frozen-backbone ResNet-18 to a validation accuracy comparable with or better than your scratch CNN, "
        "and reports the partial fine-tune result. reports/transfer_comparison.csv holds three rows with "
        "parameters, time and accuracy. models/defect_resnet_v1.pt is saved and the model card names a recommendation."),
  troubleshoot=[
   ("Downloading the pre-trained weights fails", "No internet or a blocked host. Ask the trainer for the cached weights file, or set weights=None and note honestly that the run is no longer transfer learning."),
   ("RuntimeError: expected input[N, 1, 64, 64] to have 3 channels", "The transform is not converting to 3 channels. Add Grayscale(num_output_channels=3) and Resize(224) before ToTensor."),
   ("The trainable parameter count equals the total", "requires_grad=False was applied after fc was replaced, or not at all. Freeze first, then replace the head."),
   ("Training is much slower than the scratch CNN", "Expected — 224x224 inputs are far larger. Reduce to Resize(128) and note the accuracy change, or lower the batch size."),
   ("Accuracy is stuck near the baseline", "The optimizer probably received all parameters including frozen ones. Pass only `filter(lambda p: p.requires_grad, model.parameters())`."),
  ],
  stretch=[
   "Try ResNet-34 or MobileNetV3-Small and add them to the comparison table.",
   "Freeze everything except the final BatchNorm layers and see how that compares to unfreezing layer4.",
   "Measure single-image inference time for the scratch CNN and the ResNet, and add it to the deployment argument.",
  ],
 ),

]
