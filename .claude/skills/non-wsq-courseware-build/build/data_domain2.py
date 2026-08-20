"""
Topic 2 hands-on activities - Vibe Coding Neural Networks.

Labs 7-11, mapping 1:1 onto the five published sub-topics. Single source for the
PPT activity/step slides, the Learner Guide sections, the Lesson Plan schedule
rows and the labs/labNN-*/README.md files.
"""

DOMAIN2 = [

 dict(
  num=7, topic=2,
  title="Neural Network Architectures, Activation and Loss Functions",
  objective="choose hidden and output activations and match the loss function to the output layer",
  desc=("Before you train anything, settle the three decisions that determine whether training can work at all: "
        "the architecture, the activation and the loss. You prompt for a script that plots ReLU, sigmoid and "
        "tanh next to their gradients, then proves the claim that a stack of linear layers without activations "
        "collapses into a single linear layer — by showing two networks produce identical outputs. You finish "
        "by building a decision table linking each task type to its output layer and its loss, which becomes "
        "the reference you use for the next fourteen labs."),
  build="An activations.py with plotted activations and a linear-collapse proof, plus a losses_cheatsheet.md decision table",
  services="PyTorch, matplotlib, Cursor / GitHub Copilot / Claude",
  why=("Almost every 'my model will not learn' problem traces back to one of these three choices. Learners who "
       "can state why CrossEntropyLoss wants raw logits, and why a sigmoid deep in a stack stalls learning, "
       "diagnose in seconds what otherwise costs an afternoon."),
  files=["activations.py", "losses_cheatsheet.md", "reports/activations.png"],
  steps=[
   ("Prompt for the activation comparison. Ask for the gradients as well as the functions — the gradient is what explains the behaviour.",
    "Create activations.py for a PyTorch course.\nTASK: compare the three common activation functions and show why the choice matters.\nOPERATIONS: (1) create x = torch.linspace(-6, 6, 200); (2) compute ReLU, sigmoid and tanh of x using torch.nn.functional; (3) compute the gradient of each with respect to x using autograd; (4) plot a 2-row figure - top row the three activations, bottom row their three gradients, sharing the x axis - and save it to reports/activations.png; (5) print, for each activation, the fraction of the input range where its gradient is smaller than 0.01.\nCONSTRAINTS: torch and matplotlib only, no seaborn, label every axis and subplot.\nEXPECTED OUTPUT: the saved figure plus three printed 'saturated fraction' lines."),
   ("Run it, open the figure, and answer one question from the bottom row: which activation keeps a usable gradient across the widest input range?",
    "python activations.py"),
   ("Now prove the linear-collapse claim. This is the reason activations exist at all.",
    "Add a section to activations.py called linear_collapse().\nTASK: prove that a stack of linear layers with no activation is equivalent to a single linear layer.\nOPERATIONS: (1) build net_a = nn.Sequential(nn.Linear(6, 32), nn.Linear(32, 16), nn.Linear(16, 1)) with manual seed 42; (2) analytically collapse its three weight matrices and biases into ONE equivalent nn.Linear(6, 1) by composing them; (3) run the same random input batch of shape (8, 6) through both and print the two outputs side by side plus their maximum absolute difference; (4) assert the difference is below 1e-4; (5) repeat with nn.ReLU() inserted between the layers and print the maximum absolute difference now.\nEXPECTED OUTPUT: near-zero difference without activations, clearly non-zero difference with ReLU."),
   ("Run it and confirm the assertion passes. Read the second difference — that number is what the non-linearity is buying you.",
    "python activations.py"),
   ("Build the decision table by hand in losses_cheatsheet.md. Fill it in yourself before checking it with the assistant.",
    ""),
   ("Check your table against the assistant, and specifically interrogate the classification row.",
    "I have written a table mapping task type to output layer and PyTorch loss function:\n- Regression (one continuous value) -> Linear(h, 1), no activation -> nn.MSELoss\n- Binary classification -> Linear(h, 1), no activation -> nn.BCEWithLogitsLoss\n- Multi-class classification (4 classes) -> Linear(h, 4), no activation -> nn.CrossEntropyLoss\nFor each row, confirm or correct it, and explain in one sentence what happens numerically if someone adds a softmax or sigmoid to the output layer before these losses. Be specific about which losses already apply that function internally."),
   ("Record the answer in losses_cheatsheet.md in your own words, especially the softmax warning.",
    ""),
  ],
  notes=[
   ("The saturated-fraction numbers are the quantitative version of what the plot shows. Sigmoid and tanh flatten "
    "at both ends, so their gradients approach zero over a large part of the range; ReLU has gradient exactly 1 "
    "for every positive input. That is why ReLU became the default hidden activation."),
   ("The answer is ReLU. Sigmoid's gradient peaks at 0.25 and decays fast, so in a deep stack the gradients "
    "multiply down towards nothing — the vanishing gradient problem you will meet again with plain RNNs in "
    "Lab 17. Note that ReLU's flat negative half is a real cost (dead units), just a smaller one."),
   ("Composing the layers means multiplying the weight matrices in the right order and carrying the biases "
    "through: W = W3 @ W2 @ W1 and b = W3 @ (W2 @ b1 + b2) + b3, give or take the transpose convention "
    "PyTorch uses. If the assistant gets the order wrong the assertion fails — that is a real bug worth "
    "making it fix rather than accepting a loosened tolerance."),
   ("A difference below 1e-4 without activations means three layers did exactly as much as one: 561 parameters "
    "achieving what 7 could. With ReLU the difference is large, because the network can now bend. If the "
    "assistant 'fixes' a failing assert by raising the tolerance to 1.0, reject it and make it fix the algebra."),
   ("Fill in three rows: task type, final layer, output activation, loss function. Do it from memory. The row "
    "people get wrong is multi-class — write down what you believe before you check, so you find out whether "
    "you actually knew it."),
   ("The critical answer: nn.CrossEntropyLoss applies log-softmax internally and nn.BCEWithLogitsLoss applies "
    "sigmoid internally. Adding your own softmax before CrossEntropyLoss applies it twice, which flattens the "
    "distribution, shrinks the gradients and makes the model learn slowly or not at all — while still running "
    "and still printing a falling loss. This is the single most common AI-generated PyTorch bug."),
   ("Write the warning in your own words, not copied. Something like: 'CrossEntropyLoss wants raw logits. If I "
    "see nn.Softmax in a model's output layer next to CrossEntropyLoss, one of them must go.' You will apply "
    "this exact check in Lab 9, where the assistant will very likely make the mistake for you."),
  ],
  test=("reports/activations.png shows three activations over three gradients, and the printed saturated "
        "fractions are near zero for ReLU and clearly non-zero for sigmoid and tanh. linear_collapse() asserts "
        "a difference below 1e-4 without activations and prints a visibly larger difference with ReLU. "
        "losses_cheatsheet.md has all three rows and states which losses apply softmax or sigmoid internally."),
  troubleshoot=[
   ("The gradient plot is empty or all zeros", "Gradients were computed without requires_grad on x. Follow up: 'x must be created with requires_grad=True, and use autograd.grad or backward on the sum to get elementwise gradients.'"),
   ("The linear_collapse assertion fails with a large difference", "The weight composition order or a transpose is wrong. Ask the assistant to print the shapes of each weight matrix and re-derive the composition — do not raise the tolerance."),
   ("matplotlib opens a window and blocks the script", "Add `matplotlib.use('Agg')` before importing pyplot, or call plt.savefig without plt.show. Every plot in this course is saved to a file, not shown."),
   ("reports/ does not exist", "Create it, or ask for `os.makedirs('reports', exist_ok=True)` before saving. Lab 2 created this folder — check you are running from torch-vibe/."),
   ("The assistant insists softmax before CrossEntropyLoss is fine", "It is not. Ask it to compute the loss both ways on the same logits and print the two numbers — the demonstration settles it."),
  ],
  stretch=[
   "Add LeakyReLU and GELU to the comparison and see how they handle the negative half of the range.",
   "Extend linear_collapse to five layers and confirm the equivalence still holds exactly.",
   "Compute the loss with and without an extra softmax on the same batch of logits, and record how much smaller the gradients become.",
  ],
 ),

 dict(
  num=8, topic=2,
  title="Vibe Coding a Regression Model in PyTorch",
  objective="build, train and honestly evaluate a regression network that predicts a continuous value",
  desc=("Build ForgeSight's first real model: a network predicting tool wear in microns from six sensor "
        "readings. You prompt for an nn.Module with a deliberate architecture, an MSELoss, and a training loop "
        "that reports train and validation loss each epoch. Crucially you establish a BASELINE first — "
        "predicting the training mean for every row — because a regression score means nothing on its own. "
        "You then check the prediction and target shapes explicitly, which is where the (N,) versus (N,1) trap "
        "from Lab 4 comes back to bite anyone who skipped it."),
  build="A trained tool-wear regression network reporting MAE and RMSE that beat the mean baseline",
  services="PyTorch, nn.Module, MSELoss, Cursor / GitHub Copilot / Claude",
  why=("This is the first lab where a silent bug costs you a real result rather than a printed shape. Getting "
       "the baseline, the shapes and the loss right here establishes the pattern that every remaining model in "
       "the course reuses."),
  files=["models.py", "train_wear.py", "reports/wear_curve.png"],
  steps=[
   ("Compute the baseline FIRST, before any model exists. You cannot judge a regression score without it.",
    "Create baseline_wear.py. Using load_forgesight from load_data.py, predict the MEAN of y_wear_train for every row in the validation set, and print the resulting MAE and RMSE in microns. Print the standard deviation of y_wear_train alongside them. Add a comment explaining why RMSE for a mean-predictor is close to the target's standard deviation."),
   ("Run it and write the baseline MAE down. Every number you produce for the rest of this lab is judged against it.",
    "python baseline_wear.py"),
   ("Prompt for the model and the training script. Note that the prompt names the shapes, the loss and the exact reporting format.",
    "Create models.py and train_wear.py for a PyTorch regression task.\nSHAPES: X_train is float32 (2800, 6), y_wear_train is float32 (2800,). Features are already standardised from load_data.py.\nTASK: predict tool_wear_um, a continuous value in microns.\nOPERATIONS: in models.py define class WearNet(nn.Module) with Linear(6,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,1) and NO activation on the output. In train_wear.py use nn.MSELoss and torch.optim.Adam at lr=1e-3, batch size 64 via TensorDataset and DataLoader, and train for 100 epochs.\nCONSTRAINTS: reshape the target to (N,1) so it matches the prediction shape exactly - print both shapes once before the first loss call to prove they match. Set manual seed 42. Evaluate under model.eval() and torch.no_grad(). Do not apply any activation to the output layer.\nEXPECTED OUTPUT: per-epoch train and validation loss every 10 epochs; final validation MAE and RMSE in microns; a saved plot of both loss curves at reports/wear_curve.png."),
   ("Read the generated code and check three things before running: the output layer has no activation, the target is reshaped, and eval() plus no_grad() wrap the validation pass.",
    ""),
   ("Run the training. Watch that both losses fall and that the printed shapes match.",
    "python train_wear.py"),
   ("Compare your final validation MAE against the baseline MAE you wrote down. State the improvement as a percentage.",
    ""),
   ("Open reports/wear_curve.png and read the two curves. Note whether validation loss is still falling at epoch 100 or has flattened.",
    ""),
  ],
  notes=[
   ("A mean-predictor is the honest floor for regression. Its RMSE is essentially the target's standard "
    "deviation, because that is what standard deviation measures. If your network cannot beat this, it has "
    "learned nothing, no matter how impressive the loss curve looks."),
   ("Expect a baseline MAE in the region of tens of microns. Write the exact number in review_notes.md. "
    "Learners who skip this step routinely celebrate an MAE that is worse than predicting the average."),
   ("Two clauses are doing heavy lifting. 'NO activation on the output' — a ReLU there would clamp every "
    "prediction to be non-negative, which sounds harmless for wear but silently caps the model; a sigmoid there "
    "would squash all predictions into (0,1) and make the task impossible. 'Print both shapes once before the "
    "first loss call' — this is the Lab 4 broadcast trap, made visible."),
   ("If the printed shapes are torch.Size([64, 1]) and torch.Size([64, 1]), you are safe. If you see "
    "torch.Size([64]) for the target, stop and fix it — the loss will broadcast to (64, 64) and every number "
    "after that is meaningless, without any error being raised."),
   ("Training 100 epochs on 2800 rows takes well under a minute on CPU. If the loss is not falling at all, "
    "check the learning rate first; if it explodes to nan, the target was probably not standardised or the "
    "learning rate is far too high."),
   ("A good result here is a validation MAE meaningfully below the baseline. If the improvement is under a few "
    "percent, the features may genuinely not carry much signal about wear — which is a legitimate finding to "
    "report, not a failure to hide. Say so out loud; that honesty is the professional habit being trained."),
   ("If validation loss is still falling, the model is underfitted and more epochs would help. If it has "
    "flattened while training loss keeps dropping, you are watching the beginning of overfitting — the exact "
    "pattern you will diagnose properly in Lab 14."),
  ],
  test=("baseline_wear.py prints a baseline MAE and RMSE. train_wear.py prints matching prediction and target "
        "shapes of torch.Size([N, 1]), trains with both losses falling, and reports a final validation MAE "
        "clearly BELOW the baseline MAE. reports/wear_curve.png shows both curves labelled."),
  troubleshoot=[
   ("The loss is enormous and never falls", "The target is unscaled and in the hundreds. Either standardise y as well (and invert it for reporting) or lower the learning rate — check what the assistant did to y."),
   ("Loss becomes nan after a few epochs", "Learning rate too high, or a nan in the input. Print X_train.isnan().any() and drop the lr to 1e-4."),
   ("Validation MAE is worse than the baseline", "Usually underfitting or a too-small learning rate. Train longer, or check the output layer really has no activation squashing the range."),
   ("RuntimeError about the size of tensor a and tensor b", "Prediction and target shapes disagree. Reshape the target with .view(-1, 1) — this is the Lab 4 trap."),
   ("The model predicts almost the same value for every row", "It has collapsed to the mean, which means it is not learning. Check the features really are standardised and the learning rate is not far too small."),
  ],
  stretch=[
   "Ask the assistant to add a parity plot (predicted versus actual) and read where the model is worst.",
   "Swap Adam for SGD with momentum at the same learning rate and note how differently the curve behaves.",
   "Widen the first layer to 128 units and see whether validation MAE improves or the gap to training loss widens.",
  ],
 ),

 dict(
  num=9, topic=2,
  title="Vibe Coding a Classification Model with Softmax and Cross Entropy",
  objective="build a multi-class classifier that outputs raw logits and pair it correctly with cross entropy loss",
  desc=("Predict the ForgeSight quality-control class — ok, scratch, dent or burr — from the same six sensors. "
        "This lab is deliberately a trap. You first ask the assistant for the model in a way that invites the "
        "classic mistake, and there is a strong chance it hands you a network with nn.Softmax on the output "
        "next to nn.CrossEntropyLoss. You catch it, prove numerically that it is wrong, and fix it. You then "
        "evaluate properly against a majority-class baseline with a confusion matrix, because on imbalanced "
        "classes accuracy alone will flatter a model that never predicts the rare defect at all."),
  build="A QC classifier outputting raw logits, evaluated with accuracy, per-class recall and a confusion matrix against the majority baseline",
  services="PyTorch, nn.CrossEntropyLoss, scikit-learn metrics, Cursor / GitHub Copilot / Claude",
  why=("The softmax-before-cross-entropy bug is the most common defect in AI-generated PyTorch, and it never "
       "raises an error. Catching it once, deliberately, with the numbers in front of you, is worth more than "
       "being told about it ten times."),
  files=["models.py", "train_qc.py", "reports/qc_confusion.png"],
  steps=[
   ("Establish the majority-class baseline first, and look at how imbalanced the classes actually are.",
    "Create baseline_qc.py. Using load_forgesight, print the count and percentage of each of the 4 qc_class values in the training split. Then predict the single most frequent class for every validation row and print the resulting accuracy. Print the per-class recall for that baseline as well, and add a comment on what recall is for the classes it never predicts."),
   ("Run it. Note the majority-class accuracy — this is the number your model must beat, and it is higher than most people expect.",
    "python baseline_qc.py"),
   ("Now prompt for the classifier, deliberately WITHOUT specifying the output activation. You are setting a trap for the assistant.",
    "Add class QCNet(nn.Module) to models.py and create train_qc.py.\nSHAPES: X_train is float32 (2800, 6); y_qc_train is int64 (2800,) with 4 classes.\nTASK: classify each machine reading into one of 4 quality-control classes.\nOPERATIONS: QCNet should be Linear(6,64) -> ReLU -> Linear(64,32) -> ReLU -> Linear(32,4). Train with nn.CrossEntropyLoss and Adam at lr=1e-3, batch size 64, 100 epochs.\nCONSTRAINTS: manual seed 42, evaluate under model.eval() and torch.no_grad().\nEXPECTED OUTPUT: per-epoch train and validation loss, final validation accuracy, per-class precision and recall, and a confusion matrix saved to reports/qc_confusion.png."),
   ("STOP before running. Inspect the generated QCNet output layer. Is there an nn.Softmax, nn.LogSoftmax or F.softmax anywhere after the final Linear?",
    ""),
   ("If a softmax is present, prove it is wrong before removing it rather than just deleting it.",
    "Write a short script proof_softmax.py that takes one batch of logits of shape (8, 4) with manual seed 42 and computes nn.CrossEntropyLoss twice: once on the raw logits, and once on torch.softmax(logits, dim=1). Print both loss values and both gradients with respect to the logits, and print the ratio of the gradient magnitudes. Add a comment explaining that CrossEntropyLoss applies log_softmax internally, so the second version applies softmax twice and produces much smaller gradients."),
   ("Fix the model with a targeted follow-up, then train it.",
    "In models.py, QCNet applies softmax to its output, but nn.CrossEntropyLoss already applies log_softmax internally, so the network is applying it twice and its gradients are being flattened. Remove the softmax layer so QCNet returns raw logits from the final Linear layer. Keep everything else unchanged. Where a probability is needed for reporting, apply torch.softmax at that point instead."),
   ("Train the corrected model and compare accuracy and per-class recall against the majority baseline.",
    "python train_qc.py"),
   ("Open reports/qc_confusion.png and find which class the model confuses most. Note whether it predicts the rarest class at all.",
    ""),
  ],
  notes=[
   ("With four classes and a realistic factory distribution, most readings are 'ok'. A majority-class predictor "
    "therefore scores far above 25% while being completely useless — it never flags a single defect. This is "
    "why accuracy alone cannot be the reported metric and why per-class recall matters."),
   ("Write the baseline accuracy down. If your trained model ends up close to it, the model is probably just "
    "predicting 'ok' for everything — check the confusion matrix rather than celebrating the accuracy."),
   ("The prompt names the layers but deliberately says nothing about the output activation. This is exactly how "
    "these prompts get written in real life, and it is why the bug is so common. Assistants frequently add a "
    "softmax because it 'looks like' what a classifier should do."),
   ("Look at the last layer of the Sequential or the end of forward(). Any of nn.Softmax(dim=1), "
    "F.softmax(x, dim=1) or nn.LogSoftmax before returning is the bug. If your assistant returned raw logits, "
    "it got it right — still run the proof script, because you need to see the numbers."),
   ("Expect the double-softmax loss to be noticeably different and its gradients an order of magnitude smaller. "
    "Smaller gradients mean smaller updates mean slower learning — the model still trains, the loss still "
    "falls, and it simply ends up worse. Nothing warns you. This is the whole lesson."),
   ("Note the last clause: softmax is not banned, it is relocated. You still want probabilities when reporting "
    "a confidence to a user — you just compute them at reporting time, outside the loss path."),
   ("A corrected model should beat the majority baseline on accuracy AND find a meaningful share of at least "
    "the common defect classes. If accuracy is high but recall on the rare classes is near zero, say so — that "
    "is an honest and important result about class imbalance."),
   ("The confusion matrix is the real report. Off-diagonal mass tells you which defects look alike to the "
    "model. A completely empty row means the model never predicts that class at all, which no accuracy number "
    "would have told you."),
  ],
  test=("baseline_qc.py prints class counts and the majority-class accuracy. proof_softmax.py prints two "
        "different loss values and shows the double-softmax gradients are much smaller. train_qc.py's QCNet "
        "returns raw logits with no softmax layer, and reports a validation accuracy above the majority "
        "baseline with a confusion matrix saved to reports/qc_confusion.png."),
  troubleshoot=[
   ("RuntimeError: expected scalar type Long but found Float", "y_qc is float32. CrossEntropyLoss needs int64 class indices — cast with .long(). This is the Lab 3 dtype note."),
   ("Accuracy exactly equals the majority baseline", "The model is predicting one class for everything. Check the softmax was actually removed, and try a slightly higher learning rate or more epochs."),
   ("The target has shape (N, 1) and CrossEntropyLoss complains", "Unlike MSELoss, CrossEntropyLoss wants a 1-D target of class indices. Use .squeeze() or .view(-1) — the opposite of what Lab 8 needed."),
   ("Loss starts near 1.386 and never moves", "1.386 is ln(4), the loss of a uniform guess over 4 classes. The model is not learning — check for the double softmax, then the learning rate."),
   ("The confusion matrix plot is unreadable", "Ask for class names on both axes and counts annotated in each cell. A confusion matrix without labels is not a report."),
  ],
  stretch=[
   "Add class weights to CrossEntropyLoss to penalise missing the rare defects and see how the confusion matrix changes.",
   "Print the model's softmax probabilities for ten misclassified rows and see whether it was confidently wrong or nearly right.",
   "Compare macro-averaged F1 against accuracy and decide which you would report to the plant manager.",
  ],
 ),

 dict(
  num=10, topic=2,
  title="Generating Training Loops, Optimizers and Metrics from Prompts",
  objective="refactor training into one reusable fit function and compare optimizers and learning rates with it",
  desc=("You have now written the same training loop twice. Refactor it once, properly, and never write it "
        "again. You prompt for engine.py containing a single fit() that takes a model, loaders, a loss, an "
        "optimizer and a metric function, returns a history dictionary, and works unchanged for both the "
        "regression and the classification task. With that in place, experimentation becomes cheap: you run a "
        "small sweep comparing SGD against Adam across three learning rates and produce a results table that "
        "shows learning rate matters more than the choice of optimizer."),
  build="A reusable engine.py fit/evaluate pair driving both models, plus a sweep table comparing optimizers and learning rates",
  services="PyTorch, torch.optim, DataLoader, Cursor / GitHub Copilot / Claude",
  why=("Copy-pasted training loops are where bugs breed — the fifth copy is the one missing zero_grad(). One "
       "reviewed, tested loop used everywhere is both better engineering and the thing that makes the sweep in "
       "this lab possible at all."),
  files=["engine.py", "sweep.py", "reports/sweep.csv"],
  steps=[
   ("Prompt for the reusable engine. The hard requirement is that it must work for BOTH tasks without modification.",
    "Create engine.py with two functions that work unchanged for both a regression and a classification task.\nfit(model, train_loader, val_loader, loss_fn, optimizer, epochs, metric_fn=None, device='cpu') should:\n1. Loop over epochs; for each batch call optimizer.zero_grad() BEFORE the forward pass, compute the loss, call backward, then step\n2. Accumulate the training loss weighted by batch size, not a plain mean of batch means\n3. After each epoch run a validation pass under model.eval() and torch.no_grad(), then return the model to train mode\n4. Apply metric_fn(preds, targets) on the validation set if one is given\n5. Return a history dict with keys train_loss, val_loss and val_metric, each a list of length epochs\n6. Track and restore the state_dict from the epoch with the best validation loss, and report which epoch that was\nevaluate(model, loader, loss_fn, metric_fn, device) should run one pass under eval/no_grad and return the loss and metric.\nCONSTRAINTS: no printing inside fit except an optional every-N-epochs line controlled by a verbose argument; no task-specific logic - the caller supplies loss_fn and metric_fn.\nEXPECTED OUTPUT: engine.py importable by both train_wear.py and train_qc.py."),
   ("Review the generated fit() against your Lab 6 checklist before you trust it — this function is about to run every experiment for the rest of the course.",
    ""),
   ("Rewire both existing training scripts to use it, and confirm the results still match what you got in Labs 8 and 9.",
    "Refactor train_wear.py and train_qc.py to use fit and evaluate from engine.py instead of their own inline training loops. Keep the same seeds, architectures, hyperparameters and reporting so the results are directly comparable. train_wear.py passes nn.MSELoss and an MAE metric function; train_qc.py passes nn.CrossEntropyLoss and an accuracy metric function. Delete the now-duplicated loop code from both files."),
   ("Re-run both and check the final numbers are essentially unchanged from the previous labs. A refactor that changes results is a refactor that introduced a bug.",
    "python train_wear.py\npython train_qc.py"),
   ("Now use the engine for what it was built for. Prompt for a sweep across optimizers and learning rates.",
    "Create sweep.py using fit from engine.py.\nTASK: compare optimizers and learning rates on the QC classification task.\nOPERATIONS: for each combination of optimizer in [SGD with momentum 0.9, Adam] and learning rate in [1e-2, 1e-3, 1e-4], train a FRESH QCNet for 60 epochs with manual seed 42 reset before each run, and record the best validation loss, the validation accuracy at that epoch, the epoch it occurred, and the wall-clock seconds.\nCONSTRAINTS: identical data, batch size and seed across all 6 runs - only the optimizer and learning rate change. Re-instantiate the model each run so no weights carry over.\nEXPECTED OUTPUT: a printed table sorted by best validation loss, saved to reports/sweep.csv, plus a single figure overlaying all 6 validation loss curves with a legend."),
   ("Run the sweep and read the table. Answer: does the optimizer or the learning rate account for more of the spread in results?",
    "python sweep.py"),
   ("Record the winning configuration and your answer in prompts.md, and adopt that configuration for the rest of the course.",
    ""),
  ],
  notes=[
   ("Point 2 is a real subtlety most generated loops get wrong: averaging the per-batch means is only correct "
    "when every batch is the same size, and the last batch usually is not. Weighting by batch size is the "
    "correct reduction. Point 6 is early stopping's useful half — keeping the best weights rather than the last."),
   ("Run your checklist: is zero_grad() before backward()? Is the validation pass wrapped in eval() and "
    "no_grad()? Does it return to train() afterwards — a loop that forgets this leaves dropout off for the rest "
    "of training. Is anything task-specific hiding in there, like an argmax that only makes sense for "
    "classification?"),
   ("A shared engine is only safe if it is genuinely general. If fit() contains an argmax or a .float() cast "
    "that only suits one task, it will silently corrupt the other. Making both scripts use it is the test."),
   ("'Essentially unchanged' means within normal run-to-run variation, not bit-identical — the batching order "
    "may differ slightly. If the numbers move a lot, diff the old loop against fit() and find what changed. "
    "This is a genuine regression test, and it is why you kept the seeds fixed."),
   ("Re-instantiating the model each run is the clause that makes the comparison valid. Reusing a trained model "
    "across configurations means each run starts from the previous one's weights and the table is nonsense. "
    "Assistants get this wrong regularly — check it explicitly in the generated code."),
   ("The usual finding: SGD at 1e-4 barely moves, Adam at 1e-2 is unstable, and the middle settings work. The "
    "spread across learning rates within one optimizer is typically much larger than the spread across "
    "optimizers at a sensible learning rate. Learning rate is the hyperparameter worth your attention."),
   ("You will reuse fit() in Labs 13 to 19 without modification. That is the payoff: from here on, a new model "
    "is a new nn.Module plus a metric function, not another hand-written loop with another chance to forget "
    "zero_grad()."),
  ],
  test=("engine.py's fit runs both train_wear.py and train_qc.py unchanged and reproduces the Labs 8 and 9 "
        "results. sweep.py produces reports/sweep.csv with 6 rows, a sorted printed table, and an overlay "
        "figure of 6 validation curves. You can state which factor - optimizer or learning rate - drove more "
        "of the difference."),
  troubleshoot=[
   ("fit works for regression but crashes for classification", "Task-specific logic leaked into the engine, usually a target reshape. Move it into the caller or the metric function."),
   ("All 6 sweep runs give identical results", "The model is not being re-instantiated, or the seed reset is missing. Print the initial loss of each run — identical first losses with different optimizers means the model carried over."),
   ("The best epoch is always the last one", "Validation loss is still improving, so train longer. If it is improving on every configuration, 60 epochs is too few for this comparison."),
   ("Adam at 1e-2 produces nan", "That is a legitimate result — record it as diverged rather than dropping the row. It is evidence for the learning-rate conclusion."),
   ("The refactored scripts give noticeably different results", "Check shuffle and seed placement. The DataLoader must be constructed the same way, and the seed set before model instantiation, not after."),
  ],
  stretch=[
   "Add AdamW and a cosine learning-rate schedule to the sweep and see whether either beats the winner.",
   "Extend fit with gradient clipping and confirm it rescues the diverging Adam-at-1e-2 run.",
   "Add a patience argument that stops training early when validation loss has not improved for N epochs, and check it picks the same best epoch.",
  ],
 ),

 dict(
  num=11, topic=2,
  title="Saving, Loading and Iterating on Models",
  objective="save a model's state_dict with its configuration, reload it into a fresh instance and prove the predictions are identical",
  desc=("A model that only exists in memory is not a deliverable. You prompt for a checkpoint helper that saves "
        "the state_dict together with everything needed to rebuild it — architecture arguments, the "
        "standardisation statistics, the class names, the metrics and the library versions — then reload it "
        "into a freshly constructed model in a NEW process and assert the predictions match to machine "
        "precision. That assertion is the difference between believing your model saved and knowing it did. "
        "You finish by writing the first ForgeSight model card."),
  build="A checkpoint helper saving weights plus config and statistics, a proven identical-prediction reload, and a model card",
  services="PyTorch, torch.save / torch.load, Cursor / GitHub Copilot / Claude",
  why=("Saving only the weights is the most common way to produce an unusable artifact. Without the "
       "architecture arguments the file will not load; without the standardisation statistics the model gets "
       "raw inputs it was never trained on and returns confident nonsense. Both failures happen after the "
       "project looks finished."),
  files=["checkpoint.py", "models/qc_net_v1.pt", "reload_check.py", "models/qc_model_card.md"],
  steps=[
   ("Prompt for the checkpoint helper. The key insight is that the file must contain everything needed to rebuild, not just the weights.",
    "Create checkpoint.py with save_checkpoint and load_checkpoint functions.\nsave_checkpoint(path, model, config, feat_mean, feat_std, metrics, class_names=None) must save a single dict containing: the model's state_dict; the config dict of constructor arguments needed to rebuild the class; the model class NAME as a string; feat_mean and feat_std tensors; the metrics dict; the class names; the torch version; and an ISO timestamp.\nload_checkpoint(path, model_class) must construct a fresh instance from the saved config, load the state_dict into it, call model.eval(), and return the model plus the full metadata dict.\nCONSTRAINTS: save the state_dict, never the pickled model object - explain why in a comment. Create the models/ directory if missing. Use weights_only=False on load only where required by your torch version, and note why in a comment.\nEXPECTED OUTPUT: both functions importable, with a __main__ block that round-trips a small dummy model as a self-test."),
   ("Wire it into the QC training script so a completed training run always leaves a loadable artifact behind.",
    "Update train_qc.py to call save_checkpoint at the end of training. Save to models/qc_net_v1.pt with the config needed to rebuild QCNet, the feat_mean and feat_std returned by load_forgesight, the final validation accuracy and per-class recall as metrics, and class_names=['ok','scratch','dent','burr']."),
   ("Run training so the checkpoint is written.",
    "python train_qc.py"),
   ("Now prove the reload works, from a SEPARATE script that never sees the trained object in memory.",
    "Create reload_check.py that proves the checkpoint round-trips exactly.\nOPERATIONS: (1) load models/qc_net_v1.pt with load_checkpoint, rebuilding QCNet from the saved config; (2) take the first 16 rows of the validation set; (3) run them through the reloaded model under torch.no_grad(); (4) print the predicted class indices and the saved metrics; (5) assert the reloaded model's predictions are identical to the predictions the training script produced for the same rows, which you should also save alongside the checkpoint for this purpose; (6) print the saved timestamp, torch version and class names.\nCONSTRAINTS: this script must NOT import or re-run training - it may only read the checkpoint file."),
   ("Run it and confirm the assertion passes. This is the moment you know the artifact is real.",
    "python reload_check.py"),
   ("Deliberately break it to see the failure mode you are guarding against.",
    "Add a commented-out demonstration to reload_check.py: reload the model but skip applying feat_mean and feat_std to the input rows, so the model receives raw unstandardised sensor values. Print the predictions from both paths side by side and a comment noting that no error is raised - the model simply predicts confidently and wrongly."),
   ("Write the model card. Do it in your own words — this is the document that travels with the model.",
    ""),
  ],
  notes=[
   ("Pickling the whole model object embeds your file paths and class definitions, so it breaks when the code "
    "moves or the class is renamed. A state_dict is just tensors, and the config tells you how to rebuild the "
    "shell they go into. This is why the official PyTorch recommendation is state_dict plus code."),
   ("feat_mean and feat_std are part of the model, not part of the training script. If they are not in the "
    "checkpoint, whoever loads this model six months from now has no way to preprocess input correctly, and "
    "nothing will tell them they got it wrong."),
   ("Check the file actually appeared and note its size — a few hundred kilobytes for this network. If it is "
    "suspiciously large, the whole model object was pickled rather than the state_dict."),
   ("The separate-script constraint is the real test. Loading in the same process as training can pass by "
    "accident because the trained object is still in memory. A fresh process proves the file alone is enough."),
   ("Identical means exactly identical — same architecture, same weights, eval mode, no dropout randomness. If "
    "predictions differ, the usual causes are the model left in train mode, or the config rebuilding a "
    "different architecture from the one that was trained."),
   ("This is the failure you are being inoculated against. Raw sensor values are orders of magnitude away from "
    "the standardised range the network trained on, so the predictions are garbage — but they are still valid "
    "class indices with confident probabilities. Nothing crashes. Nothing warns."),
   ("A model card states: what it predicts, what data trained it, how it scored (with the baseline for "
    "comparison), the known weaknesses from your confusion matrix, and explicitly where it must NOT be used — "
    "for example, on a machine type absent from the training data. Keep it to one page."),
  ],
  test=("models/qc_net_v1.pt exists and contains the state_dict, config, feat_mean, feat_std, metrics, class "
        "names, torch version and timestamp. reload_check.py runs in a fresh process, rebuilds QCNet from the "
        "checkpoint alone, and its assertion of identical predictions passes. models/qc_model_card.md states "
        "the metric, the baseline and at least one place the model must not be used."),
  troubleshoot=[
   ("RuntimeError: Error(s) in loading state_dict - size mismatch", "The config rebuilt a different architecture. Print the saved config and compare each layer size against QCNet's constructor."),
   ("UnpicklingError or a weights_only warning on torch.load", "Newer PyTorch defaults to weights_only=True. Pass weights_only=False for your own trusted checkpoint, and note in a comment that this is only safe for files you produced."),
   ("Predictions differ slightly between runs", "The model was left in train mode, so dropout is active. Confirm load_checkpoint calls model.eval() and the inference is inside no_grad()."),
   ("FileNotFoundError on models/qc_net_v1.pt", "Training did not reach the save call, or models/ does not exist. Add os.makedirs('models', exist_ok=True) inside save_checkpoint."),
   ("The checkpoint file is tens of megabytes", "The optimizer state or the whole model object is being saved. For this network the state_dict alone should be small."),
  ],
  stretch=[
   "Add the git commit hash to the checkpoint metadata so a model can be traced back to the exact code that made it.",
   "Save a second version with a different architecture as qc_net_v2.pt and write a script that compares the two checkpoints' metrics.",
   "Extend save_checkpoint to also store the optimizer state so training can be resumed, and prove it by resuming for 10 more epochs.",
  ],
 ),

]
