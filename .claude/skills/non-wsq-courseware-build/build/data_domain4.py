"""
Topic 4 hands-on activities - Vibe Coding Recurrent Networks for Sequence Data.

Labs 17-21, mapping 1:1 onto the five published sub-topics. Single source for the
PPT activity/step slides, the Learner Guide sections, the Lesson Plan schedule
rows and the labs/labNN-*/README.md files.
"""

DOMAIN4 = [

 dict(
  num=17, topic=4,
  title="Overview of RNNs, LSTM and GRU",
  objective="compare RNN, LSTM and GRU cells by their shapes, gates and parameter counts, and demonstrate why plain RNNs forget",
  desc=("Sequence models introduce a new axis and a new class of silent bug, so you explore the cells before "
        "you forecast with them. You prompt for a script that runs the same input batch through nn.RNN, "
        "nn.LSTM and nn.GRU, printing every output and hidden-state shape, the parameter count of each, and "
        "what LSTM returns that the others do not. You then demonstrate the vanishing-gradient problem "
        "directly: propagate a gradient back through a long sequence in an RNN and in an LSTM and compare the "
        "magnitudes at the first timestep. Finally you meet batch_first, the axis-order flag that silently "
        "trains models on nonsense."),
  build="An rnn_cells.py comparing the three cells' shapes, gates and parameter counts, plus a measured vanishing-gradient demonstration",
  services="PyTorch nn.RNN / nn.LSTM / nn.GRU, Cursor / GitHub Copilot / Claude",
  why=("Sequence bugs are the hardest to see because they never change a shape you would notice — reading the "
       "wrong axis out of an LSTM produces a tensor of exactly the right size and completely wrong contents. "
       "Understanding what each cell returns, in what order, is the only defence."),
  files=["rnn_cells.py", "reports/gradient_flow.png"],
  steps=[
   ("Prompt for the three-cell comparison. Insist on printing every returned object's shape, because that is where the confusion lives.",
    "Create rnn_cells.py comparing PyTorch's three recurrent cells.\nSHAPES: the input is a batch of 16 sequences, each 50 timesteps long with 1 feature per step, so shape (16, 50, 1) with batch_first=True.\nTASK: show exactly what each cell returns and how many parameters it has.\nOPERATIONS: build nn.RNN, nn.LSTM and nn.GRU each with input_size=1, hidden_size=32, num_layers=1, batch_first=True. For each: run the batch through it; print the shape of EVERY returned object, naming them (output, and h_n, and for LSTM also c_n); print the total parameter count; and print a one-line description of its gates - RNN has none, GRU has reset and update, LSTM has forget, input and output.\nThen print a comparison table of the three parameter counts and add a comment explaining why LSTM has roughly four times the parameters of RNN and GRU roughly three times.\nCONSTRAINTS: manual seed 42, no training. Label every printed line clearly."),
   ("Before running, predict two shapes: what shape is `output` and what shape is `h_n`? They are different, and confusing them is the classic bug.",
    ""),
   ("Run it and check your predictions.",
    "python rnn_cells.py"),
   ("Now demonstrate the batch_first trap, which is the reason sequence models silently learn nothing.",
    "Add a function batch_first_trap() to rnn_cells.py.\nOPERATIONS: take the same (16, 50, 1) input. Run it through an nn.LSTM built with batch_first=True and one built with batch_first=False, WITHOUT transposing the input for the second. Print the output shape from each and the shape of h_n from each.\nAdd a comment explaining what the second model actually did: it interpreted the batch axis as the time axis, so it treated 16 timesteps of a 50-sequence batch. Note that this raises NO error and produces a plausibly shaped tensor, which is why it is so dangerous.\nThen show the correct fix - transposing the input to (50, 16, 1) for the batch_first=False model - and confirm the output shapes now correspond."),
   ("Run it and study what happened. Confirm for yourself that nothing raised an error.",
    "python rnn_cells.py"),
   ("Measure the vanishing gradient rather than just describing it.",
    "Add a function vanishing_gradient() to rnn_cells.py.\nTASK: measure how gradient magnitude decays back through time in an RNN versus an LSTM.\nOPERATIONS: create an input of shape (1, 100, 1) with requires_grad=True. Pass it through a single-layer nn.RNN(1, 16, batch_first=True), take the LAST timestep of the output, sum it, and call backward(). Record the absolute gradient magnitude at the input for each of the 100 timesteps. Repeat with an nn.LSTM of the same size.\nPlot both gradient-magnitude curves against timestep on a LOG y axis and save to reports/gradient_flow.png. Print the ratio of gradient magnitude at timestep 0 to timestep 99 for each cell.\nCONSTRAINTS: same seed and hidden size for both so the comparison is fair."),
   ("Run it, open reports/gradient_flow.png, and read the two ratios. This is the vanishing gradient, measured.",
    "python rnn_cells.py"),
   ("Write your conclusion in prompts.md: which cell you will use in Lab 18, and the one-sentence reason.",
    ""),
  ],
  notes=[
   ("The shapes: `output` is (16, 50, 32) — the hidden state at EVERY timestep — while `h_n` is (1, 16, 32), "
    "the final hidden state only, with the leading 1 being num_layers. For forecasting you usually want the "
    "last timestep of `output`, which is `output[:, -1, :]`, or equivalently the squeezed `h_n`. Taking "
    "`output[:, 0, :]` by mistake gives you the state after ONE timestep — a right-shaped, useless tensor."),
   ("Parameter counts follow directly from the gates. An RNN has one weight set; a GRU has three (reset, "
    "update, candidate); an LSTM has four (forget, input, output, candidate). That is the whole explanation "
    "for the roughly 1:3:4 ratio you will see printed."),
   ("If your prediction of `h_n` missed the leading num_layers axis, note it. That leading 1 is the reason so "
    "much sequence code contains a `.squeeze(0)` whose purpose nobody remembers."),
   ("batch_first=False is the PyTorch DEFAULT, which is why this trap is so common: an assistant that omits the "
    "flag gives you a model expecting (seq, batch, feature) while your data is (batch, seq, feature). Nothing "
    "errors as long as the numbers happen to be compatible, and the model trains on transposed nonsense."),
   ("The lesson to carry forward: always pass batch_first explicitly, and always print the output shape of the "
    "first forward pass. Add that to your review checklist alongside the zero_grad and softmax checks."),
   ("Summing the last timestep and backpropagating to the input is a direct measurement of how far influence "
    "reaches back through time. The log y axis is necessary because the decay is exponential — on a linear "
    "axis the RNN's early timesteps are indistinguishable from zero."),
   ("Expect the RNN's gradient at timestep 0 to be orders of magnitude smaller than at timestep 99, while the "
    "LSTM's decays far more gently. That gap is precisely what the cell state and the forget gate buy you, and "
    "it is why LSTM and GRU replaced plain RNNs for anything longer than a few dozen steps."),
   ("The expected conclusion is LSTM or GRU, because the vibration series in Lab 18 uses a lookback well beyond "
    "the range where a plain RNN retains gradient. State it in your own words with the ratio you measured."),
  ],
  test=("rnn_cells.py prints output as torch.Size([16, 50, 32]) and h_n as torch.Size([1, 16, 32]) for all three "
        "cells, plus c_n for the LSTM only, with parameter counts in roughly a 1:3:4 ratio. batch_first_trap "
        "shows a wrongly shaped-but-unraised result and its fix. reports/gradient_flow.png shows the RNN "
        "gradient decaying far faster than the LSTM's on a log axis."),
  troubleshoot=[
   ("The gradient plot is empty or entirely flat", "The input was not created with requires_grad=True, or the gradient was read from the wrong tensor. Grab the gradient from the input tensor, per timestep."),
   ("RuntimeError about the input having an unexpected number of dimensions", "The (batch, seq, feature) 3-D shape is required. A 2-D input of (batch, seq) needs an explicit `.unsqueeze(-1)` to add the feature axis."),
   ("The LSTM returns two objects and the code unpacks three", "nn.LSTM returns `output, (h_n, c_n)` — the second is a tuple. Unpack as `out, (h, c) = lstm(x)`."),
   ("Parameter counts do not show the expected ratio", "Check all three were built with the same hidden_size and num_layers. Bias terms shift the ratio slightly, which is normal."),
   ("Gradient values are all nan", "The sum over a long RNN can explode rather than vanish. Reduce the sequence to 50 steps or lower the hidden size, and note that exploding gradients are the other half of the same problem."),
  ],
  stretch=[
   "Add a 2-layer LSTM and see how the h_n leading axis changes.",
   "Repeat the vanishing-gradient measurement with a GRU and place it between the RNN and the LSTM.",
   "Add gradient clipping to the RNN case and observe what it fixes and what it does not.",
  ],
 ),

 dict(
  num=18, topic=4,
  title="Vibe Coding an LSTM for Time Series Forecasting",
  objective="window and split a time series correctly and train an LSTM that beats a persistence baseline",
  desc=("Forecast ForgeSight's vibration sensor. This lab contains two traps that no error message will ever "
        "reveal. First, sequence data must be split by TIME — a random split lets the model train on data "
        "from after the period it is tested on. Second, the scaler must be fitted on the training window only. "
        "You establish the persistence baseline before modelling, prompt for the windowing and the LSTM with "
        "both constraints stated explicitly, then verify the split and the scaler yourself before trusting any "
        "score. A forecaster that cannot beat 'tomorrow equals today' has not learned the series."),
  build="An LSTM vibration forecaster with a time-ordered split and train-only scaling, measured against a persistence baseline",
  services="PyTorch nn.LSTM, pandas, engine.py, Cursor / GitHub Copilot / Claude",
  why=("Time-series leakage is the most convincing bug in machine learning — it produces beautiful plots and "
       "excellent metrics that vanish entirely in production. Learning to check the split and the scaler "
       "before reading the score is the habit that protects every forecast you build afterwards."),
  files=["windowing.py", "lstm_forecast.py", "reports/forecast.png"],
  steps=[
   ("Look at the series before modelling it, and establish the persistence baseline.",
    "Create baseline_series.py for data/vibration_series.csv, which has a timestamp column and a vibration_rms column of roughly 3000 hourly readings.\nOPERATIONS: (1) load it, print the row count, the date range and basic statistics; (2) plot the full series to reports/series_overview.png; (3) compute the PERSISTENCE baseline on the last 20% of the series - predicting that each value equals the previous value - and print its MAE and RMSE; (4) print the standard deviation of the series for comparison.\nAdd a comment explaining why persistence is the correct baseline for a forecasting task and what it means if a model cannot beat it."),
   ("Run it, open the plot, and write the persistence MAE down. This is the number your LSTM must beat.",
    "python baseline_series.py"),
   ("Prompt for the windowing, with the time-split constraint stated as the primary requirement.",
    "Create windowing.py that prepares data/vibration_series.csv for an LSTM.\nTASK: turn one long univariate series into supervised (lookback -> horizon) training pairs.\nOPERATIONS: build make_windows(series, lookback=48, horizon=1) returning X of shape (N, lookback, 1) and y of shape (N, 1). Then build get_series_loaders(lookback=48, batch_size=64) that:\n1. Loads the series in TIME ORDER - never sorted or shuffled before splitting\n2. Splits chronologically into the first 70% train, next 15% validation, final 15% test - by INDEX, not randomly\n3. Fits the scaler using the TRAINING SEGMENT ONLY, computing mean and std from the training values, then applies those same values to validation and test\n4. Windows each segment separately so no window ever spans a split boundary\n5. Returns loaders plus the scaler statistics so predictions can be converted back to the original units\nCONSTRAINTS: this is critical - do NOT use train_test_split with shuffle, and do NOT fit the scaler on the full series. Shuffling is allowed WITHIN the training loader only. Print the index ranges of the three splits and the training mean and std when run as a script.\nEXPECTED OUTPUT: printed split ranges proving they are contiguous and non-overlapping, plus X and y shapes for each split."),
   ("Verify the two traps yourself. Check the printed split ranges are contiguous and in ascending order, and find the exact line where the scaler statistics are computed.",
    "python windowing.py"),
   ("Prompt for the model and training, reusing the engine again.",
    "Create lstm_forecast.py.\nSHAPES: batches are X of (64, 48, 1) and y of (64, 1).\nOPERATIONS: add class VibrationLSTM(nn.Module) to models.py with nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2) followed by nn.Linear(64, 1). In forward, take the LAST timestep of the LSTM output with output[:, -1, :] and pass it to the Linear layer - add a comment on why this is the right slice and what output[:, 0, :] would give instead.\nTrain with fit from engine.py: nn.MSELoss, Adam lr=1e-3, 60 epochs, an MAE metric computed in the ORIGINAL units by inverting the scaling.\nCONSTRAINTS: batch_first=True must be explicit. Print the output shape of the first forward pass to prove the axis order.\nEXPECTED OUTPUT: per-epoch losses, final test MAE and RMSE in original units, printed next to the persistence baseline MAE, and a forecast-versus-actual plot of the test segment saved to reports/forecast.png."),
   ("Train it and compare the MAE against your persistence baseline.",
    "python lstm_forecast.py"),
   ("Open reports/forecast.png and look at the shape of the prediction, not just the metric.",
    ""),
   ("Prove to yourself that the time split mattered, by deliberately doing it wrong.",
    "Add a function leaky_comparison() to lstm_forecast.py that repeats the entire pipeline with two deliberate errors: a RANDOM shuffled split instead of a chronological one, and a scaler fitted on the FULL series before splitting. Train the identical model for the same 60 epochs and print the resulting test MAE next to the correct pipeline's MAE and the persistence baseline. Add a comment stating which number you would have reported if you had never checked the split, and why it is not achievable in production."),
  ],
  notes=[
   ("Persistence is a genuinely strong baseline for smooth physical signals. Vibration does not jump randomly "
    "hour to hour, so 'the same as last hour' is often quite accurate. Beating it requires the model to learn "
    "actual structure — trend, cycles, drift — rather than just tracking the level."),
   ("Write the persistence MAE in review_notes.md next to your Lab 8 and Lab 14 numbers. If your LSTM's MAE is "
    "higher than this, the honest report is that the model adds nothing, however good the plot looks."),
   ("Requirement 4 is subtle and often missed: if you window the full series first and then split, windows near "
    "the boundary contain values from both sides, so training windows include validation timesteps. Split "
    "first, then window each segment independently."),
   ("Read the printed ranges: they must look like 0-2099, 2100-2549, 2550-2999, contiguous and ascending. If "
    "the indices are scattered, a shuffle happened. And find the scaler line — it must compute from the "
    "training segment. If it uses the whole series, the future has leaked into the past."),
   ("`output[:, -1, :]` takes the hidden state after the model has seen all 48 timesteps, which is what you "
    "want for a forecast. `output[:, 0, :]` is the state after ONE timestep — same shape, and the model would "
    "be forecasting from a single reading while appearing to work perfectly."),
   ("A meaningful improvement over persistence is a genuine success here. A small improvement is a common and "
    "honest outcome on a smooth series. If your MAE is dramatically better than persistence, be suspicious "
    "before being pleased — re-check the split and the scaler first."),
   ("Look for a specific failure mode: a forecast that is simply the previous value shifted right by one step. "
    "That is the model learning persistence and nothing more. It looks excellent on a plot and is worth "
    "spotting — compare the curve's turning points against the actual series to see whether it anticipates or "
    "merely follows."),
   ("Expect the leaky pipeline to report a noticeably better MAE than the correct one. That number is the "
    "one you would have proudly presented, and it is unobtainable in production because you cannot scale "
    "today's reading using next month's statistics. This is the time-series version of the Lab 6 lesson."),
  ],
  test=("windowing.py prints three contiguous ascending index ranges and the scaler statistics computed from the "
        "training segment only. lstm_forecast.py prints the first forward pass shape as torch.Size([64, 1]), "
        "reports a test MAE in original units alongside the persistence baseline, and saves "
        "reports/forecast.png. leaky_comparison prints a better-but-unachievable MAE from the shuffled, "
        "fully-scaled pipeline."),
  troubleshoot=[
   ("The model's MAE is worse than persistence", "Common and often honest. Try a longer lookback, more epochs, or a lower learning rate — and if it still cannot beat persistence, report that finding rather than hiding it."),
   ("Loss falls to almost zero immediately", "Leakage, or the target is included in the input window. Confirm y comes from the timestep AFTER the last input timestep, not from within the window."),
   ("RuntimeError: input must have 3 dimensions, got 2", "Add the feature axis with `.unsqueeze(-1)` so the shape becomes (N, lookback, 1)."),
   ("Predictions are all the same value", "The model collapsed to the mean. Check the scaler statistics are sensible, lower the learning rate, and confirm the last-timestep slice is correct."),
   ("The forecast plot is a flat line", "You may be plotting scaled values while the actuals are in original units. Invert the scaling on both before plotting."),
  ],
  stretch=[
   "Change the horizon to 6 steps ahead and see how much the error grows relative to persistence.",
   "Swap the LSTM for a GRU with the same hidden size and compare accuracy and training time.",
   "Add the coolant temperature as a second input feature and see whether a multivariate model helps.",
  ],
 ),

 dict(
  num=19, topic=4,
  title="Tuning Sequence Models with Follow-Up Prompts",
  objective="improve a sequence model through disciplined follow-up prompts and record what each change actually bought",
  desc=("Tuning is where vibe coding either pays off or descends into random prompting. You run a structured "
        "search over the choices that matter for a sequence model — lookback length, hidden size, layer count, "
        "cell type and learning rate — changing one thing at a time and recording every result in a table. "
        "The discipline being trained is the follow-up prompt: naming a specific symptom and a specific change "
        "rather than asking the assistant to 'make it better'. You finish with a tuned model, a results table "
        "that justifies it, and an explicit note of which changes made no difference at all."),
  build="A tuning results table across lookback, hidden size, layers, cell type and learning rate, plus a justified best model",
  services="PyTorch, engine.py, Cursor / GitHub Copilot / Claude",
  why=("An assistant will happily change five things at once when asked to improve a model, leaving you unable "
       "to explain any result. One-factor-at-a-time tuning with a recorded table is slower per step and far "
       "faster overall, and it is the only version you can defend to anyone."),
  files=["tune_lstm.py", "reports/tuning_results.csv", "prompts.md"],
  steps=[
   ("First, see what an undisciplined prompt produces, so the contrast is concrete.",
    "My LSTM forecaster gets a test MAE only slightly better than a persistence baseline. Make it better."),
   ("Read what came back and count how many things it changed simultaneously. Do not run it — this is evidence, not code.",
    ""),
   ("Now the disciplined version. Prompt for a structured one-factor-at-a-time search.",
    "Create tune_lstm.py running a one-factor-at-a-time search over the vibration forecasting model.\nBASELINE CONFIG: lookback=48, hidden_size=64, num_layers=2, cell=LSTM, lr=1e-3, 40 epochs.\nRUNS: starting from that baseline, vary ONE factor at a time:\n- lookback in [12, 24, 48, 96]\n- hidden_size in [16, 32, 64, 128]\n- num_layers in [1, 2, 3]\n- cell in [LSTM, GRU]\n- lr in [1e-2, 1e-3, 1e-4]\nFor each run record: the factor changed, its value, best validation MAE in original units, test MAE in original units, epochs to best, and wall-clock seconds.\nCONSTRAINTS: reset manual seed 42 before every run and build a fresh model each time. Identical data and batch size across all runs. Include the persistence baseline MAE as a constant column so every row can be read against it.\nEXPECTED OUTPUT: a printed table grouped by factor, saved to reports/tuning_results.csv, and a figure with one subplot per factor showing test MAE against the factor's value."),
   ("Run the search. It is a few dozen short training runs — expect several minutes on CPU.",
    "python tune_lstm.py"),
   ("Read the table by factor and answer: which factor produced the largest spread in test MAE, and which produced almost none?",
    ""),
   ("Practise the disciplined follow-up on whatever your table actually showed. Adapt this to your own result.",
    "In tune_lstm.py the lookback sweep shows test MAE improving from 12 to 48 timesteps but getting worse at 96, which suggests the longer window adds noise rather than signal. Add two intermediate values, 64 and 80, to the lookback sweep only. Change nothing else - same seed, same architecture, same number of epochs - and print the extended lookback results as a single sorted table so I can see where the minimum actually sits."),
   ("Train the winning configuration properly and check it against the baseline one final time.",
    "Add a function train_best() to tune_lstm.py that takes the best configuration from reports/tuning_results.csv, trains it for 120 epochs with early stopping on validation loss, reports test MAE and RMSE in original units next to the persistence baseline, saves the model with save_checkpoint to models/vibration_lstm_v1.pt recording the full winning configuration in the metrics dict, and writes models/vibration_model_card.md stating the task, the lookback, the baseline comparison and the fact that the split was chronological."),
   ("Record in prompts.md the three follow-up prompts that worked and, just as importantly, which tuned factors turned out not to matter.",
    ""),
  ],
  notes=[
   ("Expect the assistant to return a model with a different hidden size, an extra layer, a new learning rate, "
    "bidirectionality and possibly attention — all at once. If it improves, you cannot say why; if it gets "
    "worse, you cannot say why either. This is the failure mode the rest of the lab prevents."),
   ("Count the changes. Four or five is typical. That is not tuning, it is replacement — and it is exactly what "
    "'make it better' asks for. The prompt was the problem, not the assistant."),
   ("One factor at a time is not the most sample-efficient search that exists, but it is the one that produces "
    "an explanation. Keeping the persistence baseline as a constant column is what stops a table of tiny "
    "differences from looking like progress when nothing beats the baseline."),
   ("Fresh model and reset seed per run is the clause assistants most often drop. If run 2 continues from run "
    "1's weights the whole table is meaningless — check the generated code for a model built inside the loop, "
    "not outside it."),
   ("Typical findings: lookback and learning rate produce the largest spread; hidden size shows diminishing "
    "returns past a point; num_layers beyond 2 usually adds time without accuracy; LSTM and GRU come out very "
    "close, with GRU faster. Knowing which knobs do nothing is as valuable as knowing which ones work."),
   ("Notice the structure of this follow-up: it names the file, states the observed pattern with numbers, "
    "offers an interpretation, specifies the exact change, and explicitly fences everything else with 'change "
    "nothing else'. That fence is what keeps your tuning table comparable across rounds."),
   ("The winning configuration deserves a longer, properly early-stopped run — the sweep used 40 epochs to keep "
    "the search fast, which usually undertrains. Recording the full configuration in the checkpoint means the "
    "result is reproducible rather than remembered."),
   ("Writing down what did NOT help is the part everyone skips and the part most worth having. Next time you "
    "tune a sequence model you will start from lookback and learning rate instead of adding layers."),
  ],
  test=("reports/tuning_results.csv contains one row per configuration with the factor changed, both MAE columns "
        "and the persistence baseline as a constant column. You can name the factor with the largest spread and "
        "the one with the smallest. models/vibration_lstm_v1.pt holds the winning configuration and beats the "
        "persistence baseline, and prompts.md records three working follow-up prompts."),
  troubleshoot=[
   ("Every configuration gives an identical MAE", "The model is not being rebuilt per run, or the factor is not actually reaching the constructor. Print the configuration at the start of each run to confirm."),
   ("The sweep takes too long", "Cut epochs to 25 for the search only, then train the winner properly. Note the reduction so the table stays honest."),
   ("The best configuration performs worse when retrained longer", "Overfitting at higher epoch counts. Use the early-stopped best weights, which is what train_best is for."),
   ("lookback=96 raises a shape or memory error", "Longer windows mean fewer samples and more memory per batch. Reduce the batch size for that run and note it in the table."),
   ("GRU and LSTM results are identical to several decimal places", "Suspicious — confirm the cell type is actually being switched rather than the string only being recorded."),
  ],
  stretch=[
   "Add a bidirectional LSTM to the cell sweep and consider carefully whether it is even legitimate for forecasting.",
   "Replace the grid with a small random search over the same ranges and compare how quickly each finds the best configuration.",
   "Add a 2-factor interaction check on the two most influential factors and see whether their effects are independent.",
  ],
 ),

 dict(
  num=20, topic=4,
  title="Evaluating and Visualizing Model Performance",
  objective="build the evaluation views that reveal what a single metric hides, across all three ForgeSight models",
  desc=("One number cannot describe a model. You build the standard evaluation views for each of the three "
        "ForgeSight models — a parity plot and residual analysis for tool-wear regression, a normalised "
        "confusion matrix with per-class recall for defect classification, and forecast-versus-actual with "
        "error-by-horizon for vibration. Each view is chosen to answer a question the headline metric cannot: "
        "where is the model biased, which classes does it confuse, and how fast does the forecast decay. You "
        "assemble them into one report figure and write the honest summary that goes with it."),
  build="A complete evaluation report covering all three models, with the plots and the written findings a stakeholder would need",
  services="PyTorch, matplotlib, scikit-learn metrics, Cursor / GitHub Copilot / Claude",
  why=("A single accuracy or MAE hides exactly the failures that matter — the class never predicted, the "
       "systematic bias at high values, the forecast that is really just persistence. These views are what "
       "turn a number into a decision you can defend."),
  files=["evaluate.py", "reports/evaluation_report.png", "reports/findings.md"],
  steps=[
   ("Build the regression evaluation views for the tool-wear model.",
    "Create evaluate.py with a function eval_regression() for the tool-wear model.\nOPERATIONS: load models/wear_net_v1.pt (or retrain briefly if it was not saved), predict on the test split, and produce a 3-panel figure saved to reports/eval_wear.png:\n(1) a parity plot of predicted against actual with a y=x reference line and the R-squared value in the title;\n(2) a residual plot of (predicted - actual) against actual, with a horizontal zero line;\n(3) a histogram of residuals with the mean and standard deviation printed in the title.\nAlso print test MAE, RMSE and R-squared next to the mean-predictor baseline.\nAdd a comment on what a residual plot that fans out or trends with the actual value tells you about the model."),
   ("Run it and read panel 2 specifically. Are the residuals centred on zero across the whole range, or does the model drift at high wear values?",
    "python evaluate.py"),
   ("Build the classification evaluation views.",
    "Add eval_classification() to evaluate.py for the defect CNN.\nOPERATIONS: load the best defect model, predict on the validation set, and produce a 3-panel figure saved to reports/eval_defects.png:\n(1) a confusion matrix normalised BY ROW so each cell shows the share of that true class, with class names on both axes and values annotated;\n(2) a horizontal bar chart of per-class precision, recall and F1;\n(3) a histogram of the model's maximum softmax probability, split into correct and incorrect predictions.\nPrint the classification report and the macro-averaged F1 next to the majority-class baseline accuracy.\nAdd a comment explaining why a row-normalised confusion matrix is more readable than raw counts when classes are imbalanced, and note that softmax is applied HERE for reporting, not in the model."),
   ("Run it and read panel 3. Is the model well calibrated — are its wrong answers given with lower confidence than its right ones?",
    "python evaluate.py"),
   ("Build the forecasting evaluation views, including the check for the model that only learned persistence.",
    "Add eval_forecast() to evaluate.py for the vibration LSTM.\nOPERATIONS: load models/vibration_lstm_v1.pt, predict on the test segment, and produce a 3-panel figure saved to reports/eval_forecast.png:\n(1) forecast against actual over the test period in ORIGINAL units, with the persistence baseline as a third line;\n(2) a residual-over-time plot showing whether errors grow towards the end of the period;\n(3) a scatter of predicted change against actual change from one timestep to the next - this reveals whether the model is anticipating movement or merely repeating the last value.\nPrint test MAE and RMSE against the persistence baseline, plus the correlation between predicted and actual CHANGE.\nAdd a comment explaining that a model whose predicted change correlates near zero with actual change has learned persistence and nothing more."),
   ("Run it and read panel 3 and the change-correlation figure. This is the honest test of whether your forecaster learned anything.",
    "python evaluate.py"),
   ("Assemble the single report figure and write the findings.",
    "Add build_report() to evaluate.py that assembles one combined figure saved to reports/evaluation_report.png with a row per model - regression, classification, forecasting - showing the single most informative panel from each, with a clear row label and a shared title naming the course project and the date. Then write reports/findings.md with one short section per model containing: the headline metric next to its baseline, the most important thing the plots reveal that the metric does not, and one stated limitation of the model."),
   ("Read your own findings.md aloud as if presenting to a plant manager. If any sentence would not survive a follow-up question, rewrite it.",
    ""),
  ],
  notes=[
   ("A parity plot shows immediately what R-squared summarises away. Points hugging the y=x line across the "
    "whole range is a good model; points that flatten towards the mean at the extremes is a model that is "
    "hedging — accurate on average and useless where it matters, which is usually at high wear."),
   ("Residuals that trend with the actual value mean systematic bias, not noise. Residuals that fan out mean "
    "the model is reliable for low values and unreliable for high ones. Either is a finding worth reporting; "
    "neither is visible in the MAE."),
   ("Row normalisation answers 'of the parts that really were scratches, what share did we catch?' — which is "
    "the question a factory actually asks. Raw counts make a rare class look fine simply because there are few "
    "of them. And note where softmax appears: at reporting time, never in the model, exactly as Lab 9 established."),
   ("A well-calibrated model is wrong with low confidence. If the two histograms in panel 3 overlap heavily, "
    "the model is confidently wrong a lot of the time, which matters enormously if anyone downstream plans to "
    "trust a confidence threshold to decide when to involve a human inspector."),
   ("Panel 3 is the panel that matters. A forecaster that has only learned persistence produces a "
    "predicted-change scatter that is essentially a flat cloud around zero — it never anticipates a move. A "
    "model that genuinely learned structure shows positive correlation between predicted and actual change."),
   ("Do not be discouraged by a modest change-correlation. Anticipating short-term movement in a noisy physical "
    "signal is genuinely hard, and reporting a small honest number with the plot that proves it is far more "
    "professional than reporting an MAE that hides it."),
   ("The report figure is the artifact people will actually look at. One panel per model, clearly labelled, "
    "with the date and project name, is what makes it usable a month later when nobody remembers the run."),
   ("The 'stated limitation' line is not modesty, it is scope control. 'Trained only on machine types A and B; "
    "predictions for type C are not supported' is the sentence that prevents your model being misused, and it "
    "goes straight into the model card."),
  ],
  test=("reports/eval_wear.png, reports/eval_defects.png and reports/eval_forecast.png each show three labelled "
        "panels. reports/evaluation_report.png combines one panel per model with a title and date. "
        "reports/findings.md has three sections, each naming a metric with its baseline, an insight the plot "
        "revealed that the metric did not, and a stated limitation."),
  troubleshoot=[
   ("A model file is missing", "Retrain it briefly with the saved configuration, or use the checkpoint from the relevant lab. Every model should have been saved in Labs 11, 15 and 19."),
   ("The confusion matrix rows do not sum to 1", "Normalisation was applied over the wrong axis. Row normalisation divides by the true-class totals, which is axis 1 in sklearn's normalize='true'."),
   ("R-squared is negative", "The model is worse than predicting the mean. That is a real result — check the model loaded correctly and that the scaling was inverted before comparing."),
   ("The forecast plot's three lines are indistinguishable", "Zoom into a shorter window, perhaps 200 timesteps, and use distinct line styles. A full-range plot of a smooth series hides all the detail."),
   ("Predicted and actual change correlate at almost exactly zero", "The model learned persistence. Report it honestly, and consider a longer lookback or additional features as the stated next step."),
  ],
  stretch=[
   "Add prediction intervals to the forecast plot using dropout at inference time as a rough uncertainty estimate.",
   "Add a calibration curve for the classifier and check whether its confidence matches its accuracy.",
   "Segment the regression residuals by machine type and check whether the model is systematically worse on one of them.",
  ],
 ),

 dict(
  num=21, topic=4,
  title="Packaging a Complete Deep Learning Project",
  objective="package models, code, dependencies and documentation into a project another person can run",
  desc=("Turn twenty labs of scripts into something you could hand to a colleague and walk away from. You "
        "restructure the workspace into a clean layout, pin the dependencies, write a predict.py command-line "
        "tool that loads any of the three saved models and scores new input, add a smoke test that proves the "
        "prediction path works end to end, and write the README and model cards. The real test is the last "
        "step: a partner clones your project into a fresh folder and runs it from the README alone, without "
        "asking you a single question."),
  build="A packaged, documented, tested ForgeSight project with a working CLI that another person can run unaided",
  services="Python packaging, pytest, git, Cursor / GitHub Copilot / Claude",
  why=("A project that only runs on your laptop, in your head, in the order you happen to remember, is not a "
       "deliverable. Packaging is what converts two days of learning into something with a life beyond the "
       "classroom — and the handover test is the only honest way to know you have done it."),
  files=["README.md", "requirements.txt", "predict.py", "tests/test_predict.py", ".gitignore"],
  steps=[
   ("Restructure the workspace. Ask for a plan first rather than letting the assistant move files immediately.",
    "Review my torch-vibe project folder and propose a clean package structure for handover. Do NOT move anything yet - give me the proposed layout as a tree with a one-line purpose for each folder and each top-level file.\nThe project contains: data loading for tabular and image and series data; three trained models (tool-wear regression, defect CNN, vibration LSTM); a shared training engine; a checkpoint helper; evaluation scripts; saved checkpoints; and generated reports.\nConstraints: source code separated from data, models and reports; every model loadable without running any training script; no absolute paths anywhere; data and model binaries excluded from version control but their absence explained in the README."),
   ("Review the proposed tree and adjust it before anything moves. You are the architect here, not the assistant.",
    ""),
   ("Apply the restructure and fix the imports it breaks.",
    "Apply the agreed structure. Move the files, update every import to match, and confirm each of the three training scripts and the evaluation script still run from the project root. Replace any absolute path with a path relative to the project root resolved via pathlib. Report every file you moved and every import you changed."),
   ("Write the prediction CLI — the thing that makes the project usable by someone who will never read your training code.",
    "Create predict.py, a command-line tool for the packaged project.\nUSAGE: `python predict.py --model wear --input data/sample_reading.csv`, `python predict.py --model defect --input path/to/image.png`, `python predict.py --model vibration --input data/recent_series.csv`.\nOPERATIONS: for each model - load the checkpoint, apply the SAME preprocessing the model was trained with by reading the saved statistics from the checkpoint, run inference under torch.no_grad() and model.eval(), and print a human-readable result. For the defect model print the predicted class name and its probability; for wear print the predicted microns; for vibration print the next predicted value and the persistence value alongside it for comparison.\nCONSTRAINTS: validate the input file exists and has the expected columns or format, and fail with a clear message naming what is wrong rather than a traceback. Never re-fit any scaler - always use the statistics stored in the checkpoint. Include --help text for every argument.\nEXPECTED OUTPUT: three working commands, each printing a labelled prediction."),
   ("Test all three commands yourself with the sample inputs.",
    "python predict.py --model defect --input data/defects/val/scratch/scratch_001.png\npython predict.py --model wear --input data/sample_reading.csv\npython predict.py --model vibration --input data/vibration_series.csv"),
   ("Add the smoke test that proves the prediction path works, so a future change that breaks it fails loudly.",
    "Create tests/test_predict.py using pytest.\nTESTS: (1) each of the three checkpoints loads and returns a model in eval mode; (2) each model produces an output of the expected shape for a single synthetic input; (3) the defect model's softmax probabilities sum to 1 and the predicted class is a valid index; (4) predict.py exits with a clear non-zero status and a readable message for a missing input file; (5) the wear model's prediction changes when the input changes, which catches a model that always returns the same value.\nCONSTRAINTS: tests must run without retraining anything and must not require internet access. Skip cleanly with a clear message if a checkpoint file is absent."),
   ("Run the tests and confirm they pass.",
    "python -m pytest tests/ -v"),
   ("Write the README and finalise the model cards, then pin the dependencies.",
    "Write README.md for the packaged project covering: what ForgeSight does and the three models it contains; the exact setup commands from a fresh clone including creating the virtual environment and installing requirements; how to obtain or regenerate the data, since data files are not in version control; how to run each of the three predictions with a copyable example command; how to retrain each model; the project structure as a tree; and a results table of each model's headline metric next to its baseline. Also create .gitignore excluding .venv, __pycache__, data/, models/*.pt and reports/. Then regenerate requirements.txt with pinned versions."),
   ("Run the handover test, which is the only verification that counts.",
    ""),
  ],
  notes=[
   ("Asking for the plan before the action is the pattern to carry away from this lab. A restructure that moves "
    "thirty files and rewrites the imports is exactly the kind of change you want to review as a diagram "
    "first — reverting it afterwards is far more work than reading a tree."),
   ("A reasonable layout: src/ for data, models, engine and checkpoint code; scripts/ for the training entry "
    "points; tests/ for the smoke tests; data/, models/ and reports/ for artifacts; and README.md, "
    "requirements.txt and .gitignore at the root. Push back if the proposal buries the entry points."),
   ("Absolute paths are the most common reason a project fails on someone else's machine, and they are "
    "invisible on yours. `pathlib.Path(__file__).resolve().parent.parent` gives you a project root you can "
    "build every other path from."),
   ("This is the deliverable. Everything before this lab was for you; predict.py is for someone else. The "
    "clause about reading preprocessing statistics from the checkpoint is the direct payoff of Lab 11 — "
    "without it, this tool would feed raw values to a model trained on standardised ones and print confident "
    "nonsense, exactly as you demonstrated then."),
   ("Test the error paths too, not just the happy ones. Point it at a file that does not exist and at a CSV "
    "with a missing column, and check the message names the actual problem. A tool that tracebacks at a "
    "colleague has not been handed over, it has been abandoned."),
   ("Test 5 is the one people leave out and the one that catches the most embarrassing failure: a model that "
    "loads, runs, returns the right shape and predicts the identical value for every input. All the other "
    "tests pass. Only varying the input reveals it."),
   ("Tests that need retraining or internet are tests nobody runs. Under a second, offline, is the target."),
   ("The README's real audience is a capable stranger, which in practice is you in three months. If a step "
    "assumes something you happen to know today, it is a bug in the README. The results table with baselines "
    "is what stops anyone quoting your accuracy without its context."),
   ("The handover test: swap projects with a partner, clone into a fresh folder, and follow their README "
    "without speaking. Every question you have to ask is a defect in their documentation, and every question "
    "they ask is a defect in yours. This is the most useful fifteen minutes of the two days."),
  ],
  test=("A partner clones your project into a fresh folder, follows README.md alone, creates the environment, "
        "installs from requirements.txt and successfully runs all three `python predict.py` commands without "
        "asking you anything. `python -m pytest tests/ -v` passes. No absolute path appears anywhere in the "
        "source, and each model has a model card stating its metric, its baseline and where it must not be used."),
  troubleshoot=[
   ("ModuleNotFoundError after the restructure", "Run from the project root, and add __init__.py files to the package folders — or install the project in editable mode with `pip install -e .`."),
   ("predict.py loads the model but predicts nonsense", "The preprocessing statistics from the checkpoint are not being applied. This is the exact failure demonstrated in Lab 11 — print the input tensor before inference and check its scale."),
   ("pytest cannot find the tests", "Run `python -m pytest` from the project root rather than `pytest`, so the current directory is on the path."),
   ("Your partner cannot get the data", "Data is correctly excluded from version control, but the README must say how to regenerate it — point at make_images.py and the resources folder."),
   ("requirements.txt has hundreds of entries", "pip freeze captures everything in the venv. Either keep it, which is safest for reproducibility, or list only the direct dependencies with pinned versions and say which approach you chose."),
  ],
  stretch=[
   "Add a Dockerfile so the project runs identically without any local Python setup.",
   "Wrap predict.py in a small Streamlit or Gradio interface so a non-programmer can drop in an image and see the prediction.",
   "Add a GitHub Actions workflow that runs the smoke tests on every push.",
  ],
 ),

]
