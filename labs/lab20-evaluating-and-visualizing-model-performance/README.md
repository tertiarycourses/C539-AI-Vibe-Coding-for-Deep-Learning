# Lab 20 — Evaluating and Visualizing Model Performance

> **Course:** AI Vibe Coding with PyTorch Deep Learning (`C539`) · **Topic 04:** Vibe Coding Recurrent Networks for Sequence Data  
> **Learning outcome:** Build the evaluation views that reveal what a single metric hides, across all three ForgeSight models.

## Goal

One number cannot describe a model. You build the standard evaluation views for each of the three ForgeSight models — a parity plot and residual analysis for tool-wear regression, a normalised confusion matrix with per-class recall for defect classification, and forecast-versus-actual with error-by-horizon for vibration. Each view is chosen to answer a question the headline metric cannot: where is the model biased, which classes does it confuse, and how fast does the forecast decay. You assemble them into one report figure and write the honest summary that goes with it.

## Why this lab matters

A single accuracy or MAE hides exactly the failures that matter — the class never predicted, the systematic bias at high values, the forecast that is really just persistence. These views are what turn a number into a decision you can defend.

## What you'll build

**A complete evaluation report covering all three models, with the plots and the written findings a stakeholder would need**

**Tools:** PyTorch, matplotlib, scikit-learn metrics, Cursor / GitHub Copilot / Claude

**Files you end up with:**

- `evaluate.py`
- `reports/evaluation_report.png`
- `reports/findings.md`

## Before you start

- Lab 19 completed, with your `torch-vibe/` workspace and virtual environment active.
- Your AI coding assistant open and able to see the files in the workspace.

## Steps

### 1. Build the regression evaluation views for the tool-wear model.

A parity plot shows immediately what R-squared summarises away. Points hugging the y=x line across the whole range is a good model; points that flatten towards the mean at the extremes is a model that is hedging — accurate on average and useless where it matters, which is usually at high wear.

**PROMPT** — paste this into your AI coding assistant:

```text
Create evaluate.py with a function eval_regression() for the tool-wear model.
OPERATIONS: load models/wear_net_v1.pt (or retrain briefly if it was not saved), predict on the test split, and produce a 3-panel figure saved to reports/eval_wear.png:
(1) a parity plot of predicted against actual with a y=x reference line and the R-squared value in the title;
(2) a residual plot of (predicted - actual) against actual, with a horizontal zero line;
(3) a histogram of residuals with the mean and standard deviation printed in the title.
Also print test MAE, RMSE and R-squared next to the mean-predictor baseline.
Add a comment on what a residual plot that fans out or trends with the actual value tells you about the model.
```

### 2. Run it and read panel 2 specifically. Are the residuals centred on zero across the whole range, or does the model drift at high wear values?

Residuals that trend with the actual value mean systematic bias, not noise. Residuals that fan out mean the model is reliable for low values and unreliable for high ones. Either is a finding worth reporting; neither is visible in the MAE.

**COMMAND** — run this in your terminal:

```bash
python evaluate.py
```

### 3. Build the classification evaluation views.

Row normalisation answers 'of the parts that really were scratches, what share did we catch?' — which is the question a factory actually asks. Raw counts make a rare class look fine simply because there are few of them. And note where softmax appears: at reporting time, never in the model, exactly as Lab 9 established.

**PROMPT** — paste this into your AI coding assistant:

```text
Add eval_classification() to evaluate.py for the defect CNN.
OPERATIONS: load the best defect model, predict on the validation set, and produce a 3-panel figure saved to reports/eval_defects.png:
(1) a confusion matrix normalised BY ROW so each cell shows the share of that true class, with class names on both axes and values annotated;
(2) a horizontal bar chart of per-class precision, recall and F1;
(3) a histogram of the model's maximum softmax probability, split into correct and incorrect predictions.
Print the classification report and the macro-averaged F1 next to the majority-class baseline accuracy.
Add a comment explaining why a row-normalised confusion matrix is more readable than raw counts when classes are imbalanced, and note that softmax is applied HERE for reporting, not in the model.
```

### 4. Run it and read panel 3. Is the model well calibrated — are its wrong answers given with lower confidence than its right ones?

A well-calibrated model is wrong with low confidence. If the two histograms in panel 3 overlap heavily, the model is confidently wrong a lot of the time, which matters enormously if anyone downstream plans to trust a confidence threshold to decide when to involve a human inspector.

**COMMAND** — run this in your terminal:

```bash
python evaluate.py
```

### 5. Build the forecasting evaluation views, including the check for the model that only learned persistence.

Panel 3 is the panel that matters. A forecaster that has only learned persistence produces a predicted-change scatter that is essentially a flat cloud around zero — it never anticipates a move. A model that genuinely learned structure shows positive correlation between predicted and actual change.

**PROMPT** — paste this into your AI coding assistant:

```text
Add eval_forecast() to evaluate.py for the vibration LSTM.
OPERATIONS: load models/vibration_lstm_v1.pt, predict on the test segment, and produce a 3-panel figure saved to reports/eval_forecast.png:
(1) forecast against actual over the test period in ORIGINAL units, with the persistence baseline as a third line;
(2) a residual-over-time plot showing whether errors grow towards the end of the period;
(3) a scatter of predicted change against actual change from one timestep to the next - this reveals whether the model is anticipating movement or merely repeating the last value.
Print test MAE and RMSE against the persistence baseline, plus the correlation between predicted and actual CHANGE.
Add a comment explaining that a model whose predicted change correlates near zero with actual change has learned persistence and nothing more.
```

### 6. Run it and read panel 3 and the change-correlation figure. This is the honest test of whether your forecaster learned anything.

Do not be discouraged by a modest change-correlation. Anticipating short-term movement in a noisy physical signal is genuinely hard, and reporting a small honest number with the plot that proves it is far more professional than reporting an MAE that hides it.

**COMMAND** — run this in your terminal:

```bash
python evaluate.py
```

### 7. Assemble the single report figure and write the findings.

The report figure is the artifact people will actually look at. One panel per model, clearly labelled, with the date and project name, is what makes it usable a month later when nobody remembers the run.

**PROMPT** — paste this into your AI coding assistant:

```text
Add build_report() to evaluate.py that assembles one combined figure saved to reports/evaluation_report.png with a row per model - regression, classification, forecasting - showing the single most informative panel from each, with a clear row label and a shared title naming the course project and the date. Then write reports/findings.md with one short section per model containing: the headline metric next to its baseline, the most important thing the plots reveal that the metric does not, and one stated limitation of the model.
```

### 8. Read your own findings.md aloud as if presenting to a plant manager. If any sentence would not survive a follow-up question, rewrite it.

The 'stated limitation' line is not modesty, it is scope control. 'Trained only on machine types A and B; predictions for type C are not supported' is the sentence that prevents your model being misused, and it goes straight into the model card.

## Verification — Test it

reports/eval_wear.png, reports/eval_defects.png and reports/eval_forecast.png each show three labelled panels. reports/evaluation_report.png combines one panel per model with a title and date. reports/findings.md has three sections, each naming a metric with its baseline, an insight the plot revealed that the metric did not, and a stated limitation.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| A model file is missing | Retrain it briefly with the saved configuration, or use the checkpoint from the relevant lab. Every model should have been saved in Labs 11, 15 and 19. |
| The confusion matrix rows do not sum to 1 | Normalisation was applied over the wrong axis. Row normalisation divides by the true-class totals, which is axis 1 in sklearn's normalize='true'. |
| R-squared is negative | The model is worse than predicting the mean. That is a real result — check the model loaded correctly and that the scaling was inverted before comparing. |
| The forecast plot's three lines are indistinguishable | Zoom into a shorter window, perhaps 200 timesteps, and use distinct line styles. A full-range plot of a smooth series hides all the detail. |
| Predicted and actual change correlate at almost exactly zero | The model learned persistence. Report it honestly, and consider a longer lookback or additional features as the stated next step. |

## Going further (optional)

- Add prediction intervals to the forecast plot using dropout at inference time as a rough uncertainty estimate.
- Add a calibration curve for the classifier and check whether its confidence matches its accuracy.
- Segment the regression residuals by machine type and check whether the model is systematically worse on one of them.

---

[← Lab 19: Tuning Sequence Models with Follow-Up Prompts](../lab19-tuning-sequence-models-with-follow-up-prompts/README.md) · [All labs](../README.md) · [Lab 21: Packaging a Complete Deep Learning Project →](../lab21-packaging-a-complete-deep-learning-project/README.md)

_Tertiary Infotech Academy Pte Ltd · C539 · Version v1.0 · 20 August 2026_
