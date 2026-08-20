"""
ForgeSight - baseline training script.

Trains two models on the machine telemetry in data/machines.csv:
  * WearNet - regression, predicts tool_wear_um
  * QCNet   - classification, predicts qc_class (0 ok, 1 scratch, 2 dent, 3 burr)

Usage (from your torch-vibe/ project root):
    python train_wear_buggy.py

--------------------------------------------------------------------------
TRAINER NOTE - DO NOT SHOW THIS BLOCK TO LEARNERS BEFORE LAB 6.

This script is the Lab 6 review exercise. It RUNS TO COMPLETION, prints a
falling loss and reports a respectable score, while containing five real
defects. None of them raises an error. The five, and what each does:

  1. Features are standardised on the FULL dataset before the train/test
     split (lines ~60-64). Test statistics leak into training and the
     reported score is inflated.
  2. optimizer.zero_grad() is missing from the regression loop (~line 104).
     Gradients accumulate across batches, so updates are too large and
     training is unstable.
  3. QCNet applies nn.Softmax(dim=1) to its output (~line 88) while training
     with nn.CrossEntropyLoss, which applies log_softmax internally. Softmax
     is applied twice, gradients are flattened, the model learns slowly.
  4. Both evaluations run without model.eval() and without torch.no_grad()
     (~lines 128, 150). Dropout stays active, so the reported score is noisy
     and pessimistic, and memory use is higher than it needs to be.
  5. The regression target is passed as shape (N,) while the prediction is
     (N, 1) (~line 100). MSELoss broadcasts to (N, N) and averages N-squared
     pairwise differences - an entirely wrong loss that still decreases.

Learners find three of the five unaided on average. Defect 1 is the one the
AI assistant is least likely to catch, because it is about the ORDER of
operations rather than any single wrong line.
--------------------------------------------------------------------------
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)

FEATURES = ["spindle_speed", "feed_rate", "coolant_temp",
            "vibration_rms", "spindle_load", "ambient_temp"]


def load_data(path="data/machines.csv"):
    """Load the telemetry, standardise the features and split into train/test."""
    df = pd.read_csv(path)

    X = df[FEATURES].values.astype(np.float32)
    y_wear = df["tool_wear_um"].values.astype(np.float32)
    y_qc = df["qc_class"].values.astype(np.int64)

    # Standardise the features so every column is on a comparable scale.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    n_train = int(len(X) * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    yw_train, yw_test = y_wear[:n_train], y_wear[n_train:]
    yq_train, yq_test = y_qc[:n_train], y_qc[n_train:]

    print(f"Loaded {len(df)} rows -> {len(X_train)} train / {len(X_test)} test")
    return (torch.tensor(X_train), torch.tensor(X_test),
            torch.tensor(yw_train), torch.tensor(yw_test),
            torch.tensor(yq_train), torch.tensor(yq_test))


class WearNet(nn.Module):
    """Regression network predicting tool wear in microns."""

    def __init__(self, n_features=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


class QCNet(nn.Module):
    """Classifier predicting the quality-control class."""

    def __init__(self, n_features=6, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


def train_regression(X_train, yw_train, epochs=60, lr=1e-3):
    """Train the tool-wear regression model."""
    model = WearNet(X_train.shape[1])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train, yw_train),
                        batch_size=64, shuffle=True)

    print("\nTraining WearNet (regression)")
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:>3}  train loss {total / len(X_train):10.3f}")
    return model


def train_classifier(X_train, yq_train, epochs=60, lr=1e-3):
    """Train the quality-control classifier."""
    model = QCNet(X_train.shape[1])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X_train, yq_train),
                        batch_size=64, shuffle=True)

    print("\nTraining QCNet (classification)")
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:>3}  train loss {total / len(X_train):10.4f}")
    return model


def evaluate_regression(model, X_test, yw_test):
    """Report MAE and RMSE on the held-out split."""
    pred = model(X_test).squeeze()
    mae = torch.abs(pred - yw_test).mean().item()
    rmse = torch.sqrt(((pred - yw_test) ** 2).mean()).item()
    print(f"\nWearNet   test MAE {mae:7.2f} um   RMSE {rmse:7.2f} um")
    return mae


def evaluate_classifier(model, X_test, yq_test):
    """Report accuracy on the held-out split."""
    probs = model(X_test)
    pred = probs.argmax(dim=1)
    acc = (pred == yq_test).float().mean().item()
    print(f"QCNet     test accuracy {acc:.1%}")
    return acc


def main():
    X_train, X_test, yw_train, yw_test, yq_train, yq_test = load_data()

    wear_model = train_regression(X_train, yw_train)
    qc_model = train_classifier(X_train, yq_train)

    evaluate_regression(wear_model, X_test, yw_test)
    evaluate_classifier(qc_model, X_test, yq_test)
    print("\nDone.")


if __name__ == "__main__":
    main()
