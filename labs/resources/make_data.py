#!/usr/bin/env python3
"""
Generate the ForgeSight tabular and time-series datasets for C539.

Produces two CSV files used across the course:

  machines.csv          4000 rows of machine telemetry.
                        6 sensor features + 2 targets:
                          tool_wear_um  (float)  -> regression   (Labs 8, 20)
                          qc_class      (0..3)   -> classification (Labs 9, 20)

  vibration_series.csv  3000 hourly vibration readings for the LSTM
                        forecasting labs (Labs 18-20).

Everything is synthetic and deterministic (seed 42), so every learner in the
room gets byte-identical data and the numbers quoted in the Learner Guide hold.

Run from this folder:   python make_data.py
"""
import csv
import datetime as dt
import numpy as np
from pathlib import Path

SEED = 42
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)

# --------------------------------------------------------------- machines.csv
N = 4000

# Six machines, each with its own operating point, so the data has structure a
# model can actually learn rather than pure noise.
machine_id = rng.integers(1, 7, size=N)
base_speed = np.array([0, 8200, 9000, 7600, 10400, 8800, 9600])[machine_id]

spindle_speed = base_speed + rng.normal(0, 320, N)                  # rpm
feed_rate     = 220 + 0.011 * spindle_speed + rng.normal(0, 18, N)  # mm/min
ambient_temp  = 24 + 3.5 * np.sin(np.arange(N) / 400.0) + rng.normal(0, 1.1, N)
coolant_temp  = (28 + 0.0016 * spindle_speed + 0.42 * ambient_temp
                 + rng.normal(0, 1.6, N))
spindle_load  = (46 + 0.0021 * spindle_speed + 0.030 * feed_rate
                 + rng.normal(0, 3.4, N))                           # percent
vibration_rms = (0.42 + 0.000048 * spindle_speed + 0.0092 * spindle_load
                 + 0.011 * np.maximum(coolant_temp - 40, 0)
                 + rng.normal(0, 0.075, N))                         # mm/s

# Tool wear: driven mainly by load, vibration and coolant temperature, with a
# genuine non-linear term so a neural network can beat a linear baseline.
tool_wear_um = (
    18.0
    + 1.34 * spindle_load
    + 41.0 * vibration_rms
    + 0.78 * np.maximum(coolant_temp - 38, 0) ** 1.35
    + 0.0021 * spindle_speed
    + 9.0 * np.tanh((vibration_rms - 0.85) * 4.0) * (spindle_load / 60.0)
    + rng.normal(0, 6.5, N)
)
tool_wear_um = np.clip(tool_wear_um, 5, None)

# QC class: 0 ok, 1 scratch, 2 dent, 3 burr. Driven by the same physics, so the
# classes are learnable but genuinely overlapping - and deliberately imbalanced
# ("ok" is the majority) so that accuracy alone flatters a lazy model that never
# predicts a defect. That imbalance is the whole point of Lab 9.
TARGET = np.array([0.58, 0.18, 0.16, 0.08])   # ok, scratch, dent, burr


def _z(v):
    return (v - v.mean()) / v.std()


# Each defect has a physically meaningful driver, standardised so the three are
# on a comparable scale before they compete with the "ok" logit.
z_scratch = _z(0.9 * vibration_rms + 0.010 * spindle_load - 0.00013 * spindle_speed)
z_dent    = _z(0.030 * np.maximum(coolant_temp - 41, 0) + 0.006 * spindle_load)
z_burr    = _z(0.0016 * feed_rate + 0.5 * np.maximum(vibration_rms - 1.0, 0))
z_defects = np.stack([z_scratch, z_dent, z_burr], axis=1)

# Class noise: enough overlap that the classes are genuinely hard to separate,
# so a neural network beats the baseline without ever reaching 100%.
qc_noise = rng.normal(0, 0.85, (N, 4))

# Solve for the per-class offsets that hit TARGET. A closed form does not exist
# once the argmax and the noise are involved, so nudge the offsets until the
# realised proportions match - deterministic, since the noise is already drawn.
offsets = np.zeros(3)
for _ in range(600):
    lg = np.concatenate([np.zeros((N, 1)), z_defects - offsets], axis=1) + qc_noise
    prop = np.bincount(lg.argmax(axis=1), minlength=4) / N
    err = prop[1:] - TARGET[1:]
    if np.abs(prop - TARGET).max() < 0.004:
        break
    offsets += 2.0 * err

logits = np.concatenate([np.zeros((N, 1)), z_defects - offsets], axis=1) + qc_noise
qc_class = logits.argmax(axis=1)

MACH_COLS = ["machine_id", "spindle_speed", "feed_rate", "coolant_temp",
             "vibration_rms", "spindle_load", "ambient_temp",
             "tool_wear_um", "qc_class"]
mach_rows = list(zip(
    machine_id.astype(int),
    spindle_speed.round(1), feed_rate.round(2), coolant_temp.round(2),
    vibration_rms.round(4), spindle_load.round(2), ambient_temp.round(2),
    tool_wear_um.round(2), qc_class.astype(int),
))
with open(HERE / "machines.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(MACH_COLS)
    w.writerows(mach_rows)

# ------------------------------------------------------- vibration_series.csv
# 3000 hourly readings: slow drift (tool wearing in), a daily cycle, a weekly
# maintenance dip, and autocorrelated noise. Smooth enough that persistence is
# a strong baseline - which is exactly the lesson of Lab 18.
T = 3000
t = np.arange(T)

trend    = 0.58 + 0.00019 * t
daily    = 0.085 * np.sin(2 * np.pi * t / 24.0)
weekly   = -0.055 * (np.sin(2 * np.pi * t / 168.0) > 0.92)
noise    = np.zeros(T)
eps      = rng.normal(0, 0.030, T)
for i in range(1, T):                      # AR(1) so the series is smooth
    noise[i] = 0.72 * noise[i - 1] + eps[i]

# Three maintenance resets: wear drops back after a tool change.
resets = [900, 1750, 2600]
reset_effect = np.zeros(T)
for r in resets:
    reset_effect[r:] -= 0.00019 * (np.arange(T)[r:] - r) * 0.85

vib = trend + daily + weekly + noise + reset_effect
vib = np.clip(vib, 0.05, None)

start = dt.datetime(2026, 1, 1, 0, 0)
stamps = [(start + dt.timedelta(hours=int(i))).strftime("%Y-%m-%d %H:%M:%S")
          for i in range(T)]
with open(HERE / "vibration_series.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["timestamp", "vibration_rms"])
    w.writerows(zip(stamps, vib.round(4)))

# ------------------------------------------------------------------- summary
counts = {int(k): int(v) for k, v in zip(*np.unique(qc_class, return_counts=True))}
print(f"machines.csv          {N:>5} rows x {len(MACH_COLS)} cols")
print("  qc_class counts     ", counts)
print(f"  majority class      {max(counts.values()) / N:.1%}")
print(f"  tool_wear_um        mean {tool_wear_um.mean():.1f}  "
      f"std {tool_wear_um.std():.1f}  "
      f"range {tool_wear_um.min():.1f}-{tool_wear_um.max():.1f}")
print(f"vibration_series.csv  {T:>5} rows  "
      f"mean {vib.mean():.3f}  std {vib.std():.3f}")
print(f"  persistence MAE     {np.abs(np.diff(vib[-600:])).mean():.4f} (last 20%)")
print("\nWrote:", HERE / "machines.csv")
print("Wrote:", HERE / "vibration_series.csv")
