"""Generates simulated 60-sample, 9-channel sensor streams for each firefighter
in dataset/firefighter_registry.json.

This produces demo/architecture data only so the multi-firefighter pipeline and dashboard can be built and tested
now, ready to be swapped for real per-person streams later without any
structural change.

Value ranges follow the same style as generate_live_stream.py (which this
replaces for multi-firefighter use) and stay within the distribution the
CNN was trained on (see MEAN/SCALE in live_vnnx_fireground_system_liveweb.py).
"""
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
REGISTRY_PATH = os.path.join(PROJECT, "dataset", "firefighter_registry.json")
OUTPUT_PATH = os.path.join(PROJECT, "dataset", "live_sensor_stream_multi.csv")

FEATURES = ["heart_rate", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "gas", "temperature"]

# One profile function per scenario archetype. Each returns 60 rows of the
# 9 channels. Deliberately distinct trajectories so the demo shows a mix of
# NORMAL / ELEVATED / HIGH outcomes across the roster instead of four
# identical copies of the same case.


def profile_deteriorating(rng):
    rows = []
    for t in range(60):
        if t < 20:
            hr = 75 + rng.normal(0, 2); gas = 20 + rng.normal(0, 2); temp = 30 + rng.normal(0, 0.4)
            ax = 0.10 + rng.normal(0, 0.03); ay = 0.20 + rng.normal(0, 0.03); az = 9.80 + rng.normal(0, 0.05)
            gx = 0.01 + rng.normal(0, 0.005); gy = 0.02 + rng.normal(0, 0.005); gz = 0.01 + rng.normal(0, 0.005)
        elif t < 40:
            s = t - 20
            hr = 80 + s * 0.5 + rng.normal(0, 2); gas = 20 + rng.normal(0, 2); temp = 30 + s * 0.05 + rng.normal(0, 0.4)
            ax = 0.20 + s * 0.015 + rng.normal(0, 0.03); ay = 0.20 + rng.normal(0, 0.05); az = 9.70 + rng.normal(0, 0.07)
            gx = 0.02 + s * 0.002 + rng.normal(0, 0.005); gy = 0.02 + s * 0.002 + rng.normal(0, 0.005); gz = 0.01 + s * 0.002 + rng.normal(0, 0.005)
        else:
            s = t - 40
            hr = 90 + s * 1.0 + rng.normal(0, 2); gas = 35 + s * 2.0 + rng.normal(0, 2); temp = 33 + s * 0.3 + rng.normal(0, 0.4)
            ax = 0.40 + s * 0.01 + rng.normal(0, 0.04); ay = 0.30 + rng.normal(0, 0.05); az = 9.60 - s * 0.01 + rng.normal(0, 0.06)
            gx = 0.04 + s * 0.002 + rng.normal(0, 0.006); gy = 0.04 + s * 0.002 + rng.normal(0, 0.006); gz = 0.03 + s * 0.002 + rng.normal(0, 0.006)
        rows.append([hr, ax, ay, az, gx, gy, gz, gas, temp])
    return rows


def profile_stable(rng):
    rows = []
    for t in range(60):
        hr = 74 + rng.normal(0, 2.5); gas = 18 + rng.normal(0, 2); temp = 29.5 + rng.normal(0, 0.4)
        ax = 0.09 + rng.normal(0, 0.03); ay = 0.18 + rng.normal(0, 0.03); az = 9.81 + rng.normal(0, 0.05)
        gx = 0.01 + rng.normal(0, 0.005); gy = 0.01 + rng.normal(0, 0.005); gz = 0.01 + rng.normal(0, 0.005)
        rows.append([hr, ax, ay, az, gx, gy, gz, gas, temp])
    return rows


def profile_active_moderate(rng):
    rows = []
    for t in range(60):
        s = t
        hr = 85 + s * 0.15 + rng.normal(0, 2.5); gas = 24 + rng.normal(0, 3); temp = 31 + s * 0.02 + rng.normal(0, 0.4)
        ax = 0.25 + rng.normal(0, 0.04); ay = 0.22 + rng.normal(0, 0.04); az = 9.65 + rng.normal(0, 0.08)
        gx = 0.03 + rng.normal(0, 0.006); gy = 0.03 + rng.normal(0, 0.006); gz = 0.02 + rng.normal(0, 0.006)
        rows.append([hr, ax, ay, az, gx, gy, gz, gas, temp])
    return rows


def profile_early_high_risk(rng):
    rows = []
    for t in range(60):
        s = t
        hr = 100 + s * 0.6 + rng.normal(0, 3); gas = 40 + s * 1.2 + rng.normal(0, 3); temp = 33.5 + s * 0.12 + rng.normal(0, 0.4)
        ax = 0.45 + rng.normal(0, 0.05); ay = 0.32 + rng.normal(0, 0.05); az = 9.55 - s * 0.005 + rng.normal(0, 0.07)
        gx = 0.05 + s * 0.001 + rng.normal(0, 0.006); gy = 0.05 + s * 0.001 + rng.normal(0, 0.006); gz = 0.04 + s * 0.001 + rng.normal(0, 0.006)
        rows.append([hr, ax, ay, az, gx, gy, gz, gas, temp])
    return rows


PROFILES = {
    "FF01": ("deteriorating", profile_deteriorating),
    "FF02": ("stable", profile_stable),
    "FF03": ("active_moderate", profile_active_moderate),
    "FF04": ("early_high_risk", profile_early_high_risk),
}
DEFAULT_PROFILE_CYCLE = list(PROFILES.values())


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    out_rows = []
    for i, ff in enumerate(registry["firefighters"]):
        sid = ff["sensor_stream_id"]
        name, fn = PROFILES.get(sid, DEFAULT_PROFILE_CYCLE[i % len(DEFAULT_PROFILE_CYCLE)])
        rng = np.random.default_rng(seed=hash(sid) % (2**32))
        rows = fn(rng)
        for t, row in enumerate(rows):
            out_rows.append(
                [sid, f"14:{t // 60:02d}:{t % 60:02d}"] + [round(v, 6) for v in row]
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["firefighter_id", "timestamp"] + FEATURES)
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows ({len(registry['firefighters'])} firefighters x 60 samples) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
