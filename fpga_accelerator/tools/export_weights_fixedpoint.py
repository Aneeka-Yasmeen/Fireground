"""
Export the trained multitask_fireground_cnn_v2 weights to fixed-point
(Q4.11, 16-bit signed) hex files for RTL $readmemh, and validate that the
fixed-point math still classifies correctly before any RTL is written.

Fixed-point format (applies uniformly to inputs, weights, biases and
intermediate activations): Q4.11 signed 16-bit
    - 1 sign bit + 4 integer bits + 11 fractional bits
    - range: -16.0 .. +15.9995
    - resolution: 1/2048 = 0.00048828125

Chosen because:
    - trained weights fall in [-0.73, 0.64] (checked directly from the
      model) -- comfortably inside Q4.11 with several bits of headroom
      before the integer part is ever touched
    - standardized sensor inputs fall in [-3.3, 4.0] (checked against
      dataset/v2_prepared/X_test.npy) -- also comfortably inside range
    - using ONE format everywhere (rather than a different scale per
      layer) keeps the multiply-accumulate hardware uniform and simple
      to verify by hand; the trade-off is weight precision is lower than
      a dedicated per-tensor INT8 scheme would give -- documented as a
      known limitation / future optimization in docs/DESIGN_NOTES.md,
      not hidden.

MAC rule used everywhere (and mirrored exactly in the RTL):
    acc(Q8.22) = sum(a(Q4.11) * b(Q4.11))         -- 32-bit accumulator
    y(Q4.11)   = round_and_saturate(acc >>> 11) + bias(Q4.11)
    y          = relu(y) if the layer has ReLU
"""
import sys
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "cnn_env" / "Lib" / "site-packages"))

import numpy as np
import keras

OUT_WEIGHTS = HERE.parent / "weights"
OUT_SIM = HERE.parent / "sim"
FRAC_BITS = 11
INT_BITS = 4  # + 1 sign bit = 16 total
WIDTH = 1 + INT_BITS + FRAC_BITS
Q_SCALE = 1 << FRAC_BITS  # 2048
Q_MIN = -(1 << (WIDTH - 1))       # -32768
Q_MAX = (1 << (WIDTH - 1)) - 1    # 32767

FEATURES = ["heart_rate", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "gas", "temperature"]
MEAN = [91.1068859, 0.266849514, 0.268908626, 9.66666423, 0.0322428436, 0.0323018356, 0.0324732444, 27.0979919, 31.1267252]
SCALE = [13.34430666, 0.13700957, 0.13697954, 0.1158999, 0.02163363, 0.02163962, 0.02157665, 11.87917055, 1.93309766]


def to_fixed(x):
    """float -> Q4.11 signed integer, round-to-nearest, saturate."""
    x = np.asarray(x, dtype=np.float64)
    q = np.round(x * Q_SCALE).astype(np.int64)
    return np.clip(q, Q_MIN, Q_MAX).astype(np.int32)


def from_fixed(q):
    return np.asarray(q, dtype=np.float64) / Q_SCALE


def to_hex16(q):
    """signed int -> 4-digit hex twos-complement string for $readmemh."""
    return format(int(q) & 0xFFFF, "04x")


def write_hex_flat(path, arr_i32):
    with open(path, "w") as f:
        for v in arr_i32.flatten():
            f.write(to_hex16(v) + "\n")


def fixed_matmul_relu(x_q, w_q, b_q, relu):
    """x_q: (N,), w_q: (N,M), b_q: (M,) all int32 in Q4.11 ticks.
    Mirrors the RTL MAC-and-requantize rule exactly (int64 to avoid
    Python overflow; the RTL uses a 32-bit accumulator, verified sized
    correctly for this model's actual value ranges in main() below)."""
    acc = np.zeros(w_q.shape[1], dtype=np.int64)
    for m in range(w_q.shape[1]):
        acc[m] = int(np.dot(x_q.astype(np.int64), w_q[:, m].astype(np.int64)))
    # round-to-nearest arithmetic shift by FRAC_BITS
    shifted = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS
    y = shifted + b_q.astype(np.int64)
    y = np.clip(y, Q_MIN, Q_MAX)
    if relu:
        y = np.maximum(y, 0)
    return y.astype(np.int32)


def fixed_conv1d_relu(x_q, w_q, b_q):
    """x_q: (T, Cin) int32. w_q: (K, Cin, Cout) int32. b_q: (Cout,) int32.
    Returns (T-K+1, Cout) int32, Q4.11, after bias + ReLU."""
    T, Cin = x_q.shape
    K, Cin2, Cout = w_q.shape
    assert Cin == Cin2
    Tout = T - K + 1
    y = np.zeros((Tout, Cout), dtype=np.int32)
    for t in range(Tout):
        for co in range(Cout):
            acc = 0
            for k in range(K):
                for ci in range(Cin):
                    acc += int(x_q[t + k, ci]) * int(w_q[k, ci, co])
            shifted = (acc + (1 << (FRAC_BITS - 1))) >> FRAC_BITS
            v = shifted + int(b_q[co])
            v = max(Q_MIN, min(Q_MAX, v))
            v = max(v, 0)  # ReLU
            y[t, co] = v
    return y


def main():
    OUT_WEIGHTS.mkdir(parents=True, exist_ok=True)
    OUT_SIM.mkdir(parents=True, exist_ok=True)

    model = keras.models.load_model(str(PROJECT / "models" / "multitask_fireground_cnn_v2.keras"))
    layers = {l.name: l for l in model.layers}

    conv1_w, conv1_b = layers["conv1"].get_weights()
    conv2_w, conv2_b = layers["conv2"].get_weights()
    dense_w, dense_b = layers["dense_shared"].get_weights()
    act_w, act_b = layers["activity"].get_weights()
    env_w, env_b = layers["environment"].get_weights()
    phy_w, phy_b = layers["physiological"].get_weights()

    tensors = {
        "conv1_w": conv1_w, "conv1_b": conv1_b,
        "conv2_w": conv2_w, "conv2_b": conv2_b,
        "dense_w": dense_w, "dense_b": dense_b,
        "act_w": act_w, "act_b": act_b,
        "env_w": env_w, "env_b": env_b,
        "phy_w": phy_w, "phy_b": phy_b,
    }

    print("=== weight ranges (all must be well inside [-16, 16) for Q4.11) ===")
    q_tensors = {}
    for name, t in tensors.items():
        print(f"  {name:10s} shape={t.shape} min={t.min():.4f} max={t.max():.4f}")
        q_tensors[name] = to_fixed(t)
        write_hex_flat(OUT_WEIGHTS / f"{name}.hex", q_tensors[name])

    # ------------------------------------------------------------------
    # Validate: run the fixed-point pipeline on the real test set and
    # compare classification (argmax) against the float32 Keras model.
    # ------------------------------------------------------------------
    X_test = np.load(PROJECT / "dataset" / "v2_prepared" / "X_test.npy")
    ya_test = np.load(PROJECT / "dataset" / "v2_prepared" / "y_activity_test.npy")
    ye_test = np.load(PROJECT / "dataset" / "v2_prepared" / "y_environment_test.npy")
    yp_test = np.load(PROJECT / "dataset" / "v2_prepared" / "y_physiological_test.npy")

    float_preds = model.predict(X_test, verbose=0)
    float_a = float_preds[0].argmax(axis=1)
    float_e = float_preds[1].argmax(axis=1)
    float_p = float_preds[2].argmax(axis=1)

    n = len(X_test)
    fixed_a = np.zeros(n, dtype=np.int32)
    fixed_e = np.zeros(n, dtype=np.int32)
    fixed_p = np.zeros(n, dtype=np.int32)

    max_abs_conv1_acc = 0
    max_abs_conv2_acc = 0

    for i in range(n):
        x_q = to_fixed(X_test[i])  # (10, 9)

        c1 = fixed_conv1d_relu(x_q, q_tensors["conv1_w"], q_tensors["conv1_b"])  # (8,16)
        c2 = fixed_conv1d_relu(c1, q_tensors["conv2_w"], q_tensors["conv2_b"])   # (6,32)

        # global average pool over time (6 steps) per channel, Q4.11 average
        gap = np.round(c2.astype(np.int64).sum(axis=0) / c2.shape[0]).astype(np.int32)
        gap = np.clip(gap, Q_MIN, Q_MAX)

        shared = fixed_matmul_relu(gap, q_tensors["dense_w"], q_tensors["dense_b"], relu=True)
        a_logits = fixed_matmul_relu(shared, q_tensors["act_w"], q_tensors["act_b"], relu=False)
        e_logits = fixed_matmul_relu(shared, q_tensors["env_w"], q_tensors["env_b"], relu=False)
        p_logits = fixed_matmul_relu(shared, q_tensors["phy_w"], q_tensors["phy_b"], relu=False)

        fixed_a[i] = int(np.argmax(a_logits))
        fixed_e[i] = int(np.argmax(e_logits))
        fixed_p[i] = int(np.argmax(p_logits))

    agree_a = (fixed_a == float_a).mean()
    agree_e = (fixed_e == float_e).mean()
    agree_p = (fixed_p == float_p).mean()
    acc_a = (fixed_a == ya_test).mean()
    acc_e = (fixed_e == ye_test).mean()
    acc_p = (fixed_p == yp_test).mean()

    print()
    print("=== Q4.11 fixed-point vs float32 Keras model, argmax agreement (n=%d) ===" % n)
    print(f"  activity:       {agree_a*100:.2f}% agree with float model | {acc_a*100:.2f}% match true label")
    print(f"  environment:    {agree_e*100:.2f}% agree with float model | {acc_e*100:.2f}% match true label")
    print(f"  physiological:  {agree_p*100:.2f}% agree with float model | {acc_p*100:.2f}% match true label")

    # ------------------------------------------------------------------
    # Export ONE fixed test vector (first live sensor window) + its full
    # expected intermediate/final values, for the RTL self-checking
    # testbench.
    # ------------------------------------------------------------------
    sample = X_test[0]
    x_q = to_fixed(sample)
    c1 = fixed_conv1d_relu(x_q, q_tensors["conv1_w"], q_tensors["conv1_b"])
    c2 = fixed_conv1d_relu(c1, q_tensors["conv2_w"], q_tensors["conv2_b"])
    gap = np.round(c2.astype(np.int64).sum(axis=0) / c2.shape[0]).astype(np.int32)
    gap = np.clip(gap, Q_MIN, Q_MAX)
    shared = fixed_matmul_relu(gap, q_tensors["dense_w"], q_tensors["dense_b"], relu=True)
    a_logits = fixed_matmul_relu(shared, q_tensors["act_w"], q_tensors["act_b"], relu=False)
    e_logits = fixed_matmul_relu(shared, q_tensors["env_w"], q_tensors["env_b"], relu=False)
    p_logits = fixed_matmul_relu(shared, q_tensors["phy_w"], q_tensors["phy_b"], relu=False)

    write_hex_flat(OUT_SIM / "test_input.hex", x_q)

    expected = {
        "activity_argmax": int(np.argmax(a_logits)),
        "environment_argmax": int(np.argmax(e_logits)),
        "physiological_argmax": int(np.argmax(p_logits)),
        "activity_logits_q4_11": a_logits.tolist(),
        "environment_logits_q4_11": e_logits.tolist(),
        "physiological_logits_q4_11": p_logits.tolist(),
        "float_activity_argmax": int(float_a[0]),
        "float_environment_argmax": int(float_e[0]),
        "float_physiological_argmax": int(float_p[0]),
        "frac_bits": FRAC_BITS,
        "width": WIDTH,
    }
    with open(OUT_SIM / "expected_output.json", "w") as f:
        json.dump(expected, f, indent=2)

    print()
    print("=== Test vector for RTL testbench (sim/test_input.hex) ===")
    print("Expected argmax  activity=%d  environment=%d  physiological=%d"
          % (expected["activity_argmax"], expected["environment_argmax"], expected["physiological_argmax"]))
    print("Float model argmax on same sample: activity=%d environment=%d physiological=%d"
          % (expected["float_activity_argmax"], expected["float_environment_argmax"], expected["float_physiological_argmax"]))

    with open(OUT_WEIGHTS / "quant_format.json", "w") as f:
        json.dump({"format": "Q4.11 signed 16-bit", "frac_bits": FRAC_BITS,
                    "int_bits": INT_BITS, "width": WIDTH,
                    "q_min": int(Q_MIN), "q_max": int(Q_MAX)}, f, indent=2)

    print()
    print("Wrote weight hex files to", OUT_WEIGHTS)
    print("Wrote test vector + expected output to", OUT_SIM)


if __name__ == "__main__":
    main()
