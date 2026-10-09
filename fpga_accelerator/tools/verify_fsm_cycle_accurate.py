"""
Cycle-accurate Python emulation of fireground_cnn_top.v's FSM -- same
states, same counters, same non-blocking-assignment timing (every
register update is computed from the CURRENT cycle's values and only
takes effect at the next cycle boundary, exactly like Verilog `<=`).

This is a second, independent check on top of the
by-hand cycle tracing documented in the RTL comments and
docs/DESIGN_NOTES.md: if this emulator (which mirrors the FSM's actual
control flow, not just the "ideal" batch math) agrees with the trained
float32 model,the FSM's timing is correct -- but
it is NOT a substitute for actually simulating the Verilog text itself
before synthesizing.
"""
import sys
import glob
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
sys.path.insert(0, str(PROJECT / "cnn_env" / "Lib" / "site-packages"))

import numpy as np

WEIGHTS = HERE.parent / "weights"
FRAC = 11
Q_MIN = -32768
Q_MAX = 32767

T_IN, CIN0, K = 10, 9, 3
COUT1, T1 = 16, 8
COUT2, T2 = 32, 6
DENSE_N = 32
ACT_N, ENV_N, PHY_N = 3, 2, 3


def load_hex(name, shape):
    vals = [int(l.strip(), 16) for l in open(WEIGHTS / f"{name}.hex")]
    vals = [v - 65536 if v >= 32768 else v for v in vals]
    return np.array(vals, dtype=np.int64).reshape(shape)


def requantize_relu(acc, bias, do_relu):
    shifted = (acc + (1 << (FRAC - 1))) >> FRAC
    biased = shifted + int(bias)
    biased = max(Q_MIN, min(Q_MAX, biased))
    if do_relu and biased < 0:
        return 0
    return biased


# Must match the `reg [N:0]` declarations in rtl/fireground_cnn_top.v
# exactly. This table is what lets this emulator catch under-sized
# counters (see the masking step in FsmEmulator.step()) -- it deliberately
# does NOT infer widths from the loop bounds above, since the whole point
# is to mirror what's actually written in the RTL, bugs included.
REG_WIDTHS = {
    "c1_t": 4, "c1_co": 4, "c1_k": 4, "c1_ci": 4,
    "c2_t": 6, "c2_co": 6, "c2_k": 6, "c2_ci": 6,
    "gap_c": 6, "gap_t": 3,
    "d_out": 6, "d_in": 6,
    "h_out": 6, "h_in": 6,
    "head_sel": 2, "head_n": 2,
    "best_idx": 2,
    "activity_argmax": 2, "environment_argmax": 2, "physiological_argmax": 2,
}


class FsmEmulator:
    """Mirrors fireground_cnn_top.v state-for-state, counter-for-counter."""

    S_IDLE, S_CONV1, S_CONV1_WB, S_CONV2, S_CONV2_WB, S_GAP, \
        S_DENSE, S_DENSE_WB, S_HEADS, S_HEADS_WB, S_DONE = range(11)

    def __init__(self, weights):
        self.w = weights
        self.reset()

    def reset(self):
        self.state = self.S_IDLE
        self.acc = 0
        self.done = False
        self.activity_argmax = 0
        self.environment_argmax = 0
        self.physiological_argmax = 0
        self.c1_t = self.c1_co = self.c1_k = self.c1_ci = 0
        self.c2_t = self.c2_co = self.c2_k = self.c2_ci = 0
        self.gap_c = self.gap_t = 0
        self.gap_acc = 0
        self.d_out = self.d_in = 0
        self.head_sel = self.h_out = self.h_in = 0
        self.head_n = 0
        self.best_val = -32768
        self.best_idx = 0
        self.x_mem = [0] * (T_IN * CIN0)
        self.c1_mem = [0] * (T1 * COUT1)
        self.c2_mem = [0] * (T2 * COUT2)
        self.gap_mem = [0] * COUT2
        self.sh_mem = [0] * DENSE_N
        self.cycles = 0

    def load_input(self, x_q_flat):
        self.x_mem = list(x_q_flat)

    def start(self):
        self.state = self.S_CONV1
        self.acc = 0
        self.c1_t = self.c1_co = self.c1_k = self.c1_ci = 0

    def step(self):
        """One clock edge: compute all 'next' values from current state
        (mirrors combinational logic + non-blocking assign), then commit
        them all at once (mirrors the clock edge)."""
        self.cycles += 1
        w = self.w
        nxt = dict(self.__dict__)  # shallow copy of current values as the default "no change"
        nxt["done"] = False

        s = self.state
        if s == self.S_CONV1:
            x_addr = (self.c1_t + self.c1_k) * CIN0 + self.c1_ci
            w_addr = self.c1_k * (CIN0 * COUT1) + self.c1_ci * COUT1 + self.c1_co
            nxt["acc"] = self.acc + self.x_mem[x_addr] * int(w["conv1_w"][self.c1_k, self.c1_ci, self.c1_co])
            if self.c1_ci == CIN0 - 1:
                nxt["c1_ci"] = 0
                if self.c1_k == K - 1:
                    nxt["c1_k"] = 0
                    nxt["state"] = self.S_CONV1_WB
                else:
                    nxt["c1_k"] = self.c1_k + 1
            else:
                nxt["c1_ci"] = self.c1_ci + 1

        elif s == self.S_CONV1_WB:
            c1 = list(self.c1_mem)
            c1[self.c1_t * COUT1 + self.c1_co] = requantize_relu(self.acc, int(w["conv1_b"][self.c1_co]), True)
            nxt["c1_mem"] = c1
            nxt["acc"] = 0
            if self.c1_co == COUT1 - 1:
                nxt["c1_co"] = 0
                if self.c1_t == T1 - 1:
                    nxt["c1_t"] = 0
                    nxt["c2_t"] = nxt["c2_co"] = nxt["c2_k"] = nxt["c2_ci"] = 0
                    nxt["state"] = self.S_CONV2
                else:
                    nxt["c1_t"] = self.c1_t + 1
                    nxt["state"] = self.S_CONV1
            else:
                nxt["c1_co"] = self.c1_co + 1
                nxt["state"] = self.S_CONV1

        elif s == self.S_CONV2:
            x_addr = (self.c2_t + self.c2_k) * COUT1 + self.c2_ci
            nxt["acc"] = self.acc + self.c1_mem[x_addr] * int(w["conv2_w"][self.c2_k, self.c2_ci, self.c2_co])
            if self.c2_ci == COUT1 - 1:
                nxt["c2_ci"] = 0
                if self.c2_k == K - 1:
                    nxt["c2_k"] = 0
                    nxt["state"] = self.S_CONV2_WB
                else:
                    nxt["c2_k"] = self.c2_k + 1
            else:
                nxt["c2_ci"] = self.c2_ci + 1

        elif s == self.S_CONV2_WB:
            c2 = list(self.c2_mem)
            c2[self.c2_t * COUT2 + self.c2_co] = requantize_relu(self.acc, int(w["conv2_b"][self.c2_co]), True)
            nxt["c2_mem"] = c2
            nxt["acc"] = 0
            if self.c2_co == COUT2 - 1:
                nxt["c2_co"] = 0
                if self.c2_t == T2 - 1:
                    nxt["c2_t"] = 0
                    nxt["gap_c"] = nxt["gap_t"] = 0
                    nxt["gap_acc"] = 0
                    nxt["state"] = self.S_GAP
                else:
                    nxt["c2_t"] = self.c2_t + 1
                    nxt["state"] = self.S_CONV2
            else:
                nxt["c2_co"] = self.c2_co + 1
                nxt["state"] = self.S_CONV2

        elif s == self.S_GAP:
            if self.gap_t == T2 - 1:
                gap = list(self.gap_mem)
                total = self.gap_acc + self.c2_mem[self.gap_t * COUT2 + self.gap_c]
                gap[self.gap_c] = (total + (T2 >> 1)) // T2
                nxt["gap_mem"] = gap
                nxt["gap_acc"] = 0
                nxt["gap_t"] = 0
                if self.gap_c == COUT2 - 1:
                    nxt["gap_c"] = 0
                    nxt["d_out"] = nxt["d_in"] = 0
                    nxt["acc"] = 0
                    nxt["state"] = self.S_DENSE
                else:
                    nxt["gap_c"] = self.gap_c + 1
            else:
                nxt["gap_acc"] = self.gap_acc + self.c2_mem[self.gap_t * COUT2 + self.gap_c]
                nxt["gap_t"] = self.gap_t + 1

        elif s == self.S_DENSE:
            nxt["acc"] = self.acc + self.gap_mem[self.d_in] * int(w["dense_w"][self.d_in, self.d_out])
            if self.d_in == DENSE_N - 1:
                nxt["d_in"] = 0
                nxt["state"] = self.S_DENSE_WB
            else:
                nxt["d_in"] = self.d_in + 1

        elif s == self.S_DENSE_WB:
            sh = list(self.sh_mem)
            sh[self.d_out] = requantize_relu(self.acc, int(w["dense_b"][self.d_out]), True)
            nxt["sh_mem"] = sh
            nxt["acc"] = 0
            if self.d_out == DENSE_N - 1:
                nxt["d_out"] = 0
                nxt["head_sel"] = nxt["h_out"] = nxt["h_in"] = 0
                nxt["best_val"] = -32768
                nxt["best_idx"] = 0
                nxt["state"] = self.S_HEADS
            else:
                nxt["d_out"] = self.d_out + 1
                nxt["state"] = self.S_DENSE

        elif s == self.S_HEADS:
            if self.head_sel == 0:
                wt = int(w["act_w"][self.h_in, self.h_out]); n = ACT_N
            elif self.head_sel == 1:
                wt = int(w["env_w"][self.h_in, self.h_out]); n = ENV_N
            else:
                wt = int(w["phy_w"][self.h_in, self.h_out]); n = PHY_N
            nxt["acc"] = self.acc + self.sh_mem[self.h_in] * wt
            nxt["head_n"] = n
            if self.h_in == DENSE_N - 1:
                nxt["h_in"] = 0
                nxt["state"] = self.S_HEADS_WB
            else:
                nxt["h_in"] = self.h_in + 1

        elif s == self.S_HEADS_WB:
            if self.head_sel == 0:
                bias = int(w["act_b"][self.h_out])
            elif self.head_sel == 1:
                bias = int(w["env_b"][self.h_out])
            else:
                bias = int(w["phy_b"][self.h_out])
            logit = requantize_relu(self.acc, bias, False)
            if logit > self.best_val:
                new_best_val, new_best_idx = logit, self.h_out
            else:
                new_best_val, new_best_idx = self.best_val, self.best_idx
            nxt["best_val"] = new_best_val
            nxt["best_idx"] = new_best_idx
            nxt["acc"] = 0

            if self.h_out == self.head_n - 1:
                nxt["h_out"] = 0
                if self.head_sel == 0:
                    nxt["activity_argmax"] = new_best_idx
                elif self.head_sel == 1:
                    nxt["environment_argmax"] = new_best_idx
                else:
                    nxt["physiological_argmax"] = new_best_idx
                nxt["best_val"] = -32768
                nxt["best_idx"] = 0
                if self.head_sel == 2:
                    nxt["state"] = self.S_DONE
                else:
                    nxt["head_sel"] = self.head_sel + 1
                    nxt["state"] = self.S_HEADS
            else:
                nxt["h_out"] = self.h_out + 1
                nxt["state"] = self.S_HEADS

        elif s == self.S_DONE:
            nxt["done"] = True
            nxt["state"] = self.S_IDLE

        # Apply real hardware register widths (matching the RTL's reg
        # declarations exactly), including wraparound on overflow. This
        # is what the RTL's fixed-width registers do; Python ints have no
        # such limit, so without this masking step this emulator cannot
        # catch an under-sized counter -- which is exactly the class of
        # bug that made it into the first synthesis run (c2_co was 4 bits
        # but needed 5, silently wrapping at 15 and never reaching 31,
        # hanging the FSM inside conv2 forever). See docs/DESIGN_NOTES.md.
        for field, bits in REG_WIDTHS.items():
            if field in nxt:
                nxt[field] = nxt[field] & ((1 << bits) - 1)

        for k_, v_ in nxt.items():
            setattr(self, k_, v_)

    def run(self, max_cycles=200000):
        self.start()
        for _ in range(max_cycles):
            self.step()
            if self.done:
                return self.cycles
        raise RuntimeError("FSM did not finish within max_cycles -- stuck")


def to_fixed(x):
    q = np.round(np.asarray(x, dtype=np.float64) * (1 << FRAC)).astype(np.int64)
    return np.clip(q, -32768, 32767).astype(np.int64)


def main():
    weights = {
        "conv1_w": load_hex("conv1_w", (K, CIN0, COUT1)),
        "conv1_b": load_hex("conv1_b", (COUT1,)),
        "conv2_w": load_hex("conv2_w", (K, COUT1, COUT2)),
        "conv2_b": load_hex("conv2_b", (COUT2,)),
        "dense_w": load_hex("dense_w", (DENSE_N, DENSE_N)),
        "dense_b": load_hex("dense_b", (DENSE_N,)),
        "act_w": load_hex("act_w", (DENSE_N, ACT_N)),
        "act_b": load_hex("act_b", (ACT_N,)),
        "env_w": load_hex("env_w", (DENSE_N, ENV_N)),
        "env_b": load_hex("env_b", (ENV_N,)),
        "phy_w": load_hex("phy_w", (DENSE_N, PHY_N)),
        "phy_b": load_hex("phy_b", (PHY_N,)),
    }

    sys.path.insert(0, str(PROJECT / "cnn_env" / "Lib" / "site-packages"))
    import keras
    model = keras.models.load_model(str(PROJECT / "models" / "multitask_fireground_cnn_v2.keras"))
    X_test = np.load(PROJECT / "dataset" / "v2_prepared" / "X_test.npy")
    float_preds = model.predict(X_test, verbose=0)
    float_a = float_preds[0].argmax(axis=1)
    float_e = float_preds[1].argmax(axis=1)
    float_p = float_preds[2].argmax(axis=1)

    n = len(X_test)
    agree = 0
    first_cycles = None
    for i in range(n):
        x_q = to_fixed(X_test[i]).flatten()
        emu = FsmEmulator(weights)
        emu.load_input(x_q)
        cycles = emu.run()
        if first_cycles is None:
            first_cycles = cycles
        ok = (emu.activity_argmax == float_a[i] and
              emu.environment_argmax == float_e[i] and
              emu.physiological_argmax == float_p[i])
        agree += int(ok)
        if i == 0:
            print(f"Sample 0: FSM emulator -> activity={emu.activity_argmax} "
                  f"environment={emu.environment_argmax} physiological={emu.physiological_argmax} "
                  f"in {cycles} clock cycles")
            print(f"          float model  -> activity={float_a[0]} "
                  f"environment={float_e[0]} physiological={float_p[0]}")

    print()
    print(f"Cycle-accurate FSM emulator vs float32 model: {agree}/{n} "
          f"({agree/n*100:.2f}%) exact argmax agreement across all 3 heads")
    print(f"Cycles per inference: {first_cycles}")
    if agree == n:
        print("PASS: FSM emulator (mirroring the RTL's exact control flow and timing) "
              "matches the trained model on every held-out test window.")
    else:
        print("FAIL: FSM emulator disagrees with the trained model on some windows -- "
              "do not trust the RTL until this is resolved.")


if __name__ == "__main__":
    main()
