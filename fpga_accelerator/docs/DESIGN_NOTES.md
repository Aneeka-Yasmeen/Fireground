# Design notes — Fireground AI CNN accelerator

## What this is

A first hardware accelerator for the trained multitask 1D-CNN
(`models/multitask_fireground_cnn_v2.keras`) that the Fireground AI
software pipeline already runs in TensorFlow. Same architecture, same
trained weights, same classification behaviour — implemented as
synthesizable Verilog instead of a Python/TensorFlow forward pass.

Target: Microchip Libero SoC (the project's stated target platform is a
PolarFire SoC Icicle Kit). This core is plain synthesizable Verilog-2001
with no PolarFire-specific primitives, so it should import and synthesize
in any Libero SoC project for a PolarFire, PolarFire SoC, SmartFusion2 or
IGLOO2 part; see `README.md` in this folder for the exact import steps.

## Result: first successful synthesis (Libero SoC 2026.1, Synplify Pro,
## target MPFS250T-FCVG484E)

After the `c2_co` width fix below, state machine extraction reported all
11 states reachable (previously only 5) and synthesis produced a real
netlist:

| Resource | Used | Available |
|---|---|---|
| LUTs (incl. P&R interface logic) | 5,382 | — |
| Sequential elements (SLE) | 434 | — |
| DSP blocks (MACC_PA) | 3 | 784 (0%) |
| Block RAM (RAM1K20) | 1 | 812 (0%) |
| Block RAM (RAM64x12) | 12 | 2,352 (0%) |
| Global clock buffers | 1 | — |

**Correction to the Block RAM prediction below**: the weight ROMs
(conv1_w, conv2_w, dense_w, act/env/phy_w and their biases) were indeed
mapped to logic (distributed RAM), exactly as predicted -- Synplify's own
log reports each one explicitly ("ROM ... mapped in logic"). What wasn't
anticipated: the *activation* scratchpads (`x_mem`, `c1_mem`, `c2_mem`,
`gap_mem`, `sh_mem` -- the read/write buffers, not the read-only weight
tables) were synthesized into real Block RAM primitives (13 total). The
combinational-read design choice affected the read-only weight ROMs as
expected; it didn't stop the read/write activation memories from
inferring proper Block RAM.

**Timing: does not meet the default 100 MHz constraint.** Worst slack is
-15.5 ns; estimated max frequency is ~39.2 MHz. The critical path runs
through the `sdiv` divider Synplify generated for the `/T2` (divide by 6)
step in global average pooling, chained through ~104 logic levels. This
doesn't matter for this application -- even at 39 MHz, one inference
(14,505 cycles) takes ~370 microseconds, far faster than the ~1 Hz sensor
rate this model targets -- but if a higher clock is ever needed for other
reasons on the same fabric, that divider is the thing to replace (e.g. a
fixed-point reciprocal multiply by 1/6 instead of a runtime divide, since
the divisor is a compile-time constant).

## How correctness was checked (read this before trusting the numbers)

**No Verilog simulator was available in the environment this RTL was
written in.** That matters, so here is exactly what was and wasn't
verified, in order:

1. **Fixed-point format chosen and validated in Python first, on the real
   trained weights**, before any RTL was written
   (`tools/export_weights_fixedpoint.py`). Weights were confirmed to fall
   in [-0.73, 0.64] and standardized inputs in [-3.3, 4.0] — both
   comfortably inside the chosen Q4.11 format. The *batch* fixed-point
   math (not yet the hardware's cycle-by-cycle control flow, just the
   arithmetic rules: multiply, round-shift, saturate, bias, ReLU) was run
   against all 200 held-out test windows and reached **100% argmax
   agreement** with the float32 Keras model on all three output heads.
   This validates the *number format*, not the *hardware*.

2. **The RTL's FSM was hand-traced cycle-by-cycle** while writing it. This
   caught two real bugs before they shipped:
   - Registering the MAC operands one cycle ahead of the accumulate step
     (a natural first instinct for a clean pipeline) silently dropped the
     *last* multiply-accumulate term of every inner loop, because the
     state machine advanced to the write-back state before the delayed
     product was actually added to the accumulator. Fixed by reading
     operands combinationally in the same cycle they're accumulated.
   - The output-head argmax comparison read the registered `best_idx`
     register, which lags the current cycle's comparison by one cycle —
     so whenever the *last* class evaluated in a head was the winner, the
     stale (previous) index was latched instead. Fixed by computing the
     final winner from a same-cycle combinational value, not the
     registered one.

   See the comments directly above `S_CONV1`/`S_HEADS_WB` in
   `rtl/fireground_cnn_top.v` for exactly where.

3. **A second, independent check**: `tools/verify_fsm_cycle_accurate.py`
   is a Python emulator that mirrors the RTL's FSM state-for-state and
   counter-for-counter, including replicating Verilog's non-blocking
   assignment timing (every register update is computed from the current
   cycle and only takes effect at the next cycle boundary, exactly like
   `<=`). Run against all 200 held-out test windows, it reached **100%
   exact agreement** with the float32 model on all three heads, taking
   14,505 clock cycles per inference. This is strong evidence the FSM's
   *logic and timing* are correct.

4. **Real synthesis in Libero SoC (Synplify Pro) found a real bug** that
   neither of the above caught: `c2_co` (the conv2 output-channel loop
   counter) was declared as a 4-bit register (`reg [3:0]`, max value 15)
   but needs to count 0–31 (`COUT2=32` output channels). In real hardware
   this silently wraps at 15 back to 0 and never reaches 31, so the
   condition that advances the FSM past conv2 into the rest of the
   pipeline never fires — the design hangs forever cycling inside conv2,
   `done` never asserts, and every output stays at its reset value.
   Synplify's synthesis-time reachability analysis correctly proved this
   (state machine extraction reported only 5 of 11 states reachable) and
   optimized the entire dead design down to 0 LUTs / 0 registers — which
   at first looked like a tooling problem (missing weight files, missing
   clock constraint) before the real cause was traced through the
   synthesis log.

   **This is exactly the gap flagged below**: Python integers have no
   fixed width, so `tools/verify_fsm_cycle_accurate.py` silently used an
   unbounded counter for `c2_co` and never reproduced the wraparound —
   it agreed with the float model 100% of the time while emulating
   hardware that, as originally written, would never have produced a
   result at all. The script now enforces the same register widths the
   RTL declares (`REG_WIDTHS` table, applied every cycle) specifically so
   it can catch this bug class going forward; reproducing the bug on
   purpose (forcing the old 4-bit width back in) does correctly hang the
   emulator the same way, confirming the mechanism works. Fixed in the
   RTL by widening `c2_t`/`c2_co`/`c2_k`/`c2_ci` to 6 bits (matching the
   sizing already used for the other 32-wide counters like `d_out`).

   **The practical lesson**: hand-tracing and a same-language emulator
   both check *logic*, but only something that models real fixed-width
   registers (a real Verilog simulator, or a width-aware emulator like
   the updated one here) checks *hardware*. Please still run
   `sim/tb_fireground_cnn_top.v` in an actual Verilog simulator before
   trusting future changes to this design — this bug is proof that class
   of check finds things the others structurally cannot.

**What step 3 does NOT prove**: that the Verilog *text itself* is free of
syntax slips, port-width mismatches, or anything a real simulator would
catch that a hand-written Python mirror of the same intended logic
wouldn't. **Run `sim/tb_fireground_cnn_top.v` in Libero's simulator (or
any Verilog simulator) before synthesizing** — it's a self-checking
testbench that loads the same test vector and checks for the same
expected result (`activity=2, environment=1, physiological=1`). If it
doesn't pass, do not trust the synthesis/utilization numbers that follow.

## Key design choices, and their trade-offs

### Number format: uniform Q4.11 everywhere

One fixed-point format (1 sign + 4 integer + 11 fractional bits) is used
for inputs, weights, biases, and every intermediate activation, rather
than a different scale per layer/tensor (which is what a production INT8
quantization scheme, like the project's existing VectorBlox VNNX export,
would normally use).

- **Why**: keeps the multiply-accumulate hardware uniform (one datapath
  width everywhere) and easy to verify by hand and in Python, which
  mattered a lot given there was no simulator available to fall back on.
- **Cost**: weight precision is lower than a dedicated per-tensor INT8
  scheme would give (weights only span roughly ±0.7 of the format's ±16
  range, so most of the 15-bit magnitude range is unused for weights
  specifically). It still validated at 100% classification agreement on
  the held-out set, so it isn't a functional problem for this model — but
  a smaller, denser format (e.g. INT8 weights with a separate per-layer
  scale, closer to what VectorBlox already does for the VNNX export)
  would shrink the weight ROMs and could be a worthwhile follow-up.

### No hardware softmax — argmax on raw logits instead

The trained model's three output heads apply softmax. Softmax is
monotonic (never changes which class has the highest score, only the
reported confidence), so this accelerator reports the argmax of the raw
logits directly and skips implementing `exp()`/division in hardware. This
is the right simplification if you only need the classification decision;
if a confidence score is needed on-chip, hardware softmax is a real,
separate piece of future work (a lookup-table-based exp approximation
plus a divider, or exporting logits and doing softmax on the host).

### Sequential single-MAC datapath (not parallel/pipelined)

One shared multiplier-accumulator, one MAC per clock cycle. 14,505 cycles
per inference (measured by the cycle-accurate emulator). At even a modest
50 MHz that's well under 1 ms — far faster than the ~1 Hz sensor sample
rate this model was designed for — so throughput was not a design
pressure here.

- **Why**: smallest, simplest, easiest-to-verify design for a first
  accelerator, and it's already fast enough for this application.
- **Follow-up**: if you want higher throughput (e.g. to accelerate
  batched/offline inference over many windows, or to leave more clock
  budget for other logic sharing the fabric), the natural next step is
  parallelizing across output channels within a layer (multiple MACs
  running concurrently), which trades LUT/DSP usage for cycle count.

### Weight ROMs use combinational (asynchronous) reads

Each `reg` array (`conv1_w_mem`, `conv2_w_mem`, etc.) is read directly by
address in the same cycle the accumulate happens, rather than registering
the read address one cycle ahead of a synchronous-read memory access.

- **Why**: this was the fix for the dropped-last-MAC-term bug above —
  combinational read-and-accumulate in the same cycle is much simpler to
  reason about (and to verify by hand) than a registered/pipelined read
  with a correctly-timed drain cycle at each loop boundary.
- **Cost**: PolarFire's dedicated Block RAM (µSRAM/LSRAM) primitives only
  support **synchronous** (registered-address) reads — they cannot serve
  a combinational/asynchronous read port. Describing the weight ROMs this
  way means Synplify (Libero's synthesis engine) will most likely infer
  them as **distributed RAM (LUT-based)** rather than dedicated Block
  RAM. In absolute terms this is still small — the full weight set is
  3,336 parameters × 16 bits ≈ 6.5 Kbit, which is a modest number of LUTs
  even as distributed memory — but it means this design is not making use
  of the FPGA's dedicated RAM blocks, which a more area-optimized version
  would.
- **Follow-up, if the utilization report shows this is worth fixing**:
  convert each weight ROM to a registered-address / synchronous-read
  pattern (address registered one cycle, data read the next), and add a
  one-cycle "drain" state at each inner-loop boundary so the last
  pipelined read is correctly folded into the accumulator before moving
  on. This is exactly the kind of change that needs simulation to trust —
  it reintroduces the timing subtlety that caused the original bug this
  version fixed — so it's flagged here as a deliberate next step rather
  than attempted blind.

## Interface

`fireground_cnn_top` exposes a minimal host interface:

- `x_wr_en` / `x_wr_addr` / `x_wr_data`: write one Q4.11 input sample at a
  time into the input scratchpad (`x_mem[t*9+c]`, 90 entries) before
  asserting `start`. Producing these 90 already-standardized, already
  quantized values is expected to happen on the host/software side
  (matching the standardization the software pipeline already does with
  the same MEAN/SCALE constants — see
  `src/live_vnnx_fireground_system_liveweb_v2.py`), not on this core.
- `start`: pulse one cycle to begin an inference once the input window is
  loaded.
- `done`: pulses one cycle when `activity_argmax`, `environment_argmax`,
  and `physiological_argmax` are valid.

There is no AXI/APB wrapper here — this is the compute core only. Wiring
it into a PolarFire SoC's fabric-to-processor interconnect (e.g. an APB3
register wrapper so the RISC-V side can write the input window and poll
`done`) is a separate, standard piece of integration work not attempted
here, since it depends on which interconnect/address map your specific
Libero project design uses.

## What "necessary" means here

This folder is deliberately scoped to the accelerator core: RTL, weights
derived from the actual trained model, a testbench, and the Python tools
that produced/verified them. It does not include a Libero project file
(`.prjx`) — Libero project files are tool-version-specific binary/XML
state that doesn't transfer cleanly between Libero SoC versions or target
parts, so the right way to get this into Libero is creating a fresh
project for your specific target device and importing these HDL files as
sources (see `README.md`), not shipping a pre-built project file that
might not open correctly on your Libero version.
