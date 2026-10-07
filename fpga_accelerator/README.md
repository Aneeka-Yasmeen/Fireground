# Fireground AI — CNN hardware accelerator

Synthesizable Verilog implementation of the trained multitask 1D-CNN
(`models/multitask_fireground_cnn_v2.keras`) — the same model the software
pipeline runs in TensorFlow, reimplemented as RTL for Microchip Libero SoC.

**Read `docs/DESIGN_NOTES.md` before synthesizing.** It explains exactly
how this was verified (no Verilog simulator was available while writing
it, so verification leaned on hand-tracing plus an independent
cycle-accurate Python emulator — both are documented there), and the
design trade-offs made (fixed-point format, no hardware softmax,
sequential single-MAC datapath, why the weight ROMs will likely map to
distributed RAM rather than Block RAM). None of that is hidden — read it
so the utilization numbers you get mean what you think they mean.

## Folder contents

```
fpga_accelerator/
├── rtl/
│   └── fireground_cnn_top.v         the accelerator (single file, no sub-modules)
├── weights/
│   ├── conv1_w.hex, conv1_b.hex     conv1D layer 1 (9->16ch, kernel 3)
│   ├── conv2_w.hex, conv2_b.hex     conv1D layer 2 (16->32ch, kernel 3)
│   ├── dense_w.hex, dense_b.hex     shared dense layer (32->32)
│   ├── act_w.hex, act_b.hex         activity head (32->3)
│   ├── env_w.hex, env_b.hex         environment head (32->2)
│   ├── phy_w.hex, phy_b.hex         physiological head (32->3)
│   └── quant_format.json            the fixed-point format these were quantized to
├── sim/
│   ├── tb_fireground_cnn_top.v      self-checking testbench -- RUN THIS FIRST
│   ├── test_input.hex               one real test-set sensor window, pre-quantized
│   └── expected_output.json         the classification this test vector should produce
├── tools/
│   ├── export_weights_fixedpoint.py  regenerates weights/*.hex and sim/* from the
│   │                                  trained .keras model; also validates the
│   │                                  fixed-point math against the float model
│   └── verify_fsm_cycle_accurate.py  independent Python emulation of the RTL's
│                                      exact FSM timing, checked against all 200
│                                      held-out test windows (see DESIGN_NOTES.md)
└── docs/
    └── DESIGN_NOTES.md               read this
```

## Step 1 — create a Libero SoC project

1. Open Libero SoC → **File → New Project**.
2. Pick your target device (e.g. the PolarFire SoC part on your Icicle
   Kit). This core has no device-specific primitives, so any PolarFire /
   PolarFire SoC / SmartFusion2 / IGLOO2 part should accept it.
3. Choose **Verilog** as the HDL language for the project (this core is
   plain Verilog-2001, not SystemVerilog or VHDL).

## Step 2 — import the RTL and weight files

1. In the **Design Hierarchy** pane, right-click → **Import Files**, and
   add:
   - `rtl/fireground_cnn_top.v`
2. The `$readmemh` calls inside `fireground_cnn_top.v` reference the
   weight files by relative filename (e.g. `"conv1_w.hex"`), not by path.
   Simulators and Synplify both resolve `$readmemh` paths relative to the
   working directory the tool is run from — the simplest reliable option
   is to **copy the contents of `weights/*.hex` into your Libero project's
   synthesis/simulation working directory** (or add that folder to your
   simulator's/Synplify's include search path, if your Libero version
   supports that in project settings). If you see `$readmemh` file-not-
   found warnings, this is almost always why.

## Step 3 — simulate first (don't skip this)

1. Import `sim/tb_fireground_cnn_top.v` and `sim/test_input.hex` into the
   project as simulation sources (Libero SoC → **Design Hierarchy** →
   right-click the project → **Add Simulation Source**, or use
   ModelSim/QuestaSim directly if that's your Libero's configured
   simulator).
2. Set `tb_fireground_cnn_top` as the active simulation top-level.
3. Run the simulation (Libero SoC → **Simulate**, or **Verify Pre-Synthesis
   Design** in the design flow).
4. Check the simulation log for:
   ```
   PASS: all three outputs match the Python fixed-point reference.
   ```
   If you see `FAIL` or a `TIMEOUT`, stop here — do not trust synthesis
   results until this passes. See `docs/DESIGN_NOTES.md` for what was and
   wasn't verified before this RTL was handed to you, and check the
   `$readmemh` path issue above first if outputs look uninitialized
   (all-zero).

## Step 4 — synthesize and get the utilization report

1. In Libero SoC's **Design Flow** pane, set `fireground_cnn_top` as the
   **root** (top-level) module for synthesis — the testbench is
   simulation-only and should not be included in the synthesis fileset.
2. Double-click **Synthesize** (this runs Synplify Pro under the hood).
3. After synthesis completes, open the **Synthesize log** or the
   **Reports** panel — Libero SoC generates a resource utilization report
   automatically after synthesis (look for "Device Utilization" or
   "Resource Usage" in the synthesis report, or **Reports → Synthesis
   Report** in the design flow). This reports LUT/logic element count,
   register count, Block RAM usage, and DSP block usage against your
   target device's totals.
4. If you also want post-**Place & Route** utilization (accounts for
   routing overhead, not just synthesis estimates), continue the design
   flow through **Place and Route**, then check **Reports → Place and
   Route Report**.

Given the datapath described in `docs/DESIGN_NOTES.md` (combinational
weight-ROM reads), expect the utilization report to show the weight
storage as LUT-based/distributed logic rather than dedicated Block RAM
usage — that's the documented trade-off, not a synthesis failure.

## Regenerating the weight files (only needed if you retrain the model)

```bash
python fpga_accelerator/tools/export_weights_fixedpoint.py
```

This re-reads `models/multitask_fireground_cnn_v2.keras`, re-quantizes to
Q4.11, rewrites every file in `weights/` and `sim/`, and re-validates
against the held-out test set — it will print the argmax agreement
percentage so you can confirm quantization is still lossless (or close to
it) for whatever the model looks like after retraining.
