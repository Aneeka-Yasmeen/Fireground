# Fireground AI — CNN hardware accelerator

Synthesizable Verilog implementation of the trained multitask 1D-CNN
(`models/multitask_fireground_cnn_v2.keras`) — the same model the software pipeline
runs in TensorFlow, reimplemented as RTL and synthesized for Microchip Libero SoC.

Full verification methodology, design trade-offs, and the real synthesis results
(including two bugs found by hand-tracing and one found only by reading the actual
Synplify synthesis log) are in `docs/DESIGN_NOTES.md`.

## Folder contents

fpga_accelerator/
├── rtl/
│   └── fireground_cnn_top.v       # The accelerator (single file, no sub-modules)
├── weights/
│   ├── conv1_w.hex, conv1_b.hex   # conv1D layer 1 (9->16ch, kernel 3)
│   ├── conv2_w.hex, conv2_b.hex   # conv1D layer 2 (16->32ch, kernel 3)
│   ├── dense_w.hex, dense_b.hex   # Shared dense layer (32->32)
│   ├── act_w.hex, act_b.hex       # Activity head (32->3)
│   ├── env_w.hex, env_b.hex       # Environment head (32->2)
│   ├── phy_w.hex, phy_b.hex       # Physiological head (32->3)
│   └── quant_format.json          # The fixed-point format these were quantized to
└── sim/
    ├── tb_fireground_cnn_top.v    # Self-checking testbench
    ├── test_input.hex             # One real test-set sensor window, pre-quantized
    ├── expected_output.json       # The classification this test vector produces
    ├── tools/
    │   ├── export_weights_fixedpoint.py  # Regenerated weights/hex and sim/ from the trained .keras model; 
    │   │                                 # validated the fixed-point math against the float model
    │   └── verify_fsm_cycle_accurate.py  # Independent Python emulation of the RTL's exact FSM timing, 
    │                                     # checked against all 200 held-out test windows
    └── docs/
        └── DESIGN_NOTES.md        # Verification report and design rationale
## Synthesis setup and result

This core was imported into a fresh Libero SoC 2026.1 project targeting the
MPFS250T-FCVG484E (the PolarFire SoC part on the Icicle Kit), as plain Verilog-2001
with no device-specific primitives. The weight `.hex` files were placed in the
synthesis working directory so the RTL's `$readmemh` calls (which reference them by
relative filename) could resolve, and a clock constraint was added via Libero's
Constraint Manager before synthesizing (`rtl/fireground_cnn_top.sdc`) — without one,
Synplify has no timing paths to preserve and prunes the design as unreachable logic,
which is what happened on the first attempt.
Synthesis (Synplify Pro) produced a real netlist: 5,382 LUTs, 434 sequential
elements, 3 DSP blocks (of 784), 13 Block RAMs (of 812+2,352), running at an
estimated ~39.2 MHz — roughly 2% of the device. This is a single-lane proof of
concept: it confirms the trained model maps correctly onto PolarFire SoC fabric.
The planned final design replicates this lane once per firefighter, so utilization
scales with crew size from this measured baseline. Full numbers, what limits the
clock frequency, and why the weight ROMs mapped to logic rather than Block RAM are
in `docs/DESIGN_NOTES.md`.

## What wasn't done

No Verilog simulator was available in the environment this was built in, so
`sim/tb_fireground_cnn_top.v` has not been run in an actual simulator — only hand-
traced cycle-by-cycle and cross-checked against an independent Python emulation of
the same FSM (see `docs/DESIGN_NOTES.md` for exactly what that did and didn't prove).
Running the testbench in a real simulator is the one verification step still open.
There's also no AXI/APB wrapper here — this is the compute core only; wiring it into
the PolarFire SoC's fabric-to-processor interconnect is separate integration work
not yet attempted.
