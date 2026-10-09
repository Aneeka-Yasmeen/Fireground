# Fireground AI

Multitask-CNN-based firefighter risk monitoring, accelerated in custom FPGA RTL on a
Microchip PolarFire SoC Icicle Kit. Built for the 2026 PolarFire FPGA Design Contest —
Edge Intelligence track.

**Status: v1.** The pipeline, dashboard, and FPGA accelerator below are working end to
end today. This is a first version, not a finished product — see "What's next" below
for what's planned on top of it.

## AI-Based Firefighter Safety Monitoring and Risk Assessment System

FIREGROUND AI is an AI-based firefighter monitoring system designed to analyze sensor data collected from a firefighter during fireground operations.

Each firefighter wears a sensor node reporting 9 channels (heart rate, 3-axis accelerometer, 3-axis gyroscope, gas, temperature). A rolling 10-sample window of that data is fed through a **multitask 1D CNN model** that classifies the following three things at once:

- **Activity**
- **Physiological state**
- **Environmental condition**

The outputs from these AI tasks are combined by a **Risk Decision Engine**, which determines the overall firefighter risk level as:

**NORMAL → ELEVATED → HIGH**

A dashboard shows every firefighter's status individually, and separately produces one aggregate truck/pump-operator water-demand recommendation sized to the worst reading found anywhere on scene.

The project is developed as a software prototype with the target of deploying the AI processing on the **Microchip PolarFire SoC Icicle Kit**.

---

# The problem: Firefighter safety is a real-time, multi-variable decision problem

Firefighters operate in environments where several hazards can develop simultaneously. Toxic gases can impair a firefighter, extreme temperatures can cause heat stress, smoke and structural damage can restrict movement, and intense physical exertion can lead to exhaustion or incapacitation. These conditions may change rapidly, while commanders outside the structure have limited direct visibility into each firefighter's condition.

The problem is not simply that firefighters lack sensors. The deeper problem is that measurements of a firefighter's physical condition, movement, and surrounding environment must be interpreted together and communicated to the command team in time to act.

For example, an elevated heart rate could be a normal response to strenuous activity. However, if it occurs alongside rising temperature, hazardous gas exposure, and a change from walking to crawling, the combination may indicate a developing emergency. Interpreting these signals independently can miss the significance of their combined pattern.

---

# Project Objectives

The main objectives of FIREGROUND AI are:

1. Monitor firefighter-related sensor data.
2. Process sensor data using a sliding-window approach.
3. Use a multitask AI model to analyze multiple conditions simultaneously.
4. Identify firefighter activity.
5. Identify physiological condition.
6. Identify environmental condition.
7. Combine AI outputs using a Risk Decision Engine.
8. Classify the overall firefighter condition as NORMAL, ELEVATED, or HIGH.
9. Provide a software dashboard for monitoring the results.
10. Prepare the system for deployment on the Microchip PolarFire SoC Icicle Kit.

---
## About the dataset

The sensor streams this version runs on are synthetically generated
(`src/generate_multi_firefighter_stream.py` for the live demo,
`src/generate_multitask_data.py` for model training) — they are not recordings from
real firefighters or real incidents. The one real-world data source in this project is
`dataset/fireground_ai_reference_dataset.csv`, extracted from FSRI (Fire Safety
Research Institute) live-fire case studies; the truck-level water-demand numbers are
anchored to that, not to the synthetic sensor data. Replacing the synthetic streams
with real wearable sensor data is the biggest gap between this prototype and a
deployable system.

---
## The FPGA accelerator

The trained model above is reimplemented as synthesizable Verilog and was synthesized
in Microchip Libero SoC 2026.1 (Synplify Pro) targeting the MPFS250T-FCVG484E: 5,382
LUTs, 434 sequential elements, 3 DSP blocks, 13 Block RAMs, ~39.2 MHz, roughly 2% of
the device — a single-lane proof of concept, not the final utilization figure (the
final design replicates this lane once per firefighter). Full methodology, the real
bugs found along the way, and what was and wasn’t verified: fpga_accelerator/docs/DESIGN_NOTES.md.

 ---
## Folder structure
src/ pipeline: data generation, model training, live inference, dashboard publishing
models/ trained model (multitask_fireground_cnn_v2.keras) and its scaler
dataset/ firefighter registry, FSRI reference dataset, synthetic demo sensor stream
fpga_accelerator/ the FPGA CNN accelerator: RTL, testbench, weight export, verification report
vercel_dashboard/ the live dashboard (per-firefighter view + aggregate water demand)

---

# 3. System Overview

The overall system follows the pipeline:


┌──────────────────────┐
│    Sensor Data       │
│                      │
│ Heart Rate           │
│ Accelerometer        │
│ Gyroscope            │
│ Gas                  │
│ Temperature          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Sliding Window     │
│                      │
│ 10 Sensor Samples    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Preprocessing      │
│                      │
│ Scaling / Formatting │
│ INT8 Quantization    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────┐
│      Multitask AI Model     │
│                             │
│ ┌─────────┐ ┌────────────┐ │
│ │Activity │ │Physiology  │ │
│ └─────────┘ └────────────┘ │
│                             │
│ ┌─────────────────────────┐ │
│ │     Environment         │ │
│ └─────────────────────────┘ │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│    Risk Decision Engine     │
│                             │
│ AI outputs are combined     │
│ to determine overall risk   │
└──────────────┬──────────────┘
               │
               ▼
       ┌─────────────────┐
       │  Risk Level     │
       │                 │
       │ NORMAL          │
       │ ELEVATED        │
       │ HIGH            │
       └─────────────────┘ 

---

What’s next

This is v1. Planned for later versions:

Real wearable sensor hardware in place of the synthetic stream
RISC-V-side coordination on the PolarFire SoC (currently host-side software)
Multi-hop mesh networking and PUF-based tamper-resistant firmware
Parallel per-firefighter inference lanes on the FPGA fabric

