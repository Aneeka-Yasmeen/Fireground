# FIREGROUND AI

## Project Overview

Fireground AI is a software prototype for monitoring firefighter conditions using sensor data and a multitask AI model.

The system analyzes:
- Activity
- Physiological state
- Environmental condition

The AI outputs are passed to a risk decision engine that produces:
- NORMAL
- ELEVATED
- HIGH

## Target Platform

Microchip PolarFire SoC Icicle Kit

## Current Implementation

Software-only implementation using:
- Python
- TensorFlow/Keras
- NumPy
- INT8 quantization
- Microchip VectorBlox SDK
- VNNX model
- VectorBlox C simulator

No physical hardware is currently available.

## AI Pipeline

Sensor data
    ↓
10-sample sliding window
    ↓
Preprocessing
    ↓
INT8 quantization
    ↓
VNNX AI inference
    ↓
Activity + Physiology + Environment
    ↓
Risk Decision Engine
    ↓
NORMAL / ELEVATED / HIGH

## Model Outputs

### Activity
- 0 = crawling
- 1 = standing
- 2 = walking

### Physiological State
- 0 = elevated
- 1 = high
- 2 = normal

### Environment
- 0 = normal
- 1 = risk

## Live Simulation

Input:
- 60 sensor samples
- 51 sliding windows

Current result:
- 37 NORMAL windows
- 14 ELEVATED windows
- 0 HIGH windows
- Overall status: ELEVATED

## Important Files

- VNNX model: models/fireground_v2_V250_ncomp.vnnx
- Live result: results/live_vnnx_fireground_result.json
- Dashboard: dashboard/fireground_dashboard.py
- Risk graph: results/fireground_risk_progression.png
- Architecture diagram: assets/rtosf.png

## Run the Dashboard

python3 dashboard/fireground_dashboard.py

## Run the Complete VNNX Simulation

source /home/arvind/VectorBlox-SDK/setup_vars.sh
python3 src/live_vnnx_fireground_system.py
