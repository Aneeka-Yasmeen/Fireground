# FIREGROUND AI

## AI-Based Firefighter Monitoring and Risk Assessment System

Fireground AI is a software prototype for monitoring firefighter conditions using sensor data and a multitask AI model.

The system is designed to analyze multiple aspects of firefighter condition simultaneously and convert the AI outputs into an overall firefighter risk status.

The system analyzes three major conditions:

- **Activity**
- **Physiological State**
- **Environmental Condition**

The outputs from these AI tasks are passed to a **Risk Decision Engine**, which produces one of three overall risk levels:

- **NORMAL**
- **ELEVATED**
- **HIGH**

The current implementation is a **software-only AI monitoring prototype** designed toward deployment on the **Microchip PolarFire SoC Icicle Kit**.

---

# Problem Statement

Firefighters working in fireground environments are exposed to continuously changing physical and environmental conditions. During an operation, changes in firefighter activity, physiological state, and surrounding environmental conditions can increase the level of risk.

A firefighter may change between different activities while also experiencing changes in physiological condition and environmental exposure. Monitoring these conditions separately can make it difficult to determine the overall condition of the firefighter.

Without continuous monitoring and timely identification of these changing conditions, it can be difficult to determine whether the firefighter is operating under normal, elevated, or high-risk conditions.

Therefore, there is a need for an AI-based monitoring system that can process sensor data, identify firefighter activity, physiological state, and environmental condition, and combine these outputs to determine an overall risk level.

**Fireground AI** addresses this problem by using a multitask AI model and a Risk Decision Engine to process sensor information and classify the firefighter's overall condition as:

- **NORMAL**
- **ELEVATED**
- **HIGH**

---

# Project Overview

Fireground AI is designed to monitor firefighter conditions using sensor data and artificial intelligence.

The system processes sensor information using a **10-sample sliding window**. The processed sensor data is passed through the AI pipeline to identify three major conditions:

1. **Firefighter Activity**
2. **Physiological State**
3. **Environmental Condition**

The three AI outputs are then passed to the **Risk Decision Engine**.

The Risk Decision Engine evaluates the outputs and produces a single overall firefighter risk status.

The overall concept is:

```text
Sensor Data
     ↓
10-Sample Sliding Window
     ↓
Preprocessing
     ↓
INT8 Quantization
     ↓
AI Inference
     ↓
Activity
Physiological State
Environmental Condition
     ↓
Risk Decision Engine
     ↓
NORMAL / ELEVATED / HIGH
     ↓
Dashboard
