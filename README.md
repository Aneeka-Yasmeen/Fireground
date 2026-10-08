# FIREGROUND AI

## AI-Based Firefighter Safety Monitoring and Risk Assessment System

FIREGROUND AI is an AI-based firefighter monitoring system designed to analyze sensor data collected from a firefighter during fireground operations.

The system uses a **multitask AI model** to identify three important aspects of the firefighter's condition:

- **Activity**
- **Physiological state**
- **Environmental condition**

The outputs from these AI tasks are combined by a **Risk Decision Engine**, which determines the overall firefighter risk level as:

**NORMAL → ELEVATED → HIGH**

The project is developed as a software prototype with the target of deploying the AI processing on the **Microchip PolarFire SoC Icicle Kit**.

---

# 1. Problem Statement

Firefighters working in fireground environments are exposed to continuously changing physical and environmental conditions. During an operation, changes in firefighter activity, physiological state, and surrounding environmental conditions can increase the level of risk.

Without continuous monitoring and timely identification of these conditions, it can be difficult to determine the firefighter's current risk level.

Therefore, there is a need for an AI-based monitoring system that can process sensor data, identify activity, physiological state, and environmental condition, and combine these outputs to determine an overall risk level.

**Fireground AI** addresses this problem by using a multitask AI model and a risk decision engine to classify the firefighter's condition as **NORMAL, ELEVATED, or HIGH**.

---

# 2. Project Objectives

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

# 3. System Overview

The overall system follows the pipeline:

```text
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
