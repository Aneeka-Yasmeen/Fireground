# 🔥 FIREGROUND AI

## AI-Based Firefighter Monitoring and Risk Assessment System

Fireground AI is a software prototype for monitoring firefighter conditions using sensor data and a multitask AI model.

The system analyzes three major conditions:

- **Activity**
- **Physiological State**
- **Environmental Condition**

The AI outputs are passed to a **Risk Decision Engine**, which produces one of three risk levels:

- 🟢 **NORMAL**
- 🟡 **ELEVATED**
- 🔴 **HIGH**

---

## 🚨 Problem Statement

Firefighters working in fireground environments are exposed to continuously changing physical and environmental conditions. During an operation, changes in firefighter activity, physiological state, and surrounding environmental conditions can increase the level of risk.

Without continuous monitoring and timely identification of these conditions, it can be difficult to determine the firefighter's current risk level.

Therefore, there is a need for an AI-based monitoring system that can process sensor data, identify activity, physiological state, and environmental condition, and combine these outputs to determine an overall risk level.

**Fireground AI** addresses this problem by using a multitask AI model and a Risk Decision Engine to classify the firefighter's condition as **NORMAL, ELEVATED, or HIGH**.

---

## 📌 Project Overview

Fireground AI is designed to monitor firefighter conditions using sensor data and artificial intelligence.

The system processes sensor information using a **10-sample sliding window**. The processed data is passed through the AI model to identify:

1. Firefighter activity
2. Physiological state
3. Environmental condition

These outputs are then evaluated by the Risk Decision Engine to determine the overall firefighter risk status.

### Overall System Flow

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
┌──────────────────────────────┐
│ Activity                     │
│ Physiological State          │
│ Environmental Condition      │
└──────────────────────────────┘
     ↓
Risk Decision Engine
     ↓
NORMAL / ELEVATED / HIGH
     ↓
Dashboard
