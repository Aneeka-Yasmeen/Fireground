import numpy as np
import pandas as pd

np.random.seed(42)

windows_per_class = 200
timesteps = 10

rows = []


def add_window(
    activity,
    environment,
    physiological,
    heart_rate_base,
    heart_rate_rise,
    gas_base,
    gas_rise,
    temperature_base,
    temperature_rise
):

    for t in range(timesteps):

        heart_rate = (
            heart_rate_base
            + t * heart_rate_rise
            + np.random.normal(0, 2)
        )

        gas = (
            gas_base
            + t * gas_rise
            + np.random.normal(0, 2)
        )

        temperature = (
            temperature_base
            + t * temperature_rise
            + np.random.normal(0, 0.4)
        )

        acc_x = np.random.normal(
            0.2 if activity == "standing" else 0.4,
            0.05
        )

        acc_y = np.random.normal(0.2, 0.05)

        acc_z = np.random.normal(9.7, 0.07)

        gyro_x = np.random.normal(
            0.01 if activity == "standing" else 0.04,
            0.01
        )

        gyro_y = np.random.normal(
            0.02 if activity == "standing" else 0.04,
            0.01
        )

        gyro_z = np.random.normal(
            0.01 if activity == "standing" else 0.04,
            0.01
        )

        rows.append({
            "activity": activity,
            "environment": environment,
            "physiological": physiological,
            "heart_rate": heart_rate,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
            "gyro_x": gyro_x,
            "gyro_y": gyro_y,
            "gyro_z": gyro_z,
            "gas": gas,
            "temperature": temperature
        })


# ------------------------------------------
# 1. NORMAL
# ------------------------------------------

for _ in range(windows_per_class):

    add_window(
        activity="standing",
        environment="normal",
        physiological="normal",
        heart_rate_base=75,
        heart_rate_rise=0.2,
        gas_base=20,
        gas_rise=0.2,
        temperature_base=30,
        temperature_rise=0.05
    )


# ------------------------------------------
# 2. WALKING
# ------------------------------------------

for _ in range(windows_per_class):

    add_window(
        activity="walking",
        environment="normal",
        physiological="elevated",
        heart_rate_base=80,
        heart_rate_rise=1.0,
        gas_base=20,
        gas_rise=0.3,
        temperature_base=30,
        temperature_rise=0.1
    )


# ------------------------------------------
# 3. ENVIRONMENTAL RISK
# ------------------------------------------

for _ in range(windows_per_class):

    add_window(
        activity="standing",
        environment="risk",
        physiological="elevated",
        heart_rate_base=78,
        heart_rate_rise=0.8,
        gas_base=20,
        gas_rise=4.0,
        temperature_base=30,
        temperature_rise=0.6
    )


# ------------------------------------------
# 4. COMBINED DETERIORATION
# ------------------------------------------

for _ in range(windows_per_class):

    add_window(
        activity="crawling",
        environment="risk",
        physiological="high",
        heart_rate_base=85,
        heart_rate_rise=2.0,
        gas_base=20,
        gas_rise=4.5,
        temperature_base=30,
        temperature_rise=0.7
    )


# ------------------------------------------
# Save
# ------------------------------------------

df = pd.DataFrame(rows)

df.to_csv(
    "../dataset/multitask_training_data.csv",
    index=False
)

print("Multi-task dataset created successfully!")
print()
print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print()
print("Activity distribution:")
print(df["activity"].value_counts())

print()
print("Environment distribution:")
print(df["environment"].value_counts())

print()
print("Physiological distribution:")
print(df["physiological"].value_counts())