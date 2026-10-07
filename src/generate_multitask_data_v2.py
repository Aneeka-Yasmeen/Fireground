import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

np.random.seed(123)

NUM_WINDOWS = 2000
WINDOW_SIZE = 10


# ============================================================
# STORAGE
# ============================================================

rows = []


# ============================================================
# HELPER
# ============================================================

def noise(scale):
    return np.random.normal(0, scale)


# ============================================================
# GENERATE WINDOW
# ============================================================

for window_id in range(NUM_WINDOWS):

    # --------------------------------------------------------
    # Choose a situation
    # --------------------------------------------------------

    situation = np.random.choice(
        [
            "normal_standing",
            "normal_walking",
            "elevated_walking",
            "risk_crawling",
            "risk_environment",
            "combined_high"
        ],
        p=[
            0.25,
            0.20,
            0.15,
            0.15,
            0.10,
            0.15
        ]
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if situation == "normal_standing":

        activity = "standing"
        environment = "normal"
        physiological = "normal"

    elif situation == "normal_walking":

        activity = "walking"
        environment = "normal"
        physiological = "normal"

    elif situation == "elevated_walking":

        activity = "walking"
        environment = "normal"
        physiological = "elevated"

    elif situation == "risk_crawling":

        activity = "crawling"
        environment = "normal"
        physiological = "elevated"

    elif situation == "risk_environment":

        activity = np.random.choice(
            ["walking", "crawling"]
        )
        environment = "risk"
        physiological = np.random.choice(
            ["elevated", "high"]
        )

    else:

        activity = "crawling"
        environment = "risk"
        physiological = "high"


    # --------------------------------------------------------
    # Base sensor values
    # --------------------------------------------------------

    if activity == "standing":

        hr_base = 75
        acc_motion = 0.10
        gyro_base = 0.01

    elif activity == "walking":

        hr_base = 82
        acc_motion = 0.25
        gyro_base = 0.03

    else:

        hr_base = 90
        acc_motion = 0.40
        gyro_base = 0.05


    # --------------------------------------------------------
    # Physiological effect
    # --------------------------------------------------------

    if physiological == "normal":

        hr_extra = 0

    elif physiological == "elevated":

        hr_extra = 10

    else:

        hr_extra = 20


    # --------------------------------------------------------
    # Environment effect
    # --------------------------------------------------------

    if environment == "normal":

        gas_base = 20
        temp_base = 30

    else:

        gas_base = 45
        temp_base = 34


    # --------------------------------------------------------
    # Generate 10 time steps
    # --------------------------------------------------------

    for step in range(WINDOW_SIZE):

        # gradual change within a window
        progress = step / (WINDOW_SIZE - 1)

        heart_rate = (
            hr_base
            + hr_extra
            + noise(3)
            + progress * noise(1)
        )

        # Accelerometer
        acc_x = (
            acc_motion
            + noise(0.07)
        )

        acc_y = (
            acc_motion
            + noise(0.07)
        )

        acc_z = (
            9.8
            - acc_motion * 0.5
            + noise(0.10)
        )

        # Gyroscope
        gyro_x = (
            gyro_base
            + noise(0.015)
        )

        gyro_y = (
            gyro_base
            + noise(0.015)
        )

        gyro_z = (
            gyro_base
            + noise(0.015)
        )

        # Gas
        gas = (
            gas_base
            + noise(4)
        )

        # Temperature
        temperature = (
            temp_base
            + noise(0.7)
        )

        # ----------------------------------------------------
        # Add occasional sensor disturbances
        # ----------------------------------------------------

        if np.random.random() < 0.03:

            gas += np.random.uniform(
                5,
                15
            )

        if np.random.random() < 0.02:

            temperature += np.random.uniform(
                1,
                3
            )

        if np.random.random() < 0.02:

            heart_rate += np.random.uniform(
                -8,
                8
            )

        # ----------------------------------------------------
        # Keep values physically reasonable
        # ----------------------------------------------------

        heart_rate = max(
            45,
            min(160, heart_rate)
        )

        gas = max(
            0,
            min(100, gas)
        )

        temperature = max(
            20,
            min(50, temperature)
        )

        acc_z = max(
            8.5,
            min(10.5, acc_z)
        )

        rows.append({

            "window_id": window_id,

            "timestamp":
                f"14:{window_id // 60:02d}:{step:02d}",

            "heart_rate":
                heart_rate,

            "acc_x":
                acc_x,

            "acc_y":
                acc_y,

            "acc_z":
                acc_z,

            "gyro_x":
                gyro_x,

            "gyro_y":
                gyro_y,

            "gyro_z":
                gyro_z,

            "gas":
                gas,

            "temperature":
                temperature,

            "activity":
                activity,

            "environment":
                environment,

            "physiological":
                physiological,

            "situation":
                situation
        })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

output_path = (
    "../dataset/"
    "multitask_training_data_v2.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("================================================")
print("   REALISTIC MULTI-TASK DATASET V2")
print("================================================")
print()

print(
    "Dataset created successfully!"
)

print()

print(
    "Total rows:",
    len(df)
)

print(
    "Total windows:",
    df["window_id"].nunique()
)

print(
    "Rows per window:",
    WINDOW_SIZE
)

print()

print("Activity distribution:")
print(
    df.groupby(
        "window_id"
    )["activity"].first().value_counts()
)

print()

print("Environment distribution:")
print(
    df.groupby(
        "window_id"
    )["environment"].first().value_counts()
)

print()

print("Physiological distribution:")
print(
    df.groupby(
        "window_id"
    )["physiological"].first().value_counts()
)

print()

print("Situation distribution:")
print(
    df.groupby(
        "window_id"
    )["situation"].first().value_counts()
)

print()

print("Sample data:")
print(
    df.head(10)
)

print()

print(
    "Saved to:",
    output_path
)

print()
print("================================================")