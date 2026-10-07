import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = "../dataset/multitask_training_data_v2.csv"

OUTPUT_DIR = "../dataset/v2_prepared"

SCALER_PATH = "../models/scaler_v2.pkl"


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 10

FEATURES = [
    "heart_rate",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "gas",
    "temperature"
]


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("================================================")
print("     PREPARE MULTI-TASK DATASET V2")
print("================================================")
print()

print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print("Total rows:", len(df))
print(
    "Total windows:",
    df["window_id"].nunique()
)
print()


# ============================================================
# CHECK WINDOW SIZE
# ============================================================

window_counts = df.groupby(
    "window_id"
).size()

invalid_windows = window_counts[
    window_counts != WINDOW_SIZE
]

if len(invalid_windows) > 0:

    raise ValueError(
        "Some windows do not contain exactly "
        f"{WINDOW_SIZE} samples."
    )

print("All windows contain 10 samples.")
print()


# ============================================================
# EXTRACT COMPLETE WINDOWS
# ============================================================

X = []

y_activity = []

y_environment = []

y_physiological = []

window_ids = []


for window_id, group in df.groupby(
    "window_id"
):

    group = group.sort_values(
        "timestamp"
    )

    # Sensor matrix
    sensor_window = group[
        FEATURES
    ].values.astype(
        np.float32
    )

    # Labels are constant within a window
    activity = group[
        "activity"
    ].iloc[0]

    environment = group[
        "environment"
    ].iloc[0]

    physiological = group[
        "physiological"
    ].iloc[0]

    X.append(sensor_window)

    y_activity.append(activity)

    y_environment.append(environment)

    y_physiological.append(physiological)

    window_ids.append(window_id)


X = np.array(X, dtype=np.float32)

y_activity = np.array(
    y_activity
)

y_environment = np.array(
    y_environment
)

y_physiological = np.array(
    y_physiological
)

window_ids = np.array(
    window_ids
)


# ============================================================
# LABEL ENCODING
# ============================================================

activity_classes = [
    "crawling",
    "standing",
    "walking"
]

environment_classes = [
    "normal",
    "risk"
]

physiological_classes = [
    "elevated",
    "high",
    "normal"
]


activity_to_id = {
    name: index
    for index, name
    in enumerate(activity_classes)
}

environment_to_id = {
    name: index
    for index, name
    in enumerate(environment_classes)
}

physiological_to_id = {
    name: index
    for index, name
    in enumerate(physiological_classes)
}


y_activity_encoded = np.array([
    activity_to_id[value]
    for value in y_activity
])

y_environment_encoded = np.array([
    environment_to_id[value]
    for value in y_environment
])

y_physiological_encoded = np.array([
    physiological_to_id[value]
    for value in y_physiological
])


# ============================================================
# COMPOSITE STRATIFICATION LABEL
# ============================================================

combined_labels = np.array([
    f"{a}_{e}_{p}"
    for a, e, p in zip(
        y_activity_encoded,
        y_environment_encoded,
        y_physiological_encoded
    )
])


# ============================================================
# FIRST SPLIT
# 80% TRAIN
# 20% TEMP
# ============================================================

(
    X_train,
    X_temp,
    ya_train,
    ya_temp,
    ye_train,
    ye_temp,
    yp_train,
    yp_temp,
    ids_train,
    ids_temp,
    labels_train,
    labels_temp
) = train_test_split(
    X,
    y_activity_encoded,
    y_environment_encoded,
    y_physiological_encoded,
    window_ids,
    combined_labels,
    test_size=0.20,
    random_state=42,
    stratify=combined_labels
)


# ============================================================
# SECOND SPLIT
# TEMP → 10% VALIDATION / 10% TEST
# ============================================================

(
    X_val,
    X_test,
    ya_val,
    ya_test,
    ye_val,
    ye_test,
    yp_val,
    yp_test,
    ids_val,
    ids_test
) = train_test_split(
    X_temp,
    ya_temp,
    ye_temp,
    yp_temp,
    ids_temp,
    test_size=0.50,
    random_state=42,
    stratify=labels_temp
)


# ============================================================
# FIT SCALER ON TRAINING DATA ONLY
# ============================================================

print("Fitting scaler using training data only...")

scaler = StandardScaler()

train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

scaler.fit(train_2d)


# ============================================================
# SCALE ALL SETS
# ============================================================

def scale_windows(
    X_data,
    scaler
):

    original_shape = X_data.shape

    X_2d = X_data.reshape(
        -1,
        original_shape[-1]
    )

    X_scaled = scaler.transform(
        X_2d
    )

    return X_scaled.reshape(
        original_shape
    ).astype(
        np.float32
    )


X_train = scale_windows(
    X_train,
    scaler
)

X_val = scale_windows(
    X_val,
    scaler
)

X_test = scale_windows(
    X_test,
    scaler
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)


# ============================================================
# SAVE ARRAYS
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X_train.npy",
    X_train
)

np.save(
    f"{OUTPUT_DIR}/X_val.npy",
    X_val
)

np.save(
    f"{OUTPUT_DIR}/X_test.npy",
    X_test
)

np.save(
    f"{OUTPUT_DIR}/y_activity_train.npy",
    ya_train
)

np.save(
    f"{OUTPUT_DIR}/y_activity_val.npy",
    ya_val
)

np.save(
    f"{OUTPUT_DIR}/y_activity_test.npy",
    ya_test
)

np.save(
    f"{OUTPUT_DIR}/y_environment_train.npy",
    ye_train
)

np.save(
    f"{OUTPUT_DIR}/y_environment_val.npy",
    ye_val
)

np.save(
    f"{OUTPUT_DIR}/y_environment_test.npy",
    ye_test
)

np.save(
    f"{OUTPUT_DIR}/y_physiological_train.npy",
    yp_train
)

np.save(
    f"{OUTPUT_DIR}/y_physiological_val.npy",
    yp_val
)

np.save(
    f"{OUTPUT_DIR}/y_physiological_test.npy",
    yp_test
)


# ============================================================
# SAVE WINDOW IDs
# ============================================================

np.save(
    f"{OUTPUT_DIR}/window_ids_train.npy",
    ids_train
)

np.save(
    f"{OUTPUT_DIR}/window_ids_val.npy",
    ids_val
)

np.save(
    f"{OUTPUT_DIR}/window_ids_test.npy",
    ids_test
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("================================================")
print("          DATA PREPARATION COMPLETE")
print("================================================")
print()

print("Original windows:", len(X))

print()
print("Training windows:", len(X_train))
print("Validation windows:", len(X_val))
print("Testing windows:", len(X_test))

print()

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Testing shape:", X_test.shape)

print()

print("Activity mapping:")

for index, name in enumerate(
    activity_classes
):
    print(
        index,
        "=",
        name
    )

print()

print("Environment mapping:")

for index, name in enumerate(
    environment_classes
):
    print(
        index,
        "=",
        name
    )

print()

print("Physiological mapping:")

for index, name in enumerate(
    physiological_classes
):
    print(
        index,
        "=",
        name
    )

print()

print("Scaler saved to:")
print(SCALER_PATH)

print()

print("Prepared data saved to:")
print(OUTPUT_DIR)

print()
print("================================================")