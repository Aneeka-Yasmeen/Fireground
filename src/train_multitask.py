import os
import json
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    GlobalAveragePooling1D,
    Dense
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.utils import to_categorical


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "../dataset/prepared"

MODEL_DIR = "../models"

MODEL_PATH = (
    MODEL_DIR +
    "/multitask_fireground_cnn.keras"
)

HISTORY_PATH = (
    MODEL_DIR +
    "/multitask_training_history.json"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 10
FEATURES = 9


# ============================================================
# LOAD DATA
# ============================================================

print()
print("================================================")
print("      MULTI-TASK FIREGROUND CNN V2")
print("================================================")
print()

print("Loading prepared V2 data...")

X_train = np.load(
    DATA_DIR + "/X_train.npy"
)

X_val = np.load(
    DATA_DIR + "/X_val.npy"
)

X_test = np.load(
    DATA_DIR + "/X_test.npy"
)


ya_train = np.load(
    DATA_DIR + "/y_activity_train.npy"
)

ya_val = np.load(
    DATA_DIR + "/y_activity_val.npy"
)

ya_test = np.load(
    DATA_DIR + "/y_activity_test.npy"
)


ye_train = np.load(
    DATA_DIR + "/y_environment_train.npy"
)

ye_val = np.load(
    DATA_DIR + "/y_environment_val.npy"
)

ye_test = np.load(
    DATA_DIR + "/y_environment_test.npy"
)


yp_train = np.load(
    DATA_DIR + "/y_physiological_train.npy"
)

yp_val = np.load(
    DATA_DIR + "/y_physiological_val.npy"
)

yp_test = np.load(
    DATA_DIR + "/y_physiological_test.npy"
)


# ============================================================
# PRINT DATA SHAPES
# ============================================================

print()
print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)
print()


# ============================================================
# ONE-HOT ENCODING
# ============================================================

ya_train_cat = to_categorical(
    ya_train,
    num_classes=3
)

ya_val_cat = to_categorical(
    ya_val,
    num_classes=3
)

ya_test_cat = to_categorical(
    ya_test,
    num_classes=3
)


ye_train_cat = to_categorical(
    ye_train,
    num_classes=2
)

ye_val_cat = to_categorical(
    ye_val,
    num_classes=2
)

ye_test_cat = to_categorical(
    ye_test,
    num_classes=2
)


yp_train_cat = to_categorical(
    yp_train,
    num_classes=3
)

yp_val_cat = to_categorical(
    yp_val,
    num_classes=3
)

yp_test_cat = to_categorical(
    yp_test,
    num_classes=3
)


# ============================================================
# BUILD MODEL
# ============================================================

print("Building CNN...")

inputs = Input(
    shape=(WINDOW_SIZE, FEATURES)
)


x = Conv1D(
    filters=16,
    kernel_size=3,
    activation="relu",
    name="conv1"
)(inputs)


x = Conv1D(
    filters=32,
    kernel_size=3,
    activation="relu",
    name="conv2"
)(x)


x = GlobalAveragePooling1D(
    name="global_average_pooling"
)(x)


x = Dense(
    32,
    activation="relu",
    name="dense_shared"
)(x)


# ============================================================
# THREE OUTPUT HEADS
# ============================================================

activity_output = Dense(
    3,
    activation="softmax",
    name="activity"
)(x)


environment_output = Dense(
    2,
    activation="softmax",
    name="environment"
)(x)


physiological_output = Dense(
    3,
    activation="softmax",
    name="physiological"
)(x)


# ============================================================
# CREATE MODEL
# ============================================================

model = Model(
    inputs=inputs,
    outputs=[
        activity_output,
        environment_output,
        physiological_output
    ]
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer="adam",

    loss={

        "activity":
            "categorical_crossentropy",

        "environment":
            "categorical_crossentropy",

        "physiological":
            "categorical_crossentropy"
    },

    metrics={

        "activity":
            ["accuracy"],

        "environment":
            ["accuracy"],

        "physiological":
            ["accuracy"]
    }
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    min_lr=0.00001
)


# ============================================================
# TRAIN
# ============================================================

print()
print("Starting training...")
print()


history = model.fit(

    X_train,

    {
        "activity":
            ya_train_cat,

        "environment":
            ye_train_cat,

        "physiological":
            yp_train_cat
    },

    validation_data=(

        X_val,

        {
            "activity":
                ya_val_cat,

            "environment":
                ye_val_cat,

            "physiological":
                yp_val_cat
        }
    ),

    epochs=30,

    batch_size=64,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# TEST EVALUATION
# ============================================================

print()
print("================================================")
print("             TEST EVALUATION")
print("================================================")
print()


evaluation = model.evaluate(

    X_test,

    {
        "activity":
            ya_test_cat,

        "environment":
            ye_test_cat,

        "physiological":
            yp_test_cat
    },

    verbose=1
)


print()
print("Evaluation values:")
print(evaluation)


# ============================================================
# PREDICTION ACCURACIES
# ============================================================

predictions = model.predict(
    X_test,
    verbose=0
)


activity_pred = np.argmax(
    predictions[0],
    axis=1
)

environment_pred = np.argmax(
    predictions[1],
    axis=1
)

physiological_pred = np.argmax(
    predictions[2],
    axis=1
)


activity_accuracy = np.mean(
    activity_pred == ya_test
)

environment_accuracy = np.mean(
    environment_pred == ye_test
)

physiological_accuracy = np.mean(
    physiological_pred == yp_test
)


# ============================================================
# PRINT ACCURACY
# ============================================================

print()
print("================================================")
print("          V2 TEST ACCURACY")
print("================================================")

print(
    "Activity accuracy:",
    f"{activity_accuracy:.4f}"
)

print(
    "Environment accuracy:",
    f"{environment_accuracy:.4f}"
)

print(
    "Physiological accuracy:",
    f"{physiological_accuracy:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model.save(
    MODEL_PATH
)


# ============================================================
# SAVE HISTORY
# ============================================================

history_data = {
    key: [
        float(value)
        for value in values
    ]
    for key, values
    in history.history.items()
}


with open(
    HISTORY_PATH,
    "w"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("================================================")
print("         V2 TRAINING COMPLETE")
print("================================================")

print()
print("Model saved to:")
print(MODEL_PATH)

print()
print("Training history saved to:")
print(HISTORY_PATH)

print()
print("================================================")
