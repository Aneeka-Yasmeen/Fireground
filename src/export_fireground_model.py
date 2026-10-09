import os
import json
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "../models/multitask_fireground_cnn.keras"
EXPORT_DIR = "../models/exported_v2"

os.makedirs(EXPORT_DIR, exist_ok=True)

print()
print("================================================")
print("       FIREGROUND AI V2 MODEL EXPORT")
print("================================================")
print()

print("Loading V2 model...")

model = load_model(MODEL_PATH)

print("V2 model loaded successfully.")
print()

print("Model summary:")
model.summary()

print()
print("Saving inference model...")

inference_model_path = os.path.join(
    EXPORT_DIR,
    "fireground_inference_v2.keras"
)

model.save(
    inference_model_path,
    include_optimizer=False
)

print("Saved:")
print(inference_model_path)

print()
print("Extracting model weights...")

weights = model.get_weights()

weights_path = os.path.join(
    EXPORT_DIR,
    "fireground_weights_v2.npz"
)

np.savez(
    weights_path,
    *weights
)

print("Saved:")
print(weights_path)

print()
print("Saving model information...")

model_info = {
    "model_name": "Fireground AI Multitask CNN V2",
    "input_shape": list(model.input_shape),
    "number_of_outputs": len(model.outputs),
    "output_shapes": [
        list(output.shape)
        for output in model.outputs
    ],
    "trainable_parameters": int(model.count_params()),
    "weights_count": len(weights)
}

info_path = os.path.join(
    EXPORT_DIR,
    "model_info_v2.json"
)

with open(info_path, "w") as f:
    json.dump(model_info, f, indent=4)

print("Saved:")
print(info_path)

print()
print("================================================")
print("             EXPORT COMPLETE")
print("================================================")
print()
print("Export directory:")
print(EXPORT_DIR)
print()
print("Files created:")

for filename in os.listdir(EXPORT_DIR):
    print(" -", filename)

print()
print("================================================")
