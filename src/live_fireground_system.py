#!/usr/bin/env python3
"""Fireground AI multitask CNN inference with live localhost + Vercel publishing.

Loads the trained multitask CNN (models/multitask_fireground_cnn.keras) directly
through TensorFlow/Keras, runs it over every firefighter's sensor stream from
dataset/firefighter_registry.json, and produces per-firefighter risk status plus one
aggregate truck/pump-operator water-demand recommendation sized to the worst reading
found anywhere on the roster. Results are written locally and optionally published to
a live dashboard (see vercel_publisher.py and .fireground.env.example).
"""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent

_BUNDLED_SITE_PACKAGES = PROJECT / "cnn_env" / "Lib" / "site-packages"
if _BUNDLED_SITE_PACKAGES.exists() and str(_BUNDLED_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(_BUNDLED_SITE_PACKAGES))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vercel_publisher import publish

REGISTRY_PATH = str(PROJECT / "dataset" / "firefighter_registry.json")
DATA_PATH = str(PROJECT / "dataset" / "live_sensor_stream_multi.csv")
KERAS_MODEL_PATH = str(PROJECT / "models" / "multitask_fireground_cnn.keras")
OUTPUT_PATH = str(PROJECT / "results" / "live_fireground_result.json")
WINDOW_SIZE = 10
FEATURES = ["heart_rate","acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z","gas","temperature"]
MEAN = [91.1068859,0.266849514,0.268908626,9.66666423,0.0322428436,0.0323018356,0.0324732444,27.0979919,31.1267252]
SCALE = [13.34430666,0.13700957,0.13697954,0.1158999,0.02163363,0.02163962,0.02157665,11.87917055,1.93309766]
ACTIVITY_LABELS = ["crawling","standing","walking"]
PHYSIOLOGICAL_LABELS = ["elevated","high","normal"]
ENVIRONMENT_LABELS = ["normal","risk"]

_KERAS_MODEL = None


def standardize_window(window):
    return [[(v - MEAN[j]) / SCALE[j] for j, v in enumerate(row)] for row in window]


def run_inference(window):
    global _KERAS_MODEL
    import numpy as np
    import keras

    if _KERAS_MODEL is None:
        _KERAS_MODEL = keras.models.load_model(KERAS_MODEL_PATH)

    x = np.array([standardize_window(window)], dtype="float32")
    activity_p, environment_p, physiological_p = _KERAS_MODEL.predict(x, verbose=0)

    out = {}
    for name, probs in (("activity", activity_p[0]), ("environment", environment_p[0]), ("physiological", physiological_p[0])):
        idx = int(probs.argmax())
        out[name] = {"class_index": idx, "confidence": float(probs[idx])}
    return out


def risk_score(a, e, p):
    return (2 if e == "risk" else 0) + (1 if a == "crawling" else 0) + (2 if p == "high" else 0)


def status(s):
    return "HIGH" if s >= 4 else ("ELEVATED" if s >= 2 else "NORMAL")


def event(a, e, p):
    if a == "crawling" and e == "risk" and p == "high":
        return "deterioration"
    if e == "risk":
        return "environmental_risk"
    if p == "high":
        return "physiological_alert"
    return "normal"


def sensor_payload(rows):
    return [{k: (r[k] if k == "timestamp" else float(r[k])) for k in ["timestamp"] + FEATURES} for r in rows]


def local_fireground_summary(window_rows, score):
    """
    Per-firefighter local temperature/gas reading for their immediate
    position in the building. This is individual monitoring context
    (shown on that firefighter's own card) -- it does NOT produce a
    water-supply recommendation. Water demand is calculated once, for
    the whole incident, from the worst reading found anywhere in the
    building -- see truck_water_demand().

    IMPORTANT: the compartment-temperature estimate here is a simulation
    output for the project demo, not a validated instrument reading.
    """

    temps = [float(r["temperature"]) for r in window_rows]
    gases = [float(r["gas"]) for r in window_rows]

    mean_temp = sum(temps) / len(temps)
    max_temp = max(temps)

    mean_gas = sum(gases) / len(gases)
    max_gas = max(gases)

    # Heat trend from the beginning and end of the current AI window.
    delta_temp = temps[-1] - temps[0]

    if delta_temp > 0.20:
        heat_trend = "RISING"
    elif delta_temp < -0.20:
        heat_trend = "FALLING"
    else:
        heat_trend = "STABLE"

    # --------------------------------------------------------
    # Prototype compartment-temperature estimate
    # --------------------------------------------------------
    # Current wearable temperature is around body/environment
    # sensor range. We therefore do NOT call it raw apartment
    # fire temperature. This creates a separate simulated
    # compartment estimate for the demonstration.
    # --------------------------------------------------------

    compartment_temperature = (
        40.0
        + max(0.0, mean_temp - 30.0) * 4.0
        + mean_gas * 0.55
        + score * 7.0
    )

    compartment_temperature = max(
        20.0,
        min(200.0, compartment_temperature)
    )

    if max_gas >= 50:
        gas_risk = "HIGH"
    elif max_gas >= 30:
        gas_risk = "MEDIUM"
    else:
        gas_risk = "LOW"

    if (
        score >= 4
        or compartment_temperature >= 100
        or max_gas >= 55
    ):
        compartment_risk = "HIGH"

    elif (
        score >= 2
        or compartment_temperature >= 65
        or max_gas >= 35
    ):
        compartment_risk = "ELEVATED"

    else:
        compartment_risk = "LOW"

    return {
        "overall_fire_condition": status(score),
        "gas_risk": gas_risk,
        "heat_trend": heat_trend,
        "compartment_risk": compartment_risk,
        "compartment_temperature_estimate_c": round(compartment_temperature, 1),
        "mean_wearable_temperature_c": round(mean_temp, 1),
        "max_wearable_temperature_c": round(max_temp, 1),
        "mean_gas_ppm": round(mean_gas, 1),
        "max_gas_ppm": round(max_gas, 1),
    }


# --------------------------------------------------------
# Water-demand reference values (truck / pump-operator supply)
# --------------------------------------------------------
# Flow, expected water volume, and expected duration per tier are
# anchored to case-level FSRI live-fire suppression records in
# dataset/fireground_ai_reference_dataset.csv (record_type in
# live_fire_experiment/live_fire_usage, n=40 cases with a reported
# water-use figure). These are still a lookup table, not a regression:
# with ~40 case points spanning very different building types, fitting
# a continuous curve would create false precision the data can't
# support. Flow rate is set to the standard nozzle flow actually used
# in comparable real cases (crews do not scale flow rate with fire
# severity -- they mostly use one of two standard settings), and the
# tier boundary that DOES track severity in the source data is single
# vs. simultaneous multi-line attack (single-line median 140 gal;
# the 4 two-line commercial cases used 1,398-1,842 gal). Pressure is
# fixed at the standard attack-nozzle operating pressure reported
# across the source cases (50 psi) -- pressure does not vary with
# severity in the source data either; it varies with nozzle type
# (a penetrating nozzle for hard-to-reach fires runs 75-100 psi, but
# that is a nozzle-selection fact, not a demand-level scaling).
# This remains a prototype reference estimate, not a validated
# operational flow/pressure recommendation.
# --------------------------------------------------------

WATER_REFERENCE_TABLE = {
    "LOW": dict(
        nozzle_config="single 1 3/4 in. attack line",
        flow_gpm=150,
        water_gal_range=(70, 150),
        duration_s_range=(35, 80),
        basis="single-line residential room/contents fires, FSRI case data (n=26)",
    ),
    "MEDIUM": dict(
        nozzle_config="single 1 3/4 in. attack line, extended application",
        flow_gpm=160,
        water_gal_range=(150, 260),
        duration_s_range=(80, 220),
        basis="single-line residential fires, upper range / longer-duration cases, FSRI case data",
    ),
    "HIGH": dict(
        nozzle_config="two simultaneous attack lines (primary + backup)",
        flow_gpm=310,
        water_gal_range=(400, 1850),
        duration_s_range=None,
        basis="multi-line/large-structure fires, FSRI case data (n=4 commercial cases + extended-attack outliers)",
    ),
}


def truck_water_demand(score, compartment_temperature, max_gas, source_note):
    """
    ONE water-supply recommendation for the whole incident, for the pump
    operator at the truck -- not per firefighter. Sized to the worst
    reading found anywhere in the building right now (see main(), which
    picks the max score/temperature/gas across the whole roster before
    calling this).
    """

    demand_index = (
        compartment_temperature * 0.55
        + max_gas * 0.35
        + score * 8.0
    )
    demand_index = max(0.0, min(100.0, demand_index))

    if demand_index < 35:
        demand_level = "LOW"
    elif demand_index < 65:
        demand_level = "MEDIUM"
    else:
        demand_level = "HIGH"

    ref = WATER_REFERENCE_TABLE[demand_level]

    reference_flow_lpm = round(ref["flow_gpm"] * 3.785411784)
    reference_pressure_bar = round(50 * 0.0689475729, 2)
    water_gal_lo, water_gal_hi = ref["water_gal_range"]
    duration_range = ref["duration_s_range"]

    return {
        "level": demand_level,
        "demand_index": round(demand_index, 1),
        "based_on": source_note,
        "nozzle_config": ref["nozzle_config"],
        "estimated_flow_requirement_lpm": reference_flow_lpm,
        "nozzle_pressure_target_bar": reference_pressure_bar,
        "expected_water_volume_gal_range": [water_gal_lo, water_gal_hi],
        "expected_water_volume_l_range": [
            round(water_gal_lo * 3.785411784),
            round(water_gal_hi * 3.785411784),
        ],
        "expected_duration_seconds_range": (
            list(duration_range) if duration_range else None
        ),
        "reference_basis": ref["basis"],
        "mode": "PROTOTYPE ESTIMATE (FSRI case-data reference)",
        "prototype_note": (
            "Water flow, pressure, and volume/duration ranges are anchored to case-level "
            "FSRI live-fire suppression records (dataset/fireground_ai_reference_dataset.csv), "
            "not a physics-based or validated prediction. Sized to the worst reading found "
            "anywhere in the building, for the pump operator -- not a per-firefighter value. "
            "Not validated operational firefighting guidance."
        ),
    }


SENSOR_CHANNELS = [
    {"key":"heart_rate","name":"Heart Rate","unit":"bpm"},
    {"key":"acc_x","name":"Accelerometer X","unit":"g"},
    {"key":"acc_y","name":"Accelerometer Y","unit":"g"},
    {"key":"acc_z","name":"Accelerometer Z","unit":"g"},
    {"key":"gyro_x","name":"Gyroscope X","unit":"°/s"},
    {"key":"gyro_y","name":"Gyroscope Y","unit":"°/s"},
    {"key":"gyro_z","name":"Gyroscope Z","unit":"°/s"},
    {"key":"gas","name":"Gas","unit":"ppm"},
    {"key":"temperature","name":"Temperature","unit":"°C"},
]


def load_registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)["firefighters"]


def load_multi_stream():
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_id = {}
    for r in rows:
        by_id.setdefault(r["firefighter_id"], []).append(r)
    return by_id


def process_firefighter(ff, rows):
    total = len(rows) - WINDOW_SIZE + 1
    results = []

    for start in range(total):
        wr = rows[start:start + WINDOW_SIZE]
        window = [[float(r[k]) for k in FEATURES] for r in wr]
        pred = run_inference(window)

        a = ACTIVITY_LABELS[pred["activity"]["class_index"]]
        p = PHYSIOLOGICAL_LABELS[pred["physiological"]["class_index"]]
        e = ENVIRONMENT_LABELS[pred["environment"]["class_index"]]
        score = risk_score(a, e, p)
        st = status(score)

        local_summary = local_fireground_summary(wr, score)
        ev = event(a, e, p)

        zone_status = "HIGH" if st == "HIGH" else (
            "ELEVATED" if e == "risk" or st == "ELEVATED" else "NORMAL"
        )
        priority = "HIGH" if st == "HIGH" else ("MEDIUM" if st == "ELEVATED" else "LOW")
        wn = start + 1
        ts = wr[-1]["timestamp"]

        results.append({
            "window": wn,
            "timestamp": ts,
            "firefighter": {
                "node_id": ff["id"],
                "name": ff["name"],
                "badge_number": ff["badge_number"],
                "x": ff["position"]["x"],
                "y": ff["position"]["y"],
                "zone": ff["zone"],
            },
            "ai_prediction": {
                "activity": a,
                "activity_confidence": pred["activity"]["confidence"],
                "physiological_state": p,
                "physiological_confidence": pred["physiological"]["confidence"],
                "environment": e,
                "environment_confidence": pred["environment"]["confidence"],
            },
            "risk": {
                "risk_score": score,
                "local_status": st,
                "event_type": ev,
            },
            "fireground_summary": local_summary,
            "zone": {
                "zone_id": ff["zone"],
                "zone_status": zone_status,
            },
            "command_center": {"alert_priority": priority},
        })

        print(
            f"  [{ff['id']}] Window {wn:02d}/{total} | {ts} | "
            f"Activity={a} | Physiology={p} | Environment={e} | Score={score} | {st} | "
            f"LocalTemp={local_summary['compartment_temperature_estimate_c']}C | "
            f"Gas={local_summary['gas_risk']}"
        )

    stats = {k: sum(1 for r in results if r["risk"]["local_status"] == k.upper()) for k in ("normal", "elevated", "high")}
    overall = "HIGH" if stats["high"] else ("ELEVATED" if stats["elevated"] else "NORMAL")

    return {
        "id": ff["id"],
        "name": ff["name"],
        "badge_number": ff["badge_number"],
        "zone": ff["zone"],
        "position": ff["position"],
        "data_source": ff["data_source"],
        "sensor_samples": len(rows),
        "ai_windows_processed": len(results),
        "window_statistics": stats,
        "overall_status": overall,
        "latest": results[-1] if results else None,
        "results": results,
        "sensor_stream": sensor_payload(rows),
    }


def write_payload(firefighters_out, registry):
    statuses = [f["overall_status"] for f in firefighters_out.values()]
    overall_status = "HIGH" if "HIGH" in statuses else ("ELEVATED" if "ELEVATED" in statuses else "NORMAL")

    danger_alerts = [
        {
            "id": f["id"],
            "name": f["name"],
            "zone": f["zone"],
            "status": f["overall_status"],
            "event_type": (f["latest"]["risk"]["event_type"] if f["latest"] else None),
        }
        for f in firefighters_out.values()
        if f["overall_status"] in ("ELEVATED", "HIGH")
    ]

    # --------------------------------------------------------
    # ONE water-supply recommendation for the whole incident, sized to
    # the worst score/temperature/gas reading found anywhere in the
    # building right now -- not an average, and not per firefighter.
    # A real pump operator plans for the worst zone, not a blended one.
    # --------------------------------------------------------
    worst_score = 0
    worst_temp = 20.0
    worst_gas = 0.0
    worst_source = None
    for f in firefighters_out.values():
        latest = f["latest"]
        if not latest:
            continue
        fg = latest["fireground_summary"]
        if latest["risk"]["risk_score"] > worst_score:
            worst_score = latest["risk"]["risk_score"]
        if fg["compartment_temperature_estimate_c"] > worst_temp:
            worst_temp = fg["compartment_temperature_estimate_c"]
        if fg["max_gas_ppm"] > worst_gas:
            worst_gas = fg["max_gas_ppm"]
            worst_source = f["id"]

    truck_demand = truck_water_demand(
        worst_score,
        worst_temp,
        worst_gas,
        source_note=(
            f"Worst risk score, compartment temperature, and gas reading found across the "
            f"{len(firefighters_out)}-firefighter roster (highest single reading in each "
            f"category, not necessarily all from the same person; gas peak observed at "
            f"{worst_source or 'n/a'})."
        ),
    )

    return {
        "system": "Fireground AI",
        "inference_engine": "TensorFlow/Keras direct inference (multitask_fireground_cnn.keras)",
        "generated_at": datetime.now().isoformat(),
        "window_size": WINDOW_SIZE,
        "roster_size": len(registry),
        "overall_status": overall_status,
        "overall_fireground_temperature_c": worst_temp,
        "danger_alerts": danger_alerts,
        "truck_water_demand": truck_demand,
        "firefighters": firefighters_out,
        "prototype_note": (
            "No hardware sensors are connected yet. All firefighter sensor streams are "
            "simulated for demo/architecture purposes (see dataset/firefighter_registry.json "
            "and src/generate_multi_firefighter_stream.py). Water flow, pressure, and "
            "volume/duration ranges are anchored to case-level FSRI live-fire suppression "
            "records, not a physics-based or validated prediction for any specific reading. "
            "Nothing here is validated operational firefighting guidance."
        ),
        "sensor_channels": SENSOR_CHANNELS,
    }


def main():
    for path in (DATA_PATH, REGISTRY_PATH, KERAS_MODEL_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    registry = load_registry()
    by_id = load_multi_stream()

    print("================================================")
    print(" FIREGROUND AI - MULTI-FIREFIGHTER LIVE SYSTEM")
    print("================================================")
    print(f"Roster: {len(registry)} firefighter(s)")

    firefighters_out = {}
    for ff in registry:
        sid = ff["sensor_stream_id"]
        rows = by_id.get(sid)
        if not rows:
            raise FileNotFoundError(
                f"No sensor data for firefighter_id '{sid}' in {DATA_PATH}"
            )
        print(f"\n--- {ff['id']} ({ff['name']}) | zone {ff['zone']} ---")
        firefighters_out[ff["id"]] = process_firefighter(ff, rows)

        payload = write_payload(firefighters_out, registry)
        Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(OUTPUT_PATH).write_text(json.dumps(payload, indent=4), encoding="utf-8")

    final = write_payload(firefighters_out, registry)
    Path(OUTPUT_PATH).write_text(json.dumps(final, indent=4), encoding="utf-8")

    ok, msg = publish(final)
    print()
    print("================================================")
    if ok:
        print(f"WEB UPDATE: {msg}")
    else:
        print(f"WEB UPDATE: skipped ({msg})")
    print("OVERALL STATUS:", final["overall_status"])
    if final["danger_alerts"]:
        print("DANGER ALERTS:", ", ".join(f"{a['id']} ({a['status']})" for a in final["danger_alerts"]))
    td = final["truck_water_demand"]
    print(
        f"TRUCK WATER SUPPLY: {td['level']} | FLOW={td['estimated_flow_requirement_lpm']} L/min | "
        f"PRESSURE={td['nozzle_pressure_target_bar']} bar | building temp est.={final['overall_fireground_temperature_c']}C"
    )
    print("Result saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
