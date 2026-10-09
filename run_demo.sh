#!/bin/bash
# Runs the full Fireground AI demo: multitask CNN inference -> risk engine ->
# water-demand prototype -> results JSON, then serves the web dashboard.
#
# Inference uses TensorFlow/Keras directly against
# models/multitask_fireground_cnn_v2.keras (see src/live_fireground_system.py).
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON="${FIREGROUND_PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python
fi

echo "=============================================="
echo "        FIREGROUND AI DEMONSTRATION"
echo "=============================================="
echo

"$PYTHON" src/live_fireground_system.py

echo
echo "=============================================="
echo "             DASHBOARD"
echo "=============================================="
echo
echo "Serving vercel_dashboard/ at http://localhost:8080/vercel_dashboard/"
echo "Press Ctrl+C to stop."
echo

"$PYTHON" -m http.server 8080
