#!/usr/bin/env python3
"""Publish a Fireground result snapshot to the Vercel live-data API.

Configuration is read from environment variables or a local .fireground.env file:
    FIREGROUND_VERCEL_API_URL=https://your-project.vercel.app
    FIREGROUND_VERCEL_TOKEN=<same value as Vercel FIREGROUND_INGEST_TOKEN>

The local .fireground.env file must never be committed.
"""

import json
import os
from pathlib import Path
from urllib import request, error

PROJECT = Path(__file__).resolve().parent.parent
LOCAL_ENV = PROJECT / ".fireground.env"


def _load_local_env():
    if not LOCAL_ENV.exists():
        return
    try:
        for line in LOCAL_ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def publish(payload):
    _load_local_env()
    url = os.environ.get("FIREGROUND_VERCEL_API_URL", "").strip().rstrip("/")
    token = os.environ.get("FIREGROUND_VERCEL_TOKEN", "").strip()

    if not url or not token:
        return False, "Vercel publishing is not configured"

    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        url + "/api/update",
        data=body,
        headers={
            "Authorization": "Bearer " + token,
            "Content-Type": "application/json",
            "User-Agent": "Fireground-AI-VNNX-Publisher/1.0",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=12) as resp:
            response = json.loads(resp.read().decode("utf-8"))
            if response.get("ok"):
                return True, f"published {response.get('windows', len(payload.get('results', [])))} windows"
            return False, f"API rejected update: {response}"
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return False, f"HTTP {exc.code}: {detail[:300]}"
    except Exception as exc:
        return False, str(exc)
