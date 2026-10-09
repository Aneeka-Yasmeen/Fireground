# Fireground — Vercel Visual Dashboard

This folder is a static Vercel-ready dashboard built from the verified
Fireground AI project result and the 60-sample live sensor stream.

## Deploy with Vercel Drop

1. Open https://vercel.com/drop
2. Drag this folder (or a ZIP containing it) into the page.
3. Deploy.
4. Share the resulting `vercel.app` URL.

## Deploy from GitHub

Import the `Aneeka-Yasmeen/Fireground` repository in Vercel and set the Root Directory to this folder (`vercel_dashboard`) after it is added to the repository.

## Data model

The dashboard reads live results from `GET /api/latest`, which serves whatever was most recently written to Vercel Blob storage by `POST /api/update`. 

The inference pipeline (`src/live_fireground_system.py`, via `src/vercel_publisher.py`)
pushes a fresh result snapshot to that endpoint after each run — see the root `README.md`'s "Running the demo" section and `.fireground.env.example` for how to configure the pipeline to publish to your deployed instance.



