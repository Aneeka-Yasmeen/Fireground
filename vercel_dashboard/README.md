# Fireground AI — Vercel Visual Dashboard

This folder is a static Vercel-ready dashboard built from the verified
Fireground AI project result and the 60-sample live sensor stream.

## Deploy with Vercel Drop

1. Open https://vercel.com/drop
2. Drag this folder (or a ZIP containing it) into the page.
3. Deploy.
4. Share the resulting `vercel.app` URL.

## Deploy from GitHub

Import the `ARVIND2006-hub/Fireground_AI` repository in Vercel and set
the Root Directory to this folder (`vercel_dashboard`) after it is added
to the repository.

## Data model

The dashboard currently displays the verified result snapshot stored in:
`data/dashboard_data.json`.

It is intentionally a static snapshot. The VNNX inference continues to
run on the local/target environment; it is not executed inside Vercel.
For truly live online updates, the inference system would need to publish
fresh result data to a web-accessible API/database/object store.



