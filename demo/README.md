# Localized Egocentric Vision — project page

This directory contains the static project page for **Localized
Egocentric Vision**, published to GitHub Pages. The hero demo is a real
GTEA cooking clip (`assets/real-cooking/`) with its original hand-mask
bounding boxes, localized crops, and action labels. No model predictions
are shown.

## Contents

- `index.html` — the project page (no build step, no dependencies).
- `assets/real-cooking/cooking-source-clip.mp4` — raw GTEA clip, no overlays.
- `assets/real-cooking/cooking-localized-stream.mp4` — full frame + localized crop + causal temporal strip.
- `assets/real-cooking/cooking-demo.json` — clip provenance, per-frame boxes, action intervals, annotation policy.
- `assets/real-cooking/ATTRIBUTION.md` — dataset attribution and citation.
- `.nojekyll` — disables Jekyll so files with underscores serve unchanged.

## Run locally

Open `index.html` in a browser, or serve the repository root:

```bash
python3 -m http.server 8000
```

then visit <http://localhost:8000/demo/>.

## Publish with GitHub Pages

The workflow at `.github/workflows/deploy-demo.yml` publishes this folder
automatically. Set the repository **Settings → Pages → Source** to
**GitHub Actions** once. After each push to `main` that changes `demo/`
(or the workflow), the site deploys to the repository Pages URL.

## Regenerating the demo assets

```bash
python3 scripts/make_real_cooking_demo_assets.py \
  --data data/raw/gtea \
  --out demo/assets/real-cooking
```

The script requires the GTEA raw video and hand-mask annotations. All
overlays use dataset annotations only; the interpolation policy and the
720x405 → 720x404 pixel adjustment are recorded in `cooking-demo.json`.
