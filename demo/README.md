# EgoFocus demo

This directory contains a dependency-free static showcase for the project, including the 12-second `assets/egofocus-demo.mp4` project reel displayed in the hero.

## Run locally

Open `index.html` in a browser, or serve the repository root with:

```bash
python3 -m http.server 8000
```

Then visit [http://localhost:8000/demo/](http://localhost:8000/demo/).

## Publish with GitHub Pages

The workflow at `.github/workflows/deploy-demo.yml` publishes this folder automatically. In the repository **Settings → Pages**, set **Source** to **GitHub Actions** once. After each push to `main` that changes `demo/`, the site deploys to the repository's GitHub Pages URL.

The demo deliberately uses an original stylized visual rather than redistributing EGTEA footage. Aggregate pilot metrics are real results stated in the project README; the scene-level playback is illustrative.
