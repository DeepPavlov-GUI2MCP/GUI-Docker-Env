# Eval rollout results

All evaluation runners write under `GUI-Docker-Env/results/results_*`.

## Layout

```
results/
  results_<experiment>_<timestamp>/
    pyautogui/screenshot/.../traj.jsonl
    coact/...
    preflight_augmented/.../tasks.jsonl
    images.zip
  .scripts/
    zip_images.py
    unzip_images.py
```

## Before uploading to Hugging Face

Zip screenshots so the Hub dataset stores one archive per run instead of thousands of PNG/JPG files:

```bash
source .venv/bin/activate
cd GUI-Docker-Env

python results/.scripts/zip_images.py --all
# or one run:
python results/.scripts/zip_images.py results/results_holo_gpt54_writer_traces_20260607_031517
```

This creates `images.zip` inside each run folder and removes the loose `.png`/`.jpg`/`.jpeg` files.

Target Hub repo: `tony-pitchblack/dart-gui.GUI-Docker-Env.results`

Upload with:

```bash
python scripts/upload_eval_rollouts.py \
  results/results_<name>
```

## After downloading from Hugging Face

Restore screenshots locally before inspecting traces or re-running augmentation:

```bash
source .venv/bin/activate
cd GUI-Docker-Env

python results/.scripts/unzip_images.py --all
# or one run:
python results/.scripts/unzip_images.py results/results_<name>
```

## Notes

- Run `zip_images.py` again before each Hub upload that includes screenshots.
- Trajectory JSON/JSONL, manifests, caches, and logs stay as normal files.
- `.gitattributes` in this folder marks loose image files as export-ignore for Hub packaging.
