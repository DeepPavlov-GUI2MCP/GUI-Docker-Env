# Eval rollout results

All evaluation runners write under `GUI-Docker-Env/results/results_*`.

## Layout

```
results/
  results_<experiment>_<timestamp>/
    run.zip              # single Hub artifact (flat tree inside, no nested images.zip)
  .scripts/
    zip_run.py           # before Hub upload
    unzip_run.py         # after Hub download
```

## Hub rate limits

Hugging Face throttles by **API request count**. Thousands of per-trajectory files cause
429s on download. **One `run.zip` per `results_*` folder** reduces each run to a single Hub file.

## Before uploading to Hugging Face

```bash
source .venv/bin/activate
cd GUI-Docker-Env

python results/.scripts/zip_run.py --all
# or one run:
python results/.scripts/zip_run.py results/results_holo_gpt54_writer_traces_20260607_031517
```

`zip_run.py` will:

1. Expand any existing `images.zip` into loose screenshot files (then delete `images.zip`)
2. Build one flat `run.zip` with screenshots, traj JSONL, XML, logs, etc.
3. Remove loose files so only `run.zip` remains (use `--keep-files` to retain them locally)

Target Hub repo: `tony-pitchblack/dart-gui.GUI-Docker-Env.results`

Upload (uploads only `run.zip` when present):

```bash
python scripts/upload_eval_rollouts.py results/results_<name>
```

Large runs:

```bash
export HF_XET_HIGH_PERFORMANCE=1
hf upload-large-folder tony-pitchblack/dart-gui.GUI-Docker-Env.results .tmp_hf_staging --repo-type dataset
```

## After downloading from Hugging Face

```bash
source .venv/bin/activate
cd GUI-Docker-Env

python results/.scripts/unzip_run.py --all
# or one run:
python results/.scripts/unzip_run.py results/results_<name>
```

Screenshots are restored as loose `.png`/`.jpg` files inside the run tree. There is no nested `images.zip`.

## Notes

- Do not upload `images.zip` to the Hub; use `run.zip` instead.
- `.gitattributes` marks loose image files as export-ignore for legacy loose uploads.
