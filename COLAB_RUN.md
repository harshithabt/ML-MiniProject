# Running the Earthquake Directionality Project in Google Colab

## Before you start

The project uses NGA-West2 acceleration records and flatfile metadata. These source files are not included in this repository and must be available in your own Google Drive.

## 1. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Set the project and data paths

Update these paths to match the folders in your Drive.

```python
from pathlib import Path

PROJECT = Path(
    "/content/drive/MyDrive/Earthquake_ML_MiniProject/"
    "earthquake-ground-motion-directionality"
)

AT2_DIR = Path(
    "/content/drive/MyDrive/Earthquake_ML_MiniProject/"
    "UnscaledAccRecord8878"
)

FLATFILE = Path(
    "/content/drive/MyDrive/Earthquake_ML_MiniProject/"
    "Updated_NGA_West2_flatfiles_part1/"
    "Updated_NGA_West2_Flatfile_RotD50_d005_public_version.xlsx"
)

OUT = PROJECT / "outputs"
```

Check that the paths are correct before proceeding.

```python
assert PROJECT.exists(), f"Project folder not found: {PROJECT}"
assert AT2_DIR.exists(), f"AT2 folder not found: {AT2_DIR}"
assert FLATFILE.exists(), f"Flatfile not found: {FLATFILE}"
```

## 3. Install dependencies

```python
%cd /content
!python -m pip install -q -r "{PROJECT}/requirements.txt"
```

## 4. Run the modular pipeline

```python
!python "{PROJECT}/run.py" \
    --at2 "{AT2_DIR}" \
    --flatfile "{FLATFILE}" \
    --out "{OUT}" \
    --max-km 100
```

The maximum station separation is set to 100 km for the supplementary experiment. The paper's stricter 5 km criterion must be evaluated separately.

## 5. Run the paper-informed experiment

Run this after the pipeline has generated the required pair CSV files.

```python
!python "{PROJECT}/paper_aligned_experiment.py" \
    --out "{OUT}"
```

The paper-informed script reports the available 5 km subset separately from the supplementary 100 km experiment. The current 5 km subset is too small for a reliable model comparison.

## 6. Inspect the generated outputs

```python
import json
import pandas as pd

print(json.loads((OUT / "summary.json").read_text()))

if (OUT / "station_pairs.csv").exists():
    display(pd.read_csv(OUT / "station_pairs.csv").head())

if (OUT / "paper_aligned_results.json").exists():
    print(json.loads((OUT / "paper_aligned_results.json").read_text()))
```

The exact files generated depend on which scripts have been run.

## Important notes

- The paths above are examples and must match your Drive layout.
- Do not upload raw PEER waveform records or private flatfiles to GitHub.
- The current dataset is a limited, non-random subset; results should be interpreted accordingly.
- The supplementary 100 km experiment is not an exact reproduction of the paper.
- Record the outputs from your actual run rather than inserting expected or illustrative metrics.
