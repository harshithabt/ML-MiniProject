# ML-Detecting and predicting earthquake ground motion directionality patterns

---

An exploratory, paper-informed machine-learning project that estimates differences in the orientation of maximum ground-motion response between recording stations from the same earthquake, using a limited subset of PEER NGA-West2 accelerograms and flatfile metadata.

> **Scientific status:** The pipeline runs, but the current dataset is too small for a reliable evaluation under the paper's strict 5 km station-pairing criterion. This is an exploratory implementation, **not a verified exact reproduction** of the paper. The published paper metrics shown in `outputs/paper_aligned_results.json` are reference values only and are never presented as our own results.

---

## What this project does

1. Reads horizontal NGA-West2 `.AT2` acceleration files.
2. Pairs near-orthogonal components and estimates a response-based orientation using a 5%-damped SDOF response at a configurable 1-second period.
3. Matches usable records to NGA-West2 flatfile metadata.
4. Builds same-earthquake station pairs and calculates axial orientation differences in the range 0°–90°.
5. Compares a distance-only linear benchmark, Ridge regression and an MLP, with both event-held-out exploratory evaluation and a paper-informed supplementary experiment.
6. Exports machine-readable CSV/JSON outputs and a plot.

---

## Repository structure

```text
ML-MiniProject/
├── .gitignore
├── README.md
├── requirements.txt
├── COLAB_RUN.md
├── directionality_analysis.py
├── paper_aligned_experiment.py
│
├── notebooks/
│   ├── 01_earthquake_directionality_foundations.ipynb
│   ├── 02_rotation_mathematics_validation.ipynb
│   └── 03_nga_west2_dataset_inspection.ipynb
|
├── run_googlecolab.ipynb
│
├── src/
│   ├── __init__.py
│   ├── records.py
│   ├── waveforms.py
│   ├── metadata.py
│   ├── pairs.py
│   ├── models.py
│   └── pipeline.py
│
└── outputs/
    ├── experiment_comparison.csv
    ├── model_comparison_mae.png
    ├── orientations.csv
    ├── outputs_final.txt
    ├── paper_aligned_results.json
    ├── real_acceleration_traces.png
    ├── records_with_metadata.csv
    ├── rotational_orientation_distribution.png
    ├── rotational_orientation_results.csv
    ├── same_event_pairs.csv
    ├── same_event_pairs_within_5km.csv
    └── summary.json
```

The PEER waveform files and flatfiles are **not included**. They are large and subject to their own access/redistribution terms. Keep them locally or in your private Drive; do not commit raw data to GitHub.

---

## Dataset used in the current run

- Source: a limited, non-random subset downloaded from PEER NGA-West2.
- 100 unique RSNs were present in the original downloaded folder; 93 records were usable and matched to metadata in the recorded pipeline run.
- 269 same-event station pairs were available under the supplementary 100 km radius across 9 events.
- Only 8 pairs from 5 events met the strict 5 km criterion in the paper-informed run.
- The current subset is constrained by PEER access/download limits and is not representative of the full NGA-West2 database.

These counts refer to the specific run recorded in this repository. If you rerun the pipeline with different records or flatfiles, regenerate the outputs and update the report.

## Quick start — Google Colab

1. Upload this repository ZIP to Google Drive and extract it to:
   `MyDrive/Earthquake_ML_MiniProject/earthquake-ground-motion-directionality`
2. Keep your PEER AT2 folder and NGA-West2 flatfile in your private Drive. The paths below are examples; adjust them to your own Drive layout.
3. Open `run_in_colab.ipynb` and run its cells. The notebook installs dependencies, runs the pipeline, and writes results to your Drive.
4. To run the paper-informed experiment after the pipeline has generated both pair CSVs, execute:

```bash
python paper_aligned_experiment.py --out "/content/drive/MyDrive/Earthquake_ML_MiniProject/outputs_final"
```

For local Python, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python run.py --at2 "/path/to/UnscaledAccRecord8878" \
  --flatfile "/path/to/Updated_NGA_West2_Flatfile_RotD50_d005_public_version.xlsx" \
  --out outputs
```

Check `python run.py --help` for the available arguments. The notebook currently calls `src/pipeline.py`; `run.py` is an alternate modular entry point. Keep outputs from each run together and do not mix files from different dataset runs.

---

## Results at a glance

### Strict paper-informed subset

The paper uses same-event station pairs within 5 km and records with magnitude at least 5. The current subset yielded only **8 pairs across 5 events**. An 80/10/10 split would leave too few validation/test examples, so no reliable model comparison is reported for this primary subset.

### Supplementary 100 km experiment — not paper-equivalent

A broader 100 km subset was used only to exercise the modelling workflow. It yielded 269 pairs across 9 events. With a random example-level 80/10/10 split, the recorded test metrics were:

| Model | MAE (degrees) | R² |
|---|---:|---:|
| Distance-only linear regression | 22.70 | -0.073 |
| Ridge regression (α = 1.5) | 22.91 | -0.118 |
| Validation-selected MLP | 23.15 | -0.138 |

Negative R² means these models did not outperform predicting the test-set mean under this split. The MLP was selected from a validation grid; its test score is not evidence of good generalization. Because the split is pair-level, pairs sharing events/stations may appear across splits; do not interpret these values as event-generalization results.

### Separate event-held-out exploratory evaluation

The modular pipeline also recorded an event-held-out split on the 269-pair 100 km set (53 training pairs and 216 test pairs across 6 train events and 3 test events):

| Model | MAE (degrees) | RMSE (degrees) | R² |
|---|---:|---:|---:|
| Constant-median baseline | 23.17 | 27.49 | -0.137 |
| Ridge regression (α = 10) | 22.86 | 27.43 | -0.132 |
| MLP (32, 16) | 27.04 | 33.16 | -0.654 |

The test set is dominated by a small number of events and is imbalanced by event contribution, so these are unstable exploratory results, not a robust performance claim. See `reports/FINAL_REPORT.md` and the JSON files in `outputs/`.

## Paper-reported reference values

The supplied paper reports the following reference results (as transcribed for comparison): distance benchmark R² = 0.025, MAE = 20.2°; Ridge R² = 0.245, MAE = 16.9°; neural network R² = 0.354, MAE = 13.2°. These values are **from the paper, not from our run**. The paper also reports feature selection to 21 predictors; the current feature set has not been verified to match those 21 predictors.

---

## Methodological caveats

- The 5 km primary subset is too small to estimate predictive performance reliably.
- The waveform sample is limited and non-random due to download/access constraints.
- The supplementary 100 km analysis differs from the paper's 5 km pairing rule.
- The current predictors are not confirmed to be the paper's final 21 selected features.
- Confirm that the AT2 filename azimuth suffix matches the intended component azimuth convention; inspect records against flatfile metadata.
- The 1-second response period and orientation target implementation are code choices that still require a line-by-line check against the paper's precise definition.
- Pair observations are dependent because stations and events can recur; event-held-out evaluation is more conservative but remains unstable with only nine events.
- No claim is made that the model is suitable for real-time prediction, earthquake early warning, or engineering decisions.

---

## Reproducibility and integrity

- Keep the supplied research paper in your private project records and cite it in your submission according to your course's required style.
- Do not commit PEER raw waveforms, flatfiles, credentials, or private Drive paths.
- The `outputs/` folder contains aggregate recorded results and plots only; it does not contain the original raw dataset or full pair-level records.
- Rerun from raw inputs to recreate full CSV outputs. Save a separate copy of each run's `summary.json` so results are traceable to the data used.

---

## Reference

*Detecting and Predicting Earthquake Ground Motion Directionality Patterns Using Machine Learning Tools.* Course-provided research paper (PDF). Use the author, venue and year details from the first page of the supplied PDF when preparing the formal bibliography; those fields have intentionally not been guessed here.

---
