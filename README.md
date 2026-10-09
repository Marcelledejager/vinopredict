# VinoPredict app

This folder is self-contained and can become the root of a Streamlit Cloud repository. `app.py` loads the model and evidence exported by the separate analysis notebook. It does not train a model or access the analysis folder.

## Add your exports

1. Complete the notebook run in the analysis project.
2. Copy all files from `outputs/app_artifacts` into this folder's `data` directory.
3. Copy the exported `requirements-app.txt` to this folder and rename it `requirements.txt`, replacing the starter file.
4. Use the Python minor version recorded in `data/manifest.json` for both local execution and Cloud deployment.

Required bundle contents:

```text
data/
  manifest.json
  model.joblib
  schema.json
  evaluation.json
  dataset_summary.json
  reference_profiles.csv
  feature_importance.csv
  model_comparison.csv
  calibration_curve.csv
  calibration_bins.csv
  roc_curve.csv
  precision_recall_curve.csv
  data_dictionary.csv
  example_inputs.csv
  requirements-app.txt
```

Copy the entire bundle after each notebook run. The app checks file hashes and bundle identifiers to avoid combining different runs. It also checks that model-library versions match the exported environment.

## Run locally

From this folder, after copying the exported requirements:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Substitute the notebook's Python minor version if it differs from 3.12. `run_app.bat` uses `.venv` when available, otherwise the Python on PATH.

## Streamlit Cloud

Put this folder's contents at the root of a GitHub repository, including the copied `data` bundle, `requirements.txt`, and `.streamlit/config.toml`. Choose `app.py` as the entrypoint and the same Python minor version as the notebook. The app uses only local repository artifacts. It needs no Kaggle login, API keys, secrets, or access to the analysis folder.

## App views

- **Forecast a wine:** numeric laboratory inputs, probability of Good, class at the fixed 0.50 threshold, SHAP factors, Good-wine reference quartiles, range flags and CSV download.
- **CSV forecasts:** accept up to 5,000 rows; normalize common header formats; require finite numeric measurements, non-negative concentrations, positive density, sensible pH/alcohol bounds and free SO₂ no greater than total SO₂; export predictions and range flags.
- **Model evidence:** held-out discrimination, reliability, class errors, profile-bootstrap intervals, common-fold benchmarks, forest parameters, prediction formula and importance.
- **Dataset:** source dimensions, quality counts, repeat handling, measurement dictionary and development references.

Reference ranges describe observed wines; they do not prescribe a change to production. SHAP factors explain the model's associations. A probability estimate is not a guarantee about an individual wine. Outside-range flags cover individual measurements and do not establish that every within-range combination is familiar.

The app was authored without launching or testing it, as requested. It first shows export setup instructions until you provide the notebook-generated bundle.
