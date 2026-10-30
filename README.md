# Solar Flare Prediction - Predictive Analytics Project

Predict whether a solar active region will produce an **M- or X-class flare within the next 24 hours**
from 12 hours of photospheric magnetic-field parameters. Binary, extremely imbalanced classification,
evaluated with the metrics the space-weather community uses (TSS, HSS).

Includes: data loading, EDA, preprocessing, feature engineering, feature selection comparison,
model + class-imbalance comparison, hyper-parameter tuning, probability calibration, threshold selection,
explainability (permutation importance + SHAP), auto-generated report, and a Streamlit dashboard
with a live prediction panel.

## Quick start

```bash
pip install -r requirements.txt

# 1. Run the whole pipeline (synthetic SWAN-SF-like data - works immediately, ~3-4 min)
python -m src.train --source synthetic

# 2. Open the dashboard
python -m streamlit run app/dashboard.py
```

Add `--quick` for a ~40 s smoke run. Run from the project root.

## Getting the real data - recommended route: Cleaned SWAN-SF (smaller, already imputed + normalised)

Source: EskandariNasab et al. 2024, *ApJS* 275:6 (DOI 10.3847/1538-4365/ad7c4a), repo `samresume/Cleaned-SWANSF-Dataset`
(MIT licence; the data itself is linked from `download.txt` in that repo). Please cite the paper if you use it.

**Which files to download:** the **`test/`** folder only - 5 data files + 5 label files:
`Partition{1..5}LSBZM-NormFPCKNN-impute.pkl` and `Partition{1..5}LabelsLSBZM-NormFPCKNN-impute.pkl`.
Do NOT use the `train/` folder (files with `RUS-Tomek-TimeGAN` in the name): they contain generated (TimeGAN) samples and
removed class-C windows, so they are not a faithful picture of real flare activity and must not be used for evaluation.
This project does its own imbalance handling on real windows, so it needs the unsampled files.

```bash
# put the 10 .pkl files in data/raw/cleaned_swansf/  (sub-folders are fine)
python -m src.check_data --path data/raw/cleaned_swansf        # 1. sanity checks - read the output
python -m src.train --source cleaned --max-per-partition 3000 --quick   # 2. small smoke run (minutes)
python -m src.train --source cleaned --neg-frac 0.25 --reload  # 3. real run (default neg-frac 0.25)
python -m src.train --source cleaned --neg-frac 1.0 --reload   # 4. optional: all windows (slow; best for precision/Brier)
streamlit run app/dashboard.py
```
Each partition is loaded one at a time (about 1 GB RAM per partition, float64 pickles), summarised into
mean/std/min/max/last/delta/slope per parameter (24 x 7 = 168 features), and cached in `data/processed/cleaned_features.csv`.
Only pickle files you trust should be opened: `pickle.load` can run code.

**Checklist after `check_data`:** shapes `(N, 60, 24)`; five partitions found; labels only 0/1; ~0 NaN; positive rate
roughly 1-5% per partition and not wildly different between partitions; no "suspiciously high AUC" warning
(real single parameters usually sit around 0.6-0.9).

Labels are binary here, so there is no per-flare-class breakdown (the dashboard shows M vs N only).

## Dataset Used in the Final Experiment

This project uses the **Cleaned SWAN-SF Dataset** (EskandariNasab et al., 2024), specifically its five test partitions.

The cleaned dataset provides preprocessed magnetic-field time-series data and corresponding flare labels. The final experiment uses these partitions to evaluate the model while avoiding the resampled training data.

**Dataset location:** Place the five data files and their corresponding label files in `data/raw/test/`.

The final experiment was conducted using:

```bash
python -m src.train --source cleaned --path data/raw/test --neg-frac 0.25 --reload
```

### Note on Raw SWAN-SF

The original raw SWAN-SF dataset is an alternative source dataset, but it was **not used for the final experiment or the reported performance metrics**. The results documented in this README correspond to the Cleaned SWAN-SF test partitions.

**Other options**
- Already have a table? `--source csv --path my.csv` (needs a `label` column, optionally `partition`, `flare_class`).
- A cleaned version of SWAN-SF (3D pickles) exists on GitHub (`samresume/Cleaned-SWANSF-Dataset`); flatten it with
  `summarize_mvts()` in `src/data_loading.py` and save a CSV for `--source csv`.

## Pipeline

| Stage | What happens | Where |
|---|---|---|
| Data + EDA | load, class balance per partition, missingness, duplicates | `data_loading.py`, `train.py` |
| Split | chronological partitions: train 1-3, validation 4, test 5 (no shuffling, no leakage) | `config.py` |
| Feature engineering | per-parameter mean, std, min, max, last, delta, slope | `summarize_mvts` |
| Preprocessing | inf→NaN, drop >40% missing, median impute, winsorise, signed-log skewed, standardise (train-fit only). Cleaned data is already imputed/normalised, so only the safety nets + standardise run (`--prep light`) | `preprocess.py` |
| Feature selection | compares none / correlation filter / mutual information / RF importance / RFE | `features.py` |
| Models | logistic regression (baseline), random forest, XGBoost × {none, class weights, undersampling} | `features.py` |
| Tuning | randomized search, leave-one-partition-out CV, ROC-AUC | `train.py` |
| Calibration + threshold | Platt scaling on validation; threshold maximises TSS | `model_io.py` |
| Evaluation | TSS, HSS, recall, precision, F1, ROC/PR-AUC, Brier skill, confusion matrix, 95% block-bootstrap CIs, error analysis by flare class | `metrics.py` |
| Explainability | permutation importance, SHAP | `train.py` |
| Report | figures + `reports/REPORT.md` | `report.py` |
| Dashboard | 7 pages incl. live what-if prediction and CSV batch scoring | `app/dashboard.py` |

Model, feature-selection method and threshold are chosen **only on the validation partition**. Test-partition scores are
computed for every candidate and for the final model, but nothing is chosen from them. Note that raw SWAN-SF also carries
flare-history columns (e.g. XR_MAX) that must be excluded from features to avoid label leakage; the cleaned data has none.

## Project layout

```
src/            config, data_loading, preprocess, features, metrics, model_io, train, report
app/dashboard.py
data/raw/       put SWAN-SF here        data/processed/dataset.csv   (written by train)
models/         final_model.joblib      reports/  results, tables, figures, REPORT.md
```

## Notes
### Dataset and Experimental Status

The project initially included synthetic/demo results, which should not be interpreted as scientific findings. The current experiment has been run separately using the cleaned SWAN-SF dataset (EskandariNasab et al., 2024).

The real-data experiment evaluates M/X-class flare prediction using 12-hour magnetic-field observation windows and a 24-hour prediction horizon.

### Real-Data Results

- Test ROC-AUC: approximately 0.98.
- Cross-partition ROC-AUC: approximately 0.94–0.98.
- Test TSS: approximately 0.817 at the validation-selected threshold.
- Best single feature TSS: 0.752.
- Top three features TSS: 0.815.

The results indicate strong ranking performance, with most predictive information concentrated in a small number of magnetic-field parameters.

After correcting for negative sampling, estimated precision is approximately 0.070 and HSS is approximately 0.110. The model should not be interpreted as an operational forecasting system or as producing reliably calibrated flare probabilities.

Leakage checks found no evidence of leakage in the tests performed. However, upstream dataset preprocessing and the lack of temporal metadata remain limitations.

### Scientific Context

Model performance should be compared with published SWAN-SF results using equivalent prediction horizons, flare definitions, data partitions, and evaluation metrics. Results from different experimental setups are not directly comparable.

### Future Work

- Compare performance with published SWAN-SF baselines.
- Explore LSTM and Transformer architectures on raw time-series sequences.
- Incorporate flare-history features.
- Investigate alternative prediction windows.
- Study limb-region effects.
- Compare class-imbalance strategies, including SMOTE and time-series GAN approaches.
## Model Performance

The final model was evaluated on Partition 5, using a threshold selected
on the validation partition (Partition 4).

| Metric | Result |
|---|---:|
| True Skill Statistic (TSS) | 0.8169 |
| Heidke Skill Score (HSS) | 0.3213 |
| Recall | 0.9909 |
| Precision (sampled test set) | 0.2327 |
| ROC-AUC | 0.9806 |
| PR-AUC (sampled test set) | 0.7366 |

### Prevalence Correction

The evaluation dataset was created by retaining all positive samples and
sampling 25% of negative samples. Therefore, precision and other
prevalence-sensitive metrics on the sampled test set do not directly
represent real-world performance.

After correcting for the original class prevalence:

- Precision: approximately 0.07
- HSS: approximately 0.11

These corrected values better reflect performance under the original
class distribution.
