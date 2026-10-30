"""Central configuration for the solar flare prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

# SHARP-style magnetic field parameters. SWAN-SF contains these (and more); the real-data
# loader keeps whatever numeric columns it finds. This dict is used by the synthetic data
# generator and for human-readable descriptions in the dashboard.
PARAMS = {
    "USFLUX": "Total unsigned magnetic flux",
    "MEANGAM": "Mean inclination angle of field",
    "MEANGBT": "Mean gradient of total field",
    "MEANGBZ": "Mean gradient of vertical field",
    "MEANGBH": "Mean gradient of horizontal field",
    "MEANJZD": "Mean vertical current density",
    "TOTUSJZ": "Total unsigned vertical current",
    "MEANALP": "Mean twist parameter (alpha)",
    "MEANJZH": "Mean current helicity",
    "TOTUSJH": "Total unsigned current helicity",
    "ABSNJZH": "Absolute value of net current helicity",
    "SAVNCPP": "Sum of net current per polarity",
    "MEANPOT": "Mean photospheric excess magnetic energy density",
    "TOTPOT": "Total photospheric magnetic free energy density",
    "MEANSHR": "Mean shear angle",
    "SHRGT45": "Area fraction with shear angle > 45 deg",
    "R_VALUE": "Sum of flux near polarity inversion line",
    "AREA_ACR": "Area of strong-field pixels",
}

# Chronological split, following the SWAN-SF convention of 5 time-ordered partitions.
TRAIN_PARTITIONS = [1, 2, 3]
VAL_PARTITIONS = [4]
TEST_PARTITIONS = [5]

# Binary target: positive = M or X flare within the prediction window.
POSITIVE_CLASSES = ["M", "X"]

# Cleaned SWAN-SF (EskandariNasab et al. 2024, ApJS 275:6; github.com/samresume/Cleaned-SWANSF-Dataset).
# Attribute order of the 3-D pickles (samples, timesteps, attributes), copied from that repo's README.
CLEANED_ATTRS = [
    "R_VALUE", "TOTUSJH", "TOTBSQ", "TOTPOT", "TOTUSJZ", "ABSNJZH", "SAVNCPP", "USFLUX", "TOTFZ", "MEANPOT",
    "EPSX", "EPSY", "EPSZ", "MEANSHR", "SHRGT45", "MEANGAM", "MEANGBT", "MEANGBZ", "MEANGBH", "MEANJZH",
    "TOTFY", "MEANJZD", "MEANALP", "TOTFX",
]
# Files from the repo's  test/  folder: imputed (FPCKNN) + normalised (LSBZM) ONLY - no resampling, no synthetic
# TimeGAN samples. The  train/  folder files contain generated samples and must not be used for evaluation.
CLEANED_SAMPLED_MARKERS = ("RUS", "Tomek", "TimeGAN")
