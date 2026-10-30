"""Data loading for solar flare prediction.

Canonical table produced by every loader (one row per sample = one 12-hour observation window):

    sample_id | partition (1-5, chronological) | flare_class (N/B/C/M/X) | label (0/1) | <feature columns>

Three entry points:
    * load_swansf_raw(dir)  - the public SWAN-SF benchmark (Harvard Dataverse), raw per-sample CSVs
    * load_flat_csv(path)   - any already-flattened table with a label (and optionally partition) column
    * make_synthetic(...)   - SWAN-SF-like synthetic data so the pipeline and dashboard run end to end
                              before you have downloaded the real thing
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Columns in raw SWAN-SF files that are identifiers/time stamps or label-like and must never be features.
_NON_FEATURE_PATTERNS = [
    r"time", r"^id$", r"noaa", r"harp", r"label", r"^lat", r"^lon", r"^unnamed",
]


def _is_feature_col(name: str) -> bool:
    n = name.strip().lower()
    return not any(re.search(p, n) for p in _NON_FEATURE_PATTERNS)


# --------------------------------------------------------------------------------------
# MVTS -> tabular summary features
# --------------------------------------------------------------------------------------
def summarize_mvts(x: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Summarise multivariate time series windows into tabular features.

    x: array of shape (n_samples, n_timesteps, n_params). NaNs allowed.
    Returns one column per (parameter, statistic): mean, std, min, max, last, delta (last-first), slope.
    """
    n, t, p = x.shape
    tt = np.arange(t, dtype=float)
    tt_c = tt - tt.mean()
    with np.errstate(all="ignore"):
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
        mn = np.nanmin(x, axis=1)
        mx = np.nanmax(x, axis=1)
        # last / first non-NaN value per series
        def _edge(arr, reverse):
            out = np.full((arr.shape[0], arr.shape[2]), np.nan)
            idx = range(t - 1, -1, -1) if reverse else range(t)
            for k in idx:
                cur = arr[:, k, :]
                mask = np.isnan(out) & ~np.isnan(cur)
                out[mask] = cur[mask]
            return out
        last = _edge(x, True)
        first = _edge(x, False)
        # least-squares slope with NaN handling
        xm = np.where(np.isnan(x), 0.0, x - mean[:, None, :])
        w = (~np.isnan(x)).astype(float) * tt_c[None, :, None]
        denom = (w * tt_c[None, :, None]).sum(axis=1)
        slope = np.where(denom > 0, (xm * w).sum(axis=1) / denom, np.nan)
    stats = {"mean": mean, "std": std, "min": mn, "max": mx, "last": last,
             "delta": last - first, "slope": slope}
    cols, data = [], []
    for j, name in enumerate(names):
        for s, arr in stats.items():
            cols.append(f"{name}__{s}")
            data.append(arr[:, j])
    return pd.DataFrame(np.column_stack(data), columns=cols)


def _label_from_class(flare_class: pd.Series) -> pd.Series:
    return flare_class.isin(config.POSITIVE_CLASSES).astype(int)


# --------------------------------------------------------------------------------------
# Real data: SWAN-SF raw
# --------------------------------------------------------------------------------------
def _iter_sources(root: Path):
    """Yield (pseudo_path, open_fn) for every CSV under `root`, including CSVs INSIDE .zip archives,
    so the 6 GB download never has to be unzipped. open_fn() returns a fresh binary file object."""
    import zipfile
    for f in sorted(root.rglob("*.csv")):
        yield f, (lambda f=f: open(f, "rb"))
    for z in sorted(root.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(".csv") and not name.startswith("__MACOSX"):
                    # pseudo path keeps the zip name so 'partition3.zip' -> partition 3
                    yield Path(z.stem) / name, (lambda zf=zf, name=name: zf.open(name))


def load_swansf_raw(root: str | Path, max_files: int | None = None, neg_frac: float = 1.0,
                    seed: int = config.RANDOM_STATE, chunk: int = 2000) -> pd.DataFrame:
    """Load the raw SWAN-SF benchmark, STREAMING (one file at a time, low RAM) and optionally subsampling.

    Put the downloaded partitions in `root` either unzipped (partition1/...csv) or still zipped
    (partition1.zip ...). Partition number is read from any path component like 'partition3' / 'p3'.
    Flare class is read from the FILE NAME (first letter among X/M/C/B/N, e.g. 'M1.2@...'), falling back to a
    parent folder named FL/NF.

    neg_frac: keep ALL M/X windows but only this fraction of B/C/flare-quiet windows. Decided from the file
    name BEFORE reading the file, so skipped files cost almost nothing. max_files (debugging) keeps a random
    sample of that many files PER PARTITION. neg_frac=0.05 turns a 6 GB job into
    a few minutes. It changes class prevalence (precision/Brier are then relative to the subsampled data;
    TSS, recall and false-alarm rate are unaffected).

    IMPORTANT: inspect one file from your download first. If the naming differs, adjust
    `_parse_partition` / `_parse_class` below - they are the only two places that depend on it.
    """
    root = Path(root)
    rng = np.random.default_rng(seed)
    sep, names, t = None, None, None
    buf, meta, frames = [], [], []
    scanned = kept = 0

    def flush():
        if not buf:
            return
        feats = summarize_mvts(np.stack(buf), names).astype("float32")
        frames.append(pd.concat([pd.DataFrame(meta), feats], axis=1))
        buf.clear(); meta.clear()

    keep = None
    if max_files:  # debugging slice: random `max_files` files PER PARTITION (names alone, nothing is read yet)
        by_part: dict[int, list[str]] = {}
        for path, _ in _iter_sources(root):
            by_part.setdefault(_parse_partition(path), []).append(str(path))
        keep = set()
        for lst in by_part.values():
            keep.update(lst[i] for i in rng.permutation(len(lst))[:max_files])

    for path, open_fn in _iter_sources(root):
        scanned += 1
        if keep is not None and str(path) not in keep:
            continue
        cls = _parse_class(path)
        if neg_frac < 1.0 and cls not in config.POSITIVE_CLASSES and rng.random() > neg_frac:
            continue
        if sep is None:
            with open_fn() as fh:
                header = fh.readline().decode("utf-8", "ignore")
            sep = "\t" if header.count("\t") > header.count(",") else ","
        with open_fn() as fh:
            df = pd.read_csv(fh, sep=sep)
        cols = [c for c in df.columns if _is_feature_col(c)]
        if names is None:
            names = cols
            dropped = [c for c in df.columns if c not in cols]
            print(f"    using {len(names)} parameters: {names}\n    ignored columns: {dropped}", flush=True)
        a = df.reindex(columns=names).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if t is None:
            t = a.shape[0]
        fixed = np.full((t, len(names)), np.nan)
        k = min(t, a.shape[0])
        fixed[:k] = a[:k]
        buf.append(fixed)
        meta.append({"sample_id": path.stem, "partition": _parse_partition(path), "flare_class": cls})
        kept += 1
        if len(buf) >= chunk:
            flush()
        if scanned % 20000 == 0:
            print(f"    scanned {scanned} files, kept {kept}", flush=True)
    flush()
    if not frames:
        raise FileNotFoundError(f"No CSV files found under {root} (also looked inside .zip files). See README.")
    print(f"    scanned {scanned} files, kept {kept}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["label"] = _label_from_class(out["flare_class"])
    return _order_columns(out)


def _parse_partition(path: Path) -> int:
    for part in reversed(path.parts):
        m = re.search(r"(?:partition|part|p)[_\s-]?(\d)", part, flags=re.I)
        if m:
            return int(m.group(1))
    raise ValueError(f"Cannot infer partition from path {path}; edit _parse_partition().")


def _parse_class(path: Path) -> str:
    m = re.match(r"^([XMCBN])", path.stem, flags=re.I)
    if m:
        return m.group(1).upper()
    for part in path.parts:
        if part.upper() == "FL":
            return "C"
        if part.upper() == "NF":
            return "N"
    raise ValueError(f"Cannot infer flare class from {path}; edit _parse_class().")



# --------------------------------------------------------------------------------------
# Real data: Cleaned SWAN-SF (3-D pickles, already imputed + normalised)
# --------------------------------------------------------------------------------------
def _find_cleaned_files(root: Path) -> dict[int, dict[str, Path]]:
    """Locate PartitionN<stem>.pkl (X) and PartitionNLabels<stem>.pkl (y), ignoring the resampled train/ variants."""
    found: dict[int, dict[str, Path]] = {}
    skipped = []
    for f in sorted(Path(root).rglob("*.pkl")):
        m = re.match(r"Partition(\d)_?(Labels)?(.*)\.pkl$", f.name, flags=re.I)
        if not m:
            continue
        if any(mark.lower() in m.group(3).lower() for mark in config.CLEANED_SAMPLED_MARKERS):
            skipped.append(f.name)
            continue
        found.setdefault(int(m.group(1)), {})["y" if m.group(2) else "X"] = f
    if skipped:
        print(f"    ignoring {len(skipped)} resampled/TimeGAN training files (not valid for evaluation): {skipped[:2]}...")
    return found


def load_cleaned_swansf(root, neg_frac: float = 1.0, max_samples: int | None = None,
                        seed: int = config.RANDOM_STATE, chunk: int = 5000) -> pd.DataFrame:
    """Load the Cleaned SWAN-SF *test-folder* pickles, one partition at a time (low RAM).

    Keeps ALL positive (M/X) windows; keeps `neg_frac` of the negatives (and at most `max_samples` rows per
    partition if given). Original file order inside each partition is preserved (not temporal). (sample_id = p<part>_<index>).
    Each 12-h window (60 x 24 array) is summarised with summarize_mvts() into mean/std/min/max/last/delta/slope.
    Only binary labels are available, so flare_class is just 'M' (positive) or 'N' (negative).
    """
    import pickle
    root = Path(root)
    files = _find_cleaned_files(root)
    files = {p: d for p, d in files.items() if "X" in d and "y" in d}
    if not files:
        raise FileNotFoundError(
            f"No usable Partition<N>*.pkl + Partition<N>Labels*.pkl pairs under {root}. Expected the files from the "
            "repo's  test/  folder, e.g. Partition1LSBZM-NormFPCKNN-impute.pkl and "
            "Partition1LabelsLSBZM-NormFPCKNN-impute.pkl (the train/ files with RUS/Tomek/TimeGAN are skipped).")
    print(f"    partitions found: {sorted(files)}", flush=True)
    rng = np.random.default_rng(seed)
    names, frames = None, []
    for part in sorted(files):
        with open(files[part]["X"], "rb") as fh:
            X = np.asarray(pickle.load(fh))
        with open(files[part]["y"], "rb") as fh:
            y = np.asarray(pickle.load(fh)).ravel()
        if X.ndim != 3:
            raise ValueError(f"partition {part}: expected 3-D array (samples, timesteps, attributes), got {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"partition {part}: {len(X)} samples but {len(y)} labels")
        if not set(np.unique(y).tolist()) <= {0, 1}:
            raise ValueError(f"partition {part}: labels are not binary 0/1: {np.unique(y)[:10]}")
        y = y.astype(int)
        if names is None:
            if X.shape[2] == len(config.CLEANED_ATTRS):
                names = list(config.CLEANED_ATTRS)
            else:
                names = [f"attr{i:02d}" for i in range(X.shape[2])]
                print(f"    WARNING: {X.shape[2]} attributes, expected {len(config.CLEANED_ATTRS)}; using generic names")
            print(f"    {X.shape[1]} timesteps x {len(names)} attributes: {names}", flush=True)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        n_all, n_pos, nan_frac = len(y), len(pos), float(np.isnan(X).mean())
        if neg_frac < 1.0:
            neg = rng.choice(neg, int(round(len(neg) * neg_frac)), replace=False)
        if max_samples and len(pos) + len(neg) > max_samples:
            neg = rng.choice(neg, max(max_samples - len(pos), 0), replace=False)
        idx = np.sort(np.concatenate([pos, neg]))
        feats = pd.concat([summarize_mvts(X[idx[i:i + chunk]].astype(float), names)
                           for i in range(0, len(idx), chunk)], ignore_index=True).astype("float32")
        meta = pd.DataFrame({"sample_id": [f"p{part}_{i:06d}" for i in idx], "partition": part,
                             "flare_class": np.where(y[idx] == 1, "M", "N"), "label": y[idx]})
        frames.append(pd.concat([meta, feats], axis=1))
        print(f"    partition {part}: {n_all} windows, {n_pos} positive ({n_pos / n_all:.2%}), NaN {nan_frac:.2%} "
              f"-> kept {len(idx)}", flush=True)
        del X
    return _order_columns(pd.concat(frames, ignore_index=True))

# --------------------------------------------------------------------------------------
# Real data: any flat CSV
# --------------------------------------------------------------------------------------
def load_flat_csv(path: str | Path, label_col: str = "label", partition_col: str | None = "partition",
                  class_col: str | None = "flare_class") -> pd.DataFrame:
    """Load an already-tabular dataset. If there is no partition column, one is created by
    splitting rows into 5 equal chronological chunks (so keep the file in time order!)."""
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        df.insert(0, "sample_id", np.arange(len(df)))
    if partition_col is None or partition_col not in df.columns:
        df["partition"] = np.repeat(np.arange(1, 6), int(np.ceil(len(df) / 5)))[: len(df)]
    elif partition_col != "partition":
        df = df.rename(columns={partition_col: "partition"})
    if class_col and class_col in df.columns and class_col != "flare_class":
        df = df.rename(columns={class_col: "flare_class"})
    if "flare_class" not in df.columns:
        df["flare_class"] = np.where(df[label_col] == 1, "M", "N")
    if label_col != "label":
        df = df.rename(columns={label_col: "label"})
    return _order_columns(df)


# --------------------------------------------------------------------------------------
# Synthetic SWAN-SF-like data
# --------------------------------------------------------------------------------------
def make_synthetic(n_samples: int = 12000, n_steps: int = 12, seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Generate SWAN-SF-like data: chronologically ordered active-region windows, ~3-6% major flares,
    missing values, spikes, and mild drift across partitions (solar-cycle effect).

    NOT real solar data - for developing/testing the pipeline only. The dashboard flags this.
    """
    rng = np.random.default_rng(seed)
    names = list(config.PARAMS)
    p = len(names)

    partition = np.sort(rng.integers(1, 6, size=n_samples))
    # latent magnetic "complexity" per region, partly drifting with the solar cycle
    drift = (partition - 3) * 0.08
    z = rng.normal(0.0, 1.0, n_samples) + drift

    # flare class from latent complexity
    logit_major = -3.6 + 1.5 * z
    p_major = 1 / (1 + np.exp(-logit_major))
    major = rng.random(n_samples) < p_major
    p_c = 1 / (1 + np.exp(-(-0.4 + 1.0 * z)))
    is_c = (~major) & (rng.random(n_samples) < p_c * 0.6)
    is_b = (~major) & (~is_c) & (rng.random(n_samples) < 0.25)
    flare_class = np.full(n_samples, "N", dtype=object)
    flare_class[is_b] = "B"
    flare_class[is_c] = "C"
    flare_class[major] = np.where(rng.random(major.sum()) < 0.12, "X", "M")

    # loadings: how strongly each parameter tracks complexity (R_VALUE, TOTPOT, SHRGT45... strongest)
    strong = {"R_VALUE": 0.9, "TOTPOT": 0.8, "SHRGT45": 0.7, "USFLUX": 0.6, "TOTUSJH": 0.6,
              "TOTUSJZ": 0.55, "MEANPOT": 0.5, "MEANSHR": 0.45, "AREA_ACR": 0.5}
    base_scale = rng.uniform(0.5, 3.0, p)
    x = np.zeros((n_samples, n_steps, p))
    t = np.linspace(0, 1, n_steps)
    for j, name in enumerate(names):
        load = strong.get(name, 0.15)
        level = np.exp(0.35 * (load * z + rng.normal(0, 1 - 0.4 * load, n_samples))) * base_scale[j]
        trend = 1 + 0.10 * load * np.maximum(z, 0)[:, None] * (t[None, :] - 0.5)  # growing regions ramp up
        noise = 1 + rng.normal(0, 0.03, (n_samples, n_steps))
        x[:, :, j] = level[:, None] * trend * noise

    # data-quality problems for the preprocessing stage to handle
    miss = rng.random(x.shape) < 0.012
    x[miss] = np.nan
    bad_rows = rng.random(n_samples) < 0.01
    x[bad_rows, :, rng.integers(0, p)] = np.nan   # whole parameter missing for a few samples
    spikes = rng.random(x.shape) < 0.002
    x[spikes] *= rng.uniform(5, 15)

    feats = summarize_mvts(x, names)
    out = pd.concat([pd.DataFrame({
        "sample_id": [f"syn_{i:05d}" for i in range(n_samples)],
        "partition": partition,
        "flare_class": flare_class,
    }), feats], axis=1)
    out["label"] = _label_from_class(out["flare_class"])
    return _order_columns(out)


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    front = ["sample_id", "partition", "flare_class", "label"]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest].reset_index(drop=True)


def split_by_partition(df: pd.DataFrame):
    """Chronological train / validation / test split using the partition column."""
    tr = df[df["partition"].isin(config.TRAIN_PARTITIONS)].reset_index(drop=True)
    va = df[df["partition"].isin(config.VAL_PARTITIONS)].reset_index(drop=True)
    te = df[df["partition"].isin(config.TEST_PARTITIONS)].reset_index(drop=True)
    return tr, va, te
