
# =============================================================================
# sorted_spikes_tools.py
# Utilities for loading pre-sorted spike times and plotting/analysis.
#
# Expects a parquet table with columns:
#   spikeTime (seconds), trial, cueTime (seconds, -1 if no cue), rewardTime (s),
#   situation (int 1..5), recording_num (int), cell_num (int)
#
# Main features:
# - Auto/silent loading of sorted_spikes.parquet (or a zip containing it)
# - Raster + PSTH aligned to 'cue' or 'reward' for chosen recording/cell
# - Plot all trials together, or each situation separately
# - Count spikes in a post-event window and plot histograms per situation
# - Stats:
#     (1) Within a situation: compare two windows (paired t-test or rank-sum)
#     (2) Between situations: compare groups (ANOVA or Kruskal-Wallis)
#
# Notes:
# - Time units for plotting windows are in milliseconds in the public API.
# - Binning for PSTH uses seconds internally; we expose ms in function args.
# - Figures are returned to allow further customization in notebooks.
# =============================================================================

from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from typing import Sequence, Literal, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

__all__ = [
    "SORTED_SPIKES_ZIP_URL",
    "SORTED_SPIKES_ZIP_NAME",
    "SORTED_SPIKES_PARQUET_NAME",
    "load_sorted_spikes",
    "plot_raster_psth_all",
    "plot_raster_psth_by_situation",
    "count_spikes_and_hist_by_situation",
    "stats_within_situation",
    "stats_between_situations",
]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Optionally, set this to a URL where the zip lives.
# If provided and the parquet isn't found locally, we'll download/extract it.
SORTED_SPIKES_ZIP_URL: str | None = None  # e.g., "https://.../SortedSpikes.zip"
SORTED_SPIKES_ZIP_NAME = "SortedSpikes.zip"  # expected filename if present
SORTED_SPIKES_PARQUET_NAME = "sorted_spikes.parquet"

# If your course repo already provides a get_data_dir() (e.g., in n101dd_functions.py),
# we'll try to import and use it. Otherwise we fall back to /content/data.
def _detect_data_dir() -> Path:
    try:
        from n101dd_functions import get_data_dir  # type: ignore
        return Path(get_data_dir())
    except Exception:
        # Default used by some workshop utilities
        default = Path("/content/data")
        default.mkdir(parents=True, exist_ok=True)
        return default

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def _download_zip(url, zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    # Use curl if available (fewer issues than wget on some images)
    subprocess.check_call(["bash", "-lc", f'curl -L "{url}" -o "{zip_path}"'])
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

def _ensure_parquet(data_dir: Path) -> Path:
    """
    Ensure sorted_spikes.parquet exists in data_dir, extracting/downloading if needed.
    Returns the Path to the parquet file.
    """
    parquet_path = data_dir / SORTED_SPIKES_PARQUET_NAME
    if parquet_path.exists():
        return parquet_path

    # Look for a zip beside it
    zip_path = data_dir / SORTED_SPIKES_ZIP_NAME
    if not zip_path.exists() and SORTED_SPIKES_ZIP_URL:
        try:
            _download_zip(SORTED_SPIKES_ZIP_URL, zip_path)
        except Exception:
            pass  # silent as requested

    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Extract only the parquet if present; otherwise extract all
                names = zf.namelist()
                if SORTED_SPIKES_PARQUET_NAME in names:
                    zf.extract(SORTED_SPIKES_PARQUET_NAME, path=str(data_dir))
                else:
                    zf.extractall(path=str(data_dir))
        except Exception:
            pass  # silent

    if parquet_path.exists():
        return parquet_path

    # Fallback: sometimes the zip contains nested folders:
    for root, _, files in os.walk(data_dir):
        if SORTED_SPIKES_PARQUET_NAME in files:
            return Path(root) / SORTED_SPIKES_PARQUET_NAME

    raise FileNotFoundError(
        f"Could not find {SORTED_SPIKES_PARQUET_NAME} in {data_dir}. "
        f"If it is provided in a zip, place '{SORTED_SPIKES_ZIP_NAME}' in {data_dir}, "
        f"or set SORTED_SPIKES_ZIP_URL to a downloadable zip."
    )

# --- CONFIG: set these for your workshop ---
WORKSHOP_NAME = "data101-workshop"
DATA_FOLDER_NAME = "data"               # where data will live locally
GDRIVE_FOLDER_ID = ""  # Option A (leave "" to skip)
ZIP_HTTP_URL = "https://github.com/mhburrell/Neuro101DD_Tutorial/releases/download/Week2/Data.zip"  # e.g., a GitHub Release URL to a .zip file (Option B), or leave "" to skip
ZIP_FILENAME = "Data.zip"            # only used if ZIP_HTTP_URL is set
GCS_HTTP_PREFIX = ""  # e.g., "https://storage.googleapis.com/your-bucket/workshop" (Option C), or leave "" to skip

import os, sys, shutil, subprocess, pathlib, zipfile

def get_data_dir():
    """
    Returns a pathlib.Path to the local data directory.
    Strategy:
      1) If ./data already exists, use it.
      2) Else try copying from student's Google Drive at MyDrive/{WORKSHOP_NAME}/data.
      3) Else try Google Drive shared folder (gdown) if GDRIVE_FOLDER_ID is set.
      4) Else try ZIP_HTTP_URL (e.g., GitHub Release).
      5) Else, if GCS_HTTP_PREFIX is set, just return an HTTP prefix for on-demand reads.
    """
    data_dir = pathlib.Path("/content") / DATA_FOLDER_NAME

    # 1) already there?
    if data_dir.exists() and any(data_dir.iterdir()):
        print(f"Using existing data at {data_dir}")
        return data_dir

    # 4) try a ZIP URL (e.g., GitHub Release)
    if ZIP_HTTP_URL:
        print("Downloading data zip...")
        zip_path = pathlib.Path("/content") / ZIP_FILENAME
        _download_zip(ZIP_HTTP_URL, str(zip_path), str(data_dir))
        if data_dir.exists() and any(data_dir.iterdir()):
            print(f"Data ready at {data_dir}")
            return data_dir


    raise RuntimeError(
        "Could not prepare data. Set one of: GDRIVE_FOLDER_ID, ZIP_HTTP_URL, or GCS_HTTP_PREFIX, "
        "or place data in /content/data before calling get_data_dir()."
    )

def load_sorted_spikes(silent: bool = True) -> pd.DataFrame:
    """
    Load the sorted_spikes.parquet table, extracting/downloading as needed.

    Returns
    -------
    pd.DataFrame with at least the columns described at the top of this file.
    """
    data_dir = get_data_dir()
    base = _detect_data_dir()
    parquet_path = _ensure_parquet(base)
    # Use pyarrow if available; fall back to pandas default
    try:
        df = pd.read_parquet(parquet_path, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(parquet_path)
    if not silent:
        print(f"Loaded {len(df):,} spikes from: {parquet_path}")
    return df

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _subset_df(df: pd.DataFrame, recording_num: int, cell_num: int) -> pd.DataFrame:
    sub = df[(df["recording_num"] == recording_num) & (df["cell_num"] == cell_num)].copy()
    if sub.empty:
        raise ValueError(f"No rows for recording_num={recording_num}, cell_num={cell_num}.")
    return sub

def _choose_event_col(align_to: Literal["cue", "reward"]) -> str:
    return "cueTime" if align_to == "cue" else "rewardTime"

def _ms_to_s(x_ms):
    return np.asarray(x_ms, dtype=float) / 1000.0

def _window_mask(xs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (xs >= lo) & (xs < hi)

@dataclass
class AlignedSpikes:
    rel_time_s: np.ndarray      # spike times relative to chosen event (seconds)
    trials: np.ndarray          # matching trial numbers
    situations: np.ndarray      # matching situation codes
    unique_trials: np.ndarray   # unique trial IDs in this subset

def _align_spikes(df: pd.DataFrame, align_to: Literal["cue", "reward"]) -> AlignedSpikes:
    """
    Compute relative spike times to event per trial.
    """
    event_col = _choose_event_col(align_to)
    # Drop trials where event is missing (-1)
    valid = df[df[event_col] >= 0].copy()
    if valid.empty:
        raise ValueError(f"No valid {align_to} events (all {event_col} < 0).")
    rel = valid["spikeTime"].to_numpy(float) - valid[event_col].to_numpy(float)
    trials = valid["trial"].to_numpy(int)
    situations = valid["situation"].to_numpy(int)
    uniq_trials = np.unique(trials)
    return AlignedSpikes(rel_time_s=rel, trials=trials, situations=situations, unique_trials=uniq_trials)

# -----------------------------------------------------------------------------
# Raster + PSTH (all trials together)
# -----------------------------------------------------------------------------

def plot_raster_psth_all(
    recording_num: int,
    cell_num: int,
    align_to: Literal["cue", "reward"] = "cue",
    t_window_ms: Tuple[float, float] = (-500.0, 1500.0),
    bin_size_ms: float = 20.0,
    smooth_sigma_ms: float | None = None,
    max_trials: int | None = None,
    figsize=(10, 6),
):
    """
    Plot a raster (trials x time) and a pooled PSTH for one cell.

    Returns
    -------
    fig, (ax_raster, ax_psth)
    """
    df = load_sorted_spikes(silent=True)
    sub = _subset_df(df, recording_num, cell_num)
    A = _align_spikes(sub, align_to)

    # Trial order and optional truncation
    trial_order = np.sort(A.unique_trials)
    if max_trials is not None:
        trial_order = trial_order[:max_trials]

    # Filter to the selected window for plotting
    lo_s, hi_s = _ms_to_s(np.array(t_window_ms))
    in_win = _window_mask(A.rel_time_s, lo_s, hi_s)
    rel_win = A.rel_time_s[in_win]
    trials_win = A.trials[in_win]

    # Prepare eventplot input as list per trial row
    trial_to_idx = {t: i for i, t in enumerate(trial_order)}
    rows = [[] for _ in range(len(trial_order))]
    for t, rt in zip(trials_win, rel_win):
        if t in trial_to_idx:
            rows[trial_to_idx[t]].append(rt)
    rows = [np.asarray(r, dtype=float) if len(r) else np.asarray([], dtype=float) for r in rows]

    # PSTH
    bin_s = _ms_to_s(bin_size_ms)
    edges = np.arange(lo_s, hi_s + bin_s, bin_s)
    mids = 0.5 * (edges[:-1] + edges[1:])

    # Count spikes per bin across all selected trials, then divide by (#trials * bin_width) to get Hz
    counts, _ = np.histogram(rel_win, bins=edges)
    n_trials = len(trial_order) if len(trial_order) > 0 else 1
    rates = counts / (n_trials * bin_s)

    # Optional Gaussian smoothing
    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        sigma_s = _ms_to_s(smooth_sigma_ms)
        half = int(np.ceil(3 * sigma_s / bin_s))
        if half > 0 and rates.size >= 2 * half + 1:
            xk = np.arange(-half, half + 1) * bin_s
            kernel = np.exp(-0.5 * (xk / sigma_s) ** 2)
            kernel /= np.sum(kernel)
            rates = np.convolve(rates, kernel, mode="same")

    # Plot
    fig, (ax_ras, ax_psth) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_ras.eventplot(rows, lineoffsets=np.arange(len(rows)), linelengths=0.8, colors="k", linewidths=0.6)
    ax_ras.set_ylabel("Trial")
    ax_ras.set_title(f"Recording {recording_num} • Cell {cell_num} • Align: {align_to}")
    ax_ras.set_xlim(lo_s, hi_s)

    ax_psth.bar(mids, rates, width=bin_s, align="center", edgecolor="none")
    ax_psth.set_xlabel("Time from event (s)")
    ax_psth.set_ylabel("Rate (Hz)")
    ax_psth.set_xlim(lo_s, hi_s)
    fig.tight_layout()
    return fig, (ax_ras, ax_psth)

    # -----------------------------------------------------------------------------
    # Raster + PSTH (split by situation)
    # -----------------------------------------------------------------------------

def plot_raster_psth_by_situation(
    recording_num: int,
    cell_num: int,
    align_to: Literal["cue", "reward"] = "cue",
    t_window_ms: Tuple[float, float] = (-500.0, 1500.0),
    bin_size_ms: float = 20.0,
    smooth_sigma_ms: float | None = None,
    situations: Sequence[int] | None = None,
    max_trials_per_sit: int | None = None,
    figsize=(12, 10),
):
    """
    Create a grid of rasters and PSTHs, one row per situation.
    All PSTHs share the same y-axis scale.

    Returns
    -------
    fig, axes_dict where axes_dict[sit] = (ax_raster, ax_psth)
    """
    df = load_sorted_spikes(silent=True)
    sub = _subset_df(df, recording_num, cell_num)
    A = _align_spikes(sub, align_to)

    lo_s, hi_s = _ms_to_s(np.array(t_window_ms))
    bin_s = _ms_to_s(bin_size_ms)

    if situations is None:
        situations = np.unique(A.situations)

    # Precompute PSTHs + raster rows for each situation so we can find a global y-scale
    sit_data = []
    edges = np.arange(lo_s, hi_s + bin_s, bin_s)
    mids = 0.5 * (edges[:-1] + edges[1:])

    for sit in situations:
        mask_sit = (A.situations == sit)
        rel_sit = A.rel_time_s[mask_sit]
        trial_sit = A.trials[mask_sit]

        # Window for display
        in_win = _window_mask(rel_sit, lo_s, hi_s)
        rel_win = rel_sit[in_win]
        trials_win = trial_sit[in_win]

        # Trial order (optionally truncated)
        uniq_trials = np.unique(trials_win)
        if max_trials_per_sit is not None:
            uniq_trials = uniq_trials[:max_trials_per_sit]

        # Raster rows
        trial_to_idx = {t: i for i, t in enumerate(uniq_trials)}
        rows = [[] for _ in range(len(uniq_trials))]
        for t, rt in zip(trials_win, rel_win):
            if t in trial_to_idx:
                rows[trial_to_idx[t]].append(rt)
        rows = [np.asarray(r, dtype=float) if len(r) else np.asarray([], dtype=float) for r in rows]

        # PSTH (per-situation rate)
        counts, _ = np.histogram(rel_win, bins=edges)
        n_trials = max(1, len(uniq_trials))
        rates = counts / (n_trials * bin_s)

        # Optional smoothing
        if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
            sigma_s = _ms_to_s(smooth_sigma_ms)
            half = int(np.ceil(3 * sigma_s / bin_s))
            if half > 0 and rates.size >= 2 * half + 1:
                xk = np.arange(-half, half + 1) * bin_s
                kernel = np.exp(-0.5 * (xk / sigma_s) ** 2)
                kernel /= np.sum(kernel)
                rates = np.convolve(rates, kernel, mode="same")

        sit_data.append({"sit": int(sit), "rows": rows, "rates": rates, "uniq_trials": uniq_trials})

    # determine global y-scale for PSTHs
    max_rate = 0.0
    for d in sit_data:
        if d["rates"].size:
            max_rate = max(max_rate, float(np.nanmax(d["rates"])))
    # provide a small default upper bound if all zero
    if max_rate <= 0:
        max_rate = 1.0
    ymax = max_rate * 1.05

    # Setup figure
    n_sit = len(situations)
    fig, axs = plt.subplots(n_sit, 2, figsize=figsize, sharex=False, gridspec_kw={"height_ratios": [2]*n_sit})
    if n_sit == 1:
        axs = np.array([axs])
    axes_map: Dict[int, Any] = {}

    for row, d in enumerate(sit_data):
        sit = d["sit"]
        rows = d["rows"]
        rates = d["rates"]

        ax_ras, ax_psth = axs[row, 0], axs[row, 1]
        ax_ras.eventplot(rows, lineoffsets=np.arange(len(rows)), linelengths=0.8, colors="k", linewidths=0.6)
        ax_ras.set_xlim(lo_s, hi_s)
        ax_ras.set_ylabel("Trial")
        ax_ras.set_title(f"Situation {int(sit)} – Raster")

        ax_psth.bar(mids, rates, width=bin_s, align="center", edgecolor="none")
        ax_psth.set_xlim(lo_s, hi_s)
        ax_psth.set_ylim(0, ymax)
        ax_psth.set_xlabel("Time from event (s)")
        ax_psth.set_ylabel("Rate (Hz)")
        ax_psth.set_title(f"Situation {int(sit)} – PSTH")

        axes_map[int(sit)] = (ax_ras, ax_psth)

    fig.suptitle(f"Recording {recording_num} • Cell {cell_num} • Align: {align_to}", y=0.995)
    fig.tight_layout()
    
    rel_sit = A.rel_time_s[mask_sit]
    trial_sit = A.trials[mask_sit]

    # Window for display
    in_win = _window_mask(rel_sit, lo_s, hi_s)
    rel_win = rel_sit[in_win]
    trials_win = trial_sit[in_win]

    # Trial order (optionally truncated)
    uniq_trials = np.unique(trials_win)
    if max_trials_per_sit is not None:
        uniq_trials = uniq_trials[:max_trials_per_sit]

    # Raster rows
    trial_to_idx = {t: i for i, t in enumerate(uniq_trials)}
    rows = [[] for _ in range(len(uniq_trials))]
    for t, rt in zip(trials_win, rel_win):
        if t in trial_to_idx:
            rows[trial_to_idx[t]].append(rt)
    rows = [np.asarray(r, dtype=float) if len(r) else np.asarray([], dtype=float) for r in rows]

    # PSTH (per-situation rate)
    edges = np.arange(lo_s, hi_s + bin_s, bin_s)
    mids = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(rel_win, bins=edges)
    n_trials = max(1, len(uniq_trials))
    rates = counts / (n_trials * bin_s)

    # Optional smoothing
    if smooth_sigma_ms is not None and smooth_sigma_ms > 0:
        sigma_s = _ms_to_s(smooth_sigma_ms)
        half = int(np.ceil(3 * sigma_s / bin_s))
        if half > 0 and rates.size >= 2 * half + 1:
            xk = np.arange(-half, half + 1) * bin_s
            kernel = np.exp(-0.5 * (xk / sigma_s) ** 2)
            kernel /= np.sum(kernel)
            rates = np.convolve(rates, kernel, mode="same")

    ax_ras, ax_psth = axs[row, 0], axs[row, 1]
    ax_ras.eventplot(rows, lineoffsets=np.arange(len(rows)), linelengths=0.8, colors="k", linewidths=0.6)
    ax_ras.set_xlim(lo_s, hi_s)
    ax_ras.set_ylabel("Trial")
    ax_ras.set_title(f"Situation {int(sit)} – Raster")

    ax_psth.bar(mids, rates, width=bin_s, align="center", edgecolor="none")
    ax_psth.set_xlim(lo_s, hi_s)
    ax_psth.set_xlabel("Time from event (s)")
    ax_psth.set_ylabel("Rate (Hz)")
    ax_psth.set_title(f"Situation {int(sit)} – PSTH")

    axes_map[int(sit)] = (ax_ras, ax_psth)

    fig.suptitle(f"Recording {recording_num} • Cell {cell_num} • Align: {align_to}", y=0.995)
    fig.tight_layout()
    return fig, axes_map

# -----------------------------------------------------------------------------
# Counting + Histograms per situation
# -----------------------------------------------------------------------------

def count_spikes_and_hist_by_situation(
    recording_num: int,
    cell_num: int,
    align_to: Literal["cue", "reward"],
    window_ms: Tuple[float, float],
    bins: int | str = "auto",
    situations: Sequence[int] | None = None,
    figsize=(12, 6),
):
    """
    Count spikes per trial in a window after the event; histogram per situation.

    Notes:
    - All histograms share the same x and y axes.
    - Bin edges are chosen to align with whole spike-count integers (bins centered on integers).
      The 'bins' argument is accepted for compatibility but integer-centered bins spanning the
      observed count range are used to ensure whole-number bins.
    """
    df = load_sorted_spikes(silent=True)
    sub = _subset_df(df, recording_num, cell_num)
    A = _align_spikes(sub, align_to)

    lo_s, hi_s = _ms_to_s(np.array(window_ms))

    if situations is None:
        situations = np.unique(A.situations)

    counts_dict: Dict[int, np.ndarray] = {}
    # Build counts per trial for each situation
    for sit in situations:
        mask = (A.situations == sit)
        rel = A.rel_time_s[mask]
        tri = A.trials[mask]
        uniq = np.unique(tri)
        per_trial_counts = []
        for t in uniq:
            r = rel[tri == t]
            per_trial_counts.append(int(np.sum(_window_mask(r, lo_s, hi_s))))
        counts_dict[int(sit)] = np.asarray(per_trial_counts, dtype=int)

    # Determine integer-centered bin edges spanning all counts
    all_vals = np.concatenate([v for v in counts_dict.values()]) if counts_dict else np.array([], dtype=int)
    if all_vals.size == 0:
        min_c, max_c = 0, 0
    else:
        min_c, max_c = int(np.min(all_vals)), int(np.max(all_vals))
    # Create edges so that each integer count occupies one bin (centered on integers)
    edges = np.arange(min_c - 0.5, max_c + 0.5 + 1e-9, 1.0)
    # If there is only a single possible count value, make at least one bin
    if edges.size == 0:
        edges = np.array([-0.5, 0.5])

    # Precompute hist heights to determine common y-limit
    max_count_height = 0
    hist_heights = {}
    for sit, vals in counts_dict.items():
        hist, _ = np.histogram(vals, bins=edges)
        hist_heights[sit] = hist
        if hist.size:
            max_count_height = max(max_count_height, int(np.max(hist)))

    # Setup shared axes figure
    n = len(situations)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharex=True, sharey=True)
    axes = axes.ravel()

    for i, sit in enumerate(situations):
        ax = axes[i]
        vals = counts_dict[int(sit)]
        # Use the precomputed integer edges for plotting
        ax.hist(vals, bins=edges, align="mid", edgecolor="black")
        ax.set_title(f"Situation {int(sit)}")
        ax.set_xlabel("Spike count")
        ax.set_ylabel("Trials")

    # Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Set common x and y limits
    ax_min = edges[0]
    ax_max = edges[-1]
    for ax in axes[:n]:
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(0, max(1, int(np.ceil(max_count_height * 1.05))))

    fig.suptitle(
        f"Spike counts per trial • Rec {recording_num} • Cell {cell_num} • Align {align_to} • Window {window_ms} ms",
        y=0.995,
    )
    fig.tight_layout()
    return counts_dict, fig

# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def _counts_per_trial_in_window(rel_s: np.ndarray, trials: np.ndarray, lo_s: float, hi_s: float):
    """
    Return (unique_trials, counts_per_trial) given relative spike times and trial IDs.
    """
    uniq = np.unique(trials)
    counts = np.zeros(uniq.shape[0], dtype=int)
    for i, t in enumerate(uniq):
        r = rel_s[trials == t]
        counts[i] = int(np.sum(_window_mask(r, lo_s, hi_s)))
    return uniq, counts

def stats_within_situation(
    recording_num: int,
    cell_num: int,
    situation: int,
    align_to: Literal["cue", "reward"],
    window_a_ms: Tuple[float, float],
    window_b_ms: Tuple[float, float],
    test: Literal["paired_t", "ranksum"] = "paired_t",
):
    """
    Compare two windows within the same situation using per-trial spike counts.

    test="paired_t" -> scipy.stats.ttest_rel
    test="ranksum"  -> scipy.stats.mannwhitneyu (two-sided, independent)

    Returns a dict with statistic, pvalue, n, and summary stats.
    """
    df = load_sorted_spikes(silent=True)
    sub = _subset_df(df, recording_num, cell_num)
    A = _align_spikes(sub, align_to)

    # Filter to one situation
    mask = (A.situations == situation)
    rel = A.rel_time_s[mask]
    tri = A.trials[mask]

    loA, hiA = _ms_to_s(np.array(window_a_ms))
    loB, hiB = _ms_to_s(np.array(window_b_ms))

    uniq, countsA = _counts_per_trial_in_window(rel, tri, loA, hiA)
    _, countsB = _counts_per_trial_in_window(rel, tri, loB, hiB)

    # Align trials (should already match, but be safe)
    if countsA.shape != countsB.shape:
        dA = dict(zip(uniq, countsA))
        dB = dict(zip(uniq, countsB))
        shared = sorted(set(dA).intersection(dB))
        countsA = np.array([dA[t] for t in shared], dtype=float)
        countsB = np.array([dB[t] for t in shared], dtype=float)
    else:
        countsA = countsA.astype(float)
        countsB = countsB.astype(float)

    if countsA.size < 2:
        raise ValueError("Not enough trials for a statistical test (need at least 2).")

    if test == "paired_t":
        stat, p = stats.ttest_rel(countsA, countsB, alternative="two-sided", nan_policy="omit")
    elif test == "ranksum":
        stat, p = stats.mannwhitneyu(countsA, countsB, alternative="two-sided")
    else:
        raise ValueError("test must be 'paired_t' or 'ranksum'")

    return {
        "test": test,
        "recording_num": recording_num,
        "cell_num": cell_num,
        "situation": int(situation),
        "align_to": align_to,
        "window_a_ms": tuple(window_a_ms),
        "window_b_ms": tuple(window_b_ms),
        "n_trials": int(countsA.size),
        "statistic": float(stat),
        "pvalue": float(p),
        "meanA": float(np.mean(countsA)),
        "meanB": float(np.mean(countsB)),
        "stdA": float(np.std(countsA, ddof=1) if countsA.size > 1 else 0.0),
        "stdB": float(np.std(countsB, ddof=1) if countsB.size > 1 else 0.0),
    }

def stats_between_situations(
    recording_num: int,
    cell_num: int,
    align_to: Literal["cue", "reward"],
    window_ms: Tuple[float, float],
    situations: Sequence[int],
    test: Literal["anova", "kruskal"] = "anova",
):
    """
    Compare a single window across multiple situations (2-4 groups).

    test="anova"   -> scipy.stats.f_oneway
    test="kruskal" -> scipy.stats.kruskal

    Returns a dict with statistic, pvalue, group sizes, and per-group means.
    """
    if len(situations) < 2 or len(situations) > 4:
        raise ValueError("Provide between 2 and 4 situations to compare.")

    df = load_sorted_spikes(silent=True)
    sub = _subset_df(df, recording_num, cell_num)
    A = _align_spikes(sub, align_to)

    lo_s, hi_s = _ms_to_s(np.array(window_ms))

    group_counts = []
    sizes = []
    means = []
    for sit in situations:
        mask = (A.situations == sit)
        rel = A.rel_time_s[mask]
        tri = A.trials[mask]
        _, c = _counts_per_trial_in_window(rel, tri, lo_s, hi_s)
        group_counts.append(c.astype(float))
        sizes.append(int(c.size))
        means.append(float(np.mean(c)) if c.size else float("nan"))

    if any(s < 2 for s in sizes):
        raise ValueError("Each group should have at least 2 trials for a robust test.")

    if test == "anova":
        stat, p = stats.f_oneway(*group_counts)
    elif test == "kruskal":
        stat, p = stats.kruskal(*group_counts)
    else:
        raise ValueError("test must be 'anova' or 'kruskal'")

    return {
        "test": test,
        "recording_num": recording_num,
        "cell_num": cell_num,
        "align_to": align_to,
        "window_ms": tuple(window_ms),
        "situations": [int(s) for s in situations],
        "group_sizes": sizes,
        "group_means": means,
        "statistic": float(stat),
        "pvalue": float(p),
    }
