"""Neuro101DD data-analysis tutorial: Part 3 analysis tools (2026).

Alignment, spike counting, plotting and the regressions for Part 3.

You build the spike table yourself with `build_sorted_spikes()`, passing the
filter band and detection threshold you settled on in Part 2. Those values
determine which spikes are in the table, so they determine every number you
report. Change one and you must rebuild.

Long-format table, one row per spike:
    spikeTime      seconds from the start of the trial
    trial          1-based trial number
    cueTime        seconds, -1 on uncued trials
    outcomeTime    seconds; the time juice arrived, or would have
    predicted_ml   volume the cue predicted, NaN on uncued trials
    delivered_ml   volume actually delivered
    delta          delivered - predicted, in value units
    matched        True when delivered_ml == predicted_ml
    recording_num  1..10
    cell_num       1..n units found in that recording
"""
from __future__ import annotations

import json
import pathlib
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# NumPy 2 compatibility, in case this module is imported before the setup cell.
for _old, _new in [("in1d", "isin"), ("alltrue", "all"), ("sometrue", "any")]:
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

__all__ = [
    "build_sorted_spikes", "load_sorted_spikes", "recording_summary",
    "plot_raster_psth_all", "plot_raster_psth_by_volume",
    "count_spikes_by_volume", "response_table",
    "cue_response_vs_volume", "outcome_response_vs_volume",
    "outcome_response_vs_error", "same_reward_different_cue",
    "stats_before_after",
]

DATA_DIR = pathlib.Path("/content/data")
PARQUET = "sorted_spikes.parquet"
VOLUMES_ML = (0.05, 0.15, 0.25, 0.40, 0.60)
CUE_LETTER = dict(zip(VOLUMES_ML, "ABCDE"))
# Perceptually ordered and color-blind safe: the five volumes are an ordered
# variable, so they get a sequential ramp rather than five arbitrary hues.
VOL_COLOR = dict(zip(VOLUMES_ML, ["#c9d8e8", "#8fb4d4", "#4e86b8", "#2a5c8a", "#14324d"]))


def _data_dir() -> pathlib.Path:
    for c in (DATA_DIR, pathlib.Path("data"), pathlib.Path(".")):
        if (c / "manifest.json").exists() or any(c.glob("recording*_events.json")):
            return c
    raise FileNotFoundError(
        "Could not find your recordings. Run the data-download cell first."
    )


# --------------------------------------------------------------------- build
def build_sorted_spikes(
    detect_threshold: float = 5.0,
    freq_min: float = 300.0,
    freq_max: float = 6000.0,
    recordings: Sequence[int] | None = None,
    out_path: str | pathlib.Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sort every recording with your parameters and build the spike table.

    Takes a couple of minutes for ten recordings. Change a parameter and you
    must run this again. Record the values you used; they belong in your
    methods paragraph.
    """
    import spikeinterface as si
    import spikeinterface.sorters as ss
    import spikeinterface.preprocessing as sp

    base = _data_dir()
    if recordings is None:
        recordings = sorted(
            int(p.name.replace("recording", "").replace("_events.json", ""))
            for p in base.glob("recording*_events.json")
        )

    rows = []
    for n in recordings:
        rec = si.read_binary_folder(base / f"recording{n}")
        events = json.loads((base / f"recording{n}_events.json").read_text())
        filt = sp.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)
        sorting = ss.run_sorter(
            "tridesclous", filt, folder=f"_sort_rec{n}", remove_existing_folder=True,
            detect_threshold=detect_threshold, verbose=False,
        )
        fs = float(rec.get_sampling_frequency())
        unit_ids = list(sorting.get_unit_ids())
        if verbose:
            total = sum(
                len(sorting.get_unit_spike_train(unit_id=u, segment_index=k))
                for u in unit_ids for k in range(rec.get_num_segments())
            )
            dur = rec.get_num_segments() * rec.get_num_samples(segment_index=0) / fs
            print(f"recording {n:>2}: {len(unit_ids)} unit(s), "
                  f"{total} spikes, {total/dur:.2f} Hz overall")

        for cell_num, u in enumerate(unit_ids, start=1):
            for k, ev in enumerate(events):
                st = sorting.get_unit_spike_train(unit_id=u, segment_index=k) / fs
                if st.size == 0:
                    continue
                pred = ev["predicted_ml"]
                rows.append(pd.DataFrame({
                    "spikeTime": st,
                    "trial": ev["trial"],
                    "cueTime": -1.0 if ev["cue_time"] is None else ev["cue_time"],
                    "outcomeTime": ev["outcome_time"],
                    "predicted_ml": np.nan if pred is None else pred,
                    "delivered_ml": ev["delivered_ml"],
                    "delta": ev["delta"],
                    "matched": (pred is not None) and (pred == ev["delivered_ml"]),
                    "recording_num": n,
                    "cell_num": cell_num,
                }))

    df = pd.concat(rows, ignore_index=True)
    out = pathlib.Path(out_path) if out_path else base / PARQUET
    try:
        df.to_parquet(out, index=False)
        if verbose:
            print(f"\nsaved {len(df):,} spikes to {out}")
            print(f"parameters used: detect_threshold={detect_threshold}, "
                  f"freq_min={freq_min}, freq_max={freq_max}")
    except Exception as e:                                   # pyarrow missing
        if verbose:
            print(f"(could not save parquet: {e}) -- the table is still returned")
    return df


def load_sorted_spikes(silent: bool = False) -> pd.DataFrame:
    """Load the table you built earlier in this session."""
    path = _data_dir() / PARQUET
    if not path.exists():
        raise FileNotFoundError("No sorted_spikes.parquet yet -- run build_sorted_spikes() first.")
    df = pd.read_parquet(path)
    if not silent:
        print(f"{len(df):,} spikes, recordings {sorted(int(v) for v in df.recording_num.unique())}")
    return df


def recording_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unit: how many trials it appears in, its spike count and its
    firing rate. No waveform information -- for that, see the template plots
    from run_sorting() in Part 2.
    """
    out = []
    for (n, c), g in df.groupby(["recording_num", "cell_num"]):
        n_trials = int(g.trial.nunique())
        # trial length is fixed by the task, so rate = spikes / (trials x length)
        dur = n_trials * 4.0
        out.append({"recording_num": n, "cell_num": c, "trials": n_trials,
                    "spikes": len(g), "firing_rate_hz": round(len(g) / dur, 2)})
    return pd.DataFrame(out).sort_values(["recording_num", "cell_num"]).reset_index(drop=True)


# ------------------------------------------------------------------ helpers
def _one(df: pd.DataFrame, recording_num: int, cell_num: int) -> pd.DataFrame:
    sub = df[(df.recording_num == recording_num) & (df.cell_num == cell_num)]
    if sub.empty:
        raise ValueError(f"nothing for recording {recording_num}, cell {cell_num}")
    return sub


def _event_col(align_to: str) -> str:
    if align_to not in ("cue", "outcome"):
        raise ValueError("align_to must be 'cue' or 'outcome'")
    return "cueTime" if align_to == "cue" else "outcomeTime"


def _aligned(sub: pd.DataFrame, align_to: str) -> pd.DataFrame:
    col = _event_col(align_to)
    v = sub[sub[col] >= 0].copy()
    v["rel"] = v.spikeTime - v[col]
    return v


def _counts(sub: pd.DataFrame, align_to: str, window_ms: Sequence[float]) -> pd.DataFrame:
    """Spikes per trial in a window, one row per trial, with its condition."""
    lo, hi = np.asarray(window_ms, float) / 1000.0
    col = _event_col(align_to)
    v = sub[sub[col] >= 0]
    trials = v[["trial", col, "predicted_ml", "delivered_ml", "delta", "matched"]].drop_duplicates("trial")
    inwin = v[(v.spikeTime - v[col] >= lo) & (v.spikeTime - v[col] < hi)]
    counts = inwin.groupby("trial").size().rename("spikes")
    out = trials.set_index("trial").join(counts).fillna({"spikes": 0})
    out["rate_hz"] = out.spikes / (hi - lo)
    return out.reset_index()


def _fit(x, y, xlabel, ylabel, title, ax=None, color="#2a5c8a"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    lr = stats.linregress(x, y)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 3.4))
    jitter = (np.random.default_rng(0).random(len(x)) - 0.5) * (np.ptp(x) * 0.012 + 1e-9)
    ax.plot(x + jitter, y, "o", ms=4, alpha=0.45, color=color, mec="none")
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, "-", lw=1.8, color="#9d1b28")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nslope {lr.slope:.2f}   r² {lr.rvalue**2:.3f}   p {lr.pvalue:.3g}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return {"slope": float(lr.slope), "intercept": float(lr.intercept),
            "r_squared": float(lr.rvalue ** 2), "p_value": float(lr.pvalue),
            "n_trials": int(len(x))}


# --------------------------------------------------------------- raster/PSTH
def plot_raster_psth_all(recording_num: int, cell_num: int = 1,
                         align_to: Literal["cue", "outcome"] = "cue",
                         t_window_ms: Sequence[float] = (-500, 1500),
                         bin_size_ms: float = 50, df: pd.DataFrame | None = None):
    """Raster and PSTH over all trials, in the order they were recorded.

    Trials appear in recording order, so anything that changes partway through
    the session shows up as a change down the page.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    sub = _one(df, recording_num, cell_num)
    v = _aligned(sub, align_to)
    lo, hi = np.asarray(t_window_ms, float) / 1000.0
    trials = np.sort(sub.trial.unique())
    idx = {t: i for i, t in enumerate(trials)}

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.3]})
    per = [v.rel[(v.trial == t) & (v.rel >= lo) & (v.rel < hi)].to_numpy() for t in trials]
    a1.eventplot(per, lineoffsets=np.arange(len(trials)), linelengths=0.85,
                 colors="k", linewidths=0.6)
    a1.axvline(0, color="#9d1b28", lw=1.2)
    a1.set_ylabel("Trial (recording order)")
    a1.set_ylim(-0.5, len(trials) - 0.5)
    a1.set_title(f"Recording {recording_num}, cell {cell_num} — aligned to {align_to}", fontsize=10)

    edges = np.arange(lo, hi + 1e-9, bin_size_ms / 1000.0)
    rel = np.concatenate(per) if len(per) else np.empty(0)
    counts, _ = np.histogram(rel, bins=edges)
    a2.bar(edges[:-1], counts / (len(trials) * bin_size_ms / 1000.0),
           width=bin_size_ms / 1000.0, align="edge", color="#4e86b8", edgecolor="none")
    a2.axvline(0, color="#9d1b28", lw=1.2)
    a2.set_xlabel(f"Time from {align_to} (s)"); a2.set_ylabel("Firing rate (Hz)")
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig


def plot_raster_psth_by_volume(recording_num: int, cell_num: int = 1,
                               align_to: Literal["cue", "outcome"] = "cue",
                               t_window_ms: Sequence[float] = (-500, 1500),
                               bin_size_ms: float = 50, matched_only: bool = True,
                               df: pd.DataFrame | None = None):
    """One raster and PSTH per volume, stacked so they can be compared.

    With align_to='cue' the panels are the volume the cue predicted. With
    align_to='outcome' they are the volume delivered. Uncued trials get their
    own panel when align_to='outcome'.

    matched_only=True (the default) uses only fully predicted trials; set it to
    False to include the mismatch trials. The PSTH panels share a y-axis, so
    heights are comparable between them.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    sub = _one(df, recording_num, cell_num)
    if matched_only:
        sub = sub[sub.matched]
    key = "predicted_ml" if align_to == "cue" else "delivered_ml"
    groups = [(v, sub[sub[key] == v]) for v in VOLUMES_ML]
    if align_to == "outcome" and not matched_only:
        groups.append(("uncued", sub[sub.predicted_ml.isna()]))
    groups = [(k, g) for k, g in groups if not g.empty]

    lo, hi = np.asarray(t_window_ms, float) / 1000.0
    edges = np.arange(lo, hi + 1e-9, bin_size_ms / 1000.0)
    fig, axes = plt.subplots(len(groups), 2, figsize=(9.6, 1.7 * len(groups)),
                             squeeze=False, sharex=True)
    # Share the PSTH y-axis across panels. With free axes the panels look alike
    # however different the rates are, which would make this figure actively
    # misleading in a write-up.
    for r in range(1, len(groups)):
        axes[r][1].sharey(axes[0][1])
    for r, (k, g) in enumerate(groups):
        ar, ap = axes[r]
        v = _aligned(g, align_to)
        trials = np.sort(g.trial.unique())
        per = [v.rel[(v.trial == t) & (v.rel >= lo) & (v.rel < hi)].to_numpy() for t in trials]
        col = VOL_COLOR.get(k, "#6b6b6b")
        ar.eventplot(per, lineoffsets=np.arange(len(trials)), linelengths=0.85,
                     colors="k", linewidths=0.7)
        ar.axvline(0, color="#9d1b28", lw=1.1)
        label = f"{k} ml (cue {CUE_LETTER[k]})" if k in CUE_LETTER else str(k)
        ar.set_ylabel(label, fontsize=9)
        ar.set_ylim(-0.5, max(len(trials) - 0.5, 0.5))
        counts, _ = np.histogram(np.concatenate(per) if per else np.empty(0), bins=edges)
        ap.bar(edges[:-1], counts / (max(len(trials), 1) * bin_size_ms / 1000.0),
               width=bin_size_ms / 1000.0, align="edge", color=col, edgecolor="none")
        ap.axvline(0, color="#9d1b28", lw=1.1)
        ap.set_ylabel("Hz", fontsize=9)
        for ax in (ar, ap):
            ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ar.set_title(f"Recording {recording_num}, cell {cell_num} — "
                         f"aligned to {align_to}", fontsize=10, loc="left")
    axes[-1][0].set_xlabel(f"Time from {align_to} (s)")
    axes[-1][1].set_xlabel(f"Time from {align_to} (s)")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------ counting
def count_spikes_by_volume(recording_num: int, cell_num: int = 1,
                           align_to: Literal["cue", "outcome"] = "cue",
                           window_ms: Sequence[float] = (50, 300),
                           matched_only: bool = True, plot: bool = True,
                           df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Spike counts per trial in one window, summarized by volume.

    A row per volume: number of trials, the smallest and largest single-trial
    count, the mean count and the mean rate. Look at the min and max as well as
    the mean.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    sub = _one(df, recording_num, cell_num)
    if matched_only:
        sub = sub[sub.matched]
    per_trial = _counts(sub, align_to, window_ms)
    key = "predicted_ml" if align_to == "cue" else "delivered_ml"

    out = (per_trial.groupby(key)
           .agg(trials=("spikes", "size"), min_spikes=("spikes", "min"),
                max_spikes=("spikes", "max"), mean_spikes=("spikes", "mean"),
                mean_rate_hz=("rate_hz", "mean"))
           .round(2).reset_index())

    if plot:
        vols = out[key].tolist()
        fig, axes = plt.subplots(1, len(vols), figsize=(2.1 * len(vols), 2.3),
                                 sharey=True, squeeze=False)
        for ax, v in zip(axes[0], vols):
            d = per_trial[per_trial[key] == v].spikes
            ax.hist(d, bins=np.arange(-0.5, max(3, d.max()) + 1.5),
                    color=VOL_COLOR.get(v, "#4e86b8"), edgecolor="white", linewidth=0.6)
            ax.set_title(f"{v} ml", fontsize=9)
            ax.set_xlabel("spikes/trial", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0][0].set_ylabel("trials", fontsize=8)
        plt.suptitle(f"Recording {recording_num} cell {cell_num} — {align_to}, "
                     f"{window_ms[0]}–{window_ms[1]} ms", fontsize=9)
        plt.tight_layout()
    return out


def response_table(recording_num: int, cell_num: int = 1,
                   cue_window_ms: Sequence[float] = (50, 300),
                   outcome_window_ms: Sequence[float] = (50, 350),
                   df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Everything for one cell in one table: cue and outcome response by
    condition, matched and probe trials separately. Handy for the write-up."""
    df = load_sorted_spikes(silent=True) if df is None else df
    sub = _one(df, recording_num, cell_num)
    cue = _counts(sub, "cue", cue_window_ms).rename(columns={"rate_hz": "cue_rate_hz"})
    out = _counts(sub, "outcome", outcome_window_ms).rename(columns={"rate_hz": "outcome_rate_hz"})
    m = cue[["trial", "cue_rate_hz"]].merge(
        out[["trial", "outcome_rate_hz", "predicted_ml", "delivered_ml", "delta", "matched"]],
        on="trial", how="outer")
    g = (m.groupby(["predicted_ml", "delivered_ml"], dropna=False)
          .agg(trials=("trial", "size"), delta=("delta", "first"),
               cue_rate_hz=("cue_rate_hz", "mean"), outcome_rate_hz=("outcome_rate_hz", "mean"))
          .round(2).reset_index())
    return g.sort_values(["predicted_ml", "delivered_ml"]).reset_index(drop=True)


# --------------------------------------------------------------- regressions
def cue_response_vs_volume(recording_num: int, cell_num: int = 1,
                           window_ms: Sequence[float] = (50, 300),
                           df: pd.DataFrame | None = None) -> dict:
    """Q3.3 — regress the cue response on the volume the cue predicts.

    Fully predicted trials only, all five volumes in one fit.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    c = _counts(_one(df, recording_num, cell_num), "cue", window_ms)
    c = c[c.matched]
    return _fit(c.predicted_ml, c.spikes, "Volume predicted by the cue (ml)",
                f"Spikes in {window_ms[0]}–{window_ms[1]} ms",
                f"Cue response — recording {recording_num}, cell {cell_num}")


def outcome_response_vs_volume(recording_num: int, cell_num: int = 1,
                               window_ms: Sequence[float] = (50, 350),
                               df: pd.DataFrame | None = None) -> dict:
    """Q3.4 — on fully predicted trials, regress the outcome response on the
    volume delivered.

    Uses only trials where the volume delivered was the volume cued. Work out
    what you expect the slope to be before you run it.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    c = _counts(_one(df, recording_num, cell_num), "outcome", window_ms)
    c = c[c.matched]
    return _fit(c.delivered_ml, c.spikes, "Volume delivered (ml)",
                f"Spikes in {window_ms[0]}–{window_ms[1]} ms",
                f"Outcome response, fully predicted trials — rec {recording_num}, cell {cell_num}",
                color="#1f7a6b")


def outcome_response_vs_error(recording_num: int, cell_num: int = 1,
                              window_ms: Sequence[float] = (50, 350),
                              df: pd.DataFrame | None = None) -> dict:
    """Regress the outcome response on `delta`, using every cued trial.

    delta is negative on trials where the animal got less than the cue
    predicted, positive where it got more, zero on fully predicted trials.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    c = _counts(_one(df, recording_num, cell_num), "outcome", window_ms)
    c = c[c.predicted_ml.notna()]
    return _fit(c.delta, c.spikes, "Prediction error (delivered − predicted)",
                f"Spikes in {window_ms[0]}–{window_ms[1]} ms",
                f"Outcome response vs prediction error — rec {recording_num}, cell {cell_num}",
                color="#9d1b28")


def same_reward_different_cue(recording_num: int, cell_num: int = 1,
                              window_ms: Sequence[float] = (50, 350),
                              df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Q3.5 — the outcome response for 0.25 ml and 0.40 ml, split by which cue
    preceded them: a smaller cue, their own cue, and a larger cue.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    c = _counts(_one(df, recording_num, cell_num), "outcome", window_ms)
    rows = []
    for got in (0.25, 0.40):
        for pred in (0.05, got, 0.60):
            g = c[(c.delivered_ml == got) & (c.predicted_ml == pred)]
            if g.empty:
                continue
            rows.append({"delivered_ml": got, "predicted_ml": pred,
                         "cue": CUE_LETTER[pred], "delta": round(float(g.delta.iloc[0]), 2),
                         "trials": len(g), "mean_spikes": round(float(g.spikes.mean()), 2),
                         "rate_hz": round(float(g.rate_hz.mean()), 2)})
    return pd.DataFrame(rows)


def stats_before_after(recording_num: int, cell_num: int = 1,
                       align_to: Literal["cue", "outcome"] = "outcome",
                       before_ms: Sequence[float] = (-350, -50),
                       after_ms: Sequence[float] = (50, 350),
                       subset: str = "matched",
                       test: Literal["ranksum", "ttest"] = "ranksum",
                       df: pd.DataFrame | None = None) -> dict:
    """Is there a response at all? Compares spike counts before and after the
    event, trial by trial.

    subset: 'matched', 'probe', 'uncued', or 'all'.

    Spike counts from a slowly firing neuron are not normally distributed, so
    the default is the non-parametric Wilcoxon rank-sum test.
    """
    df = load_sorted_spikes(silent=True) if df is None else df
    sub = _one(df, recording_num, cell_num)
    if subset == "matched":
        sub = sub[sub.matched]
    elif subset == "probe":
        sub = sub[(~sub.matched) & sub.predicted_ml.notna()]
    elif subset == "uncued":
        sub = sub[sub.predicted_ml.isna()]
    b = _counts(sub, align_to, before_ms).set_index("trial").spikes
    a = _counts(sub, align_to, after_ms).set_index("trial").spikes
    common = b.index.intersection(a.index)
    b, a = b.loc[common].to_numpy(), a.loc[common].to_numpy()
    if test == "ranksum":
        s, p = stats.ranksums(a, b)
    else:
        s, p = stats.ttest_rel(a, b)
    return {"n_trials": int(len(common)), "mean_before": round(float(b.mean()), 2),
            "mean_after": round(float(a.mean()), 2), "test": test,
            "statistic": round(float(s), 3), "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05)}
