import os, math, numpy as np, pandas as pd
import spikeinterface as si
import spikeinterface.preprocessing as sp
import spikeinterface.sorters as ss
import spikeinterface.widgets as sw
from scipy.signal import welch
from IPython.display import display

"""
si_plot_audio_utils.py

Utilities for visualizing, listening to, and spike-sorting single-channel, multi-segment
SpikeInterface recordings.

Contents
--------
Plotting:
- plot_segment(...)
- plot_concatenated(...)
- plot_segment_with_spikes_and_threshold_mad(...)

Audio:
- play_audio_segment(...)
- play_audio_concatenated_segments(...)

Spike sorting (Tridesclous) + metrics:
- run_tridesclous_sorting(...)
- compute_average_firing_rate(...)

Templates (via SortingAnalyzer):
- extract_templates_via_sorting_analyzer(...)
- plot_templates_from_array(...)
"""

from typing import Optional, Tuple
import warnings

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.signal import resample_poly
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    resample_poly = None

# ------------------------------
# Helpers
# ------------------------------

def _assert_single_channel(recording):
    n_ch = recording.get_num_channels()
    if n_ch != 1:
        raise ValueError(f"This utility expects a single-channel recording; got {n_ch} channels.")


def _get_segment_samples(recording, segment_index: int) -> int:
    tr = recording.get_traces(segment_index=segment_index)
    return int(tr.shape[0])


def _downsample_for_plot(y: np.ndarray, max_points: int) -> tuple[np.ndarray, int]:
    n = y.shape[0]
    if n <= max_points:
        return y, 1
    step = int(np.ceil(n / max_points))
    return y[::step], step


def _median_and_mad(x: np.ndarray) -> Tuple[float, float]:
    """Return (median, MAD) where MAD = median(|x - median(x)|)."""
    med = float(np.median(x)) if x.size else 0.0
    mad = float(np.median(np.abs(x - med))) if x.size else 0.0
    return med, mad


# ------------------------------
# Plotting
# ------------------------------

def plot_segment(recording,
                 segment_index: int,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 ax: Optional[plt.Axes] = None,
                 max_points: int = 500_000) -> plt.Axes:
    _assert_single_channel(recording)
    fs = float(recording.get_sampling_frequency())

    seg_n = _get_segment_samples(recording, segment_index)
    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = seg_n / fs
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    start_frame = int(np.clip(np.floor(start_time * fs), 0, seg_n))
    end_frame = int(np.clip(np.ceil(end_time * fs), start_frame, seg_n))

    tr = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    tr = np.asarray(tr).reshape(-1)
    t = np.arange(tr.shape[0], dtype=float) / fs + (start_frame / fs)

    y_ds, step = _downsample_for_plot(tr, max_points=max_points)
    t_ds = t[::step]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ds, y_ds, linewidth=0.8)
    ax.set_title(f"Segment {segment_index} ({start_time:.3f}s — {end_time:.3f}s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.margins(x=0)
    return ax


def plot_concatenated(recording,
                      ax: Optional[plt.Axes] = None,
                      max_points: int = 1_000_000) -> plt.Axes:
    _assert_single_channel(recording)
    fs = float(recording.get_sampling_frequency())
    n_seg = int(recording.get_num_segments())

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    time_offset = 0.0
    total_samples = 0
    seg_lengths = []
    for seg in range(n_seg):
        seg_n = _get_segment_samples(recording, seg)
        seg_lengths.append(seg_n)
        total_samples += seg_n

    if total_samples == 0:
        warnings.warn("Recording appears to be empty.")
        return ax

    global_step = int(np.ceil(total_samples / max_points))
    global_step = max(global_step, 1)

    for seg, seg_n in enumerate(seg_lengths):
        tr = recording.get_traces(segment_index=seg).reshape(-1)
        y_ds = tr[::global_step]
        t_start = time_offset
        t_seg = t_start + (np.arange(y_ds.shape[0]) * (global_step / fs))
        ax.plot(t_seg, y_ds, linewidth=0.6)
        time_offset += seg_n / fs

    ax.set_title("Concatenated recording (all segments)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.margins(x=0)
    return ax


def plot_segment_with_spikes_and_threshold_mad(
        recording,
        sorting,
        segment_index: int,
        detect_threshold_mad: float,
        start_time: float | None = None,
        end_time: float | None = None,
        polarity: str = "neg",
        ax: Optional[plt.Axes] = None,
        max_points: int = 500_000) -> plt.Axes:
    """
    Plot a time window with spikes and a threshold derived from MAD units.

    detect_threshold_mad is converted to amplitude thresholds using the window's
    median and MAD:
        thr_neg = median - detect_threshold_mad * MAD
        thr_pos = median + detect_threshold_mad * MAD
    """
    fs = float(recording.get_sampling_frequency())
    _assert_single_channel(recording)

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        seg_n = _get_segment_samples(recording, segment_index)
        end_time = seg_n / fs

    ax = plot_segment(recording, segment_index, start_time, end_time, ax=ax, max_points=max_points)

    start_frame = int(np.floor(start_time * fs))
    end_frame = int(np.ceil(end_time * fs))
    tr = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index).reshape(-1)
    t = np.arange(tr.shape[0]) / fs + (start_frame / fs)

    med, mad = _median_and_mad(tr)
    thr_neg = med - detect_threshold_mad * mad
    thr_pos = med + detect_threshold_mad * mad

    if polarity in ("neg", "both"):
        ax.hlines(thr_neg, xmin=start_time, xmax=end_time, linestyles='--', linewidth=1.0, label=f'-{detect_threshold_mad:.2f} MAD')
    if polarity in ("pos", "both"):
        ax.hlines(thr_pos, xmin=start_time, xmax=end_time, linestyles='--', linewidth=1.0, label=f'+{detect_threshold_mad:.2f} MAD')

    unit_ids = sorting.get_unit_ids()
    spike_times_s = []
    for uid in unit_ids:
        st_frames = sorting.get_unit_spike_train(unit_id=uid, segment_index=segment_index)
        st_s = st_frames / fs
        mask = (st_s >= start_time) & (st_s <= end_time)
        spike_times_s.extend(list(st_s[mask]))

    if spike_times_s:
        y_at_spikes = np.interp(spike_times_s, t, tr)
        ax.scatter(spike_times_s, y_at_spikes, s=20, marker='x', label='spikes')

    ax.legend(loc='best')
    ax.set_title(f"Segment {segment_index} with spikes & ±{detect_threshold_mad} MAD threshold")
    return ax


# ------------------------------
# Audio
# ------------------------------

def play_audio_segment(recording,
                       segment_index: int,
                       start_time: float = 0.0,
                       duration: float = 30.0,
                       target_fs: int = 44100,
                       normalize: bool = True):
    from IPython.display import Audio

    _assert_single_channel(recording)
    fs = float(recording.get_sampling_frequency())
    seg_n = _get_segment_samples(recording, segment_index)

    if duration <= 0:
        raise ValueError("duration must be positive")

    start_frame = int(np.clip(np.floor(start_time * fs), 0, max(seg_n - 1, 0)))
    end_frame = int(np.clip(start_frame + int(np.round(duration * fs)), 0, seg_n))

    if end_frame <= start_frame:
        warnings.warn("Requested window is empty; returning silence.")
        data = np.zeros(int(target_fs * max(duration, 0.01)), dtype=np.float32)
        return Audio(data, rate=target_fs)

    tr = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index)
    x = np.asarray(tr).astype(np.float32).reshape(-1)

    if x.size > 0:
        x = x - np.mean(x, dtype=np.float64)

    if int(round(fs)) != int(target_fs):
        if _HAS_SCIPY:
            import math
            g = math.gcd(int(round(fs)), int(target_fs))
            up = int(target_fs // g)
            down = int(round(fs) // g)
            x = resample_poly(x, up=up, down=down)
        else:
            t_in = np.linspace(0, 1, num=x.shape[0], endpoint=False)
            num_out = int(np.round(x.shape[0] * (target_fs / fs)))
            t_out = np.linspace(0, 1, num=num_out, endpoint=False)
            x = np.interp(t_out, t_in, x).astype(np.float32)

    if normalize:
        peak = np.max(np.abs(x)) if x.size else 1.0
        if peak > 0:
            x = 0.99 * (x / peak)

    return Audio(x, rate=int(target_fs))


def play_audio_concatenated_segments(recording,
                                     start_segment: int,
                                     n_segments: int = 5,
                                     gap_duration: float = 1.0,
                                     target_fs: int = 44100,
                                     normalize: bool = True,
                                     auto_display: bool = True):
    """
    Concatenate N consecutive segments with a silence gap and return an IPython Audio widget.
    In Google Colab, set `auto_display=True` (default) so the widget is displayed immediately.
    """
    from IPython.display import Audio, display

    _assert_single_channel(recording)
    fs = float(recording.get_sampling_frequency())
    total_segments = int(recording.get_num_segments())

    if start_segment < 0 or start_segment >= total_segments:
        raise IndexError(f"start_segment {start_segment} out of range (0..{total_segments-1})")

    end_segment = min(start_segment + n_segments, total_segments)

    parts = []
    for seg in range(start_segment, end_segment):
        tr = recording.get_traces(segment_index=seg)
        x = np.asarray(tr).astype(np.float32).reshape(-1)
        if x.size > 0:
            x = x - np.mean(x, dtype=np.float64)
        parts.append(x)
        if seg < end_segment - 1 and gap_duration > 0:
            parts.append(np.zeros(int(round(gap_duration * fs)), dtype=np.float32))

    x = np.concatenate(parts) if parts else np.zeros(int(target_fs * max(0.5, n_segments * 5.0)), dtype=np.float32)

    # Resample if needed
    if int(round(fs)) != int(target_fs):
        if _HAS_SCIPY:
            import math
            g = math.gcd(int(round(fs)), int(target_fs))
            up = int(target_fs // g)
            down = int(round(fs) // g)
            x = resample_poly(x, up=up, down=down).astype(np.float32, copy=False)
        else:
            t_in = np.linspace(0, 1, num=x.shape[0], endpoint=False, dtype=np.float32)
            num_out = int(np.round(x.shape[0] * (target_fs / fs)))
            t_out = np.linspace(0, 1, num=num_out, endpoint=False, dtype=np.float32)
            x = np.interp(t_out, t_in, x).astype(np.float32)

    if normalize and x.size:
        peak = float(np.max(np.abs(x)))
        if peak > 0:
            x = 0.99 * (x / peak)
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    from IPython.display import Audio
    audio = Audio(x, rate=int(target_fs))

    if auto_display:
        try:
            from IPython.display import display
            display(audio)
        except Exception:
            pass

    return audio

# ------------------------------
# Spike sorting (Tridesclous) + metrics
# ------------------------------

def run_tridesclous_sorting(recording,
                            output_folder: str = "tdc_sort",
                            detect_threshold: float = 5.0,
                            **sorter_kwargs):
    _assert_single_channel(recording)

    try:
        import spikeinterface.full as si
    except Exception:
        import spikeinterface as si

    params = dict(detect_threshold=detect_threshold)
    params.update(sorter_kwargs)

    sorting = si.run_sorter(
        sorter_name="tridesclous",
        recording=recording,
        output_folder=output_folder,
        remove_existing=True,
        verbose=True,
        **params,
    )
    return sorting, params


def compute_average_firing_rate(sorting, recording) -> tuple[float, dict]:
    fs = float(recording.get_sampling_frequency())
    n_seg = int(recording.get_num_segments())

    total_samples = 0
    for seg in range(n_seg):
        total_samples += _get_segment_samples(recording, seg)
    total_duration = total_samples / fs if fs > 0 else 0.0
    if total_duration == 0:
        return 0.0, {}

    unit_ids = sorting.get_unit_ids()
    per_unit_hz = {}
    total_spikes = 0
    for uid in unit_ids:
        count = 0
        for seg in range(n_seg):
            st = sorting.get_unit_spike_train(unit_id=uid, segment_index=seg)
            count += int(len(st))
        per_unit_hz[uid] = count / total_duration
        total_spikes += count

    overall_hz = total_spikes / total_duration
    return overall_hz, per_unit_hz


# ------------------------------
# Templates via SortingAnalyzer
# ------------------------------

def extract_templates_via_sorting_analyzer(recording,
                                           sorting,
                                           folder: str | None = None,
                                           max_spikes_per_unit: int = 300,
                                           ms_before: float = 1.0,
                                           ms_after: float = 2.0):
    """
    Use SpikeInterface's SortingAnalyzer to compute and return average templates.
    Mirrors:
        sr1 = si.create_sorting_analyzer(sorting, recording)
        sr1.compute('random_spikes')
        sr1.compute('waveforms')
        sr1.compute('templates')
        av_templates = sr1.get_extension('templates').get_data(operator='average')
    """
    try:
        import spikeinterface.full as si
    except Exception:
        import spikeinterface as si

    sr1 = si.create_sorting_analyzer(sorting, recording, output_folder=folder, remove_if_exists=True)
    sr1.compute('random_spikes', max_spikes_per_unit=max_spikes_per_unit)
    sr1.compute('waveforms', ms_before=ms_before, ms_after=ms_after, return_scaled=True)
    sr1.compute('templates')

    templ_ext = sr1.get_extension('templates')
    av_templates = templ_ext.get_data(operator="average")
    av_templates = np.asarray(av_templates)
    unit_ids = list(sr1.sorting.get_unit_ids())

    if av_templates.ndim == 3 and av_templates.shape[1] == 1:
        av_templates = av_templates[:, 0, :]

    return av_templates, unit_ids


import matplotlib.pyplot as plt
import numpy as np

def plot_templates_in_ms(av_templates, sampling_frequency=22000):
    """
    Plots average unit templates with the x-axis in milliseconds.

    Args:
        av_templates (np.ndarray): Array of average templates with shape (num_units, num_samples, num_channels).
        sampling_frequency (int): The sampling frequency of the recording in Hz.
    """
    num_units, num_samples, num_channels = av_templates.shape

    # Calculate the time axis in milliseconds
    time_in_ms = np.arange(num_samples) / sampling_frequency * 1000

    plt.figure(figsize=(10, 4))
    for unit_index in range(num_units):
        # Assuming you want to plot the template for the first channel if there are multiple channels
        plt.plot(time_in_ms, av_templates[unit_index, :, 0], label=f'Unit {unit_index}')

    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Average Unit Templates')
    plt.legend()
    plt.show()


def plot_templates_from_array(templates: np.ndarray, unit_ids=None, ax: Optional[plt.Axes] = None):
    """
    Plot templates from an array of shape (n_units, n_samples, n_channels).
    If n_channels == 1, plots each unit as a 1D trace.
    X axis is in milliseconds (ms) for 22 kHz sample rate.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    T = templates
    # (n_units, n_samples, n_channels)
    if T.ndim == 3 and T.shape[2] == 1:
        # Only one channel, squeeze to (n_units, n_samples)
        T = T[:, :, 0]
    elif T.ndim == 3:
        # More than one channel, plot only channel 0
        T = T[:, :, 0]
    elif T.ndim == 2:
        # Already (n_units, n_samples)
        pass
    else:
        raise ValueError(f"Unexpected template shape {templates.shape}; expected (n_units, n_samples, n_channels) or (n_units, n_samples).")

    n_units = T.shape[0]
    n_samples = T.shape[1]
    fs = 22000.0  # 22 kHz
    t_ms = np.arange(n_samples) / fs * 1000.0

    if unit_ids is None:
        unit_ids = list(range(n_units))

    # Plot each unit's template
    for i in range(n_units):
        ax.plot(t_ms, T[i], label=f"unit {unit_ids[i]}")
    ax.set_title("Unit templates (average)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    if n_units <= 15:
        ax.legend(loc='best')
    return ax

# --- CONFIG: set these for your workshop ---
WORKSHOP_NAME = "data101-workshop"
DATA_FOLDER_NAME = "data"               # where data will live locally
GDRIVE_FOLDER_ID = ""  # Option A (leave "" to skip)
ZIP_HTTP_URL = "https://github.com/mhburrell/Neuro101DD_Tutorial/releases/download/Data/Data.zip"  # e.g., a GitHub Release URL to a .zip file (Option B), or leave "" to skip
ZIP_FILENAME = "Data.zip"            # only used if ZIP_HTTP_URL is set
GCS_HTTP_PREFIX = ""  # e.g., "https://storage.googleapis.com/your-bucket/workshop" (Option C), or leave "" to skip

# --- LOADER ---
import os, sys, shutil, subprocess, pathlib, zipfile

def _ensure(pkg):
    """Pip install a package if missing (quietly)."""
    try:
        __import__(pkg)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

def _download_with_gdown(folder_id, dest):
    _ensure("gdown")
    import gdown
    os.makedirs(dest, exist_ok=True)
    # download folder (handles many files)
    gdown.download_folder(
        url=f"https://drive.google.com/drive/folders/{folder_id}",
        output=dest,
        quiet=False, use_cookies=False
    )

def _download_zip(url, zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    # Use curl if available (fewer issues than wget on some images)
    subprocess.check_call(["bash", "-lc", f'curl -L "{url}" -o "{zip_path}"'])
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

def _maybe_copy_from_drive(source_rel, dest):
    """If student mounted Drive and placed data there, copy it once."""
    drive_path = "/content/drive/MyDrive"
    source = pathlib.Path(drive_path) / source_rel
    if source.exists():
        print(f"Copying data from Drive: {source}")
        if dest.exists():
            return
        shutil.copytree(source, dest)

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

    # 3) try shared Google Drive folder via gdown
    if GDRIVE_FOLDER_ID:
        print("Downloading data from shared Google Drive folder...")
        _download_with_gdown(GDRIVE_FOLDER_ID, str(data_dir))
        if data_dir.exists() and any(data_dir.iterdir()):
            print(f"Data ready at {data_dir}")
            return data_dir

    # 4) try a ZIP URL (e.g., GitHub Release)
    if ZIP_HTTP_URL:
        print("Downloading data zip...")
        zip_path = pathlib.Path("/content") / ZIP_FILENAME
        _download_zip(ZIP_HTTP_URL, str(zip_path), str(data_dir))
        if data_dir.exists() and any(data_dir.iterdir()):
            print(f"Data ready at {data_dir}")
            return data_dir

    # 5) if using GCS HTTP prefix, return a virtual path (you'll read via HTTP)
    if GCS_HTTP_PREFIX:
        print("No local data; you set GCS_HTTP_PREFIX, so read directly via HTTP/streaming.")
        return pathlib.Path(GCS_HTTP_PREFIX)  # treat as a 'virtual' path

    raise RuntimeError(
        "Could not prepare data. Set one of: GDRIVE_FOLDER_ID, ZIP_HTTP_URL, or GCS_HTTP_PREFIX, "
        "or place data in /content/data before calling get_data_dir()."
    )

# Example convenience loader
def load_csv(name, **read_csv_kwargs):
    """
    Load a CSV named `name` (e.g., 'iris.csv').
    If using GCS_HTTP_PREFIX, this builds the HTTP URL instead of local path.
    """
    import pandas as pd
    base = get_data_dir()
    # If base looks like an HTTP prefix, assemble URL; else local path
    if str(base).startswith("http://") or str(base).startswith("https://"):
        url = f"{str(base).rstrip('/')}/{name}"
        print(f"Reading remotely: {url}")
        return pd.read_csv(url, **read_csv_kwargs)
    else:
        path = base / name
        print(f"Reading local: {path}")
        return pd.read_csv(path, **read_csv_kwargs)
    
import os
import spikeinterface as si  # assuming this is the `si` you're using
from pathlib import Path

def load_recording(num):


    data_dir = get_data_dir()
    if not (1 <= num <= 7):
        raise ValueError("num must be between 1 and 7")

    rec_path = os.path.join(data_dir, "Data", f"recording{num}")
    return si.read_binary_folder(rec_path)

def plot_trial(recording_number,trial_num=1):
  recording = load_recording(recording_number)
  plot_segment(recording,segment_index = trial_num)

def plot_all_trials(recording_number):
  recording = load_recording(recording_number)
  plot_concatenated(recording)

def listen_recording(recording_number,start_trial=1):
  recording = load_recording(recording_number)
  play_audio_concatenated_segments(recording,start_segment=start_trial)


def show_filter_effects(
    recording,
    freq_min:float = 300.0,
    freq_max:float = 6000.0,
    notch_hz:float | None = None,   # set to 50 or 60 to notch
    reference:str = 'global',       # 'global' or None
    preview_seconds:float = 2.0
):
    """
    Loads a recording, applies simple filtering, and shows:
      - raw vs filtered trace (first few seconds)
      - PSD before/after
    Returns the filtered Recording (for chaining if desired).
    """
    rec = recording
    recording_idx = 0
    fs = 22000 #rec.sampling_frequency
    seg = 1
    raw = rec.get_traces(segment_index=seg)[:, 0]
    t = np.arange(raw.size) / fs
    win = int(min(preview_seconds * fs, raw.size))

    # Filtering pipeline
    rec_f = sp.bandpass_filter(rec, freq_min=freq_min, freq_max=freq_max)
    if notch_hz:
        try:
            rec_f = sp.notch_filter(rec_f, freq=notch_hz, q=30)
        except Exception as e:
            print("Notch filter skipped:", e)


    filt = rec_f.get_traces(segment_index=seg)[:, 0]

    # Raw vs filtered (time snippet)
    plt.figure()
    plt.plot(t[:win], raw[:win], label="raw", alpha=0.7)
    plt.plot(t[:win], filt[:win], label="filtered", alpha=0.7)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    plt.title(f"Recording {recording_idx}: Raw vs Filtered ({preview_seconds:g}s)")
    plt.legend(); plt.show()

    # PSD before/after
    f1, P1 = welch(raw, fs=fs, nperseg=min(2048, raw.size))
    f2, P2 = welch(filt, fs=fs, nperseg=min(2048, filt.size))
    plt.figure()
    plt.semilogy(f1, P1, alpha=0.8, label="raw")
    plt.semilogy(f2, P2, alpha=0.8, label="filtered")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD")
    plt.title(f"Recording {recording_idx}: PSD (raw vs filtered)")
    plt.legend(); plt.show()

    return rec_f

def filter_recording(recording_number,
    freq_min:float = 300.0,
    freq_max:float = 6000.0,
    notch_hz:float | None = None,   # set to 50 or 60 to notch
    reference:str = 'global',       # 'global' or None
    preview_seconds:float = 2.0,
    start_segment = 1):
  recording = load_recording(recording_number)
  filt_recording = show_filter_effects(recording,freq_min=freq_min,freq_max=freq_max, notch_hz=notch_hz,preview_seconds=preview_seconds)
  play_audio_concatenated_segments(filt_recording,start_segment = start_segment)

import sys, os
import contextlib



def run_tridesclous_sorting(recording,
                            detect_threshold: float = 5.0,
                            **sorter_kwargs):
    _assert_single_channel(recording)

    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):

            try:
                import spikeinterface.full as si
            except Exception:
                import spikeinterface as si

            params = dict(detect_threshold=detect_threshold)
            params.update(sorter_kwargs)

            sorting = si.run_sorter(
                sorter_name="tridesclous",
                recording=recording,
                remove_existing_folder=True,
                **params,
            )
    return sorting, params


def compute_average_firing_rate(sorting, recording) -> tuple[float, dict]:
  with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
          fs = float(recording.get_sampling_frequency())
          n_seg = int(recording.get_num_segments())

          total_samples = 0
          for seg in range(n_seg):
              total_samples += _get_segment_samples(recording, seg)
          total_duration = total_samples / fs if fs > 0 else 0.0
          if total_duration == 0:
              return 0.0, {}

          unit_ids = sorting.get_unit_ids()
          per_unit_hz = {}
          total_spikes = 0
          for uid in unit_ids:
              count = 0
              for seg in range(n_seg):
                  st = sorting.get_unit_spike_train(unit_id=uid, segment_index=seg)
                  count += int(len(st))
              per_unit_hz[uid] = count / total_duration
              total_spikes += count

          overall_hz = total_spikes / total_duration
  return overall_hz, per_unit_hz


# ------------------------------
# Templates via SortingAnalyzer
# ------------------------------

def extract_templates_via_sorting_analyzer(recording,
                                           sorting,
                                           max_spikes_per_unit: int = 300,
                                           ms_before: float = 1.0,
                                           ms_after: float = 2.0):
    """
    Use SpikeInterface's SortingAnalyzer to compute and return average templates.
    Mirrors:
        sr1 = si.create_sorting_analyzer(sorting, recording)
        sr1.compute('random_spikes')
        sr1.compute('waveforms')
        sr1.compute('templates')
        av_templates = sr1.get_extension('templates').get_data(operator='average')
    """
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
          try:
              import spikeinterface.full as si
          except Exception:
              import spikeinterface as si

          sr1 = si.create_sorting_analyzer(sorting, recording)
          sr1.compute('random_spikes', max_spikes_per_unit=max_spikes_per_unit)
          sr1.compute('waveforms', ms_before=ms_before, ms_after=ms_after)
          sr1.compute('templates')

          templ_ext = sr1.get_extension('templates')
          av_templates = templ_ext.get_data(operator="average")
          av_templates = np.asarray(av_templates)
          unit_ids = list(sr1.sorting.get_unit_ids())

          if av_templates.ndim == 3 and av_templates.shape[1] == 1:
              av_templates = av_templates[:, 0, :]

    return av_templates, unit_ids


def plot_templates_from_array(templates: np.ndarray, unit_ids=None, ax: Optional[plt.Axes] = None):
    """
    Plot templates from an array of shape (n_channels, n_samples, n_units).
    If n_channels == 1, plots each unit as a 1D trace.
    X axis is in milliseconds (ms) for 22 kHz sample rate.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    T = templates
    if T.ndim == 3 and T.shape[0] == 1:
        # (1, n_samples, n_units) -> (n_units, n_samples)
        T = T[0].T
    elif T.ndim == 3:
        # (n_channels, n_samples, n_units) -> (n_units, n_samples)
        # Only plot channel 0
        T = T[0].T
    elif T.ndim == 2:
        # (n_samples, n_units) -> (n_units, n_samples)
        T = T.T
    else:
        raise ValueError(f"Unexpected template shape {templates.shape}; expected (n_channels, n_samples, n_units) or (n_samples, n_units).")

    n_units = T.shape[0]
    n_samples = T.shape[1]
    fs = 22000.0  # 22 kHz
    t_ms = np.arange(n_samples) / fs * 1000.0

    if unit_ids is None:
        unit_ids = list(range(n_units))

    for i in range(n_units):
        ax.plot(t_ms, T[i], label=f"unit {unit_ids[i]}")
    ax.set_title("Unit templates (average)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    if n_units <= 15:
        ax.legend(loc='best')
    return ax

def plot_segment_with_spikes_and_threshold_mad(
        recording,
        sorting,
        segment_index: int,
        detect_threshold_mad: float,
        start_time: float | None = None,
        end_time: float | None = None,
        polarity: str = "neg",
        ax: Optional[plt.Axes] = None,
        max_points: int = 500_000) -> plt.Axes:
    """
    Plot a time window with spikes and a threshold derived from MAD units.

    detect_threshold_mad is converted to amplitude thresholds using the window's
    median and MAD:
        thr_neg = median - detect_threshold_mad * MAD
        thr_pos = median + detect_threshold_mad * MAD
    """
    fs = float(recording.get_sampling_frequency())
    _assert_single_channel(recording)

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        seg_n = _get_segment_samples(recording, segment_index)
        end_time = seg_n / fs

    ax = plot_segment(recording, segment_index, start_time, end_time, ax=ax, max_points=max_points)

    start_frame = int(np.floor(start_time * fs))
    end_frame = int(np.ceil(end_time * fs))
    tr = recording.get_traces(start_frame=start_frame, end_frame=end_frame, segment_index=segment_index).reshape(-1)
    t = np.arange(tr.shape[0]) / fs + (start_frame / fs)

    med, mad = _median_and_mad(tr)
    thr_neg = med - detect_threshold_mad * mad
    thr_pos = med + detect_threshold_mad * mad

    if polarity in ("neg", "both"):
        ax.hlines(thr_neg, xmin=start_time, xmax=end_time, linestyles='--', linewidth=1.0, label=f'-{detect_threshold_mad:.2f} MAD')
    if polarity in ("pos", "both"):
        ax.hlines(thr_pos, xmin=start_time, xmax=end_time, linestyles='--', linewidth=1.0, label=f'+{detect_threshold_mad:.2f} MAD')

    unit_ids = sorting.get_unit_ids()
    spike_times_s = []
    for uid in unit_ids:
        st_frames = sorting.get_unit_spike_train(unit_id=uid, segment_index=segment_index)
        st_s = st_frames / fs
        mask = (st_s >= start_time) & (st_s <= end_time)
        spike_times_s.extend(list(st_s[mask]))

    if spike_times_s:
        y_at_spikes = np.interp(spike_times_s, t, tr)
        ax.scatter(spike_times_s, y_at_spikes, s=20, marker='x', label='spikes')

    ax.legend(loc='best')
    ax.set_title(f"Segment {segment_index} with spikes & ±{detect_threshold_mad} MAD threshold")
    return ax

def run_sorting(recording_number,
                detect_threshold: float = 5.0,
                filter = False,
                freq_min:float = 300.0,
                freq_max:float = 6000.0,
                **sorter_kwargs):
  recording = load_recording(recording_number)
  if filter:
    recording = show_filter_effects(recording,freq_min = freq_min, freq_max=freq_max)
  
  sorting, params = run_tridesclous_sorting(recording, detect_threshold=detect_threshold, **sorter_kwargs)

  overall_hz, per_unit_hz = compute_average_firing_rate(sorting, recording)
  print(f"Average firing rate (all units): {overall_hz:.3f} Hz")
  for uid, hz in per_unit_hz.items():
    print(f"  Unit {uid}: {hz:.3f} Hz")

  av_templates, unit_ids = extract_templates_via_sorting_analyzer(recording = recording, sorting = sorting)
  plot_templates_in_ms(av_templates = av_templates)
  plot_segment_with_spikes_and_threshold_mad(recording = recording, sorting=sorting,segment_index=1,detect_threshold_mad=detect_threshold,polarity='both')
  plot_raster_and_psth_per_unit(sorting=sorting,recording=recording)

  import numpy as np
import matplotlib.pyplot as plt

def plot_raster_and_psth_per_unit(
    sorting,
    recording,
    bin_size: float = 0.01,   # seconds
    units=None,
    max_units: int | None = None,
    smooth_sigma: float | None = None,  # optional Gaussian smoothing (in seconds)
    figsize_scale: float = 2.2,
):
    """
    Make a per-unit raster (segments as rows, time within segment on x) and a PSTH.

    Raster:
      • For each unit, rows correspond to segments (0..N-1), and x is time within each segment.
      • Uses matplotlib.eventplot for clean, fast rendering.

    PSTH:
      • Pooled across segments, computed on segment-relative times (0 at each segment start).
      • Each time bin is normalized by the number of segments that *contribute to that bin*
        (i.e., segments whose duration covers that bin), giving firing rate in Hz.
      • Optionally smoothed with a Gaussian of std = smooth_sigma (seconds).

    Parameters
    ----------
    sorting : SpikeInterface BaseSorting
    recording : SpikeInterface BaseRecording
    bin_size : float
        PSTH bin width in seconds.
    units : list or None
        Subset of unit IDs to plot. By default, all units.
    max_units : int or None
        If given, truncate to the first `max_units` units (after applying `units` filter).
    smooth_sigma : float or None
        If provided, apply Gaussian smoothing to the PSTH with std = smooth_sigma (seconds).
    figsize_scale : float
        Scales the figure height relative to the number of units.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : dict
        Mapping unit_id -> (ax_raster, ax_psth).
    """
    fs = float(recording.get_sampling_frequency())
    n_seg = int(recording.get_num_segments())
    unit_ids = sorting.get_unit_ids() if units is None else list(units)
    if max_units is not None:
        unit_ids = unit_ids[:max_units]
    n_units = len(unit_ids)

    # Determine per-segment durations (seconds)
    seg_durs = []
    for seg in range(n_seg):
        # Using trace shape to get sample count for compatibility
        seg_n = recording.get_traces(segment_index=seg).shape[0]
        seg_durs.append(seg_n / fs if fs > 0 else 0.0)
    seg_durs = np.asarray(seg_durs, dtype=float)
    if n_seg == 0 or np.all(seg_durs <= 0):
        raise ValueError("Recording has no non-empty segments.")

    # Bin edges for PSTH: use from t=0 to max segment duration.
    # We'll correct for segments that don't reach some bins via 'contributors'.
    max_dur = float(np.max(seg_durs))
    nbins = int(np.ceil(max_dur / bin_size))
    edges = np.linspace(0.0, nbins * bin_size, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])  # for plotting

    # Precompute contributors per bin (how many segments cover each bin)
    # A segment "contributes" to a bin if the bin's RIGHT edge is <= segment duration.
    bin_rights = edges[1:]
    contributors = np.zeros(nbins, dtype=float)
    for dur in seg_durs:
        contributors += (bin_rights <= dur).astype(float)
    contributors[contributors == 0] = 1.0  # avoid divide-by-zero

    # Prepare figure
    fig_height = max(2.5, figsize_scale * n_units)
    fig, axs = plt.subplots(n_units, 2, figsize=(12, fig_height), sharex=False)
    if n_units == 1:
        axs = np.array([axs])  # make 2D for uniform indexing

    axes_map = {}

    # Optional Gaussian smoothing kernel for PSTH
    kernel = None
    if smooth_sigma is not None and smooth_sigma > 0:
        # Kernel length: +/- 3 sigma
        half_width = int(np.ceil(3 * smooth_sigma / bin_size))
        xk = np.arange(-half_width, half_width + 1, dtype=float) * bin_size
        kernel = np.exp(-0.5 * (xk / smooth_sigma) ** 2)
        kernel /= np.sum(kernel)

    for ui, uid in enumerate(unit_ids):
        ax_raster, ax_psth = axs[ui, 0], axs[ui, 1]

        # Collect per-segment spike times (relative to segment start) for this unit
        per_seg_times = []
        for seg in range(n_seg):
            st_frames = sorting.get_unit_spike_train(unit_id=uid, segment_index=seg)
            st_sec = np.asarray(st_frames, dtype=float) / fs
            # Keep those within [0, seg_durs[seg]] just in case
            if st_sec.size:
                st_sec = st_sec[(st_sec >= 0.0) & (st_sec <= seg_durs[seg])]
            per_seg_times.append(st_sec)

        # --- Raster (segments as rows) ---
        # eventplot expects a list of arrays; lineoffsets enumerate rows
        ax_raster.eventplot(
            per_seg_times,
            lineoffsets=np.arange(n_seg),
            linelengths=0.8,
            colors="k",
            linewidths=0.75,
        )
        ax_raster.set_ylabel("Segment")
        ax_raster.set_xlabel("Time within segment (s)")
        ax_raster.set_title(f"Unit {uid} – Raster")
        ax_raster.set_ylim(-0.5, n_seg - 0.5)
        # X-limits: up to the max segment duration for context
        ax_raster.set_xlim(0, max_dur)

        # --- PSTH (pooled across segments, normalized by contributors/binwidth) ---
        counts = np.zeros(nbins, dtype=float)
        for seg in range(n_seg):
            ts = per_seg_times[seg]
            if ts.size == 0:
                continue
            # Only include spikes that fall within the defined edges
            ts_clip = ts[(ts >= edges[0]) & (ts < edges[-1])]
            if ts_clip.size:
                c, _ = np.histogram(ts_clip, bins=edges)
                counts += c

        # Convert to rate: spikes per (s * contributing segments)
        rates = counts / (bin_size * contributors)

        # Optional smoothing
        if kernel is not None and rates.size >= kernel.size:
            rates = np.convolve(rates, kernel, mode="same")

        ax_psth.bar(mids, rates, width=bin_size, align="center", edgecolor="none")
        ax_psth.set_xlabel("Time within segment (s)")
        ax_psth.set_ylabel("Firing rate (Hz)")
        ax_psth.set_title(f"Unit {uid} – PSTH")
        ax_psth.set_xlim(0, max_dur)

        axes_map[uid] = (ax_raster, ax_psth)

    # Tidy layout
    fig.tight_layout()
    return fig, axes_map  