
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt, spectrogram, welch
from scipy.stats import (false_discovery_control, friedmanchisquare, mannwhitneyu,
                         spearmanr, ttest_ind, wilcoxon)

# Frequency bands (Hz)
BANDS = {
    "delta": (0.1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "low gamma": (30, 60),
    "high gamma": (60, 200),
}


# Loading

def load_data(data_dir, patient):
    """Load signal (n_samples, n_channels), events (n_samples,) and channel names (n_channels,)."""
    data_dir = Path(data_dir)
    data = np.load(data_dir / f"{patient}_sEEG.enret.npy")
    events = np.load(data_dir / f"{patient}_events.enret.npy").astype(str)
    channel_names = np.load(data_dir / f"{patient}_channelNames.enret.npy").reshape(-1).astype(str)
    return data, events, channel_names


def bandpass_filter(data, low, high, sr, order=3):
    """Zero-phase Butterworth band-pass filter along time (axis 0)."""
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, data, axis=0)



# Events and trials

def find_onsets(events, label, exact=False):
    """Sample indices of an event. exact=False also matches e.g. 'Encoding_3' for 'Encoding'."""
    if exact:
        return np.where(events == label)[0]
    return np.where(np.char.find(events, label) >= 0)[0]


def extract_epochs(data, onsets, start, length):
    """Cut data[onset + start : onset + start + length] for every onset.

    Returns (n_trials, length, n_channels).
    """
    return np.stack([data[onset + start:onset + start + length] for onset in onsets])


def level_windows(events, phase, levels, encoding_post, retrieval_length):
    """One window per level block (a block runs from its level marker to the next one).

    encoding:  from the second-to-last encoding event to encoding_post samples after the last one
    retrieval: retrieval_length samples starting at the first retrieval event

    Returns dict: level -> list of (start, end).
    """
    level_onsets = {level: find_onsets(events, level, exact=True) for level in levels}
    all_level_onsets = np.sort(np.concatenate(list(level_onsets.values())))
    phase_onsets = find_onsets(events, phase.capitalize())

    windows = {}
    for level, onsets in level_onsets.items():
        windows[level] = []
        for onset in onsets:
            later = all_level_onsets[all_level_onsets > onset]
            block_end = later[0] if len(later) > 0 else np.inf
            in_block = phase_onsets[(phase_onsets > onset) & (phase_onsets < block_end)]

            if phase == "encoding" and len(in_block) >= 2:
                windows[level].append((in_block[-2], in_block[-1] + encoding_post))
            elif phase == "retrieval" and len(in_block) >= 1:
                windows[level].append((in_block[0], in_block[0] + retrieval_length))
    return windows


def same_length(windows, percentile=80):
    """Cut or extend all windows to the length at the given percentile."""
    lengths = np.sort([end - start for w in windows.values() for start, end in w])
    length = int(lengths[int(len(lengths) * percentile / 100)])
    return {level: [(start, start + length) for start, _ in w] for level, w in windows.items()}, length



# Power and spectrograms

def band_power(data, windows, sr):
    """Area under the Welch PSD for every window and channel.

    Returns dict: level -> (n_trials, n_channels).
    """
    power = {}
    for level, w in windows.items():
        trials = []
        for start, end in w:
            freqs, psd = welch(data[start:end], fs=sr, nperseg=sr // 2, axis=0)
            trials.append(np.trapezoid(psd, freqs, axis=0))
        power[level] = np.array(trials)
    return power


def log_spectrograms(trials, sr, nperseg, noverlap, nfft, fmax, sigma):
    """Log-power spectrograms of one channel, smoothed along frequency.

    trials: (n_trials, n_samples)
    Returns f (n_freqs,), t (n_times,) in s, S (n_freqs, n_times, n_trials).
    """
    f, t, S = spectrogram(trials, fs=sr, nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1)
    S = np.moveaxis(S, 0, -1)
    t = t - (nperseg / 2) / sr

    keep = f <= fmax
    S = gaussian_filter1d(S[keep], sigma=sigma, axis=0, mode="reflect")
    return f[keep], t, np.log(S + 1e-6)


def frequency_band(freq):
    """Name of the band in BANDS that a frequency falls into."""
    for name, (low, high) in BANDS.items():
        if low <= freq < high:
            return name
    return "delta" if freq < BANDS["delta"][0] else "unknown"


# Statistics

def fdr(p, axis=None):
    """Benjamini-Hochberg FDR. axis=0 corrects each column separately."""
    return false_discovery_control(p, axis=axis)


def compare_to_baseline(condition, baseline):
    """Mann-Whitney U test at every time point and channel. Returns p (n_times, n_channels)."""
    return mannwhitneyu(condition, baseline, axis=0).pvalue


def level_tests(power):
    """Friedman test across levels and pairwise Wilcoxon signed-rank tests, per channel.

    Trials are paired by their order within each level, so every level needs the same number.
    Returns p_friedman (n_channels,) and dict (levelA, levelB) -> p (n_channels,).
    """
    counts = [len(v) for v in power.values()]
    if len(set(counts)) > 1:
        raise ValueError(f"Levels have different numbers of trials: {counts}")

    p_friedman = friedmanchisquare(*power.values(), axis=0).pvalue
    p_pairs = {(a, b): wilcoxon(power[a], power[b], axis=0).pvalue
               for a, b in combinations(power, 2)}
    return p_friedman, p_pairs


def load_correlation(power):
    """Spearman correlation between memory load (1, 2, 3) and power, per channel."""
    load = np.concatenate([np.full(len(v), i + 1) for i, v in enumerate(power.values())])
    values = np.concatenate(list(power.values()))
    results = [spearmanr(load, values[:, ch]) for ch in range(values.shape[1])]
    rho = np.array([r.statistic for r in results])
    p = np.array([r.pvalue for r in results])
    return rho, p


def tf_ttest(spec_a, spec_b):
    """Welch's t-test per time-frequency bin (trials on the last axis). Returns t, p."""
    result = ttest_ind(spec_a, spec_b, axis=-1, equal_var=False)
    return result.statistic, result.pvalue


def significant_windows(sig, times, min_duration):
    """Continuous stretches where sig is True lasting at least min_duration. Returns [(start, end)]."""
    windows = []
    start = None
    for i in range(len(sig)):
        if sig[i] and start is None:
            start = i
        if start is not None and (not sig[i] or i == len(sig) - 1):
            end = i if sig[i] else i - 1
            if times[end] - times[start] >= min_duration:
                windows.append((times[start], times[end]))
            start = None
    return windows


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""



# Plots

def plot_erp(time_ms, condition, baseline, sig_windows, label, color, title, save_path=None):
    """Mean +- SEM of condition and baseline (n_trials, n_times) with significant windows shaded."""
    plt.figure(figsize=(6, 4))

    for trials, name, c in [(condition, label, color), (baseline, "Baseline", "#4D4D4D")]:
        mean = trials.mean(axis=0)
        sem = trials.std(axis=0) / np.sqrt(len(trials))
        plt.plot(time_ms, mean, color=c, linewidth=0.95, label=name)
        plt.fill_between(time_ms, mean - sem, mean + sem, color=c, alpha=0.2)

    for start, end in sig_windows:
        plt.axvspan(start, end, ymin=0.025, ymax=0.975, facecolor=(1, 0.84, 0, 0.1),
                    edgecolor=(0, 0, 0, 0.5), linestyle="--", linewidth=0.75, zorder=0)

    plt.axvline(0, color="black", linestyle="--", linewidth=0.75, alpha=0.8)
    plt.axhline(0, color="black", linestyle="--", linewidth=0.75, alpha=0.8)
    plt.xlabel("Time from onset (ms)")
    plt.ylabel("Voltage (µV)")
    plt.title(title)
    plt.legend(frameon=False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_levels(values, labels, colors, pair_p, ylabel, title, save_path=None):
    """Boxplot per level with significance brackets.

    values: list of 1D arrays, pair_p: dict (i, j) -> p with i, j positions in values.
    """
    plt.figure(figsize=(7, 5))
    boxes = plt.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False,
                        medianprops={"color": "black", "linewidth": 2})
    for box, color in zip(boxes["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.6)

    # y-limits from the whiskers, with space on top for the brackets
    whiskers = np.concatenate([w.get_ydata() for w in boxes["whiskers"]])
    y_range = whiskers.max() - whiskers.min()
    y_min = whiskers.min() - 0.05 * y_range
    y_max = whiskers.max() + 0.20 * y_range
    plt.ylim(y_min, y_max)

    # widest bracket on top, the others below it
    y = y_min + 0.93 * (y_max - y_min)
    for (i, j), p in sorted(pair_p.items(), key=lambda x: x[0][1] - x[0][0], reverse=True):
        if stars(p):
            plt.plot([i + 1, i + 1, j + 1, j + 1], [y, y + 0.01 * y_range, y + 0.01 * y_range, y],
                     color="black", linewidth=1.5)
            plt.text((i + j + 2) / 2, y, stars(p), ha="center", va="bottom", fontsize=20)
            y -= 0.09 * y_range

    plt.ylabel(ylabel, fontsize=16)
    plt.title(title)
    plt.xticks(fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_tmap(t, f, t_values, sig, title, save_path=None):
    """t-values over time and frequency, significant bins outlined."""
    vmax = np.ceil(np.nanmax(np.abs(t_values)))

    plt.figure(figsize=(7, 4))
    plt.pcolormesh(t, f, t_values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="nearest")
    plt.colorbar(label="t-value", ticks=[-vmax, 0, vmax])
    if sig.any():
        plt.contour(t, f, sig.astype(float), levels=[0.5], colors="k", linewidths=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.yticks([4, 8, 30, 60, 100])
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
