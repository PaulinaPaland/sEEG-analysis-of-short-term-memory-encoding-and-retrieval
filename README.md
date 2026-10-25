# sEEG analysis of short-term memory encoding and retrieval

Code from my bachelor's thesis,"Neural mechanisms of memory encoding and retrieval under varying cognitive loads: an seeg investigation"

For my thesis, I analysed intracranial EEG (stereo-EEG) recordings from a patient performing a spatial short-term memory task with three memory load levels. I investigated neural activity during memory encoding and retrieval, comparing different frequency bands (theta, low gamma, and high gamma) against a fixation baseline. The analysis included signal preprocessing, time-frequency analysis, and statistical comparisons to identify task-related changes in neural activity.

## Contents

| File | |
|---|---|
| `01_erp.ipynb` | Event-related potentials after encoding or retrieval onset vs. baseline |
| `02_power.ipynb` | Band power across the three load levels |
| `03_spectrogram.ipynb` | Time-frequency differences between load levels |
| `helpers.py` | Functions shared by the notebooks (loading, filtering, trial extraction, statistics, plots) |

Each notebook runs either the encoding or the retrieval analysis, depending on `PHASE` in the first cell. The ERP and power notebooks also take a frequency band (`BAND`): theta (4–8 Hz), low gamma (30–60 Hz) or high gamma (60–200 Hz). The band limits are set in `helpers.py`.

## Methods

**ERP.** The signal was band-pass filtered to the theta, low gamma or high gamma band and cut into trials from 0.5 s before to 0.5 s after each encoding or retrieval onset. The baseline was the 1 s after each fixation cross. At every time point and channel, trials were compared with the baseline using a Mann-Whitney U test, with FDR correction across time points within each channel. For the figures, neighbouring contacts were averaged.

**Band power.** For each load-level block I took one window: for encoding, from the second-to-last encoding event to 1400 samples (about 1.4 s) after the last one; for retrieval, the first second after retrieval onset. Power in each band was the area under the Welch power spectrum of the band-pass filtered signal. For each channel, the levels were compared with a Friedman test and pairwise Wilcoxon signed-rank tests, and power was correlated with load (Spearman). P-values were FDR-corrected across channels.

**Time-frequency.** For selected channels I computed log-power spectrograms (0.1–100 Hz) of each trial, using the same windows as above, cut to equal length. Load levels were compared pairwise with Welch's t-test at every time-frequency bin.

## Data

The patient data are not included here. The notebooks expect three NumPy files per recording in a `data/` folder:

- `<id>_sEEG.enret.npy`: signal, samples × channels, sampled at 1024 Hz
- `<id>_events.enret.npy`: one event label per sample (e.g. `FixationCross`, `Level2`, `Encoding_1`, `Retrieval`)
- `<id>_channelNames.enret.npy`: channel names

## Running the code

```bash
pip install -r requirements.txt
jupyter notebook
```

Set `DATA_DIR`, `PATIENT` and `PHASE` in the first cell of a notebook and run it. Figures and tables are saved to `results/<phase>/`.
