# MIT-BIH preprocessing pipeline (Day 3)

## Scope and reproducibility

This initial pipeline reads the 48 WFDB `RECORDS` entries from the locally preserved PhysioNet MIT-BIH Arrhythmia Database v1.0.0. It filters continuous records, extracts beat-centered windows, normalizes each window, and stores labels with record, annotation, and pseudonymous grouping provenance. It does not train a model, augment or balance classes, or make train/validation/test splits. The versioned parameters are in [`configs/preprocessing/mitbih_v1.json`](../configs/preprocessing/mitbih_v1.json); software versions and counts are recorded in the generated dataset manifest.

## Pipeline and parameters

1. **Load/validate:** `ecg_io.py` uses WFDB physical units (mV), checks `.hea`, `.dat`, and `.atr` presence and any listed SHA-256 entries, enforces 360 Hz, two channels, finite `[samples, channels]` signals, and in-range annotation locations. Raw files are read only.
2. **Filter:** `filtering.py` applies a fourth-order Butterworth bandpass prototype with cutoffs 0.5 and 40 Hz at 360 Hz. SciPy designs second-order sections, applied with forward-backward `sosfiltfilt` (`padtype="odd"`) along time. A Butterworth response provides a smooth passband; the high-pass attenuates baseline wander and the low-pass attenuates higher-frequency noise while retaining much ECG morphology. Forward-backward filtering has zero phase shift; it is noncausal and uses the full record. A bandpass made from an order-4 prototype has an eighth-order one-pass transfer function, and forward/backward application squares the magnitude response. These are starting research parameters, not a clinically validated optimum. A 40 Hz upper cutoff will not remove the documented possible 30 Hz playback artifact.
3. **Segment:** `segmentation.py` centers one fixed window on each recognized beat annotation: 72 samples (0.2 s) before and 144 samples (0.4 s) after the annotated sample, total 216 samples (0.6 s). The annotated sample is at index 72. Incomplete boundary windows are dropped and counted; no padding is performed. Every segment records source record, pseudonymous patient group, source symbol, annotation sample, and half-open `[start,end)` offsets. Neighboring windows may overlap; all beat annotations are retained independently.
4. **Normalize:** `normalization.py` performs per-segment, per-channel z-score normalization over the 216 time points (`ddof=0`, epsilon `1e-8`, output `float32`). This avoids estimating shared scaling statistics from the whole cohort. It also removes absolute per-window amplitude information and may amplify low-variance windows; such windows fail validation rather than being silently altered.
5. **Label:** the initial mapping is identity mapping for the 15 beat symbols counted on Day 2: `N`, `L`, `R`, `V`, `/`, `A`, `f`, `F`, `j`, `a`, `E`, `J`, `Q`, `e`, `S`. No clinical classes are merged or renamed. The eight known non-beat/event symbols (`+`, `~`, `!`, `"`, `x`, `|`, `[`, `]`) do not create beat windows and are counted in record summaries as skipped non-beats. Any symbol outside those lists raises an error. Identity labels are source labels only; the eventual task definition remains unresolved.
6. **Store:** one compressed NPZ per record, plus `manifest.json`, `class_distribution.csv`, and `record_summary.csv`. NPZ files use numeric/string arrays and are opened with `allow_pickle=False`. Output is written to a staging directory and renamed on success; existing output is never overwritten.

## Patient grouping and leakage considerations

The official MIT-BIH record directory documents records 201 and 202 as coming from the same analog tape, so they receive the same pseudonymous group key (`mitdb_group_201_202`). Other records get record-derived keys only; they are **not asserted to be patient identifiers**. This known pair should remain grouped for any future split. Subject identities/dependencies beyond this documented pair need investigation before a subject-level evaluation is claimed. Filtering occurs on each complete record before window extraction, so any eventual evaluation design should split at the source-record/group level before deriving model datasets and consider the effect of whole-record zero-phase filtering. This pipeline creates no splits.

## Validation and limitations

Validation checks source files and available checksums, sampling frequency, channel count, array dimensions, finite inputs/outputs, annotation bounds and count alignment, configured labels, segment size, label-to-source-symbol identity, annotation-to-window offsets, and normalized per-channel mean/std. Unit tests also exercise filtering, boundary exclusion, known event-symbol exclusion, unknown-symbol failure, flat-window rejection, and metadata mismatch.

The 216-sample window, filter band, per-window normalization, treatment of uncommon beat symbols, and eventual target taxonomy require research justification and sensitivity analyses before model experiments. Segment counts describe successfully extracted windows after boundary handling; they do not establish clinical utility or population prevalence.

## Local outputs

The full derived dataset is written to `data/processed/mitbih_v1/` and excluded from Git because it is reproducible from the raw data and config. The validation notebook is [`notebooks/02_preprocessing_validation.ipynb`](../notebooks/02_preprocessing_validation.ipynb); its plot is saved under `results/figures/`.
