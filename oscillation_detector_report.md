# Oscillation detector analysis report (quick)

## Summary
We analyzed mesoscopic ROI activity to detect transient oscillatory bursts using a narrow‑band filter and Hilbert envelope. Bursts were defined as contiguous epochs where the envelope exceeded a robust threshold derived from the median absolute deviation (MAD). Detected bursts were summarized per session, and time‑aligned traces were visualized alongside pupil diameter.

## Data & signal selection
- Primary signal: `mesomap` → ROI `R_VISp`.
- Pupil trace: `pupil` → `pupil_diameter_mm`.
- Sampling rate: $f_s = 50\,\mathrm{Hz}$ (assumed uniform).
- Alignment: All traces share the same duration and are plotted over the same time axis; overlay uses the available sample counts without resampling.

## Preprocessing
1. Band‑pass filter: Butterworth (order 4) in the $3.1$–$4.3\,\mathrm{Hz}$ range.
2. Analytic amplitude: Hilbert envelope of the band‑passed signal.

## Burst detection
Let $e(t)$ be the envelope. A robust threshold is computed as:

$$
\theta = \mathrm{median}(e) + k\cdot 1.4826\,\mathrm{MAD}(e)
$$

with $k = 0.5$. Bursts are contiguous samples where $e(t) > \theta$.
- Minimum duration: $1.0\,\mathrm{s}$.
- Merge gap: $0.1\,\mathrm{s}$ between bursts.

## Parameters used
- Sampling rate: $50\,\mathrm{Hz}$
- Band: $3.1$–$4.3\,\mathrm{Hz}$
- Filter order: $4$
- Threshold factor: $k = 0.5$
- Minimum burst duration: $1\,\mathrm{s}$
- Merge gap: $0.1\,\mathrm{s}$

## Outputs
- Burst table: start time, end time, duration, peak envelope.
- Plot: raw ROI trace (with shaded bursts), band‑pass + envelope, and pupil diameter trace (with the same shaded burst epochs) on a shared time axis.
