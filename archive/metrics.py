# =============================================================
# rPPG System Evaluation: Core Statistical Metrics
# Ground Truth: CMS50D Pulse Oximeter (Pulse_Rate_Hardware)
# rPPG Estimates: HR_FFT and HR_Peak from the pipeline
#
# Also includes SpO2 and Waveform-based analysis suggestions.
#
# USAGE: Run standalone in Google Colab or any Python env.
# Requires: pandas, numpy, scipy, matplotlib, sklearn
# Input file: mohanesh_period_01_cms50d_sync_data_20260603_192934.csv
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from google.colab.patches import cv2_imshow
from google.colab import files


# =============================================================
# SECTION 0: LOAD & CLEAN DATA
# =============================================================

uploaded = files.upload()
if not uploaded:
    raise RuntimeError("No video file uploaded.")
CSV_path = list(uploaded.keys())[0]
print("Uploaded:", CSV_path)


df_raw = pd.read_csv(CSV_path)

# Drop trailing NaN rows (data logger footer artifact)
df = df_raw.dropna(subset=["Pulse_Rate_Hardware", "Waveform", "HR_FFT", "HR_Peak"]).copy()
df = df.reset_index(drop=True)

print("=" * 60)
print("  DATA LOADING SUMMARY")
print("=" * 60)
print(f"  Raw rows loaded      : {len(df_raw)}")
print(f"  Clean rows after NaN : {len(df)}")
print(f"  Columns              : {list(df.columns)}")
print(f"  Recording duration   : ~{len(df) / 30:.1f} seconds (@ ~30 fps)")


# =============================================================
# SECTION 1: DEFINE GROUND TRUTH AND ESTIMATES
# =============================================================
# Ground truth: Pulse_Rate_Hardware (oximeter hardware HR output)
# This is a running average from the CMS50D device.
#
# Your rPPG system gives ONE final consensus HR per video.
# You must supply it below after running your pipeline.
# HR_FFT and HR_Peak from this CSV are the oximeter's own
# intermediate estimates — included for reference analysis.
# =============================================================

gt_series = df["Pulse_Rate_Hardware"].values.astype(float)

# --- YOUR rPPG FINAL CONSENSUS HR (from your pipeline output) ---
# Replace this with the actual value printed by your rPPG script.
# e.g. if your script printed "Final Consensus HR: 63.45 bpm"
# set: RPPG_CONSENSUS_HR = 63.45
RPPG_CONSENSUS_HR = rppg_results["consensus_hr"]

# For multi-video evaluation: list one value per video
# RPPG_ESTIMATES  = [63.45, 61.20, 66.10, ...]   # your rPPG values
# GT_REFERENCES   = [63.0,  62.0,  65.0,  ...]   # oximeter mean per video
# (each GT value = mean of Pulse_Rate_Hardware over that video)

# --- Derive a single scalar ground truth for this video ---
# We use the median (robust to the discrete stepping of oximeter hardware)
GT_SCALAR = float(np.median(gt_series))
GT_MEAN   = float(np.mean(gt_series))
GT_STD    = float(np.std(gt_series))

print(f"\n  Ground Truth (oximeter) median : {GT_SCALAR:.2f} BPM")
print(f"  Ground Truth (oximeter) mean   : {GT_MEAN:.2f} BPM")
print(f"  Ground Truth (oximeter) std    : {GT_STD:.2f} BPM")
print(f"  Ground Truth range             : "
      f"{gt_series.min():.1f} – {gt_series.max():.1f} BPM")

if RPPG_CONSENSUS_HR is not None:
    print(f"\n  rPPG Consensus HR              : {RPPG_CONSENSUS_HR:.2f} BPM")
else:
    print("\n  rPPG Consensus HR : NOT SET — fill RPPG_CONSENSUS_HR above.")
    print("  Single-video metrics will be skipped until set.")


# =============================================================
# SECTION 2: SINGLE-VIDEO SCALAR METRICS
# (Your system → one value; oximeter → one reference value)
# =============================================================

print("\n" + "=" * 60)
print("  SECTION 2: SINGLE-VIDEO SCALAR METRICS")
print("=" * 60)

if RPPG_CONSENSUS_HR is not None:

    error          = RPPG_CONSENSUS_HR - GT_SCALAR
    abs_error      = abs(error)
    pct_error      = abs_error / GT_SCALAR * 100.0

    print(f"\n  rPPG estimate          : {RPPG_CONSENSUS_HR:.2f} BPM")
    print(f"  Ground truth (median)  : {GT_SCALAR:.2f} BPM")
    print(f"  Signed error           : {error:+.2f} BPM  "
          f"({'overestimate' if error > 0 else 'underestimate'})")
    print(f"  Absolute error (MAE)   : {abs_error:.2f} BPM")
    print(f"  Percentage error (MAPE): {pct_error:.2f} %")
    print(f"  RMSE (single value)    : {abs_error:.2f} BPM  "
          f"(same as MAE for n=1)")

    # Clinical acceptability check
    # IEEE/ANSI standard for HR monitors: ±5 BPM or ±5% (whichever is larger)
    clinical_pass = abs_error <= max(5.0, 0.05 * GT_SCALAR)
    print(f"\n  Clinical accuracy check (±5 BPM / ±5% threshold): "
          f"{'PASS ✓' if clinical_pass else 'FAIL ✗'}")

else:
    print("  [Skipped — set RPPG_CONSENSUS_HR to enable]")


# =============================================================
# SECTION 3: MULTI-VIDEO METRICS
# (Run your pipeline on all videos, collect values here)
#
# Even with 1 video now, this section shows you what to compute
# across your full dataset of 20 videos.
# =============================================================

print("\n" + "=" * 60)
print("  SECTION 3: MULTI-VIDEO EVALUATION METRICS")
print("=" * 60)

# ----- PLACEHOLDER: replace with your actual collected results -----
# Format: one entry per video
# rppg_estimates[i] = your system's output for video i
# gt_references[i]  = median(Pulse_Rate_Hardware) for video i
#
# Example (fill in your real values):
# rppg_estimates = np.array([63.45, 61.20, 66.10, 70.80, 58.90])
# gt_references  = np.array([63.00, 62.00, 65.00, 71.00, 60.00])

# Run this block once per video, accumulating results
rppg_estimates = np.append(rppg_estimates, rppg_results["consensus_hr"])
gt_references  = np.append(gt_references,  GT_SCALAR)   # from that video's CSV
if RPPG_CONSENSUS_HR is not None and len(rppg_estimates) == 0:
    # Seed with the single video we have
    rppg_estimates = np.array([RPPG_CONSENSUS_HR])
    gt_references  = np.array([GT_SCALAR])

if len(rppg_estimates) == 0:
    print("  [Skipped — populate rppg_estimates and gt_references above]")

elif len(rppg_estimates) == 1:
    print(f"  Only 1 video provided. Full multi-video metrics need >= 2.")
    print(f"  Showing available single-video results only (see Section 2).")

else:
    errors     = rppg_estimates - gt_references
    abs_errors = np.abs(errors)

    # --- 3a: MAE ---
    mae = float(np.mean(abs_errors))

    # --- 3b: RMSE ---
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # --- 3c: MAPE ---
    mape = float(np.mean(abs_errors / (gt_references + 1e-8)) * 100.0)

    # --- 3d: Pearson correlation ---
    if len(rppg_estimates) >= 3:
        r_pearson, p_pearson = stats.pearsonr(gt_references, rppg_estimates)
    else:
        r_pearson, p_pearson = float('nan'), float('nan')

    # --- 3e: Spearman correlation ---
    if len(rppg_estimates) >= 3:
        r_spearman, p_spearman = stats.spearmanr(gt_references, rppg_estimates)
    else:
        r_spearman, p_spearman = float('nan'), float('nan')

    # --- 3f: Bland-Altman statistics ---
    ba_means = (rppg_estimates + gt_references) / 2.0
    ba_diffs = rppg_estimates - gt_references

    ba_bias   = float(np.mean(ba_diffs))
    ba_std    = float(np.std(ba_diffs, ddof=1))
    loa_upper = ba_bias + 1.96 * ba_std
    loa_lower = ba_bias - 1.96 * ba_std

    # --- 3g: Within ±5 BPM fraction ---
    within_5 = float(np.mean(abs_errors <= 5.0) * 100.0)
    within_3 = float(np.mean(abs_errors <= 3.0) * 100.0)

    print(f"\n  N videos evaluated     : {len(rppg_estimates)}")
    print(f"\n  -- Error Metrics --")
    print(f"  MAE                    : {mae:.3f} BPM")
    print(f"  RMSE                   : {rmse:.3f} BPM")
    print(f"  MAPE                   : {mape:.3f} %")
    print(f"\n  -- Correlation --")
    print(f"  Pearson r              : {r_pearson:.4f}  (p = {p_pearson:.4f})")
    print(f"  Spearman ρ             : {r_spearman:.4f}  (p = {p_spearman:.4f})")
    print(f"\n  -- Bland-Altman --")
    print(f"  Bias (mean diff)       : {ba_bias:+.3f} BPM")
    print(f"  Std of differences     : {ba_std:.3f} BPM")
    print(f"  Upper LoA (+1.96 SD)   : {loa_upper:+.3f} BPM")
    print(f"  Lower LoA (-1.96 SD)   : {loa_lower:+.3f} BPM")
    print(f"\n  -- Clinical Accuracy --")
    print(f"  Within ±5 BPM          : {within_5:.1f} %")
    print(f"  Within ±3 BPM          : {within_3:.1f} %")

    # --- 3h: Interpret ---
    print(f"\n  -- Interpretation --")
    if mae < 5.0:
        print(f"  MAE {mae:.2f} BPM → ACCEPTABLE (clinical threshold: <5 BPM)")
    else:
        print(f"  MAE {mae:.2f} BPM → EXCEEDS clinical threshold of 5 BPM")

    if abs(ba_bias) < 2.0:
        print(f"  Bland-Altman bias {ba_bias:+.2f} BPM → LOW SYSTEMATIC BIAS")
    elif abs(ba_bias) < 5.0:
        print(f"  Bland-Altman bias {ba_bias:+.2f} BPM → MODERATE SYSTEMATIC BIAS")
    else:
        print(f"  Bland-Altman bias {ba_bias:+.2f} BPM → HIGH SYSTEMATIC BIAS — recalibrate")


# =============================================================
# SECTION 4: OXIMETER TIME-SERIES ANALYSIS
# Evaluates whether HR_FFT from the oximeter (converging series)
# is useful as a framewise reference.
# =============================================================

print("\n" + "=" * 60)
print("  SECTION 4: OXIMETER INTERNAL TIME-SERIES")
print("=" * 60)

# HR_FFT from the oximeter converges from a garbage initial value
# (560 BPM) — trim the convergence region for any framewise analysis
CONVERGENCE_TRIM = 30   # frames to skip at start

hr_fft_series   = df["HR_FFT"].values[CONVERGENCE_TRIM:].astype(float)
hr_peak_series  = df["HR_Peak"].values[CONVERGENCE_TRIM:].astype(float)
gt_trim_series  = gt_series[CONVERGENCE_TRIM:]
frame_times     = (df.index[CONVERGENCE_TRIM:].values) / 30.0  # seconds at ~30fps

print(f"\n  HR_FFT (oximeter) converged mean  : {np.mean(hr_fft_series):.2f} BPM")
print(f"  HR_Peak (oximeter) mean           : {np.mean(hr_peak_series):.2f} BPM")
print(f"  Ground truth (hw) mean            : {np.mean(gt_trim_series):.2f} BPM")

# Internal agreement: HR_FFT vs Pulse_Rate_Hardware (same device)
internal_diff   = hr_fft_series - gt_trim_series
internal_mae    = float(np.mean(np.abs(internal_diff)))
internal_bias   = float(np.mean(internal_diff))

print(f"\n  Oximeter internal: HR_FFT vs Hardware HR")
print(f"  MAE                               : {internal_mae:.3f} BPM")
print(f"  Bias                              : {internal_bias:+.3f} BPM")
print(f"  (Serves as a reference floor — "
      f"your rPPG error should be benchmarked against this)")

# Final oximeter convergence value (last 30 frames, stable)
stable_window  = hr_fft_series[-30:]
stable_mean    = float(np.mean(stable_window))
stable_std     = float(np.std(stable_window))
print(f"\n  HR_FFT stable value (last 30 frames): "
      f"{stable_mean:.2f} ± {stable_std:.2f} BPM")
print(f"  → Use {stable_mean:.2f} BPM as the most accurate ground truth "
      f"for this video if comparing to a single rPPG output.")


# =============================================================
# SECTION 5: WAVEFORM-DERIVED SpO2 ANALYSIS
# The CMS50D Waveform column is the raw PPG signal.
# We can derive: respiratory rate, pulse amplitude variability,
# perfusion index proxy, and cross-validate SpO2.
# =============================================================

print("\n" + "=" * 60)
print("  SECTION 5: WAVEFORM & SpO2 ANALYSIS")
print("=" * 60)

waveform   = df["Waveform"].values.astype(float)
spo2_series = df["SpO2"].values.astype(float)
fs_ppg     = 30.0   # approximate frame rate

# --- 5a: SpO2 descriptive stats ---
print(f"\n  SpO2 Statistics:")
print(f"  Mean     : {np.mean(spo2_series):.2f} %")
print(f"  Std      : {np.std(spo2_series):.2f} %")
print(f"  Min      : {np.min(spo2_series):.1f} %")
print(f"  Max      : {np.max(spo2_series):.1f} %")
print(f"  Range    : {np.max(spo2_series) - np.min(spo2_series):.1f} %")

# Normal SpO2 >= 95%; SpO2 < 92% = clinically significant
pct_normal  = float(np.mean(spo2_series >= 95.0) * 100.0)
pct_low     = float(np.mean(spo2_series < 92.0)  * 100.0)
print(f"  Normal (≥95%) frames : {pct_normal:.1f} %")
print(f"  Low    (<92%) frames : {pct_low:.1f} %")

if pct_low > 5.0:
    print("  WARNING: >5% frames below 92% SpO2 — verify sensor contact.")

# --- 5b: Perfusion Index proxy from waveform ---
# PI = AC component / DC component of PPG
# Higher PI = better peripheral perfusion = more reliable rPPG
waveform_detrended = waveform - np.mean(waveform)
pi_proxy = float(np.std(waveform_detrended) / (np.mean(waveform) + 1e-8) * 100.0)
print(f"\n  Waveform Perfusion Index proxy : {pi_proxy:.2f} %")
print(f"  (PI >0.5% → good signal quality; <0.2% → weak peripheral pulse)")

# --- 5c: Pulse amplitude variability (PAV) ---
# PAV reflects respiratory modulation of blood pressure — correlates with RR
peaks_ppg, props = find_peaks(
    waveform_detrended,
    distance=int(fs_ppg * 60.0 / 120.0),   # min distance at max 120 BPM
    prominence=0.1 * np.std(waveform_detrended)
)

if len(peaks_ppg) >= 4:
    peak_amplitudes = waveform_detrended[peaks_ppg]
    pav_std  = float(np.std(peak_amplitudes))
    pav_cv   = float(pav_std / (np.mean(np.abs(peak_amplitudes)) + 1e-8) * 100.0)
    print(f"\n  Pulse Amplitude Variability (PAV):")
    print(f"  Detected PPG peaks : {len(peaks_ppg)}")
    print(f"  Amplitude std      : {pav_std:.3f}")
    print(f"  Amplitude CV       : {pav_cv:.2f} %")
    print(f"  (High PAV → strong respiratory modulation of pulse pressure)")
else:
    print(f"\n  PAV: Not enough PPG peaks detected ({len(peaks_ppg)}).")

# --- 5d: HR from waveform (cross-check) ---
if len(peaks_ppg) >= 2:
    peak_times_ppg = peaks_ppg / fs_ppg
    rr_ppg = np.diff(peak_times_ppg)
    hr_from_waveform = float(np.median(60.0 / rr_ppg))
    print(f"\n  HR from raw waveform peaks : {hr_from_waveform:.2f} BPM")
    print(f"  Ground truth (hw median)   : {GT_SCALAR:.2f} BPM")
    print(f"  Waveform HR vs GT error    : {hr_from_waveform - GT_SCALAR:+.2f} BPM")


# =============================================================
# SECTION 6: VISUALIZATIONS
# =============================================================

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.38)

# Color palette
C_GT    = "#1565C0"   # deep blue  = ground truth
C_RPPG  = "#C62828"   # deep red   = your rPPG
C_FFT   = "#2E7D32"   # deep green = HR_FFT series
C_PEAK  = "#F57F17"   # amber      = HR_Peak series
C_SPO2  = "#6A1B9A"   # purple     = SpO2
C_WAVE  = "#00838F"   # teal       = waveform

fig.suptitle(
    "rPPG System Evaluation — Physiological Signal Dashboard\n"
    f"Recording: mohanesh_period_01  |  GT median: {GT_SCALAR:.1f} BPM  |  "
    f"rPPG: {'%.2f BPM' % RPPG_CONSENSUS_HR if RPPG_CONSENSUS_HR else 'not set'}",
    fontsize=13, fontweight="bold"
)

# --- Plot 1: Ground truth HR over time ---
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(frame_times, gt_trim_series, color=C_GT, linewidth=1.5,
         label="Oximeter HR (ground truth)")
ax1.axhline(GT_SCALAR, color=C_GT, linestyle="--", alpha=0.6,
            label=f"GT median = {GT_SCALAR:.1f} BPM")
if RPPG_CONSENSUS_HR is not None:
    ax1.axhline(RPPG_CONSENSUS_HR, color=C_RPPG, linestyle="--", linewidth=2,
                label=f"rPPG = {RPPG_CONSENSUS_HR:.1f} BPM")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Heart Rate (BPM)")
ax1.set_title("Ground Truth HR vs rPPG Estimate Over Time")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# --- Plot 2: HR_FFT convergence ---
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(frame_times, hr_fft_series, color=C_FFT, linewidth=1,
         alpha=0.8, label="Oximeter HR_FFT")
ax2.plot(frame_times, gt_trim_series, color=C_GT, linewidth=1.5,
         alpha=0.6, label="GT Hardware HR")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("BPM")
ax2.set_title("Oximeter HR_FFT Convergence")
ax2.legend(fontsize=7)
ax2.grid(alpha=0.3)

# --- Plot 3: Raw PPG waveform ---
ax3 = fig.add_subplot(gs[1, :2])
wf_time = np.arange(len(waveform)) / fs_ppg
ax3.plot(wf_time, waveform, color=C_WAVE, linewidth=0.9, alpha=0.85,
         label="Raw PPG waveform")
if len(peaks_ppg) > 0:
    ax3.plot(peaks_ppg / fs_ppg, waveform[peaks_ppg], "x",
             color=C_RPPG, markersize=5, label="Detected peaks")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Waveform amplitude")
ax3.set_title("CMS50D PPG Waveform with Peak Detection")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# --- Plot 4: SpO2 over time ---
ax4 = fig.add_subplot(gs[1, 2])
spo2_time = np.arange(len(spo2_series)) / fs_ppg
ax4.plot(spo2_time, spo2_series, color=C_SPO2, linewidth=1.5)
ax4.axhline(95.0, color="orange", linestyle="--", linewidth=1,
            label="Normal threshold (95%)")
ax4.axhline(92.0, color="red", linestyle="--", linewidth=1,
            label="Low threshold (92%)")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("SpO2 (%)")
ax4.set_title("SpO2 Over Time")
ax4.set_ylim(85, 102)
ax4.legend(fontsize=7)
ax4.grid(alpha=0.3)

# --- Plot 5: Bland-Altman (multi-video, or placeholder) ---
ax5 = fig.add_subplot(gs[2, 0])
if len(rppg_estimates) >= 2:
    ax5.scatter(ba_means, ba_diffs, color=C_RPPG, alpha=0.7, s=40, zorder=3)
    ax5.axhline(ba_bias,   color="black",  linestyle="-",  linewidth=1.5,
                label=f"Bias = {ba_bias:+.2f}")
    ax5.axhline(loa_upper, color="red",    linestyle="--", linewidth=1,
                label=f"+1.96 SD = {loa_upper:+.2f}")
    ax5.axhline(loa_lower, color="red",    linestyle="--", linewidth=1,
                label=f"-1.96 SD = {loa_lower:+.2f}")
    ax5.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax5.set_xlabel("Mean of rPPG & GT (BPM)")
    ax5.set_ylabel("rPPG − GT (BPM)")
    ax5.set_title("Bland-Altman Plot")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.3)
elif RPPG_CONSENSUS_HR is not None:
    # Single-video Bland-Altman placeholder
    ax5.scatter([np.mean([RPPG_CONSENSUS_HR, GT_SCALAR])],
                [RPPG_CONSENSUS_HR - GT_SCALAR],
                color=C_RPPG, s=80, zorder=3, label="This video")
    ax5.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax5.axhline(RPPG_CONSENSUS_HR - GT_SCALAR, color=C_RPPG,
                linestyle="--", linewidth=1,
                label=f"Error = {RPPG_CONSENSUS_HR - GT_SCALAR:+.2f} BPM")
    ax5.set_xlabel("Mean of rPPG & GT (BPM)")
    ax5.set_ylabel("rPPG − GT (BPM)")
    ax5.set_title("Bland-Altman (single video)")
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
else:
    ax5.text(0.5, 0.5, "Set RPPG_CONSENSUS_HR\nand run multi-video\nto see Bland-Altman",
             ha="center", va="center", transform=ax5.transAxes, fontsize=9,
             color="gray")
    ax5.set_title("Bland-Altman Plot")

# --- Plot 6: Correlation scatter (multi-video or placeholder) ---
ax6 = fig.add_subplot(gs[2, 1])
if len(rppg_estimates) >= 2:
    ax6.scatter(gt_references, rppg_estimates, color=C_RPPG, alpha=0.7, s=40)
    lim_min = min(gt_references.min(), rppg_estimates.min()) - 2
    lim_max = max(gt_references.max(), rppg_estimates.max()) + 2
    ax6.plot([lim_min, lim_max], [lim_min, lim_max], "k--",
             linewidth=1, label="Identity (perfect)")
    # Regression line
    slope, intercept, r_val, *_ = stats.linregress(gt_references, rppg_estimates)
    x_fit = np.array([lim_min, lim_max])
    ax6.plot(x_fit, slope * x_fit + intercept, color=C_RPPG, linewidth=1.5,
             label=f"Fit: r={r_val:.3f}")
    ax6.set_xlim(lim_min, lim_max)
    ax6.set_ylim(lim_min, lim_max)
    ax6.set_xlabel("Ground Truth (BPM)")
    ax6.set_ylabel("rPPG Estimate (BPM)")
    ax6.set_title("Correlation: rPPG vs GT")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
elif RPPG_CONSENSUS_HR is not None:
    ax6.scatter([GT_SCALAR], [RPPG_CONSENSUS_HR], color=C_RPPG, s=80)
    lim = [min(GT_SCALAR, RPPG_CONSENSUS_HR) - 5,
           max(GT_SCALAR, RPPG_CONSENSUS_HR) + 5]
    ax6.plot(lim, lim, "k--", linewidth=1, label="Identity")
    ax6.set_xlabel("Ground Truth (BPM)")
    ax6.set_ylabel("rPPG Estimate (BPM)")
    ax6.set_title("Correlation (single video)")
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
else:
    ax6.text(0.5, 0.5, "Needs multi-video data",
             ha="center", va="center", transform=ax6.transAxes, fontsize=9,
             color="gray")
    ax6.set_title("Correlation Scatter")

# --- Plot 7: Pulse amplitude variability ---
ax7 = fig.add_subplot(gs[2, 2])
if len(peaks_ppg) >= 4:
    peak_frame_times = peaks_ppg / fs_ppg
    ax7.plot(peak_frame_times, peak_amplitudes, "o-",
             color=C_WAVE, markersize=4, linewidth=1, label="Pulse amplitude")
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Amplitude (a.u.)")
    ax7.set_title(f"Pulse Amplitude Variability\nCV = {pav_cv:.1f}%")
    ax7.legend(fontsize=8)
    ax7.grid(alpha=0.3)
else:
    ax7.text(0.5, 0.5, "Not enough peaks\ndetected",
             ha="center", va="center", transform=ax7.transAxes, fontsize=9,
             color="gray")
    ax7.set_title("Pulse Amplitude Variability")

plt.savefig("rppg_evaluation_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Dashboard saved → rppg_evaluation_dashboard.png")


# =============================================================
# SECTION 7: PRINTABLE SUMMARY TABLE
# =============================================================

print("\n" + "=" * 60)
print("  EVALUATION SUMMARY TABLE")
print("=" * 60)

rows = [
    ("Ground Truth (oximeter median)",   f"{GT_SCALAR:.2f} BPM"),
    ("Ground Truth (oximeter mean)",     f"{GT_MEAN:.2f} BPM"),
    ("Ground Truth std",                 f"{GT_STD:.2f} BPM"),
    ("rPPG Consensus HR",
     f"{RPPG_CONSENSUS_HR:.2f} BPM" if RPPG_CONSENSUS_HR else "not set"),
]

if RPPG_CONSENSUS_HR:
    rows += [
        ("Signed error",     f"{RPPG_CONSENSUS_HR - GT_SCALAR:+.2f} BPM"),
        ("MAE",              f"{abs(RPPG_CONSENSUS_HR - GT_SCALAR):.2f} BPM"),
        ("MAPE",             f"{abs(RPPG_CONSENSUS_HR - GT_SCALAR)/GT_SCALAR*100:.2f} %"),
        ("Clinical pass (±5 BPM)", "YES ✓" if clinical_pass else "NO ✗"),
    ]

rows += [
    ("SpO2 mean",           f"{np.mean(spo2_series):.2f} %"),
    ("SpO2 range",          f"{np.min(spo2_series):.0f}–{np.max(spo2_series):.0f} %"),
    ("Perfusion index proxy", f"{pi_proxy:.2f} %"),
    ("HR from waveform",
     f"{hr_from_waveform:.2f} BPM" if len(peaks_ppg) >= 2 else "N/A"),
    ("Waveform HR vs GT",
     f"{hr_from_waveform - GT_SCALAR:+.2f} BPM" if len(peaks_ppg) >= 2 else "N/A"),
]

if len(peaks_ppg) >= 4:
    rows.append(("Pulse Amplitude CV",   f"{pav_cv:.2f} %"))

for label, value in rows:
    print(f"  {label:<38}: {value}")

print("=" * 60)