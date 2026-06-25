"""Post-recording evaluation of a live rPPG session against the CMS50D.

Ported from evaluate_live_recording_metrics() in the live scripts (logic
unchanged; globals replaced by parameters, display calls removed). Consumes
the live sync CSV and writes a 6-panel evaluation dashboard PNG plus two
summary CSVs into out_dir, returning a JSON-safe summary dict.

Most metrics require the CMS50D ground-truth columns; when the oximeter
was not connected those columns are empty and the corresponding metrics
come back as None, while the live-only panels still render.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                # Console/stream encoding (e.g. Windows cp1252) can't represent
                # some characters we print (±, —, ✓). Degrade gracefully on that
                # stream instead of aborting the whole run.
                enc = getattr(s, "encoding", None) or "ascii"
                s.write(data.encode(enc, errors="replace").decode(enc))

    def flush(self):
        for s in self._streams:
            s.flush()


def _jsafe(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float, int)):
        f = float(v)
        return None if np.isnan(f) else round(f, 4)
    return v


def contiguous_motion_segments(df, time_col="Elapsed_Time_sec",
                               class_col="Live_Motion_Class"):
    if class_col not in df.columns or time_col not in df.columns or len(df) == 0:
        return []
    classes = df[class_col].fillna("unknown").astype(str).values
    times = df[time_col].values.astype(float)
    segments = []
    start_idx = 0
    current_class = classes[0]
    segment_id = 1
    for i in range(1, len(df)):
        if classes[i] != current_class:
            segments.append({
                "segment_id": segment_id, "motion_class": current_class,
                "start_idx": start_idx, "end_idx": i - 1,
                "start_sec": float(times[start_idx]), "end_sec": float(times[i - 1]),
            })
            segment_id += 1
            start_idx = i
            current_class = classes[i]
    segments.append({
        "segment_id": segment_id, "motion_class": current_class,
        "start_idx": start_idx, "end_idx": len(df) - 1,
        "start_sec": float(times[start_idx]), "end_sec": float(times[-1]),
    })
    return segments


def evaluate_live_recording(csv_path, out_dir, target_fps=30.0,
                            warmup_seconds=12.0):
    """Run the post-recording evaluation. Returns a result dict with the
    summary metrics, segment rows, and generated file names."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "live_eval_log.txt"

    with open(log_path, "w", encoding="utf-8") as fh:
        with contextlib.redirect_stdout(_Tee(sys.stdout, fh)):
            result = _evaluate(csv_path, out_dir, target_fps, warmup_seconds)

    result["log"] = log_path.name
    return result


def _evaluate(csv_path, out_dir, target_fps, warmup_seconds):
    print("\n" + "=" * 60)
    print("  LIVE RECORDING METRIC EVALUATION")
    print("=" * 60)
    print("CSV:", csv_path)

    df_raw = pd.read_csv(csv_path)
    if len(df_raw) == 0:
        print("[Eval] CSV is empty. Skipping evaluation.")
        return {"available": False, "reason": "empty_csv"}

    if "Timestamp" in df_raw.columns:
        ts = pd.to_datetime(df_raw["Timestamp"], errors="coerce")
        if ts.notna().sum() >= 2:
            df_raw["Elapsed_Time_sec"] = (ts - ts.dropna().iloc[0]).dt.total_seconds()
        else:
            df_raw["Elapsed_Time_sec"] = np.arange(len(df_raw), dtype=float) / max(target_fps, 1)
    else:
        df_raw["Elapsed_Time_sec"] = np.arange(len(df_raw), dtype=float) / max(target_fps, 1)

    col_gt_hr = "CMS_Pulse_Rate_Hardware" if "CMS_Pulse_Rate_Hardware" in df_raw.columns else "Pulse_Rate_Hardware"
    col_wave = "CMS_Waveform" if "CMS_Waveform" in df_raw.columns else "Waveform"
    col_spo2 = "CMS_SpO2" if "CMS_SpO2" in df_raw.columns else "SpO2"
    col_cms_fft = "CMS_HR_FFT" if "CMS_HR_FFT" in df_raw.columns else "HR_FFT"
    col_cms_peak = "CMS_HR_Peak" if "CMS_HR_Peak" in df_raw.columns else "HR_Peak"
    col_rppg_hr = "Live_rPPG_HR"
    col_live_rr = "Live_RR"

    if col_rppg_hr not in df_raw.columns:
        print("[Eval] Missing required column:", col_rppg_hr)
        return {"available": False, "reason": "missing_rppg_hr"}

    numeric_cols = [
        col_gt_hr, col_wave, col_spo2, col_cms_fft, col_cms_peak,
        col_rppg_hr, col_live_rr,
        "Live_Green_FFT", "Live_POS_FFT", "Live_Green_Peak", "Live_POS_Peak",
        "Live_Green_Reliability", "Live_POS_Reliability", "Live_FS",
    ]
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_eval = df_raw[df_raw["Elapsed_Time_sec"] >= warmup_seconds].copy()
    if len(df_eval) == 0:
        df_eval = df_raw.copy()

    has_gt = col_gt_hr in df_eval.columns
    df_gt = df_eval[df_eval[col_gt_hr].notna()].copy() if has_gt else df_eval.iloc[0:0]
    df_pair = (
        df_eval[df_eval[col_gt_hr].notna() & df_eval[col_rppg_hr].notna()].copy()
        if has_gt else df_eval.iloc[0:0]
    )

    print(f"Rows total        : {len(df_raw)}")
    print(f"Rows after warmup : {len(df_eval)}")
    print(f"Paired HR rows    : {len(df_pair)}")

    rppg_valid = df_eval[col_rppg_hr].dropna().values.astype(float)
    rr_valid = (df_eval[col_live_rr].dropna().values.astype(float)
                if col_live_rr in df_eval.columns else np.array([]))
    RPPG_CONSENSUS_HR = float(np.median(rppg_valid)) if len(rppg_valid) else None
    RPPG_MEAN_HR = float(np.mean(rppg_valid)) if len(rppg_valid) else None
    RPPG_LAST_HR = float(rppg_valid[-1]) if len(rppg_valid) else None
    RR_FINAL = float(np.median(rr_valid)) if len(rr_valid) else None

    GT_SCALAR = GT_MEAN = GT_STD = GT_STABLE = None
    gt_series = np.array([])
    if len(df_gt) > 0:
        gt_series = df_gt[col_gt_hr].values.astype(float)
        GT_SCALAR = float(np.median(gt_series))
        GT_MEAN = float(np.mean(gt_series))
        GT_STD = float(np.std(gt_series, ddof=1)) if len(gt_series) >= 2 else 0.0
        GT_STABLE = float(np.mean(gt_series[-30:]))
        print(f"\nGT median HR      : {GT_SCALAR:.2f} BPM")
        print(f"GT mean/std       : {GT_MEAN:.2f} / {GT_STD:.2f} BPM")
        print(f"GT stable end     : {GT_STABLE:.2f} BPM")
    else:
        print("\n[Eval] No CMS hardware HR (oximeter not connected) — "
              "live-only panels will render; GT metrics are N/A.")

    print(f"Live rPPG median  : {RPPG_CONSENSUS_HR:.2f} BPM" if RPPG_CONSENSUS_HR is not None else "Live rPPG median  : N/A")
    print(f"Live rPPG mean    : {RPPG_MEAN_HR:.2f} BPM" if RPPG_MEAN_HR is not None else "Live rPPG mean    : N/A")
    print(f"Live rPPG last    : {RPPG_LAST_HR:.2f} BPM" if RPPG_LAST_HR is not None else "Live rPPG last    : N/A")
    print(f"Live RR median    : {RR_FINAL:.1f} br/min" if RR_FINAL is not None else "Live RR median    : N/A")

    error = abs_error = pct_error = clinical_pass = None
    if RPPG_CONSENSUS_HR is not None and GT_SCALAR is not None:
        error = RPPG_CONSENSUS_HR - GT_SCALAR
        abs_error = abs(error)
        pct_error = abs_error / (GT_SCALAR + 1e-8) * 100.0
        clinical_pass = abs_error <= max(5.0, 0.05 * GT_SCALAR)
        print(f"\nSigned error      : {error:+.2f} BPM")
        print(f"MAE               : {abs_error:.2f} BPM")
        print(f"MAPE              : {pct_error:.2f} %")
        print(f"Clinical check    : {'PASS' if clinical_pass else 'FAIL'}")

    point_mae = point_rmse = point_mape = point_corr = None
    if len(df_pair) >= 2:
        p_gt = df_pair[col_gt_hr].values.astype(float)
        p_rppg = df_pair[col_rppg_hr].values.astype(float)
        pe = p_rppg - p_gt
        point_mae = float(np.mean(np.abs(pe)))
        point_rmse = float(np.sqrt(np.mean(pe ** 2)))
        point_mape = float(np.mean(np.abs(pe) / (p_gt + 1e-8)) * 100.0)
        if len(df_pair) >= 3:
            try:
                point_corr = float(stats.pearsonr(p_gt, p_rppg)[0])
            except Exception:
                point_corr = None
        print("\nPoint-by-point live metrics:")
        print(f"MAE               : {point_mae:.2f} BPM")
        print(f"RMSE              : {point_rmse:.2f} BPM")
        print(f"MAPE              : {point_mape:.2f} %")
        print(f"Pearson r         : {point_corr:.3f}" if point_corr is not None else "Pearson r         : N/A")

    # Segment-level summary by motion class.
    segment_rows = []
    for seg in contiguous_motion_segments(df_eval):
        seg_df = df_eval.iloc[seg["start_idx"]:seg["end_idx"] + 1]
        seg_gt = seg_df[col_gt_hr].dropna().values.astype(float) if has_gt else np.array([])
        seg_hr = seg_df[col_rppg_hr].dropna().values.astype(float)
        seg_rr = (seg_df[col_live_rr].dropna().values.astype(float)
                  if col_live_rr in seg_df.columns else np.array([]))
        seg_gt_med = float(np.median(seg_gt)) if len(seg_gt) else None
        seg_hr_med = float(np.median(seg_hr)) if len(seg_hr) else None
        seg_rr_med = float(np.median(seg_rr)) if len(seg_rr) else None
        seg_err = (seg_hr_med - seg_gt_med) if seg_hr_med is not None and seg_gt_med is not None else None
        segment_rows.append({
            "Segment_ID": seg["segment_id"], "Motion_Class": seg["motion_class"],
            "Start_sec": seg["start_sec"], "End_sec": seg["end_sec"],
            "Duration_sec": seg["end_sec"] - seg["start_sec"],
            "GT_HR_Median": seg_gt_med, "Live_rPPG_HR_Median": seg_hr_med,
            "Signed_Error": seg_err,
            "Abs_Error": abs(seg_err) if seg_err is not None else None,
            "Live_RR_Median": seg_rr_med,
            "HR_Sample_Count": int(len(seg_hr)), "GT_Sample_Count": int(len(seg_gt)),
        })
    segment_df = pd.DataFrame(segment_rows)
    segment_csv = out_dir / "live_motion_segment_metric_summary.csv"
    segment_df.to_csv(segment_csv, index=False)

    print("\nMotion-segment metric summary:")
    if len(segment_df) == 0:
        print("No motion segments found.")
    else:
        for _, row in segment_df.iterrows():
            hr_txt = "N/A" if pd.isna(row["Live_rPPG_HR_Median"]) else f"{row['Live_rPPG_HR_Median']:.2f} BPM"
            gt_txt = "N/A" if pd.isna(row["GT_HR_Median"]) else f"{row['GT_HR_Median']:.2f} BPM"
            err_txt = "N/A" if pd.isna(row["Signed_Error"]) else f"{row['Signed_Error']:+.2f} BPM"
            print(f"  Segment {int(row['Segment_ID']):02d} | {row['Motion_Class']:<11} | "
                  f"{row['Start_sec']:.1f}-{row['End_sec']:.1f}s | "
                  f"rPPG={hr_txt} | GT={gt_txt} | error={err_txt}")

    fs_ppg = 30.0
    HAS_CMS_FFT = col_cms_fft in df_raw.columns and df_raw[col_cms_fft].notna().sum() > 0
    HAS_WAVEFORM = col_wave in df_raw.columns and df_raw[col_wave].notna().sum() > 0
    HAS_SPO2 = col_spo2 in df_raw.columns and df_raw[col_spo2].notna().sum() > 0

    stable_fft = int_mae = None
    if HAS_CMS_FFT and has_gt:
        clean = df_eval[[col_gt_hr, col_cms_fft]].dropna()
        if len(clean) > 0:
            hr_fft_s = clean[col_cms_fft].values.astype(float)
            gt_for_fft = clean[col_gt_hr].values.astype(float)
            stable_fft = float(np.mean(hr_fft_s[-30:]))
            int_mae = float(np.mean(np.abs(hr_fft_s - gt_for_fft)))
            print(f"\nCMS HR_FFT stable : {stable_fft:.2f} BPM")
            print(f"CMS HR_FFT MAE    : {int_mae:.3f} BPM")

    hr_wf = pi_proxy = pav_cv = None
    peaks_ppg = np.array([])
    waveform = None
    if HAS_WAVEFORM:
        waveform = df_raw[col_wave].dropna().values.astype(float)
        if len(waveform) >= 10:
            wf_det = waveform - np.mean(waveform)
            peaks_ppg, _ = find_peaks(wf_det, distance=int(fs_ppg * 60.0 / 180.0),
                                      prominence=0.1 * np.std(wf_det))
            if len(peaks_ppg) >= 4:
                hr_wf = float(np.median(60.0 / np.diff(peaks_ppg / fs_ppg)))
                pi_proxy = float(np.std(wf_det) / (np.mean(waveform) + 1e-8) * 100.0)
                pav_cv = float(np.std(wf_det[peaks_ppg]) /
                               (np.mean(np.abs(wf_det[peaks_ppg])) + 1e-8) * 100.0)
                print(f"\nHR from waveform  : {hr_wf:.2f} BPM")
                if GT_SCALAR is not None:
                    print(f"Waveform HR err   : {hr_wf - GT_SCALAR:+.2f} BPM")
                print(f"Perfusion Index   : {pi_proxy:.2f} %")
                print(f"Pulse Amp CV      : {pav_cv:.2f} %")

    spo2 = None
    spo2_median = spo2_min = spo2_max = None
    if HAS_SPO2:
        spo2 = df_raw[col_spo2].dropna().values.astype(float)
        if len(spo2) > 0:
            spo2_median = float(np.median(spo2))
            spo2_min = float(spo2.min())
            spo2_max = float(spo2.max())
            print(f"\nSpO2 median       : {spo2_median:.1f} %")
            print(f"SpO2 mean/std     : {np.mean(spo2):.2f} / {np.std(spo2):.2f} %")
            print(f"SpO2 range        : {spo2_min:.0f}-{spo2_max:.0f} %")

    dashboard_name = _render_dashboard(
        out_dir, df_raw, df_eval, segment_df,
        col_gt_hr, col_rppg_hr, col_cms_fft, col_wave,
        GT_SCALAR, RPPG_CONSENSUS_HR, RR_FINAL,
        HAS_CMS_FFT, HAS_WAVEFORM, has_gt,
        waveform, peaks_ppg, fs_ppg,
    )

    summary = {
        "GT_Median_HR": GT_SCALAR, "GT_Mean_HR": GT_MEAN, "GT_Std_HR": GT_STD,
        "Live_rPPG_HR_Median": RPPG_CONSENSUS_HR, "Live_rPPG_HR_Mean": RPPG_MEAN_HR,
        "Live_rPPG_HR_Last": RPPG_LAST_HR, "Signed_Error": error, "MAE": abs_error,
        "MAPE_percent": pct_error, "Clinical_Pass": clinical_pass,
        "Pointwise_MAE": point_mae, "Pointwise_RMSE": point_rmse,
        "Pointwise_MAPE_percent": point_mape, "Pointwise_Pearson_r": point_corr,
        "Live_RR_Median": RR_FINAL, "CMS_HR_FFT_Stable": stable_fft,
        "CMS_HR_FFT_MAE": int_mae, "HR_from_CMS_Waveform": hr_wf,
        "Perfusion_Index_Proxy": pi_proxy, "Pulse_Amplitude_CV": pav_cv,
        "SpO2_Median": spo2_median,
        "SpO2_Mean": float(np.mean(spo2)) if spo2 is not None and len(spo2) else None,
        "SpO2_Std": float(np.std(spo2)) if spo2 is not None and len(spo2) else None,
        "SpO2_Min": spo2_min, "SpO2_Max": spo2_max,
        "Valid_Live_HR_Count": int(len(rppg_valid)),
        "Valid_GT_Count": int(len(gt_series)), "Paired_Count": int(len(df_pair)),
    }
    summary_csv = out_dir / "live_rppg_metric_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\n" + "=" * 60)
    print("  COMPLETE LIVE EVALUATION SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:<28}: {v}")
    print("=" * 60)

    return {
        "available": True,
        "has_ground_truth": GT_SCALAR is not None,
        "summary": {k: _jsafe(v) for k, v in summary.items()},
        "segments": [{k: _jsafe(v) for k, v in r.items()} for r in segment_rows],
        "dashboard": dashboard_name,
        "segment_summary_csv": segment_csv.name,
        "metric_summary_csv": summary_csv.name,
    }


def _render_dashboard(out_dir, df_raw, df_eval, segment_df,
                      col_gt_hr, col_rppg_hr, col_cms_fft, col_wave,
                      GT_SCALAR, RPPG_CONSENSUS_HR, RR_FINAL,
                      HAS_CMS_FFT, HAS_WAVEFORM, has_gt,
                      waveform, peaks_ppg, fs_ppg):
    C_GT, C_RPPG, C_FFT, C_WAVE = "#1565C0", "#C62828", "#2E7D32", "#00838F"
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30)
    fig.suptitle(
        "Live rPPG Evaluation\n"
        f"GT: {('%.1f BPM' % GT_SCALAR) if GT_SCALAR is not None else 'N/A'} | "
        f"Live rPPG median: {('%.1f BPM' % RPPG_CONSENSUS_HR) if RPPG_CONSENSUS_HR is not None else 'N/A'} | "
        f"RR: {('%.1f br/min' % RR_FINAL) if RR_FINAL is not None else 'N/A'}",
        fontsize=12, fontweight="bold",
    )

    ax1 = fig.add_subplot(gs[0, 0])
    if has_gt:
        ax1.plot(df_raw["Elapsed_Time_sec"], df_raw[col_gt_hr], color=C_GT, linewidth=1.3, label="CMS GT HR")
    ax1.plot(df_raw["Elapsed_Time_sec"], df_raw[col_rppg_hr], color=C_RPPG, linewidth=1.8, label="Live rPPG HR")
    if GT_SCALAR is not None:
        ax1.axhline(GT_SCALAR, color=C_GT, linestyle="--", alpha=0.5, label=f"GT median={GT_SCALAR:.1f}")
    if RPPG_CONSENSUS_HR is not None:
        ax1.axhline(RPPG_CONSENSUS_HR, color=C_RPPG, linestyle="--", alpha=0.7, label=f"rPPG median={RPPG_CONSENSUS_HR:.1f}")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("HR (BPM)")
    ax1.set_title("CMS GT HR vs Live rPPG HR"); ax1.legend(fontsize=7); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    if HAS_CMS_FFT and has_gt:
        ax2.plot(df_raw["Elapsed_Time_sec"], df_raw[col_cms_fft], color=C_FFT, linewidth=1.0, alpha=0.8, label="CMS HR_FFT")
        ax2.plot(df_raw["Elapsed_Time_sec"], df_raw[col_gt_hr], color=C_GT, linewidth=1.3, alpha=0.5, label="CMS Hardware HR")
        ax2.set_title("CMS HR_FFT Convergence"); ax2.legend(fontsize=7)
    elif HAS_WAVEFORM and waveform is not None:
        wf_time = np.arange(len(waveform)) / fs_ppg
        ax2.plot(wf_time, waveform, color=C_WAVE, linewidth=0.9, alpha=0.8, label="CMS waveform")
        if len(peaks_ppg) >= 4:
            ax2.plot(peaks_ppg / fs_ppg, waveform[peaks_ppg], "x", color=C_RPPG, markersize=5, label="Peaks")
        ax2.set_title("CMS PPG Waveform"); ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "No CMS waveform/FFT", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("BPM / amplitude"); ax2.grid(alpha=0.3)

    seg_valid = (segment_df.dropna(subset=["GT_HR_Median", "Live_rPPG_HR_Median"])
                 if len(segment_df) else pd.DataFrame())

    ax3 = fig.add_subplot(gs[1, 0])
    if len(seg_valid) >= 2:
        ba_means = (seg_valid["GT_HR_Median"].values + seg_valid["Live_rPPG_HR_Median"].values) / 2.0
        ba_diffs = seg_valid["Live_rPPG_HR_Median"].values - seg_valid["GT_HR_Median"].values
        ba_bias = float(np.mean(ba_diffs)); ba_std = float(np.std(ba_diffs, ddof=1))
        ax3.scatter(ba_means, ba_diffs, color=C_RPPG, alpha=0.75, s=55)
        ax3.axhline(ba_bias, color="black", linestyle="-", label=f"Bias {ba_bias:+.2f}")
        ax3.axhline(ba_bias + 1.96 * ba_std, color="red", linestyle="--", label=f"+LoA {ba_bias + 1.96 * ba_std:+.2f}")
        ax3.axhline(ba_bias - 1.96 * ba_std, color="red", linestyle="--", label=f"-LoA {ba_bias - 1.96 * ba_std:+.2f}")
        for i, (xv, yv) in enumerate(zip(ba_means, ba_diffs)):
            ax3.annotate(f"S{i+1}", (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax3.legend(fontsize=7); ax3.set_title("Bland-Altman by motion segment")
    elif RPPG_CONSENSUS_HR is not None and GT_SCALAR is not None:
        ax3.scatter([np.mean([RPPG_CONSENSUS_HR, GT_SCALAR])], [RPPG_CONSENSUS_HR - GT_SCALAR], color=C_RPPG, s=80)
        ax3.axhline(0, color="gray", linestyle=":"); ax3.set_title("Bland-Altman overall")
    else:
        ax3.text(0.5, 0.5, "No GT for Bland-Altman", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Mean HR (BPM)"); ax3.set_ylabel("rPPG - GT (BPM)"); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    if len(seg_valid) > 0:
        errs_bar = seg_valid["Live_rPPG_HR_Median"].values - seg_valid["GT_HR_Median"].values
        labels = [f"S{int(s)}\n{m}" for s, m in zip(seg_valid["Segment_ID"], seg_valid["Motion_Class"])]
        colors_b = ["#C62828" if e > 0 else "#1565C0" for e in errs_bar]
        ax4.bar(range(len(errs_bar)), errs_bar, color=colors_b, alpha=0.8, edgecolor="white")
        ax4.axhline(0, color="black", linewidth=1)
        ax4.axhline(5, color="orange", linestyle="--", linewidth=1, label="±5 BPM")
        ax4.axhline(-5, color="orange", linestyle="--", linewidth=1)
        ax4.set_xticks(range(len(errs_bar))); ax4.set_xticklabels(labels, fontsize=7)
        ax4.set_ylabel("Error (BPM)"); ax4.set_title("Per-motion-segment error"); ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, "No segment HR vs GT", ha="center", va="center", transform=ax4.transAxes)
    ax4.grid(alpha=0.3)

    name = "live_rppg_evaluation_dashboard.png"
    plt.savefig(out_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDashboard saved -> {name}")
    return name
