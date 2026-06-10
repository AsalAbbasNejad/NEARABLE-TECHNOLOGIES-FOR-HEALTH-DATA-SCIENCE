# =========================
# Respiratory Rate Estimation from Tachogram
# Append this block after the peak detection visualization
# in your existing rPPG pipeline.
#
# Method: HRV-Derived Respiration (HDR)
# - Extracts R-R intervals from POS peaks
# - Resamples to uniform time grid (cubic spline)
# - Applies PSD (Welch method) in the respiratory band
# - Refines peak with parabolic interpolation
# - Cross-validates with bandpass envelope method
# =========================

from scipy.interpolate import CubicSpline
from scipy.signal import welch, butter, filtfilt, hilbert


# =========================
# Config
# =========================
RR_MIN_BPM     = 10      # Minimum plausible respiratory rate (breaths/min)
RR_MAX_BPM     = 40      # Maximum plausible respiratory rate (breaths/min)
RR_RESAMPLE_FS = 4.0     # Hz — resample tachogram to this uniform rate
                         # 4 Hz gives good resolution for 0.1–0.7 Hz resp band

RR_MIN_PEAKS_REQUIRED = 10   # Need at least this many R-R intervals to proceed
RR_MIN_DURATION_SEC   = 20.0 # Need at least this many seconds of signal


# =========================
# Step 1: Extract R-R intervals from POS peaks
# pos_peaks and relative_time already exist from your pipeline
# =========================

print("\n" + "="*50)
print("  RESPIRATORY RATE FROM TACHOGRAM")
print("="*50)

if len(pos_peaks) < RR_MIN_PEAKS_REQUIRED:
    print(f"[RR] Not enough peaks detected ({len(pos_peaks)})."
          f" Need at least {RR_MIN_PEAKS_REQUIRED}. Skipping RR estimation.")
else:
    # Peak timestamps (seconds)
    peak_times = relative_time[pos_peaks]

    # R-R intervals: time difference between consecutive peaks (in seconds)
    rr_intervals = np.diff(peak_times)          # unit: seconds

    # Timestamps for each R-R interval (use midpoint between consecutive peaks)
    rr_times = 0.5 * (peak_times[:-1] + peak_times[1:])

    duration_covered = rr_times[-1] - rr_times[0]

    print(f"[RR] Detected peaks     : {len(pos_peaks)}")
    print(f"[RR] R-R intervals      : {len(rr_intervals)}")
    print(f"[RR] Tachogram duration : {duration_covered:.1f} s")
    print(f"[RR] Mean R-R interval  : {np.mean(rr_intervals)*1000:.1f} ms  "
          f"({60.0/np.mean(rr_intervals):.1f} BPM)")

    if duration_covered < RR_MIN_DURATION_SEC:
        print(f"[RR] Duration too short ({duration_covered:.1f}s < {RR_MIN_DURATION_SEC}s)."
              " RR estimate may be unreliable.")

    # =========================
    # Step 2: Resample tachogram to uniform time grid (cubic spline)
    # R-R intervals are unevenly spaced → must interpolate before spectral analysis
    # =========================

    t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / RR_RESAMPLE_FS)

    cs = CubicSpline(rr_times, rr_intervals)
    rr_resampled = cs(t_uniform)

    # Detrend to remove slow drift (posture, mean HR shift)
    rr_resampled = detrend(rr_resampled)

    # =========================
    # Step 3a: PSD via Welch method
    # Window chosen to give ~0.01 Hz frequency resolution
    # Respiratory band: RR_MIN_BPM/60 to RR_MAX_BPM/60 Hz
    # =========================

    nperseg = min(len(rr_resampled), int(RR_RESAMPLE_FS * 32))  # 32-second windows
    nperseg = max(nperseg, 16)  # Safety floor

    freqs_welch, psd = welch(
        rr_resampled,
        fs=RR_RESAMPLE_FS,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling='density'
    )

    # Convert respiratory band limits to Hz
    rr_low_hz  = RR_MIN_BPM / 60.0
    rr_high_hz = RR_MAX_BPM / 60.0

    # Mask to respiratory band only
    resp_mask = (freqs_welch >= rr_low_hz) & (freqs_welch <= rr_high_hz)

    if not np.any(resp_mask):
        print("[RR] No frequency content in respiratory band. Check signal length.")
    else:
        resp_freqs = freqs_welch[resp_mask]
        resp_psd   = psd[resp_mask]

        # Find dominant peak in respiratory PSD
        peak_idx_welch = int(np.argmax(resp_psd))

        # Refine with parabolic interpolation (reuse your existing function)
        delta = parabolic_interpolation(resp_psd, peak_idx_welch)
        freq_step = resp_freqs[1] - resp_freqs[0] if len(resp_freqs) > 1 else 0.0

        rr_freq_welch    = float(resp_freqs[peak_idx_welch]) + delta * freq_step
        rr_welch_bpm     = rr_freq_welch * 60.0
        rr_welch_bpm     = float(np.clip(rr_welch_bpm, RR_MIN_BPM, RR_MAX_BPM))

        # Peak sharpness: ratio of peak PSD to mean PSD in band
        # Higher = cleaner respiratory signal
        peak_sharpness = float(resp_psd[peak_idx_welch] / (np.mean(resp_psd) + 1e-10))

        print(f"\n[RR] Welch PSD dominant frequency : {rr_freq_welch:.4f} Hz")
        print(f"[RR] Respiratory Rate (Welch)     : {rr_welch_bpm:.1f} breaths/min")
        print(f"[RR] PSD peak sharpness           : {peak_sharpness:.2f}  "
              f"(>3 = clean, <2 = noisy/unreliable)")

        # =========================
        # Step 3b: Cross-validate with bandpass envelope method
        # Independent method — bandpass tachogram in resp band, extract envelope
        # =========================

        try:
            b_resp, a_resp = butter(
                3,
                [rr_low_hz, rr_high_hz],
                btype='bandpass',
                fs=RR_RESAMPLE_FS
            )

            padlen = 3 * max(len(a_resp), len(b_resp))

            if len(rr_resampled) > padlen:
                rr_bandpassed = filtfilt(b_resp, a_resp, rr_resampled)

                # Instantaneous frequency via Hilbert transform
                analytic_signal  = hilbert(rr_bandpassed)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_freq  = np.diff(instantaneous_phase) / (2.0 * np.pi / RR_RESAMPLE_FS)

                # Median instantaneous frequency → respiratory rate
                valid_freq_mask = (
                    (instantaneous_freq >= rr_low_hz) &
                    (instantaneous_freq <= rr_high_hz)
                )

                if np.sum(valid_freq_mask) > 5:
                    rr_hilbert_hz  = float(np.median(instantaneous_freq[valid_freq_mask]))
                    rr_hilbert_bpm = rr_hilbert_hz * 60.0
                    rr_hilbert_bpm = float(np.clip(rr_hilbert_bpm, RR_MIN_BPM, RR_MAX_BPM))

                    print(f"[RR] Respiratory Rate (Hilbert)   : {rr_hilbert_bpm:.1f} breaths/min")

                    # Agreement check between two methods
                    method_diff = abs(rr_welch_bpm - rr_hilbert_bpm)
                    print(f"[RR] Method agreement diff        : {method_diff:.1f} breaths/min  "
                          f"({'GOOD' if method_diff <= 3.0 else 'MODERATE' if method_diff <= 6.0 else 'POOR'})")

                    # Final fused estimate: weighted average
                    # Welch carries more weight (more robust for short signals)
                    if method_diff <= 6.0:
                        rr_final = 0.65 * rr_welch_bpm + 0.35 * rr_hilbert_bpm
                    else:
                        # Methods disagree too much — fall back to Welch alone
                        rr_final = rr_welch_bpm
                        print("[RR] Methods disagree — using Welch estimate only.")
                else:
                    rr_final = rr_welch_bpm
                    print("[RR] Hilbert method: insufficient valid frames, using Welch only.")
            else:
                rr_final = rr_welch_bpm
                print("[RR] Signal too short for bandpass cross-validation.")

        except Exception as e:
            rr_final = rr_welch_bpm
            print(f"[RR] Hilbert method failed ({e}), using Welch only.")

        print(f"\n[RR] >>> FINAL RESPIRATORY RATE : {rr_final:.1f} breaths/min <<<")

        # Qualitative confidence note
        if peak_sharpness >= 3.0:
            confidence = "HIGH — strong periodic respiratory modulation in tachogram"
        elif peak_sharpness >= 2.0:
            confidence = "MODERATE — RSA present but some noise"
        else:
            confidence = "LOW — weak RSA signal; result may be unreliable"
        print(f"[RR] Confidence                   : {confidence}")

        # =========================
        # Plot 1: Resampled tachogram
        # =========================
        plt.figure(figsize=(14, 4))
        plt.plot(rr_times, rr_intervals * 1000, alpha=0.5, label="R-R intervals (raw)")
        plt.plot(t_uniform, cs(t_uniform) * 1000, linewidth=1.5, label="Cubic spline resampled")
        plt.xlabel("Time (s)")
        plt.ylabel("R-R Interval (ms)")
        plt.title("Tachogram: R-R Intervals Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # =========================
        # Plot 2: PSD in respiratory band
        # =========================
        plt.figure(figsize=(12, 4))
        plt.plot(resp_freqs * 60.0, resp_psd, linewidth=2, label="PSD in respiratory band")
        plt.axvline(
            rr_welch_bpm,
            color='red',
            linestyle='--',
            label=f"Welch peak = {rr_welch_bpm:.1f} breaths/min"
        )
        if 'rr_hilbert_bpm' in dir():
            plt.axvline(
                rr_hilbert_bpm,
                color='orange',
                linestyle='-.',
                label=f"Hilbert = {rr_hilbert_bpm:.1f} breaths/min"
            )
        plt.axvline(
            rr_final,
            color='green',
            linestyle=':',
            linewidth=2,
            label=f"Final estimate = {rr_final:.1f} breaths/min"
        )
        plt.xlabel("Rate (breaths/min)")
        plt.ylabel("PSD (ms²/Hz)")
        plt.title("Power Spectral Density of Tachogram — Respiratory Band")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # =========================
        # Plot 3: Bandpass-filtered tachogram showing respiratory oscillation
        # =========================
        if len(rr_resampled) > padlen:
            plt.figure(figsize=(14, 4))
            plt.plot(t_uniform, rr_resampled, alpha=0.4, label="Detrended tachogram")
            plt.plot(t_uniform, rr_bandpassed, linewidth=2, color='red',
                     label=f"Respiratory component ({RR_MIN_BPM}–{RR_MAX_BPM} BPM band)")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude (ms)")
            plt.title("Respiratory Sinus Arrhythmia (RSA) Component in Tachogram")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        # =========================
        # Summary alongside HR
        # =========================
        print("\n" + "="*50)
        print("  COMBINED PHYSIOLOGICAL SUMMARY")
        print("="*50)
        if hr_consensus is not None:
            print(f"  Heart Rate      : {hr_consensus:.1f} BPM")
        print(f"  Respiratory Rate: {rr_final:.1f} breaths/min")
        if hr_consensus is not None:
            ratio = hr_consensus / rr_final
            print(f"  HR / RR ratio   : {ratio:.1f}  (normal resting ≈ 4–5)")
        print("="*50)