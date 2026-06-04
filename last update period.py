# ==============================
# Robust rPPG HR Estimation Code
# GREEN + POS + CHROM
# Final selection improved
# ==============================

!pip install opencv-python scipy matplotlib pandas -q

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
from google.colab import files
import os

# ==============================
# Upload video
# ==============================

uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print("Uploaded video:", video_path)

# ==============================
# Parameters
# ==============================

LOW_BPM = 40
HIGH_BPM = 140
LOW_HZ = LOW_BPM / 60
HIGH_HZ = HIGH_BPM / 60

OUTPUT_DIR = "rppg_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Helper functions
# ==============================

def bandpass_filter(signal, fs, low=LOW_HZ, high=HIGH_HZ, order=4):
    signal = np.asarray(signal, dtype=np.float64)

    if len(signal) < 20:
        return signal

    nyq = 0.5 * fs
    low = low / nyq
    high = high / nyq

    if high >= 1:
        high = 0.99

    b, a = butter(order, [low, high], btype="band")

    try:
        return filtfilt(b, a, signal)
    except:
        return signal


def normalize_signal(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def remove_outliers(x, z_thresh=3.0):
    x = np.asarray(x, dtype=np.float64)
    z = np.abs((x - np.median(x)) / (np.std(x) + 1e-8))
    x_clean = x.copy()
    bad = z > z_thresh
    x_clean[bad] = np.median(x)
    return x_clean


def fft_hr(signal, fs):
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal, fs)

    n = len(signal)
    if n < 20:
        return None, 0

    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(signal))

    mask = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    freqs_band = freqs[mask]
    mag_band = fft_mag[mask]

    if len(mag_band) == 0:
        return None, 0

    idx = np.argmax(mag_band)
    hr = freqs_band[idx] * 60

    sorted_mag = np.sort(mag_band)
    if len(sorted_mag) > 5:
        conf = mag_band[idx] / (np.mean(sorted_mag[:-1]) + 1e-8)
    else:
        conf = 0

    return hr, conf


def peak_hr(signal, fs):
    signal = normalize_signal(signal)
    signal = bandpass_filter(signal, fs)

    min_distance = int(fs * 60 / HIGH_BPM)
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=0.15)

    if len(peaks) < 3:
        return None, 0, peaks

    intervals = np.diff(peaks) / fs
    intervals = intervals[(intervals > 60/HIGH_BPM) & (intervals < 60/LOW_BPM)]

    if len(intervals) < 2:
        return None, 0, peaks

    hr_values = 60 / intervals
    hr = np.median(hr_values)
    conf = 1 / (np.std(hr_values) + 1e-8)

    return hr, conf, peaks


def pos_method(rgb, fs, window_sec=1.6):
    rgb = np.asarray(rgb, dtype=np.float64)
    n = rgb.shape[0]
    win = int(window_sec * fs)

    if n < win:
        return np.zeros(n)

    H = np.zeros(n)

    for start in range(0, n - win):
        end = start + win
        C = rgb[start:end].T

        mean_color = np.mean(C, axis=1)
        Cn = C / (mean_color[:, None] + 1e-8)

        S1 = Cn[1] - Cn[2]
        S2 = Cn[1] + Cn[2] - 2 * Cn[0]

        alpha = np.std(S1) / (np.std(S2) + 1e-8)
        h = S1 + alpha * S2

        H[start:end] += h - np.mean(h)

    return normalize_signal(H)


def chrom_method(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)

    R = rgb[:, 0]
    G = rgb[:, 1]
    B = rgb[:, 2]

    Rn = R / (np.mean(R) + 1e-8)
    Gn = G / (np.mean(G) + 1e-8)
    Bn = B / (np.mean(B) + 1e-8)

    X = 3 * Rn - 2 * Gn
    Y = 1.5 * Rn + Gn - 1.5 * Bn

    alpha = np.std(X) / (np.std(Y) + 1e-8)
    S = X - alpha * Y

    return normalize_signal(S)


def signal_quality(signal, fs):
    f, pxx = welch(signal, fs=fs, nperseg=min(256, len(signal)))

    mask = (f >= LOW_HZ) & (f <= HIGH_HZ)
    if np.sum(mask) == 0:
        return 0

    band_power = np.sum(pxx[mask])
    total_power = np.sum(pxx) + 1e-8
    return band_power / total_power


def valid_hr(hr):
    return hr is not None and LOW_BPM <= hr <= HIGH_BPM


def choose_final_hr(results):
    candidates = []

    for method, r in results.items():
        if valid_hr(r["fft_hr"]):
            candidates.append({
                "method": method,
                "type": "fft",
                "hr": r["fft_hr"],
                "conf": r["fft_conf"],
                "quality": r["quality"]
            })

        if valid_hr(r["peak_hr"]):
            candidates.append({
                "method": method,
                "type": "peak",
                "hr": r["peak_hr"],
                "conf": r["peak_conf"],
                "quality": r["quality"]
            })

    if len(candidates) == 0:
        return None, "no_valid_candidate"

    # ===================================================
    # NEW RULE 1:
    # If POS FFT and CHROM FFT agree, trust them strongly
    # ===================================================

    pos_fft = None
    chrom_fft = None

    for c in candidates:
        if c["method"] == "POS" and c["type"] == "fft":
            pos_fft = c["hr"]
        if c["method"] == "CHROM" and c["type"] == "fft":
            chrom_fft = c["hr"]

    if pos_fft is not None and chrom_fft is not None:
        if abs(pos_fft - chrom_fft) <= 4:
            return (pos_fft + chrom_fft) / 2, "pos_chrom_fft_agreement"

    # ===================================================
    # NEW RULE 2:
    # If POS peak and CHROM peak agree, use them
    # ===================================================

    pos_peak = None
    chrom_peak = None

    for c in candidates:
        if c["method"] == "POS" and c["type"] == "peak":
            pos_peak = c["hr"]
        if c["method"] == "CHROM" and c["type"] == "peak":
            chrom_peak = c["hr"]

    if pos_peak is not None and chrom_peak is not None:
        if abs(pos_peak - chrom_peak) <= 5:
            return (pos_peak + chrom_peak) / 2, "pos_chrom_peak_agreement"

    # ===================================================
    # NEW RULE 3:
    # If POS FFT and CHROM peak are close, use average
    # ===================================================

    if pos_fft is not None and chrom_peak is not None:
        if abs(pos_fft - chrom_peak) <= 6:
            return (pos_fft + chrom_peak) / 2, "pos_fft_chrom_peak_agreement"

    # ===================================================
    # NEW RULE 4:
    # If CHROM FFT and POS peak are close, use average
    # ===================================================

    if chrom_fft is not None and pos_peak is not None:
        if abs(chrom_fft - pos_peak) <= 6:
            return (chrom_fft + pos_peak) / 2, "chrom_fft_pos_peak_agreement"

    # ===================================================
    # Cluster-based selection
    # ===================================================

    hrs = np.array([c["hr"] for c in candidates])
    used = np.zeros(len(hrs), dtype=bool)

    best_cluster = None
    best_score = -1

    for i in range(len(hrs)):
        if used[i]:
            continue

        cluster_idx = np.where(np.abs(hrs - hrs[i]) <= 7)[0]
        cluster = [candidates[j] for j in cluster_idx]

        methods = len(set(c["method"] for c in cluster))
        types = len(set(c["type"] for c in cluster))
        avg_quality = np.mean([c["quality"] for c in cluster])
        avg_conf = np.mean([c["conf"] for c in cluster])

        score = (
            3.0 * methods +
            1.5 * types +
            0.8 * avg_quality +
            0.3 * avg_conf
        )

        if score > best_score:
            best_score = score
            best_cluster = cluster

        used[cluster_idx] = True

    if best_cluster is not None:
        weighted_sum = 0
        weight_total = 0

        for c in best_cluster:
            weight = 1.0

            if c["method"] in ["POS", "CHROM"]:
                weight *= 1.4

            if c["type"] == "fft":
                weight *= 1.2

            weight *= (1 + c["quality"])
            weight *= (1 + min(c["conf"], 5))

            weighted_sum += c["hr"] * weight
            weight_total += weight

        return weighted_sum / weight_total, "agreement_cluster_weighted"

    # fallback
    best = max(candidates, key=lambda c: c["quality"] + 0.3 * c["conf"])
    return best["hr"], "best_quality_fallback"


# ==============================
# Read video
# ==============================

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 1:
    fps = 30

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps

print("Video FPS:", fps)
print("Frames:", frame_count)
print("Duration:", duration)

# ==============================
# Face detector
# No MediaPipe needed
# ==============================

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(face_cascade_path)

rgb_values = []
green_values = []
times = []

preview_frames = []
roi_records = []

last_face = None
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = frame_idx / fps
    frame_idx += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        last_face = (x, y, w, h)
    elif last_face is not None:
        x, y, w, h = last_face
    else:
        continue

    # Smooth ROI by using detected face box
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    # Dynamic ROIs
    forehead_x1 = int(x + 0.30 * w)
    forehead_x2 = int(x + 0.70 * w)
    forehead_y1 = int(y + 0.12 * h)
    forehead_y2 = int(y + 0.28 * h)

    left_x1 = int(x + 0.18 * w)
    left_x2 = int(x + 0.40 * w)
    left_y1 = int(y + 0.42 * h)
    left_y2 = int(y + 0.65 * h)

    right_x1 = int(x + 0.60 * w)
    right_x2 = int(x + 0.82 * w)
    right_y1 = int(y + 0.42 * h)
    right_y2 = int(y + 0.65 * h)

    rois = [
        rgb_frame[forehead_y1:forehead_y2, forehead_x1:forehead_x2],
        rgb_frame[left_y1:left_y2, left_x1:left_x2],
        rgb_frame[right_y1:right_y2, right_x1:right_x2],
    ]

    valid_rois = [r for r in rois if r.size > 0]

    if len(valid_rois) == 0:
        continue

    roi_pixels = np.concatenate([r.reshape(-1, 3) for r in valid_rois], axis=0)

    mean_rgb = np.mean(roi_pixels, axis=0)

    rgb_values.append(mean_rgb)
    green_values.append(mean_rgb[1])
    times.append(t)

    roi_records.append({
        "time": t,
        "R": mean_rgb[0],
        "G": mean_rgb[1],
        "B": mean_rgb[2],
        "face_x": x,
        "face_y": y,
        "face_w": w,
        "face_h": h
    })

    if len(preview_frames) < 6 and frame_idx % int(max(fps, 1)) == 0:
        pf = rgb_frame.copy()

        cv2.rectangle(pf, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.rectangle(
            pf,
            (forehead_x1, forehead_y1),
            (forehead_x2, forehead_y2),
            (255, 0, 0),
            2
        )

        cv2.rectangle(
            pf,
            (left_x1, left_y1),
            (left_x2, left_y2),
            (0, 0, 255),
            2
        )

        cv2.rectangle(
            pf,
            (right_x1, right_y1),
            (right_x2, right_y2),
            (0, 0, 255),
            2
        )

        cv2.putText(
            pf,
            "forehead",
            (forehead_x1, forehead_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        cv2.putText(
            pf,
            "left_cheek",
            (left_x1, left_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        cv2.putText(
            pf,
            "right_cheek",
            (right_x1, right_y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        preview_frames.append(pf)

cap.release()

rgb_values = np.asarray(rgb_values)
green_values = np.asarray(green_values)
times = np.asarray(times)

print("Extracted samples:", len(green_values))
print("Estimated sampling rate:", len(green_values) / (times[-1] - times[0]))

fs = len(green_values) / (times[-1] - times[0])

# ==============================
# Preview frames
# ==============================

print("Preview frames:")
print("Green rectangle = face")
print("Red rectangle = forehead")
print("Blue rectangles = cheeks")

for pf in preview_frames:
    plt.figure(figsize=(10, 5))
    plt.imshow(pf)
    plt.axis("off")
    plt.show()

# ==============================
# Signal processing
# ==============================

green_raw = remove_outliers(green_values)
green_norm = normalize_signal(green_raw)
green_filtered = bandpass_filter(green_norm, fs)

pos_raw = pos_method(rgb_values, fs)
pos_filtered = bandpass_filter(pos_raw, fs)

chrom_raw = chrom_method(rgb_values)
chrom_filtered = bandpass_filter(chrom_raw, fs)

signals = {
    "GREEN": green_filtered,
    "POS": pos_filtered,
    "CHROM": chrom_filtered
}

results = {}

for name, sig in signals.items():
    f_hr, f_conf = fft_hr(sig, fs)
    p_hr, p_conf, peaks = peak_hr(sig, fs)
    q = signal_quality(sig, fs)

    results[name] = {
        "fft_hr": f_hr,
        "fft_conf": f_conf,
        "peak_hr": p_hr,
        "peak_conf": p_conf,
        "quality": q,
        "peaks": peaks
    }

final_hr, final_method = choose_final_hr(results)

# ==============================
# Print results
# ==============================

print("\n========== GLOBAL RESULTS ==========\n")

for name, r in results.items():
    print(f"--- {name} ---")
    print(f"FFT HR:  {r['fft_hr']:.2f} bpm | conf: {r['fft_conf']:.2f}" if r["fft_hr"] else "FFT HR: None")
    print(f"Peak HR: {r['peak_hr']:.2f} bpm | conf: {r['peak_conf']:.2f}" if r["peak_hr"] else "Peak HR: None")
    print(f"Quality: {r['quality']:.2f}")
    print()

print("========== FINAL RESULT ==========")
print(f"Final HR: {final_hr:.2f} bpm")
print(f"Final method: {final_method}")

# ==============================
# Save CSV
# ==============================

df = pd.DataFrame(roi_records)
df["green_filtered"] = green_filtered
df["pos_filtered"] = pos_filtered
df["chrom_filtered"] = chrom_filtered
df["final_hr"] = final_hr

csv_path = os.path.join(OUTPUT_DIR, "rppg_results.csv")
df.to_csv(csv_path, index=False)

print("CSV saved:", csv_path)

# ==============================
# Plot filtered signals
# ==============================

plt.figure(figsize=(14, 5))
plt.plot(times, green_filtered, label="Green filtered")
plt.plot(times, pos_filtered, label="POS filtered")
plt.plot(times, chrom_filtered, label="CHROM filtered")
plt.title(f"Filtered rPPG Signals | Final HR={final_hr:.2f} bpm")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()

# ==============================
# FFT spectrum plot
# ==============================

plt.figure(figsize=(14, 5))

for name, sig in signals.items():
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    mag = np.abs(np.fft.rfft(normalize_signal(sig)))

    bpm = freqs * 60
    mask = (bpm >= LOW_BPM) & (bpm <= HIGH_BPM)

    plt.plot(bpm[mask], mag[mask], label=f"{name} spectrum")

plt.axvline(final_hr, linestyle="--", label=f"Final HR={final_hr:.2f}")
plt.title("FFT Spectrum Comparison")
plt.xlabel("Heart Rate (BPM)")
plt.ylabel("FFT Magnitude")
plt.grid(True)
plt.legend()
plt.show()

# ==============================
# POS peak plot
# ==============================

pos_peaks = results["POS"]["peaks"]

plt.figure(figsize=(14, 5))
plt.plot(times, pos_filtered, label="POS filtered")

if len(pos_peaks) > 0:
    plt.scatter(times[pos_peaks], pos_filtered[pos_peaks], marker="x", label="Detected POS peaks")

plt.title("Peak Detection on POS rPPG Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.show()
