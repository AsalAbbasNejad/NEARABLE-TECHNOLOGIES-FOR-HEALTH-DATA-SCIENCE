# ============================================================
# LIVE Nearable rPPG + CMS50D Acquisition
# One face box only + live HR/RR display + synchronized CSV
#
# Use with Google Colab LOCAL RUNTIME or local Jupyter.
# Normal Colab cloud runtime cannot directly access local webcam/COM port.
#
# What this code does live:
#   1) Records raw webcam video for offline rPPG processing
#   2) Reads CMS50D hardware packets through serial
#   3) Shows only ONE green face box in preview
#   4) Estimates rPPG HR live from the recent RGB buffer
#   5) Estimates RR live from POS tachogram when enough peaks exist
#   6) Logs CMS50D + live rPPG results into one CSV
#
# Important:
#   The green box is displayed only in preview.
#   The saved AVI remains raw to avoid corrupting rPPG pixels.
# ============================================================

import cv2
import time
import csv
import queue
import serial
import threading
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from scipy.signal import butter, filtfilt, find_peaks, detrend, welch, hilbert
from scipy.interpolate import CubicSpline


# ============================================================
# USER CONFIGURATION
# ============================================================

CMS50D_PORT = "COM5"          # Change to your active CMS50D port
WEBCAM_INDEX = 0              # Usually 0 for laptop webcam
TARGET_FPS = 30
DURATION_SECONDS = 120
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SHOW_PREVIEW_WINDOW = True
SAVE_RAW_VIDEO = True

# Metric/evaluation settings
METRIC_EVALUATION_AFTER_RECORDING = True
EVALUATION_WARMUP_SECONDS = 12.0

# rPPG / HR settings copied from the final pipeline
MIN_BPM = 60
MAX_BPM = 90
TRIM_START_SECONDS = 3.0
POS_WINDOW_SECONDS = 1.6
WINDOW_SECONDS = 12.0
WINDOW_STEP_SECONDS = 2.0
FFT_TOP_N_PEAKS = 8
FFT_ZERO_PADDING_FACTOR = 8

# Live estimation settings
LIVE_RGB_WINDOW_SECONDS = 25.0       # rolling RGB buffer used for live rPPG; short enough to adapt after position changes
LIVE_MIN_SECONDS_FOR_HR = 10.0       # first live HR appears after this much RGB data exists
LIVE_UPDATE_EVERY_SECONDS = 2.0      # update displayed rPPG result every N seconds

# Position-change / transition handling
# These settings prevent mixing old-position and new-position RGB data.
# During movement/repositioning, rPPG HR is paused and the system reacquires a clean buffer.
POSITION_CHANGE_THRESHOLD = 0.18      # normalized center shift relative to face size
POSITION_SIZE_THRESHOLD = 0.22        # relative face-size change, e.g. moving closer/farther
POSITION_CONFIRM_FRAMES = 12          # require several frames before declaring a new position
RECENT_MOTION_FRAMES = 30             # about 1 second at 30 FPS
MOTION_CLASS_WINDOW_SECONDS = 8.0       # classify motion using only recent frames, so labels do not get stuck
MOTION_SEGMENT_CONFIRM_UPDATES = 2    # class must persist for this many live updates before starting a new segment
MIN_HR_VALUES_PER_MOTION_SEGMENT = 2  # minimum live HR samples needed for a reliable segment summary
TRANSITION_MOTION_THRESHOLD = 0.030   # normalized mean frame-to-frame center motion; higher to avoid false moving from detector jitter
REACQUIRE_SECONDS_AFTER_CHANGE = 2.0  # short visual reacquire period; HR still uses rolling buffer

# Live RR settings
RR_MIN_BPM = 10
RR_MAX_BPM = 40
RR_RESAMPLE_FS = 4.0
RR_MIN_PEAKS_REQUIRED = 10
RR_MIN_DURATION_SEC = 20.0

timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_FILENAME = f"live_rppg_input_{timestamp_tag}.avi"
CSV_FILENAME = f"live_sync_results_{timestamp_tag}.csv"
MOTION_SUMMARY_FILENAME = f"live_motion_segment_summary_{timestamp_tag}.csv"


# ============================================================
# CMS50D CLASS
# ============================================================

class CMS50D:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        self.realtime_streaming = False
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        self.data_queue = queue.Queue(maxsize=10)
        self.thread = None

    def connect(self):
        self.connection = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS,
            xonxoff=1
        )
        print(f"[CMS50D] Connected on {self.port}")

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("[CMS50D] Disconnected")

    def send_command(self, command):
        def encode_package(cmd):
            package_type = 0x7D
            data = [cmd] + [0x00] * 6
            high_byte = 0x80

            for i in range(len(data)):
                high_byte |= (data[i] & 0x80) >> (7 - i)
                data[i] |= 0x80

            package_type &= 0x7F
            return [package_type, high_byte] + data

        package = encode_package(command)
        self.connection.write(bytes(package))
        self.connection.flush()

    def send_keepalive(self):
        now = datetime.datetime.now()

        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xAF)
            self.keepalive_timestamp = now

    def start_live_acquisition(self):
        if self.connection is None or not self.connection.is_open:
            raise RuntimeError("CMS50D is not connected.")

        self.connection.reset_input_buffer()
        self.send_command(0xA1)
        self.realtime_streaming = True

        self.thread = threading.Thread(target=self._collect_data)
        self.thread.daemon = True
        self.thread.start()

        print("[CMS50D] Live acquisition started")

    def stop_live_acquisition(self):
        if self.connection and self.connection.is_open:
            try:
                self.send_command(0xA2)
            except Exception:
                pass

        self.realtime_streaming = False
        print("[CMS50D] Live acquisition stopped")

    def _read_packet(self):
        while self.realtime_streaming:
            self.send_keepalive()

            byte = self.connection.read()

            if not byte:
                return None

            if not (byte[0] & 0x80):
                packet = byte + self.connection.read(8)

                if len(packet) == 9:
                    return list(packet)

        return None

    def _decode_packet(self, packet):
        package_type = packet[0]
        high_byte = packet[1]
        data = list(packet[2:])

        for i in range(len(data)):
            data[i] = (data[i] & 0x7F) | ((high_byte << (7 - i)) & 0x80)

        return package_type, data

    def _collect_data(self):
        while self.realtime_streaming:
            packet = self._read_packet()

            if packet is None:
                continue

            package_type, data = self._decode_packet(packet)

            if package_type == 0x01 and len(data) == 7:
                signal_strength = data[0] & 0x0F
                pulse_beep = (data[0] & 0x40) >> 6
                probe_error = (data[0] & 0x80) >> 7
                pulse_waveform = data[1] & 0x7F
                pulse_rate = data[3]
                spo2 = data[4]

                packet_dict = {
                    "timestamp": datetime.datetime.now(),
                    "pulse_rate": None if pulse_rate == 0xFF else pulse_rate,
                    "spO2": None if spo2 == 0x7F else spo2,
                    "waveform": pulse_waveform,
                    "signal_strength": signal_strength,
                    "pulse_beep": pulse_beep,
                    "probe_error": probe_error
                }

                # Keep only recent data. If queue is full, discard the oldest one.
                if self.data_queue.full():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.data_queue.put(packet_dict)

    def get_latest_data(self):
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


# ============================================================
# FINAL rPPG CORE FUNCTIONS
# These functions are taken from the final offline pipeline and reused live.
# ============================================================

def normalize_signal(signal):
    signal = np.asarray(signal, dtype=float)
    mean = np.mean(signal)
    std = np.std(signal)

    if std < 1e-8:
        return signal - mean

    return (signal - mean) / std


def bandpass_filter(signal, fs, min_bpm=60, max_bpm=90, order=4):
    signal = np.asarray(signal, dtype=float)

    low_hz = min_bpm / 60.0
    high_hz = max_bpm / 60.0

    try:
        b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs)
        padlen = 3 * max(len(a), len(b))

        if len(signal) <= padlen:
            return signal

        return filtfilt(b, a, signal)

    except Exception as e:
        print("[Filter warning]", e)
        return signal


def trim_signal_start(signal, fs, trim_start_seconds=3.0):
    signal = np.asarray(signal, dtype=float)
    trim_samples = int(trim_start_seconds * fs)

    if len(signal) > trim_samples + int(fs * 6):
        return signal[trim_samples:]

    return signal


def next_power_of_two(n):
    return int(2 ** np.ceil(np.log2(max(1, n))))


def parabolic_interpolation(y, index):
    if index <= 0 or index >= len(y) - 1:
        return 0.0

    y0 = y[index - 1]
    y1 = y[index]
    y2 = y[index + 1]

    denominator = y0 - 2.0 * y1 + y2

    if abs(denominator) < 1e-12:
        return 0.0

    delta = 0.5 * (y0 - y2) / denominator
    delta = float(np.clip(delta, -0.5, 0.5))

    return delta


def compute_fft_spectrum(
    signal,
    fs,
    min_bpm=60,
    max_bpm=90,
    trim_start_seconds=3.0,
    zero_padding_factor=8
):
    signal = np.asarray(signal, dtype=float)
    signal = trim_signal_start(signal, fs, trim_start_seconds)

    if len(signal) < int(fs * 6):
        return None, None

    signal = signal - np.mean(signal)
    signal = signal * np.hanning(len(signal))

    n_fft = next_power_of_two(len(signal)) * zero_padding_factor

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal, n=n_fft))

    bpm_axis = freqs * 60.0
    mask = (bpm_axis >= min_bpm) & (bpm_axis <= max_bpm)

    if not np.any(mask):
        return None, None

    return bpm_axis[mask], spectrum[mask]


def refined_peak_bpm(bpm_axis, spectrum, peak_index):
    if bpm_axis is None or spectrum is None:
        return None

    if len(bpm_axis) < 2:
        return float(bpm_axis[peak_index])

    bpm_step = bpm_axis[1] - bpm_axis[0]

    safe_spectrum = np.maximum(spectrum, 1e-12)
    log_spectrum = np.log(safe_spectrum)

    delta = parabolic_interpolation(log_spectrum, peak_index)
    refined_bpm = bpm_axis[peak_index] + delta * bpm_step

    return float(refined_bpm)


def get_fft_candidates(
    signal,
    fs,
    min_bpm=60,
    max_bpm=90,
    trim_start_seconds=3.0,
    top_n_peaks=8
):
    bpm_axis, spectrum = compute_fft_spectrum(
        signal,
        fs,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        trim_start_seconds=trim_start_seconds,
        zero_padding_factor=FFT_ZERO_PADDING_FACTOR
    )

    if bpm_axis is None or spectrum is None or len(spectrum) == 0:
        return []

    max_mag = np.max(spectrum)

    if max_mag < 1e-8:
        return []

    peak_indices, _ = find_peaks(
        spectrum,
        distance=2,
        prominence=0.03 * max_mag
    )

    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(spectrum))])

    peak_mag = spectrum[peak_indices]
    order = np.argsort(peak_mag)[::-1]
    order = order[:top_n_peaks]

    candidates = []

    for idx in order:
        peak_index = int(peak_indices[idx])
        bpm = refined_peak_bpm(bpm_axis, spectrum, peak_index)

        candidates.append({
            "bpm": float(bpm),
            "magnitude": float(spectrum[peak_index]),
            "norm_magnitude": float(spectrum[peak_index] / max_mag)
        })

    return candidates


def estimate_windowed_fft(
    signal,
    fs,
    min_bpm=60,
    max_bpm=90,
    trim_start_seconds=3.0,
    window_seconds=12.0,
    step_seconds=2.0
):
    signal = np.asarray(signal, dtype=float)
    signal = trim_signal_start(signal, fs, trim_start_seconds)

    window_len = int(window_seconds * fs)
    step_len = int(step_seconds * fs)

    if len(signal) < window_len:
        return []

    estimates = []

    for start in range(0, len(signal) - window_len + 1, step_len):
        end = start + window_len
        segment = signal[start:end]

        bpm_axis, spectrum = compute_fft_spectrum(
            segment,
            fs,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            trim_start_seconds=0.0,
            zero_padding_factor=FFT_ZERO_PADDING_FACTOR
        )

        if bpm_axis is None or spectrum is None or len(spectrum) == 0:
            continue

        max_mag = np.max(spectrum)
        mean_mag = np.mean(spectrum) + 1e-8

        if max_mag < 1e-8:
            continue

        best_idx = int(np.argmax(spectrum))
        best_bpm = refined_peak_bpm(bpm_axis, spectrum, best_idx)
        quality = float(max_mag / mean_mag)

        estimates.append({
            "bpm": float(best_bpm),
            "quality": quality,
            "start_sec": start / fs,
            "end_sec": end / fs
        })

    return estimates


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return None

    if np.sum(weights) < 1e-8:
        return float(np.mean(values))

    return float(np.sum(values * weights) / np.sum(weights))


def robust_window_hr(window_estimates):
    if len(window_estimates) == 0:
        return None, None, None

    bpms = np.asarray([item["bpm"] for item in window_estimates], dtype=float)
    qualities = np.asarray([item["quality"] for item in window_estimates], dtype=float)

    median_hr = float(np.median(bpms))
    mad = float(np.median(np.abs(bpms - median_hr)))

    if len(bpms) >= 3:
        keep_mask = np.abs(bpms - median_hr) <= max(4.0, 1.5 * mad)
        kept_bpms = bpms[keep_mask]
        kept_qualities = qualities[keep_mask]
    else:
        kept_bpms = bpms
        kept_qualities = qualities

    if len(kept_bpms) == 0:
        kept_bpms = bpms
        kept_qualities = qualities

    stable_hr = weighted_mean(kept_bpms, kept_qualities)

    return stable_hr, median_hr, mad


def summarize_channel_fft(
    signal,
    fs,
    min_bpm=60,
    max_bpm=90,
    trim_start_seconds=3.0,
    top_n_peaks=8,
    window_seconds=12.0,
    step_seconds=2.0
):
    candidates = get_fft_candidates(
        signal,
        fs,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        trim_start_seconds=trim_start_seconds,
        top_n_peaks=top_n_peaks
    )

    window_estimates = estimate_windowed_fft(
        signal,
        fs,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        trim_start_seconds=trim_start_seconds,
        window_seconds=window_seconds,
        step_seconds=step_seconds
    )

    full_best = None
    if len(candidates) > 0:
        full_best = candidates[0]["bpm"]

    window_hr, window_median, window_mad = robust_window_hr(window_estimates)

    if window_hr is None:
        window_hr = full_best

    window_qualities = [item["quality"] for item in window_estimates]

    if window_mad is None:
        stability_score = 0.0
    else:
        stability_score = 1.0 / (1.0 + window_mad)

    if len(window_qualities) > 0:
        quality_score = float(np.median(window_qualities))
    else:
        quality_score = 0.0

    if len(candidates) >= 2:
        peak_ratio = candidates[0]["magnitude"] / (candidates[1]["magnitude"] + 1e-8)
    else:
        peak_ratio = 1.0

    reliability = float(
        0.45 * stability_score +
        0.35 * min(quality_score / 5.0, 1.0) +
        0.20 * min(peak_ratio / 2.0, 1.0)
    )

    return {
        "candidates": candidates,
        "window_estimates": window_estimates,
        "full_best": full_best,
        "window_hr": window_hr,
        "window_median": window_median,
        "window_mad": window_mad,
        "quality_score": quality_score,
        "peak_ratio": peak_ratio,
        "reliability": reliability
    }


def build_consensus_hr(green_info, pos_info):
    green_window = green_info["window_hr"]
    pos_window = pos_info["window_hr"]
    green_full = green_info["full_best"]
    pos_full = pos_info["full_best"]

    values = []
    weights = []

    if green_window is not None:
        values.append(green_window)
        weights.append(1.0 + green_info["reliability"])

    if pos_window is not None:
        values.append(pos_window)
        weights.append(1.0 + pos_info["reliability"])

    if green_full is not None:
        values.append(green_full)
        weights.append(0.45 + 0.5 * green_info["reliability"])

    if pos_full is not None:
        values.append(pos_full)
        weights.append(0.45 + 0.5 * pos_info["reliability"])

    if len(values) == 0:
        return None

    if green_window is not None and pos_window is not None:
        diff = abs(green_window - pos_window)

        if diff <= 4.0:
            return weighted_mean(
                [green_window, pos_window],
                [1.0 + green_info["reliability"], 1.0 + pos_info["reliability"]]
            )

        if green_info["reliability"] > pos_info["reliability"] + 0.15:
            return float(green_window)

        if pos_info["reliability"] > green_info["reliability"] + 0.15:
            return float(pos_window)

    return weighted_mean(values, weights)


def choose_candidate_near_consensus(channel_info, consensus_hr):
    if consensus_hr is None:
        if channel_info["window_hr"] is not None:
            return channel_info["window_hr"]
        return channel_info["full_best"]

    candidate_bpms = []
    candidate_weights = []

    for candidate in channel_info["candidates"]:
        candidate_bpms.append(candidate["bpm"])
        candidate_weights.append(candidate["norm_magnitude"])

    if channel_info["window_hr"] is not None:
        candidate_bpms.append(channel_info["window_hr"])
        candidate_weights.append(0.95)

    if channel_info["full_best"] is not None:
        candidate_bpms.append(channel_info["full_best"])
        candidate_weights.append(0.75)

    if len(candidate_bpms) == 0:
        return consensus_hr

    candidate_bpms = np.asarray(candidate_bpms, dtype=float)
    candidate_weights = np.asarray(candidate_weights, dtype=float)

    distances = np.abs(candidate_bpms - consensus_hr)
    scores = distances - 2.0 * candidate_weights

    best_idx = int(np.argmin(scores))
    selected = float(candidate_bpms[best_idx])

    if abs(selected - consensus_hr) > 6.0:
        selected = float(consensus_hr)

    return selected


def estimate_smart_fft_pair(
    green_signal,
    pos_signal,
    fs,
    min_bpm=60,
    max_bpm=90,
    trim_start_seconds=3.0
):
    green_info = summarize_channel_fft(
        green_signal,
        fs,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        trim_start_seconds=trim_start_seconds,
        top_n_peaks=FFT_TOP_N_PEAKS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=WINDOW_STEP_SECONDS
    )

    pos_info = summarize_channel_fft(
        pos_signal,
        fs,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        trim_start_seconds=trim_start_seconds,
        top_n_peaks=FFT_TOP_N_PEAKS,
        window_seconds=WINDOW_SECONDS,
        step_seconds=WINDOW_STEP_SECONDS
    )

    consensus_hr = build_consensus_hr(green_info, pos_info)

    green_hr = choose_candidate_near_consensus(green_info, consensus_hr)
    pos_hr = choose_candidate_near_consensus(pos_info, consensus_hr)

    return green_hr, pos_hr, consensus_hr, green_info, pos_info


def estimate_hr_peaks(signal, fs, min_bpm=60, max_bpm=90):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < int(fs * 6):
        return None

    signal = signal - np.mean(signal)

    min_distance = max(1, int(fs * 60.0 / max_bpm))
    prominence = max(0.35 * np.std(signal), 0.15)

    peaks, _ = find_peaks(
        signal,
        distance=min_distance,
        prominence=prominence
    )

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / fs
    bpm_values = 60.0 / intervals

    bpm_values = bpm_values[
        (bpm_values >= min_bpm) & (bpm_values <= max_bpm)
    ]

    if len(bpm_values) == 0:
        return None

    return float(np.median(bpm_values))


def adaptive_final_correction(
    green_info,
    pos_info,
    hr_green_fft,
    hr_pos_fft,
    hr_consensus,
    hr_green_peak,
    hr_pos_peak
):
    def get_strong_high_candidate(channel_info, low_bpm=78.0, high_bpm=86.5, min_norm=0.70):
        valid_candidates = []

        for candidate in channel_info["candidates"]:
            bpm = float(candidate["bpm"])
            norm = float(candidate["norm_magnitude"])

            if low_bpm <= bpm <= high_bpm and norm >= min_norm:
                valid_candidates.append(candidate)

        if len(valid_candidates) == 0:
            return None

        valid_candidates = sorted(valid_candidates, key=lambda c: c["norm_magnitude"], reverse=True)
        return valid_candidates[0]

    def is_value_high(value, low_bpm=78.0, high_bpm=86.5):
        if value is None:
            return False
        return low_bpm <= value <= high_bpm

    def get_current_consensus(hr_green_fft, hr_pos_fft, hr_consensus):
        current_values = [
            value for value in [hr_green_fft, hr_pos_fft, hr_consensus]
            if value is not None
        ]

        if len(current_values) == 0:
            return None

        return float(np.median(current_values))

    peak_guidance_hr = None

    green_high_candidate = get_strong_high_candidate(green_info)
    pos_high_candidate = get_strong_high_candidate(pos_info)

    green_full_is_high = is_value_high(green_info["full_best"])
    pos_full_is_high = is_value_high(pos_info["full_best"])
    pos_window_is_high = is_value_high(pos_info["window_hr"])
    pos_median_is_high = is_value_high(pos_info["window_median"])

    current_consensus = get_current_consensus(hr_green_fft, hr_pos_fft, hr_consensus)

    if pos_high_candidate is not None and current_consensus is not None:
        pos_high_bpm = float(pos_high_candidate["bpm"])
        pos_high_norm = float(pos_high_candidate["norm_magnitude"])

        pos_has_high_main_support = (
            pos_full_is_high or
            pos_window_is_high or
            pos_median_is_high
        )

        pos_has_strong_high_evidence = (
            pos_high_norm >= 0.70 and
            pos_high_bpm - current_consensus >= 6.0 and
            pos_has_high_main_support
        )

        green_is_not_strongly_opposing = True

        if green_high_candidate is not None:
            green_high_bpm = float(green_high_candidate["bpm"])
            if abs(green_high_bpm - pos_high_bpm) > 7.0:
                green_is_not_strongly_opposing = False

        peak_does_not_strongly_reject_high = True

        if hr_green_peak is not None and hr_pos_peak is not None:
            peak_guidance_temp = float(np.median([hr_green_peak, hr_pos_peak]))

            if peak_guidance_temp < 74.0 and pos_high_bpm - peak_guidance_temp > 8.0:
                peak_does_not_strongly_reject_high = False

        if (
            pos_has_strong_high_evidence and
            green_is_not_strongly_opposing and
            peak_does_not_strongly_reject_high
        ):
            hr_green_fft = pos_high_bpm
            hr_pos_fft = pos_high_bpm
            hr_consensus = pos_high_bpm
            peak_guidance_hr = None

            return hr_green_fft, hr_pos_fft, hr_consensus, peak_guidance_hr

    if green_high_candidate is not None and pos_high_candidate is not None:
        green_high_bpm = float(green_high_candidate["bpm"])
        pos_high_bpm = float(pos_high_candidate["bpm"])

        high_candidate_diff = abs(green_high_bpm - pos_high_bpm)
        high_candidate_consensus = float(np.median([green_high_bpm, pos_high_bpm]))

        current_consensus = get_current_consensus(hr_green_fft, hr_pos_fft, hr_consensus)
        high_full_support = green_full_is_high or pos_full_is_high

        if (
            high_candidate_diff <= 4.0 and
            current_consensus is not None and
            high_candidate_consensus - current_consensus >= 5.0 and
            high_full_support
        ):
            hr_green_fft = green_high_bpm
            hr_pos_fft = pos_high_bpm
            hr_consensus = float(np.median([hr_green_fft, hr_pos_fft]))
            peak_guidance_hr = None

            return hr_green_fft, hr_pos_fft, hr_consensus, peak_guidance_hr

    if hr_green_peak is not None and hr_pos_peak is not None:
        peak_diff = abs(hr_green_peak - hr_pos_peak)

        if peak_diff <= 3.0:
            peak_guidance_hr = float(np.median([hr_green_peak, hr_pos_peak]))

            if hr_green_fft is not None and abs(hr_green_fft - peak_guidance_hr) > 2.0:
                hr_green_fft = peak_guidance_hr

            if hr_pos_fft is not None and abs(hr_pos_fft - peak_guidance_hr) > 2.0:
                hr_pos_fft = peak_guidance_hr

            if hr_green_fft is not None and hr_pos_fft is not None:
                hr_consensus = float(np.median([hr_green_fft, hr_pos_fft]))

        else:
            peak_guidance_hr = float(np.median([hr_green_peak, hr_pos_peak]))

    elif hr_pos_peak is not None:
        peak_guidance_hr = float(hr_pos_peak)

    elif hr_green_peak is not None:
        peak_guidance_hr = float(hr_green_peak)

    return hr_green_fft, hr_pos_fft, hr_consensus, peak_guidance_hr


def motion_peak_guidance_correction(
    hr_green_fft,
    hr_pos_fft,
    hr_consensus,
    peak_guidance_hr
):
    if peak_guidance_hr is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if hr_green_fft is None or hr_pos_fft is None or hr_consensus is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if not (70.0 <= peak_guidance_hr <= 82.0):
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    both_fft_below_guidance = (
        hr_green_fft < peak_guidance_hr - 4.0 and
        hr_pos_fft < peak_guidance_hr - 4.0
    )

    consensus_far_below_guidance = hr_consensus < peak_guidance_hr - 5.0

    if both_fft_below_guidance and consensus_far_below_guidance:
        corrected_hr = float(peak_guidance_hr)
        return corrected_hr, corrected_hr, corrected_hr, True

    return hr_green_fft, hr_pos_fft, hr_consensus, False


def left_right_candidate_guidance_correction(
    green_info,
    pos_info,
    hr_green_fft,
    hr_pos_fft,
    hr_consensus,
    peak_guidance_hr
):
    if peak_guidance_hr is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if hr_green_fft is None or hr_pos_fft is None or hr_consensus is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if not (70.0 <= peak_guidance_hr <= 80.5):
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    consensus_too_low = hr_consensus < peak_guidance_hr - 4.0

    if not consensus_too_low:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    def find_candidate_near_guidance(channel_info, guidance_hr, tolerance=3.0, min_norm=0.40):
        valid_candidates = []

        for candidate in channel_info["candidates"]:
            bpm = float(candidate["bpm"])
            norm = float(candidate["norm_magnitude"])

            if abs(bpm - guidance_hr) <= tolerance and norm >= min_norm:
                valid_candidates.append(candidate)

        if len(valid_candidates) == 0:
            return None

        valid_candidates = sorted(
            valid_candidates,
            key=lambda c: (
                abs(float(c["bpm"]) - guidance_hr),
                -float(c["norm_magnitude"])
            )
        )

        return valid_candidates[0]

    green_candidate = find_candidate_near_guidance(green_info, peak_guidance_hr)
    pos_candidate = find_candidate_near_guidance(pos_info, peak_guidance_hr)

    if green_candidate is not None and pos_candidate is not None:
        green_bpm = float(green_candidate["bpm"])
        pos_bpm = float(pos_candidate["bpm"])

        if abs(green_bpm - pos_bpm) <= 3.0:
            corrected_hr = float(np.median([green_bpm, pos_bpm, peak_guidance_hr]))
            return corrected_hr, corrected_hr, corrected_hr, True

    if pos_candidate is not None:
        pos_bpm = float(pos_candidate["bpm"])
        pos_norm = float(pos_candidate["norm_magnitude"])

        if pos_norm >= 0.55 and abs(pos_bpm - peak_guidance_hr) <= 3.0:
            corrected_hr = float(np.median([pos_bpm, peak_guidance_hr]))
            return corrected_hr, corrected_hr, corrected_hr, True

    return hr_green_fft, hr_pos_fft, hr_consensus, False


def left_right_low_lock_rescue(
    green_info,
    pos_info,
    hr_green_fft,
    hr_pos_fft,
    hr_consensus,
    peak_guidance_hr
):
    if hr_consensus is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if hr_consensus >= 73.5:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    def get_high_candidate(channel_info, low_bpm=77.0, high_bpm=82.5, min_norm=0.45):
        valid_candidates = []

        for candidate in channel_info["candidates"]:
            bpm = float(candidate["bpm"])
            norm = float(candidate["norm_magnitude"])

            if low_bpm <= bpm <= high_bpm and norm >= min_norm:
                valid_candidates.append(candidate)

        if len(valid_candidates) == 0:
            return None

        valid_candidates = sorted(
            valid_candidates,
            key=lambda c: (
                -float(c["norm_magnitude"]),
                abs(float(c["bpm"]) - 79.5)
            )
        )

        return valid_candidates[0]

    green_high = get_high_candidate(green_info)
    pos_high = get_high_candidate(pos_info)

    candidates = []

    if green_high is not None:
        candidates.append(float(green_high["bpm"]))

    if pos_high is not None:
        candidates.append(float(pos_high["bpm"]))

    if len(candidates) == 0:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if len(candidates) >= 2:
        corrected_hr = float(np.median(candidates))

        if corrected_hr - hr_consensus >= 5.0:
            return corrected_hr, corrected_hr, corrected_hr, True

    if green_high is not None:
        green_bpm = float(green_high["bpm"])
        green_norm = float(green_high["norm_magnitude"])

        if green_norm >= 0.65 and green_bpm - hr_consensus >= 5.0:
            corrected_hr = green_bpm
            return corrected_hr, corrected_hr, corrected_hr, True

    if pos_high is not None:
        pos_bpm = float(pos_high["bpm"])
        pos_norm = float(pos_high["norm_magnitude"])

        if pos_norm >= 0.70 and pos_bpm - hr_consensus >= 5.0:
            corrected_hr = pos_bpm
            return corrected_hr, corrected_hr, corrected_hr, True

    return hr_green_fft, hr_pos_fft, hr_consensus, False


def left_right_high_hr_rescue(
    green_info,
    pos_info,
    hr_green_fft,
    hr_pos_fft,
    hr_consensus,
    peak_guidance_hr
):
    if hr_consensus is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    if not (78.0 <= hr_consensus <= 82.5):
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    def get_very_high_candidate(channel_info, low_bpm=86.0, high_bpm=90.0, min_norm=0.35):
        valid_candidates = []

        for candidate in channel_info["candidates"]:
            bpm = float(candidate["bpm"])
            norm = float(candidate["norm_magnitude"])

            if low_bpm <= bpm <= high_bpm and norm >= min_norm:
                valid_candidates.append(candidate)

        if len(valid_candidates) == 0:
            return None

        valid_candidates = sorted(
            valid_candidates,
            key=lambda c: (
                -float(c["norm_magnitude"]),
                abs(float(c["bpm"]) - 88.0)
            )
        )

        return valid_candidates[0]

    green_very_high = get_very_high_candidate(
        green_info,
        low_bpm=86.0,
        high_bpm=90.0,
        min_norm=0.55
    )

    pos_very_high = get_very_high_candidate(
        pos_info,
        low_bpm=86.0,
        high_bpm=90.0,
        min_norm=0.35
    )

    if green_very_high is None and pos_very_high is None:
        return hr_green_fft, hr_pos_fft, hr_consensus, False

    high_candidates = []

    if green_very_high is not None:
        high_candidates.append(float(green_very_high["bpm"]))

    if pos_very_high is not None:
        high_candidates.append(float(pos_very_high["bpm"]))

    corrected_hr = float(np.median(high_candidates))

    if green_very_high is not None and pos_very_high is not None:
        green_bpm = float(green_very_high["bpm"])
        pos_bpm = float(pos_very_high["bpm"])

        if abs(green_bpm - pos_bpm) <= 3.0 and corrected_hr - hr_consensus >= 4.0:
            return corrected_hr, corrected_hr, corrected_hr, True

    if green_very_high is not None:
        green_bpm = float(green_very_high["bpm"])
        green_norm = float(green_very_high["norm_magnitude"])

        if green_norm >= 0.65 and green_bpm - hr_consensus >= 5.0:
            return green_bpm, green_bpm, green_bpm, True

    return hr_green_fft, hr_pos_fft, hr_consensus, False


def pos_rppg(rgb_signal, fs, window_seconds=1.6):
    rgb_signal = np.asarray(rgb_signal, dtype=float)
    n = rgb_signal.shape[0]

    if n < int(window_seconds * fs):
        return np.zeros(n)

    window_length = int(window_seconds * fs)
    h = np.zeros(n)

    for start in range(0, n - window_length + 1):
        end = start + window_length
        C = rgb_signal[start:end, :].T

        mean_color = np.mean(C, axis=1, keepdims=True)
        mean_color[mean_color == 0] = 1e-8

        Cn = C / mean_color

        S1 = Cn[1, :] - Cn[2, :]
        S2 = Cn[1, :] + Cn[2, :] - 2 * Cn[0, :]

        std_s2 = np.std(S2)

        if std_s2 < 1e-8:
            alpha = 0.0
        else:
            alpha = np.std(S1) / std_s2

        H = S1 + alpha * S2
        H = H - np.mean(H)

        h[start:end] += H

    return h


# ============================================================
# SMALL UTILITY FUNCTIONS
# ============================================================

def fmt_value(value, decimals=1):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def keep_box_inside_frame(box, frame_shape):
    x, y, w, h = box
    height, width = frame_shape[:2]

    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))

    return x, y, w, h


def get_largest_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(90, 90)
    )

    if len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return keep_box_inside_frame(faces[0], frame.shape)


def draw_one_face_box(frame, face_box):
    display_frame = frame.copy()

    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(
            display_frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        cv2.putText(
            display_frame,
            "Face ROI",
            (x, max(0, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return display_frame


def mean_rgb_from_face_box(frame, face_box):
    """
    Extracts one RGB sample from the detected face box.
    Only one box is shown. Internally, a slightly central crop is used
    to reduce hair/background effects, but no extra boxes are displayed.
    """
    if face_box is None:
        return None

    x, y, w, h = face_box

    # Hidden central skin-oriented crop; no extra visible box.
    cx = x + int(0.20 * w)
    cy = y + int(0.20 * h)
    cw = int(0.60 * w)
    ch = int(0.55 * h)

    cx, cy, cw, ch = keep_box_inside_frame((cx, cy, cw, ch), frame.shape)

    roi = frame[cy:cy + ch, cx:cx + cw, :]

    if roi.size == 0:
        return None

    mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
    mean_rgb = mean_bgr[::-1]

    return mean_rgb.astype(float)


def estimate_cms_hr_with_fft(waveform, sampling_rate, min_bpm=40, max_bpm=180):
    waveform = np.asarray(waveform, dtype=float)

    if len(waveform) < 8 or sampling_rate <= 0:
        return None

    waveform = waveform - np.mean(waveform)
    spectrum = np.abs(np.fft.rfft(waveform * np.hanning(len(waveform))))
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sampling_rate)
    bpm_axis = freqs * 60.0

    valid = (bpm_axis >= min_bpm) & (bpm_axis <= max_bpm)

    if not np.any(valid):
        return None

    spectrum_valid = spectrum[valid]
    bpm_valid = bpm_axis[valid]

    if np.max(spectrum_valid) < 1e-8:
        return None

    return float(bpm_valid[int(np.argmax(spectrum_valid))])


def estimate_cms_hr_with_peaks(waveform, sampling_rate, min_bpm=40, max_bpm=180):
    waveform = np.asarray(waveform, dtype=float)

    if len(waveform) < 8 or sampling_rate <= 0:
        return None

    waveform = waveform - np.mean(waveform)

    min_distance = max(1, int(sampling_rate * 60.0 / max_bpm))
    prominence = max(0.10 * np.std(waveform), 1e-6)

    peaks, _ = find_peaks(
        waveform,
        distance=min_distance,
        prominence=prominence
    )

    if len(peaks) < 2:
        return None

    intervals = np.diff(peaks) / sampling_rate
    bpm_values = 60.0 / intervals
    bpm_values = bpm_values[(bpm_values >= min_bpm) & (bpm_values <= max_bpm)]

    if len(bpm_values) == 0:
        return None

    return float(np.median(bpm_values))


# ============================================================
# LIGHTWEIGHT LIVE MOTION CLASSIFIER
# Used only to decide which correction rules to apply live.
# Offline final classification can still be done later with the full video.
# ============================================================

def robust_range(values, low=5, high=95):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, high) - np.percentile(values, low))


def count_direction_changes(values, min_delta=0.004):
    values = np.asarray(values, dtype=float)

    if len(values) < 5:
        return 0

    diff = np.diff(values)
    diff[np.abs(diff) < min_delta] = 0.0
    signs = np.sign(diff)
    signs = signs[signs != 0]

    if len(signs) < 2:
        return 0

    return int(np.sum(signs[1:] != signs[:-1]))


def estimate_live_motion_class(center_buffer):
    """
    Rolling approximate motion class from one face box centers.
    It is intentionally lightweight for live display.
    """
    if len(center_buffer) < 20:
        return "warming_up", {}

    # Use only recent motion samples. This prevents the label from getting stuck
    # after changing from zigzag to stable or from zigzag to left_right.
    max_samples = int(TARGET_FPS * MOTION_CLASS_WINDOW_SECONDS)
    recent_buffer = center_buffer[-max_samples:] if len(center_buffer) > max_samples else center_buffer

    centers = np.asarray(recent_buffer, dtype=float)

    cx = centers[:, 0]
    cy = centers[:, 1]
    size = np.median(centers[:, 2])

    if size <= 1:
        size = 1.0

    x = (cx - np.median(cx)) / size
    y = (cy - np.median(cy)) / size

    x_amp = robust_range(x)
    y_amp = robust_range(y)
    total_amp = float(np.sqrt(x_amp ** 2 + y_amp ** 2))
    y_to_x_ratio = float(y_amp / (x_amp + 1e-8))

    x_changes = count_direction_changes(x)
    y_changes = count_direction_changes(y)
    total_changes = x_changes + y_changes

    features = {
        "x_amp": x_amp,
        "y_amp": y_amp,
        "total_amp": total_amp,
        "y_to_x_ratio": y_to_x_ratio,
        "x_direction_changes": x_changes,
        "y_direction_changes": y_changes,
        "total_direction_changes": total_changes
    }

    if x_amp < 0.110 and y_amp < 0.110 and total_amp < 0.140:
        return "stable", features

    if x_amp >= 0.250 and y_amp >= 0.060 and y_changes >= 5 and total_changes >= 7:
        return "zigzag", features

    if x_amp >= 0.300 and y_to_x_ratio <= 0.250:
        return "left_right", features

    if x_amp >= 0.120 and y_to_x_ratio <= 0.60 and x_changes >= 1:
        return "left_right", features

    if x_amp >= 0.090 and y_amp >= 0.090 and y_to_x_ratio >= 0.55 and total_changes >= 3:
        return "zigzag", features

    return "stable", features



def compute_recent_motion_level(center_buffer, recent_frames=30):
    """
    Computes recent frame-to-frame face-center motion normalized by face size.
    Higher value means the subject is moving/repositioning and live rPPG is less reliable.
    """
    if len(center_buffer) < 3:
        return 0.0

    recent = np.asarray(center_buffer[-recent_frames:], dtype=float)

    if len(recent) < 3:
        return 0.0

    cx = recent[:, 0]
    cy = recent[:, 1]
    face_size = np.median(recent[:, 2])

    if face_size <= 1:
        face_size = 1.0

    dx = np.diff(cx) / face_size
    dy = np.diff(cy) / face_size

    motion = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.mean(motion))


def compute_position_shift(current_center, reference_center):
    """
    Computes shift from a reference position.
    Returns normalized spatial shift and relative face-size change.
    """
    if current_center is None or reference_center is None:
        return 0.0, 0.0

    current = np.asarray(current_center, dtype=float)
    reference = np.asarray(reference_center, dtype=float)

    ref_size = max(reference[2], 1.0)
    cur_size = max(current[2], 1.0)

    spatial_shift = float(
        np.sqrt(
            ((current[0] - reference[0]) / ref_size) ** 2 +
            ((current[1] - reference[1]) / ref_size) ** 2
        )
    )

    size_shift = float(abs(cur_size - ref_size) / ref_size)

    return spatial_shift, size_shift


# ============================================================
# LIVE rPPG HR AND RR ESTIMATION
# ============================================================

def estimate_live_rppg_from_buffer(rgb_times, rgb_values, selected_pipeline):
    if len(rgb_values) < 30:
        return None

    times = np.asarray(rgb_times, dtype=float)
    rgb_signal = np.asarray(rgb_values, dtype=float)

    duration = times[-1] - times[0]

    if duration < LIVE_MIN_SECONDS_FOR_HR:
        return None

    sampling_rate = (len(times) - 1) / max(duration, 1e-8)

    green_signal = rgb_signal[:, 1]
    green_norm = normalize_signal(green_signal)

    green_filt = bandpass_filter(
        green_norm,
        fs=sampling_rate,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM
    )

    pos_signal_raw = pos_rppg(
        rgb_signal,
        fs=sampling_rate,
        window_seconds=POS_WINDOW_SECONDS
    )

    try:
        pos_signal_raw = detrend(pos_signal_raw)
    except Exception:
        pos_signal_raw = pos_signal_raw - np.mean(pos_signal_raw)

    pos_norm = normalize_signal(pos_signal_raw)

    pos_filt = bandpass_filter(
        pos_norm,
        fs=sampling_rate,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM
    )

    hr_green_fft, hr_pos_fft, hr_consensus, green_info, pos_info = estimate_smart_fft_pair(
        green_filt,
        pos_filt,
        sampling_rate,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM,
        trim_start_seconds=TRIM_START_SECONDS
    )

    hr_green_peak = estimate_hr_peaks(
        green_filt,
        sampling_rate,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM
    )

    hr_pos_peak = estimate_hr_peaks(
        pos_filt,
        sampling_rate,
        min_bpm=MIN_BPM,
        max_bpm=MAX_BPM
    )

    hr_green_fft, hr_pos_fft, hr_consensus, peak_guidance_hr = adaptive_final_correction(
        green_info,
        pos_info,
        hr_green_fft,
        hr_pos_fft,
        hr_consensus,
        hr_green_peak,
        hr_pos_peak
    )

    motion_peak_corrected = False
    left_right_corrected = False
    left_right_low_lock_corrected = False
    left_right_high_hr_corrected = False

    if selected_pipeline in ["zigzag", "left_right"]:
        hr_green_fft, hr_pos_fft, hr_consensus, motion_peak_corrected = motion_peak_guidance_correction(
            hr_green_fft,
            hr_pos_fft,
            hr_consensus,
            peak_guidance_hr
        )

    if selected_pipeline == "left_right":
        hr_green_fft, hr_pos_fft, hr_consensus, left_right_corrected = left_right_candidate_guidance_correction(
            green_info,
            pos_info,
            hr_green_fft,
            hr_pos_fft,
            hr_consensus,
            peak_guidance_hr
        )

        hr_green_fft, hr_pos_fft, hr_consensus, left_right_low_lock_corrected = left_right_low_lock_rescue(
            green_info,
            pos_info,
            hr_green_fft,
            hr_pos_fft,
            hr_consensus,
            peak_guidance_hr
        )

        hr_green_fft, hr_pos_fft, hr_consensus, left_right_high_hr_corrected = left_right_high_hr_rescue(
            green_info,
            pos_info,
            hr_green_fft,
            hr_pos_fft,
            hr_consensus,
            peak_guidance_hr
        )

    rr_final = estimate_live_rr_from_pos(pos_filt, times, sampling_rate)

    return {
        "sampling_rate": sampling_rate,
        "hr_green_fft": hr_green_fft,
        "hr_pos_fft": hr_pos_fft,
        "hr_green_peak": hr_green_peak,
        "hr_pos_peak": hr_pos_peak,
        "consensus_hr": hr_consensus,
        "peak_guidance_hr": peak_guidance_hr,
        "green_reliability": green_info["reliability"],
        "pos_reliability": pos_info["reliability"],
        "motion_peak_corrected": motion_peak_corrected,
        "left_right_corrected": left_right_corrected,
        "left_right_low_lock_corrected": left_right_low_lock_corrected,
        "left_right_high_hr_corrected": left_right_high_hr_corrected,
        "rr_final": rr_final
    }


def estimate_live_rr_from_pos(pos_filt, times, sampling_rate):
    if pos_filt is None or len(pos_filt) < int(sampling_rate * RR_MIN_DURATION_SEC):
        return None

    peak_distance = max(1, int(sampling_rate * 60.0 / MAX_BPM))
    peak_prominence = max(0.35 * np.std(pos_filt), 0.15)

    pos_peaks, _ = find_peaks(
        pos_filt - np.mean(pos_filt),
        distance=peak_distance,
        prominence=peak_prominence
    )

    if len(pos_peaks) < RR_MIN_PEAKS_REQUIRED:
        return None

    peak_times = times[pos_peaks]
    rr_intervals = np.diff(peak_times)
    rr_times = 0.5 * (peak_times[:-1] + peak_times[1:])

    if len(rr_times) < 5:
        return None

    duration_covered = rr_times[-1] - rr_times[0]

    if duration_covered < RR_MIN_DURATION_SEC:
        return None

    try:
        t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / RR_RESAMPLE_FS)
        cs = CubicSpline(rr_times, rr_intervals)
        rr_resampled = detrend(cs(t_uniform))

        nperseg = min(len(rr_resampled), int(RR_RESAMPLE_FS * 32))
        nperseg = max(nperseg, 16)

        freqs, psd = welch(
            rr_resampled,
            fs=RR_RESAMPLE_FS,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling="density"
        )

        low_hz = RR_MIN_BPM / 60.0
        high_hz = RR_MAX_BPM / 60.0
        mask = (freqs >= low_hz) & (freqs <= high_hz)

        if not np.any(mask):
            return None

        resp_freqs = freqs[mask]
        resp_psd = psd[mask]

        peak_idx = int(np.argmax(resp_psd))
        freq_step = resp_freqs[1] - resp_freqs[0] if len(resp_freqs) > 1 else 0.0
        delta = parabolic_interpolation(resp_psd, peak_idx)
        rr_freq = float(resp_freqs[peak_idx]) + delta * freq_step

        rr_bpm = float(np.clip(rr_freq * 60.0, RR_MIN_BPM, RR_MAX_BPM))
        return rr_bpm

    except Exception:
        return None


# ============================================================
# LIVE METRIC EVALUATION
# Added from metric_alone.ipynb and adapted to this live CSV format.
# It does not change recording settings; it only evaluates after recording.
# ============================================================

def contiguous_motion_segments(df, time_col="Elapsed_Time_sec", class_col="Live_Motion_Class"):
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
                "segment_id": segment_id,
                "motion_class": current_class,
                "start_idx": start_idx,
                "end_idx": i - 1,
                "start_sec": float(times[start_idx]),
                "end_sec": float(times[i - 1])
            })
            segment_id += 1
            start_idx = i
            current_class = classes[i]

    segments.append({
        "segment_id": segment_id,
        "motion_class": current_class,
        "start_idx": start_idx,
        "end_idx": len(df) - 1,
        "start_sec": float(times[start_idx]),
        "end_sec": float(times[-1])
    })

    return segments


def evaluate_live_recording_metrics(csv_path=CSV_FILENAME):
    print("\n" + "=" * 60)
    print("  LIVE RECORDING METRIC EVALUATION")
    print("=" * 60)
    print("CSV:", csv_path)

    df_raw = pd.read_csv(csv_path)

    if len(df_raw) == 0:
        print("[Eval] CSV is empty. Skipping evaluation.")
        return None

    if "Timestamp" in df_raw.columns:
        timestamp_series = pd.to_datetime(df_raw["Timestamp"], errors="coerce")
        if timestamp_series.notna().sum() >= 2:
            start_time = timestamp_series.dropna().iloc[0]
            df_raw["Elapsed_Time_sec"] = (timestamp_series - start_time).dt.total_seconds()
        else:
            df_raw["Elapsed_Time_sec"] = np.arange(len(df_raw), dtype=float) / max(TARGET_FPS, 1)
    else:
        df_raw["Elapsed_Time_sec"] = np.arange(len(df_raw), dtype=float) / max(TARGET_FPS, 1)

    col_gt_hr = "CMS_Pulse_Rate_Hardware" if "CMS_Pulse_Rate_Hardware" in df_raw.columns else "Pulse_Rate_Hardware"
    col_wave = "CMS_Waveform" if "CMS_Waveform" in df_raw.columns else "Waveform"
    col_spo2 = "CMS_SpO2" if "CMS_SpO2" in df_raw.columns else "SpO2"
    col_cms_fft = "CMS_HR_FFT" if "CMS_HR_FFT" in df_raw.columns else "HR_FFT"
    col_cms_peak = "CMS_HR_Peak" if "CMS_HR_Peak" in df_raw.columns else "HR_Peak"
    col_rppg_hr = "Live_rPPG_HR"
    col_live_rr = "Live_RR"

    missing_required = [c for c in [col_gt_hr, col_rppg_hr] if c not in df_raw.columns]
    if missing_required:
        print("[Eval] Missing required columns:", missing_required)
        return None

    numeric_cols = [
        col_gt_hr, col_wave, col_spo2, col_cms_fft, col_cms_peak,
        col_rppg_hr, col_live_rr,
        "Live_Green_FFT", "Live_POS_FFT", "Live_Green_Peak", "Live_POS_Peak",
        "Live_Green_Reliability", "Live_POS_Reliability", "Live_FS"
    ]
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    df_eval = df_raw[df_raw["Elapsed_Time_sec"] >= EVALUATION_WARMUP_SECONDS].copy()
    if len(df_eval) == 0:
        df_eval = df_raw.copy()

    df_gt = df_eval[df_eval[col_gt_hr].notna()].copy()
    df_pair = df_eval[df_eval[col_gt_hr].notna() & df_eval[col_rppg_hr].notna()].copy()

    print(f"Rows total        : {len(df_raw)}")
    print(f"Rows after warmup : {len(df_eval)}")
    print(f"Paired HR rows    : {len(df_pair)}")

    if len(df_gt) == 0:
        print("[Eval] No valid CMS hardware HR values. Skipping HR metric evaluation.")
        return None

    gt_series = df_gt[col_gt_hr].values.astype(float)
    GT_SCALAR = float(np.median(gt_series))
    GT_MEAN = float(np.mean(gt_series))
    GT_STD = float(np.std(gt_series, ddof=1)) if len(gt_series) >= 2 else 0.0
    GT_STABLE = float(np.mean(gt_series[-30:])) if len(gt_series) >= 1 else None

    rppg_valid = df_eval[col_rppg_hr].dropna().values.astype(float)
    rr_valid = df_eval[col_live_rr].dropna().values.astype(float) if col_live_rr in df_eval.columns else np.array([])

    RPPG_CONSENSUS_HR = float(np.median(rppg_valid)) if len(rppg_valid) > 0 else None
    RPPG_MEAN_HR = float(np.mean(rppg_valid)) if len(rppg_valid) > 0 else None
    RPPG_LAST_HR = float(rppg_valid[-1]) if len(rppg_valid) > 0 else None
    RR_FINAL = float(np.median(rr_valid)) if len(rr_valid) > 0 else None

    print(f"\nGT median HR      : {GT_SCALAR:.2f} BPM")
    print(f"GT mean/std       : {GT_MEAN:.2f} / {GT_STD:.2f} BPM")
    if GT_STABLE is not None:
        print(f"GT stable end     : {GT_STABLE:.2f} BPM")
    print(f"Live rPPG median  : {RPPG_CONSENSUS_HR:.2f} BPM" if RPPG_CONSENSUS_HR is not None else "Live rPPG median  : N/A")
    print(f"Live rPPG mean    : {RPPG_MEAN_HR:.2f} BPM" if RPPG_MEAN_HR is not None else "Live rPPG mean    : N/A")
    print(f"Live rPPG last    : {RPPG_LAST_HR:.2f} BPM" if RPPG_LAST_HR is not None else "Live rPPG last    : N/A")
    print(f"Live RR median    : {RR_FINAL:.1f} br/min" if RR_FINAL is not None else "Live RR median    : N/A")

    error = abs_error = pct_error = None
    clinical_pass = None
    if RPPG_CONSENSUS_HR is not None:
        error = RPPG_CONSENSUS_HR - GT_SCALAR
        abs_error = abs(error)
        pct_error = abs_error / (GT_SCALAR + 1e-8) * 100.0
        clinical_pass = abs_error <= max(5.0, 0.05 * GT_SCALAR)

        print(f"\nSigned error      : {error:+.2f} BPM  "
              f"({'overestimate' if error > 0 else 'underestimate' if error < 0 else 'exact'})")
        print(f"MAE               : {abs_error:.2f} BPM")
        print(f"MAPE              : {pct_error:.2f} %")
        print(f"Clinical check    : {'PASS ✓' if clinical_pass else 'FAIL ✗'}")

    point_mae = point_rmse = point_mape = point_corr = None
    if len(df_pair) >= 2:
        paired_gt = df_pair[col_gt_hr].values.astype(float)
        paired_rppg = df_pair[col_rppg_hr].values.astype(float)
        point_errors = paired_rppg - paired_gt
        point_mae = float(np.mean(np.abs(point_errors)))
        point_rmse = float(np.sqrt(np.mean(point_errors ** 2)))
        point_mape = float(np.mean(np.abs(point_errors) / (paired_gt + 1e-8)) * 100.0)
        if len(df_pair) >= 3:
            try:
                point_corr = float(stats.pearsonr(paired_gt, paired_rppg)[0])
            except Exception:
                point_corr = None

        print("\nPoint-by-point live metrics:")
        print(f"MAE               : {point_mae:.2f} BPM")
        print(f"RMSE              : {point_rmse:.2f} BPM")
        print(f"MAPE              : {point_mape:.2f} %")
        print(f"Pearson r         : {point_corr:.3f}" if point_corr is not None else "Pearson r         : N/A")

    # Segment-level summary by motion class changes.
    segment_rows = []
    segments = contiguous_motion_segments(df_eval)
    for seg in segments:
        seg_df = df_eval.iloc[seg["start_idx"]:seg["end_idx"] + 1].copy()
        seg_gt = seg_df[col_gt_hr].dropna().values.astype(float)
        seg_hr = seg_df[col_rppg_hr].dropna().values.astype(float)
        seg_rr = seg_df[col_live_rr].dropna().values.astype(float) if col_live_rr in seg_df.columns else np.array([])

        seg_gt_median = float(np.median(seg_gt)) if len(seg_gt) > 0 else None
        seg_hr_median = float(np.median(seg_hr)) if len(seg_hr) > 0 else None
        seg_rr_median = float(np.median(seg_rr)) if len(seg_rr) > 0 else None
        seg_error = seg_hr_median - seg_gt_median if seg_hr_median is not None and seg_gt_median is not None else None

        segment_rows.append({
            "Segment_ID": seg["segment_id"],
            "Motion_Class": seg["motion_class"],
            "Start_sec": seg["start_sec"],
            "End_sec": seg["end_sec"],
            "Duration_sec": seg["end_sec"] - seg["start_sec"],
            "GT_HR_Median": seg_gt_median,
            "Live_rPPG_HR_Median": seg_hr_median,
            "Signed_Error": seg_error,
            "Abs_Error": abs(seg_error) if seg_error is not None else None,
            "Live_RR_Median": seg_rr_median,
            "HR_Sample_Count": int(len(seg_hr)),
            "GT_Sample_Count": int(len(seg_gt))
        })

    segment_df = pd.DataFrame(segment_rows)
    segment_summary_path = "live_motion_segment_metric_summary.csv"
    segment_df.to_csv(segment_summary_path, index=False)

    print("\nMotion-segment metric summary:")
    if len(segment_df) == 0:
        print("No motion segments found.")
    else:
        for _, row in segment_df.iterrows():
            hr_txt = "N/A" if pd.isna(row["Live_rPPG_HR_Median"]) else f"{row['Live_rPPG_HR_Median']:.2f} BPM"
            gt_txt = "N/A" if pd.isna(row["GT_HR_Median"]) else f"{row['GT_HR_Median']:.2f} BPM"
            err_txt = "N/A" if pd.isna(row["Signed_Error"]) else f"{row['Signed_Error']:+.2f} BPM"
            print(
                f"  Segment {int(row['Segment_ID']):02d} | {row['Motion_Class']:<11} | "
                f"{row['Start_sec']:.1f}-{row['End_sec']:.1f}s | "
                f"rPPG={hr_txt} | GT={gt_txt} | error={err_txt}"
            )
    print(f"Segment summary saved → {segment_summary_path}")

    fs_ppg = 30.0
    HAS_CMS_FFT = col_cms_fft in df_raw.columns and df_raw[col_cms_fft].notna().sum() > 0
    HAS_WAVEFORM = col_wave in df_raw.columns and df_raw[col_wave].notna().sum() > 0
    HAS_SPO2 = col_spo2 in df_raw.columns and df_raw[col_spo2].notna().sum() > 0

    stable_fft = int_mae = None
    if HAS_CMS_FFT:
        cms_fft_clean = df_eval[[col_gt_hr, col_cms_fft]].dropna()
        if len(cms_fft_clean) > 0:
            hr_fft_s = cms_fft_clean[col_cms_fft].values.astype(float)
            gt_for_fft = cms_fft_clean[col_gt_hr].values.astype(float)
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
            peaks_ppg, _ = find_peaks(
                wf_det,
                distance=int(fs_ppg * 60.0 / 180.0),
                prominence=0.1 * np.std(wf_det)
            )
            if len(peaks_ppg) >= 4:
                hr_wf = float(np.median(60.0 / np.diff(peaks_ppg / fs_ppg)))
                pi_proxy = float(np.std(wf_det) / (np.mean(waveform) + 1e-8) * 100.0)
                pav_cv = float(
                    np.std(wf_det[peaks_ppg]) /
                    (np.mean(np.abs(wf_det[peaks_ppg])) + 1e-8) * 100.0
                )
                print(f"\nHR from waveform  : {hr_wf:.2f} BPM")
                print(f"Waveform HR err   : {hr_wf - GT_SCALAR:+.2f} BPM")
                print(f"Perfusion Index   : {pi_proxy:.2f} %")
                print(f"Pulse Amp CV      : {pav_cv:.2f} %")

    spo2 = None
    if HAS_SPO2:
        spo2 = df_raw[col_spo2].dropna().values.astype(float)
        if len(spo2) > 0:
            print(f"\nSpO2 mean/std     : {np.mean(spo2):.2f} / {np.std(spo2):.2f} %")
            print(f"SpO2 range        : {spo2.min():.0f}–{spo2.max():.0f} %")

    C_GT = "#1565C0"
    C_RPPG = "#C62828"
    C_FFT = "#2E7D32"
    C_WAVE = "#00838F"

    fig = plt.figure(figsize=(18, 10))
    gs_eval = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    fig.suptitle(
        f"Live rPPG Evaluation\n"
        f"GT: {GT_SCALAR:.1f} BPM | "
        f"Live rPPG median: {'%.1f BPM' % RPPG_CONSENSUS_HR if RPPG_CONSENSUS_HR is not None else 'N/A'} | "
        f"RR: {'%.1f br/min' % RR_FINAL if RR_FINAL is not None else 'N/A'}",
        fontsize=12,
        fontweight="bold"
    )

    ax1 = fig.add_subplot(gs_eval[0, 0])
    ax1.plot(df_raw["Elapsed_Time_sec"], df_raw[col_gt_hr], color=C_GT, linewidth=1.3, label="CMS GT HR")
    ax1.plot(df_raw["Elapsed_Time_sec"], df_raw[col_rppg_hr], color=C_RPPG, linewidth=1.8, label="Live rPPG HR")
    ax1.axhline(GT_SCALAR, color=C_GT, linestyle="--", alpha=0.5, label=f"GT median={GT_SCALAR:.1f}")
    if RPPG_CONSENSUS_HR is not None:
        ax1.axhline(RPPG_CONSENSUS_HR, color=C_RPPG, linestyle="--", alpha=0.7, label=f"rPPG median={RPPG_CONSENSUS_HR:.1f}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("HR (BPM)")
    ax1.set_title("CMS GT HR vs Live rPPG HR")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs_eval[0, 1])
    if HAS_CMS_FFT:
        ax2.plot(df_raw["Elapsed_Time_sec"], df_raw[col_cms_fft], color=C_FFT, linewidth=1.0, alpha=0.8, label="CMS HR_FFT")
        ax2.plot(df_raw["Elapsed_Time_sec"], df_raw[col_gt_hr], color=C_GT, linewidth=1.3, alpha=0.5, label="CMS Hardware HR")
        ax2.set_title("CMS HR_FFT Convergence")
        ax2.legend(fontsize=7)
    elif HAS_WAVEFORM and waveform is not None:
        wf_time = np.arange(len(waveform)) / fs_ppg
        ax2.plot(wf_time, waveform, color=C_WAVE, linewidth=0.9, alpha=0.8, label="CMS waveform")
        if len(peaks_ppg) >= 4:
            ax2.plot(peaks_ppg / fs_ppg, waveform[peaks_ppg], "x", color=C_RPPG, markersize=5, label="Peaks")
        ax2.set_title("CMS PPG Waveform")
        ax2.legend(fontsize=7)
    else:
        ax2.text(0.5, 0.5, "No CMS waveform/FFT", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("BPM / amplitude")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs_eval[0, 2])
    if HAS_SPO2 and spo2 is not None and len(spo2) > 0:
        spo2_time = df_raw.loc[df_raw[col_spo2].notna(), "Elapsed_Time_sec"].values
        ax3.plot(spo2_time, spo2, color="#6A1B9A", linewidth=1.5)
        ax3.axhline(95, color="orange", linestyle="--", linewidth=1, label="95%")
        ax3.axhline(92, color="red", linestyle="--", linewidth=1, label="92%")
        ax3.set_ylim(85, 102)
        ax3.legend(fontsize=7)
        ax3.set_title("SpO2")
    else:
        ax3.text(0.5, 0.5, "SpO2 N/A", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("SpO2 (%)")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs_eval[1, 0])
    seg_valid = segment_df.dropna(subset=["GT_HR_Median", "Live_rPPG_HR_Median"]) if len(segment_df) else pd.DataFrame()
    if len(seg_valid) >= 2:
        ba_means = (seg_valid["GT_HR_Median"].values + seg_valid["Live_rPPG_HR_Median"].values) / 2.0
        ba_diffs = seg_valid["Live_rPPG_HR_Median"].values - seg_valid["GT_HR_Median"].values
        ba_bias = float(np.mean(ba_diffs))
        ba_std = float(np.std(ba_diffs, ddof=1))
        loa_upper = ba_bias + 1.96 * ba_std
        loa_lower = ba_bias - 1.96 * ba_std
        ax4.scatter(ba_means, ba_diffs, color=C_RPPG, alpha=0.75, s=55)
        ax4.axhline(ba_bias, color="black", linestyle="-", label=f"Bias {ba_bias:+.2f}")
        ax4.axhline(loa_upper, color="red", linestyle="--", label=f"+LoA {loa_upper:+.2f}")
        ax4.axhline(loa_lower, color="red", linestyle="--", label=f"−LoA {loa_lower:+.2f}")
        for i, (xv, yv) in enumerate(zip(ba_means, ba_diffs)):
            ax4.annotate(f"S{i+1}", (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax4.legend(fontsize=7)
        ax4.set_title("Bland-Altman by motion segment")
    elif RPPG_CONSENSUS_HR is not None:
        ax4.scatter([np.mean([RPPG_CONSENSUS_HR, GT_SCALAR])], [RPPG_CONSENSUS_HR - GT_SCALAR], color=C_RPPG, s=80)
        ax4.axhline(0, color="gray", linestyle=":")
        ax4.set_title("Bland-Altman overall")
    else:
        ax4.text(0.5, 0.5, "No rPPG HR", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_xlabel("Mean HR (BPM)")
    ax4.set_ylabel("rPPG − GT (BPM)")
    ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs_eval[1, 1])
    if len(seg_valid) >= 2:
        x_gt = seg_valid["GT_HR_Median"].values.astype(float)
        y_rppg = seg_valid["Live_rPPG_HR_Median"].values.astype(float)
        ax5.scatter(x_gt, y_rppg, color=C_RPPG, alpha=0.75, s=55)
        lim_min = min(x_gt.min(), y_rppg.min()) - 3
        lim_max = max(x_gt.max(), y_rppg.max()) + 3
        ax5.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="Identity")
        if len(seg_valid) >= 3:
            sl, ic, rv, *_ = stats.linregress(x_gt, y_rppg)
            xf = np.array([lim_min, lim_max])
            ax5.plot(xf, sl * xf + ic, color=C_RPPG, linewidth=1.5, label=f"r={rv:.3f}")
        for i, (xv, yv) in enumerate(zip(x_gt, y_rppg)):
            ax5.annotate(f"S{i+1}", (xv, yv), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax5.set_xlim(lim_min, lim_max)
        ax5.set_ylim(lim_min, lim_max)
        ax5.legend(fontsize=7)
        ax5.set_title("Segment correlation")
    elif len(df_pair) >= 5:
        sample_pair = df_pair.iloc[::max(1, len(df_pair)//200)]
        ax5.scatter(sample_pair[col_gt_hr], sample_pair[col_rppg_hr], color=C_RPPG, alpha=0.35, s=16)
        ax5.set_title("Pointwise live correlation")
    elif RPPG_CONSENSUS_HR is not None:
        ax5.scatter([GT_SCALAR], [RPPG_CONSENSUS_HR], color=C_RPPG, s=80)
        lim = [min(GT_SCALAR, RPPG_CONSENSUS_HR) - 5, max(GT_SCALAR, RPPG_CONSENSUS_HR) + 5]
        ax5.plot(lim, lim, "k--", linewidth=1)
        ax5.set_title("Correlation overall")
    else:
        ax5.text(0.5, 0.5, "No paired data", ha="center", va="center", transform=ax5.transAxes)
    ax5.set_xlabel("GT HR (BPM)")
    ax5.set_ylabel("Live rPPG HR (BPM)")
    ax5.grid(alpha=0.3)

    ax6 = fig.add_subplot(gs_eval[1, 2])
    if len(seg_valid) > 0:
        errs_bar = seg_valid["Live_rPPG_HR_Median"].values - seg_valid["GT_HR_Median"].values
        labels = [f"S{int(s)}\n{m}" for s, m in zip(seg_valid["Segment_ID"], seg_valid["Motion_Class"])]
        colors_b = ["#C62828" if e > 0 else "#1565C0" for e in errs_bar]
        ax6.bar(range(len(errs_bar)), errs_bar, color=colors_b, alpha=0.8, edgecolor="white")
        ax6.axhline(0, color="black", linewidth=1)
        ax6.axhline(5, color="orange", linestyle="--", linewidth=1, label="±5 BPM")
        ax6.axhline(-5, color="orange", linestyle="--", linewidth=1)
        ax6.set_xticks(range(len(errs_bar)))
        ax6.set_xticklabels(labels, fontsize=7, rotation=0)
        ax6.set_ylabel("Error (BPM)")
        ax6.set_title("Per-motion-segment error")
        ax6.legend(fontsize=7)
    else:
        ax6.text(0.5, 0.5, "No segment HR", ha="center", va="center", transform=ax6.transAxes)
    ax6.grid(alpha=0.3)

    dashboard_path = "live_rppg_evaluation_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nDashboard saved → {dashboard_path}")

    summary_rows = [{
        "GT_Median_HR": GT_SCALAR,
        "GT_Mean_HR": GT_MEAN,
        "GT_Std_HR": GT_STD,
        "Live_rPPG_HR_Median": RPPG_CONSENSUS_HR,
        "Live_rPPG_HR_Mean": RPPG_MEAN_HR,
        "Live_rPPG_HR_Last": RPPG_LAST_HR,
        "Signed_Error": error,
        "MAE": abs_error,
        "MAPE_percent": pct_error,
        "Clinical_Pass": clinical_pass,
        "Pointwise_MAE": point_mae,
        "Pointwise_RMSE": point_rmse,
        "Pointwise_MAPE_percent": point_mape,
        "Pointwise_Pearson_r": point_corr,
        "Live_RR_Median": RR_FINAL,
        "CMS_HR_FFT_Stable": stable_fft,
        "CMS_HR_FFT_MAE": int_mae,
        "HR_from_CMS_Waveform": hr_wf,
        "Perfusion_Index_Proxy": pi_proxy,
        "Pulse_Amplitude_CV": pav_cv,
        "SpO2_Mean": float(np.mean(spo2)) if spo2 is not None and len(spo2) > 0 else None,
        "SpO2_Std": float(np.std(spo2)) if spo2 is not None and len(spo2) > 0 else None,
        "Valid_Live_HR_Count": int(len(rppg_valid)),
        "Valid_GT_Count": int(len(gt_series)),
        "Paired_Count": int(len(df_pair))
    }]

    summary_df = pd.DataFrame(summary_rows)
    summary_path = "live_rppg_metric_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("  COMPLETE LIVE EVALUATION SUMMARY")
    print("=" * 60)
    for key, value in summary_rows[0].items():
        print(f"  {key:<28}: {value}")
    print("=" * 60)
    print(f"Summary saved   → {summary_path}")
    print(f"Segments saved  → {segment_summary_path}")
    print(f"Dashboard saved → {dashboard_path}")

    return {
        "summary": summary_df,
        "segments": segment_df,
        "dashboard_path": dashboard_path,
        "summary_path": summary_path,
        "segment_summary_path": segment_summary_path
    }



# ============================================================
# MAIN LIVE ACQUISITION LOOP
# ============================================================

def run_live_acquisition():
    print("============================================================")
    print("LIVE RGB rPPG + CMS50D ACQUISITION")
    print("============================================================")
    print(f"Video output : {VIDEO_FILENAME}")
    print(f"CSV output   : {CSV_FILENAME}")
    print(f"Duration     : {DURATION_SECONDS} s")
    print(f"Target FPS   : {TARGET_FPS}")
    print(f"CMS50D port  : {CMS50D_PORT}")
    print("============================================================")

    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened. Check WEBCAM_INDEX.")

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        VIDEO_FILENAME,
        fourcc,
        TARGET_FPS,
        (actual_width, actual_height)
    )

    if not out.isOpened():
        cap.release()
        raise RuntimeError("VideoWriter could not be opened.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    monitor = CMS50D(port=CMS50D_PORT)
    monitor.connect()

    csv_file = open(CSV_FILENAME, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Timestamp",
        "Frame_Index",

        "CMS_Waveform",
        "CMS_SpO2",
        "CMS_Pulse_Rate_Hardware",
        "CMS_HR_FFT",
        "CMS_HR_Peak",
        "CMS_Signal_Strength",
        "CMS_Probe_Error",

        "Live_Motion_Class",
        "Motion_Segment_ID",
        "Position_Segment",
        "Position_State",
        "Position_Shift",
        "Recent_Motion_Level",
        "RGB_Buffer_Seconds",
        "Live_rPPG_HR",
        "Live_RR",
        "Live_Green_FFT",
        "Live_POS_FFT",
        "Live_Green_Peak",
        "Live_POS_Peak",
        "Live_Green_Reliability",
        "Live_POS_Reliability",
        "Live_FS"
    ])

    frame_count = 0
    frame_interval = 1.0 / TARGET_FPS

    # CMS waveform rolling buffer
    cms_times = []
    cms_waveform = []

    # rPPG RGB rolling buffer
    rgb_times = []
    rgb_values = []

    # motion rolling buffer: cx, cy, face_size
    center_buffer = []

    # live results cache
    live_result = None
    live_motion_class = "warming_up"
    live_motion_features = {}
    last_live_update = 0.0

    # Position/transition state
    position_segment_id = 1
    position_reference = None
    position_change_counter = 0
    last_position_change_time = -1e9
    position_state = "initializing"
    position_shift = 0.0
    size_shift = 0.0
    recent_motion_level = 0.0

    # Motion segment state.
    # Each continuous motion condition gets one final HR summary.
    motion_segment_id = 0
    current_motion_segment = None
    motion_segments = []
    pending_motion_class = None
    pending_motion_count = 0

    print("\nReady.")
    print("Press ENTER to start live recording.")
    input()

    monitor.start_live_acquisition()
    start_time = time.time()

    print("\nLive recording started.")
    print("Press 'q' in preview window or Ctrl+C to stop early.")

    try:
        while time.time() - start_time < DURATION_SECONDS:
            loop_start = time.time()
            elapsed_total = time.time() - start_time
            current_datetime = datetime.datetime.now()

            ret, raw_frame = cap.read()

            if not ret:
                print("[Warning] Webcam frame drop.")
                break

            frame_count += 1

            if SAVE_RAW_VIDEO:
                out.write(raw_frame)

            # Face box and one-box RGB sample
            face_box = get_largest_face(raw_frame, face_cascade)
            rgb_sample = mean_rgb_from_face_box(raw_frame, face_box)

            current_center = None

            if face_box is not None:
                x, y, w, h = face_box
                current_center = [x + w / 2.0, y + h / 2.0, max(w, h)]
                center_buffer.append(current_center)

                # Keep motion buffer recent only
                max_motion_len = int(TARGET_FPS * LIVE_RGB_WINDOW_SECONDS)
                if len(center_buffer) > max_motion_len:
                    center_buffer = center_buffer[-max_motion_len:]

                # Initialize or update reference position
                if position_reference is None:
                    position_reference = current_center
                    position_segment_id = 1
                    position_state = "stable"

                position_shift, size_shift = compute_position_shift(current_center, position_reference)
                recent_motion_level = compute_recent_motion_level(center_buffer, RECENT_MOTION_FRAMES)

                possible_new_position = (
                    position_shift >= POSITION_CHANGE_THRESHOLD or
                    size_shift >= POSITION_SIZE_THRESHOLD
                )

                if possible_new_position:
                    position_change_counter += 1
                else:
                    position_change_counter = max(0, position_change_counter - 1)

                if position_change_counter >= POSITION_CONFIRM_FRAMES:
                    # New position confirmed. Reset rPPG buffers so old and new positions are not mixed.
                    position_segment_id += 1
                    position_reference = current_center
                    position_change_counter = 0
                    last_position_change_time = elapsed_total
                    position_state = "reacquiring"

                    # Do not wipe the live HR completely.
                    # The rolling RGB buffer is short, so old-position samples naturally disappear.
                    # Keeping it prevents HR from becoming permanently NA after a detected transition.

                    print(
                        f"[Position] New position detected -> segment {position_segment_id} "
                        f"at {elapsed_total:.1f}s. rPPG buffer reset."
                    )

            else:
                recent_motion_level = 0.0
                position_state = "face_lost"

            is_recently_changed = (elapsed_total - last_position_change_time) < REACQUIRE_SECONDS_AFTER_CHANGE
            is_currently_moving = recent_motion_level >= TRANSITION_MOTION_THRESHOLD

            if face_box is not None:
                if is_recently_changed:
                    position_state = "reacquiring"
                elif is_currently_moving:
                    position_state = "moving"
                else:
                    position_state = "stable"

            # Always add RGB samples when the face is detected.
            # The rolling window is short, so the estimate adapts to the new position.
            # We still show position_state so you know if the current value may be less reliable.
            if rgb_sample is not None:
                rgb_times.append(elapsed_total)
                rgb_values.append(rgb_sample)

                # Keep only rolling RGB window
                while len(rgb_times) > 2 and rgb_times[-1] - rgb_times[0] > LIVE_RGB_WINDOW_SECONDS:
                    rgb_times.pop(0)
                    rgb_values.pop(0)

            # Read CMS packets
            latest_packet = None

            while True:
                packet = monitor.get_latest_data()

                if packet is None:
                    break

                latest_packet = packet

                if packet["waveform"] is not None:
                    cms_times.append(packet["timestamp"])
                    cms_waveform.append(packet["waveform"])

            # Keep last 10 seconds of CMS waveform
            if cms_times:
                cutoff = current_datetime - datetime.timedelta(seconds=10)
                recent = [(t, w) for t, w in zip(cms_times, cms_waveform) if t >= cutoff]

                if recent:
                    cms_times, cms_waveform = map(list, zip(*recent))
                else:
                    cms_times, cms_waveform = [], []

            # CMS quick HR monitoring
            cms_hr_fft = None
            cms_hr_peak = None

            if len(cms_times) > 8:
                dt_window = (cms_times[-1] - cms_times[0]).total_seconds()
                cms_fs = len(cms_times) / dt_window if dt_window > 0 else 60.0
                cms_hr_fft = estimate_cms_hr_with_fft(cms_waveform, cms_fs)
                cms_hr_peak = estimate_cms_hr_with_peaks(cms_waveform, cms_fs)

            # Latest CMS data for this frame
            if latest_packet is not None:
                waveform_value = latest_packet["waveform"]
                spo2_value = latest_packet["spO2"]
                hardware_hr = latest_packet["pulse_rate"]
                signal_strength = latest_packet["signal_strength"]
                probe_error = latest_packet["probe_error"]
            else:
                waveform_value = None
                spo2_value = None
                hardware_hr = None
                signal_strength = None
                probe_error = None

            # Update live rPPG every few seconds.
            if elapsed_total - last_live_update >= LIVE_UPDATE_EVERY_SECONDS:
                live_motion_class, live_motion_features = estimate_live_motion_class(center_buffer)

                # -----------------------------
                # Motion segment detection
                # -----------------------------
                # A new segment starts only if the new motion class persists for
                # MOTION_SEGMENT_CONFIRM_UPDATES live updates. This prevents flickering.
                if live_motion_class != "warming_up":
                    if current_motion_segment is None:
                        motion_segment_id += 1
                        current_motion_segment = {
                            "id": motion_segment_id,
                            "motion_class": live_motion_class,
                            "start_sec": elapsed_total,
                            "end_sec": None,
                            "hr_values": [],
                            "rr_values": [],
                            "green_reliability": [],
                            "pos_reliability": []
                        }
                        motion_segments.append(current_motion_segment)
                        pending_motion_class = None
                        pending_motion_count = 0

                        print(
                            f"[Motion Segment] Start segment {motion_segment_id}: "
                            f"{live_motion_class} at {elapsed_total:.1f}s"
                        )

                    elif live_motion_class != current_motion_segment["motion_class"]:
                        if pending_motion_class == live_motion_class:
                            pending_motion_count += 1
                        else:
                            pending_motion_class = live_motion_class
                            pending_motion_count = 1

                        if pending_motion_count >= MOTION_SEGMENT_CONFIRM_UPDATES:
                            current_motion_segment["end_sec"] = elapsed_total

                            motion_segment_id += 1
                            current_motion_segment = {
                                "id": motion_segment_id,
                                "motion_class": live_motion_class,
                                "start_sec": elapsed_total,
                                "end_sec": None,
                                "hr_values": [],
                                "rr_values": [],
                                "green_reliability": [],
                                "pos_reliability": []
                            }
                            motion_segments.append(current_motion_segment)

                            pending_motion_class = None
                            pending_motion_count = 0

                            print(
                                f"[Motion Segment] Start segment {motion_segment_id}: "
                                f"{live_motion_class} at {elapsed_total:.1f}s"
                            )
                    else:
                        pending_motion_class = None
                        pending_motion_count = 0

                if live_motion_class == "warming_up":
                    correction_class = "stable"
                else:
                    correction_class = live_motion_class

                new_live_result = estimate_live_rppg_from_buffer(
                    rgb_times,
                    rgb_values,
                    selected_pipeline=correction_class
                )

                if new_live_result is not None:
                    live_result = new_live_result

                    # Add one HR sample to the current motion segment per live update.
                    if current_motion_segment is not None:
                        if live_result.get("consensus_hr") is not None:
                            current_motion_segment["hr_values"].append(float(live_result["consensus_hr"]))

                        if live_result.get("rr_final") is not None:
                            current_motion_segment["rr_values"].append(float(live_result["rr_final"]))

                        if live_result.get("green_reliability") is not None:
                            current_motion_segment["green_reliability"].append(float(live_result["green_reliability"]))

                        if live_result.get("pos_reliability") is not None:
                            current_motion_segment["pos_reliability"].append(float(live_result["pos_reliability"]))

                last_live_update = elapsed_total

            # Prepare live values
            live_hr = live_result["consensus_hr"] if live_result is not None else None
            live_rr = live_result["rr_final"] if live_result is not None else None
            live_green_fft = live_result["hr_green_fft"] if live_result is not None else None
            live_pos_fft = live_result["hr_pos_fft"] if live_result is not None else None
            live_green_peak = live_result["hr_green_peak"] if live_result is not None else None
            live_pos_peak = live_result["hr_pos_peak"] if live_result is not None else None
            live_green_rel = live_result["green_reliability"] if live_result is not None else None
            live_pos_rel = live_result["pos_reliability"] if live_result is not None else None
            live_fs = live_result["sampling_rate"] if live_result is not None else None
            rgb_buffer_seconds = (rgb_times[-1] - rgb_times[0]) if len(rgb_times) >= 2 else 0.0
            current_motion_segment_id = current_motion_segment["id"] if current_motion_segment is not None else None

            # Write synchronized CSV row
            csv_writer.writerow([
                current_datetime.strftime("%Y-%m-%d %H:%M:%S.%f"),
                frame_count,

                waveform_value,
                spo2_value,
                hardware_hr,
                cms_hr_fft,
                cms_hr_peak,
                signal_strength,
                probe_error,

                live_motion_class,
                current_motion_segment_id,
                position_segment_id,
                position_state,
                position_shift,
                recent_motion_level,
                rgb_buffer_seconds,
                live_hr,
                live_rr,
                live_green_fft,
                live_pos_fft,
                live_green_peak,
                live_pos_peak,
                live_green_rel,
                live_pos_rel,
                live_fs
            ])
            csv_file.flush()

            # Preview
            if SHOW_PREVIEW_WINDOW:
                display_frame = draw_one_face_box(raw_frame, face_box)

                cv2.putText(
                    display_frame,
                    f"Frame {frame_count} | Time {elapsed_total:.1f}/{DURATION_SECONDS:.1f}s | Motion: {live_motion_class} | MSeg: {fmt_value(current_motion_segment_id,0)} | Pos {position_segment_id}: {position_state}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"CMS HR: {fmt_value(hardware_hr, 0)} | SpO2: {fmt_value(spo2_value, 0)} | CMS FFT: {fmt_value(cms_hr_fft)}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"LIVE rPPG HR: {fmt_value(live_hr)} bpm | RR: {fmt_value(live_rr)} br/min | RGB {rgb_buffer_seconds:.1f}s",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Green FFT: {fmt_value(live_green_fft)} | POS FFT: {fmt_value(live_pos_fft)} | POS Peak: {fmt_value(live_pos_peak)}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"Rel G/P: {fmt_value(live_green_rel, 2)} / {fmt_value(live_pos_rel, 2)} | FS: {fmt_value(live_fs)} | shift {position_shift:.2f} | move {recent_motion_level:.3f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 100),
                    2
                )

                cv2.imshow("LIVE rPPG + CMS50D - One Face Box", display_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            # FPS control
            elapsed_loop = time.time() - loop_start
            sleep_time = frame_interval - elapsed_loop

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nLive recording interrupted by user.")

    finally:
        print("\nShutting down...")

        try:
            monitor.stop_live_acquisition()
        except Exception as e:
            print("[Warning] Could not stop CMS50D cleanly:", e)

        try:
            monitor.disconnect()
        except Exception as e:
            print("[Warning] Could not disconnect CMS50D cleanly:", e)

        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0.0

        # Finalize last motion segment and save one HR output per motion segment.
        if current_motion_segment is not None and current_motion_segment["end_sec"] is None:
            current_motion_segment["end_sec"] = total_time

        print("\n============================================================")
        print("MOTION SEGMENT HR SUMMARY")
        print("============================================================")
        print("Each row below is one continuous motion segment with one final HR value.")

        with open(MOTION_SUMMARY_FILENAME, "w", newline="") as summary_file:
            summary_writer = csv.writer(summary_file)
            summary_writer.writerow([
                "Motion_Segment_ID",
                "Motion_Class",
                "Start_sec",
                "End_sec",
                "Duration_sec",
                "Final_HR_Median",
                "Final_HR_Mean",
                "HR_Sample_Count",
                "Final_RR_Median",
                "RR_Sample_Count",
                "Green_Reliability_Median",
                "POS_Reliability_Median"
            ])

            for segment in motion_segments:
                hr_values = np.asarray(segment["hr_values"], dtype=float)
                rr_values = np.asarray(segment["rr_values"], dtype=float)
                green_rel = np.asarray(segment["green_reliability"], dtype=float)
                pos_rel = np.asarray(segment["pos_reliability"], dtype=float)

                start_sec = float(segment["start_sec"])
                end_sec = float(segment["end_sec"]) if segment["end_sec"] is not None else total_time
                duration_sec = end_sec - start_sec

                final_hr_median = float(np.median(hr_values)) if len(hr_values) > 0 else None
                final_hr_mean = float(np.mean(hr_values)) if len(hr_values) > 0 else None
                final_rr_median = float(np.median(rr_values)) if len(rr_values) > 0 else None
                green_rel_median = float(np.median(green_rel)) if len(green_rel) > 0 else None
                pos_rel_median = float(np.median(pos_rel)) if len(pos_rel) > 0 else None

                summary_writer.writerow([
                    segment["id"],
                    segment["motion_class"],
                    start_sec,
                    end_sec,
                    duration_sec,
                    final_hr_median,
                    final_hr_mean,
                    len(hr_values),
                    final_rr_median,
                    len(rr_values),
                    green_rel_median,
                    pos_rel_median
                ])

                hr_text = f"{final_hr_median:.2f} bpm" if final_hr_median is not None else "NA"
                rr_text = f"{final_rr_median:.2f} br/min" if final_rr_median is not None else "NA"

                reliability_note = ""
                if len(hr_values) < MIN_HR_VALUES_PER_MOTION_SEGMENT:
                    reliability_note = "  [low samples]"

                print(
                    f"Segment {segment['id']:02d} | "
                    f"{segment['motion_class']:<10} | "
                    f"{start_sec:6.1f}s -> {end_sec:6.1f}s | "
                    f"HR={hr_text} | RR={rr_text} | "
                    f"N_HR={len(hr_values)}{reliability_note}"
                )

        print(f"Motion summary saved: {MOTION_SUMMARY_FILENAME}")

        print("\n============================================================")
        print("LIVE ACQUISITION REPORT")
        print("============================================================")
        print(f"Video saved    : {VIDEO_FILENAME}")
        print(f"CSV saved      : {CSV_FILENAME}")
        print(f"Frames logged  : {frame_count}")
        print(f"Run time       : {total_time:.2f} s")
        print(f"Average FPS    : {avg_fps:.2f}")
        print(f"Position segments detected: {position_segment_id}")
        print(f"Motion segments detected  : {motion_segment_id}")
        if live_result is not None:
            print(f"Last live HR   : {fmt_value(live_result['consensus_hr'])} bpm")
            print(f"Last live RR   : {fmt_value(live_result['rr_final'])} br/min")
        print("============================================================")


# ============================================================
# RUN
# ============================================================

run_live_acquisition()

if METRIC_EVALUATION_AFTER_RECORDING:
    evaluation_outputs = evaluate_live_recording_metrics(CSV_FILENAME)
