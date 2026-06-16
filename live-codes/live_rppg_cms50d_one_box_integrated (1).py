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

from scipy.signal import butter, filtfilt, find_peaks, detrend, welch, hilbert
from scipy.interpolate import CubicSpline


# ============================================================
# USER CONFIGURATION
# ============================================================

CMS50D_PORT = "COM7"          # Change to your active CMS50D port
WEBCAM_INDEX = 0              # Usually 0 for laptop webcam
TARGET_FPS = 30
DURATION_SECONDS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

SHOW_PREVIEW_WINDOW = True
SAVE_RAW_VIDEO = True

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
LIVE_RGB_WINDOW_SECONDS = 25.0       # rolling RGB buffer used for live rPPG
LIVE_MIN_SECONDS_FOR_HR = 12.0       # do not estimate before this much data exists
LIVE_UPDATE_EVERY_SECONDS = 2.0      # update displayed rPPG result every N seconds

# Live RR settings
RR_MIN_BPM = 10
RR_MAX_BPM = 40
RR_RESAMPLE_FS = 4.0
RR_MIN_PEAKS_REQUIRED = 10
RR_MIN_DURATION_SEC = 20.0

timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_FILENAME = f"live_rppg_input_{timestamp_tag}.avi"
CSV_FILENAME = f"live_sync_results_{timestamp_tag}.csv"


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

    centers = np.asarray(center_buffer, dtype=float)

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

            if face_box is not None:
                x, y, w, h = face_box
                center_buffer.append([x + w / 2.0, y + h / 2.0, max(w, h)])

                # Keep motion buffer recent only
                max_motion_len = int(TARGET_FPS * LIVE_RGB_WINDOW_SECONDS)
                if len(center_buffer) > max_motion_len:
                    center_buffer = center_buffer[-max_motion_len:]

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

            # Update live rPPG every few seconds
            if elapsed_total - last_live_update >= LIVE_UPDATE_EVERY_SECONDS:
                live_motion_class, live_motion_features = estimate_live_motion_class(center_buffer)

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
                    f"Frame {frame_count} | Time {elapsed_total:.1f}/{DURATION_SECONDS:.1f}s | Motion: {live_motion_class}",
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
                    f"LIVE rPPG HR: {fmt_value(live_hr)} bpm | RR: {fmt_value(live_rr)} br/min",
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
                    f"Reliability G/P: {fmt_value(live_green_rel, 2)} / {fmt_value(live_pos_rel, 2)} | Live FS: {fmt_value(live_fs)}",
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

        print("\n============================================================")
        print("LIVE ACQUISITION REPORT")
        print("============================================================")
        print(f"Video saved    : {VIDEO_FILENAME}")
        print(f"CSV saved      : {CSV_FILENAME}")
        print(f"Frames logged  : {frame_count}")
        print(f"Run time       : {total_time:.2f} s")
        print(f"Average FPS    : {avg_fps:.2f}")
        if live_result is not None:
            print(f"Last live HR   : {fmt_value(live_result['consensus_hr'])} bpm")
            print(f"Last live RR   : {fmt_value(live_result['rr_final'])} br/min")
        print("============================================================")


# ============================================================
# RUN
# ============================================================

run_live_acquisition()
