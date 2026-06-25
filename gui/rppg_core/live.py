"""Live rPPG session: webcam-driven HR/RR/motion estimation with an
optional CMS50D oximeter overlay.

All signal processing is reused from rppg_core.pipeline (no duplicated
DSP). The live-only helpers (single-box ROI, rolling-buffer estimation,
lightweight motion class) are ported from
live_rppg_cms50d_original_settings_with_metrics.py with their logic
unchanged. The CMS50D parts activate only when a serial device is
provided/detected, so the camera-only path runs with no extra hardware.
"""

from __future__ import annotations

import base64
import csv
import datetime
import time

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import welch

from . import pipeline as P
from .device import CMS50D

# ------------------------------------------------------------
# Live settings (from the original live script)
# ------------------------------------------------------------
TARGET_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

LIVE_RGB_WINDOW_SECONDS = 25.0
# First live HR appears after this much RGB data. The two scripts differ:
# 12 s in the 30s/original script, 10 s in the 2-min motion-segment script.
LIVE_MIN_SECONDS_FOR_HR_ORIGINAL = 12.0
LIVE_MIN_SECONDS_FOR_HR_SEGMENT = 10.0
LIVE_MIN_SECONDS_FOR_HR = LIVE_MIN_SECONDS_FOR_HR_ORIGINAL  # default
LIVE_UPDATE_EVERY_SECONDS = 2.0

RR_MIN_BPM = 10
RR_MAX_BPM = 40
RR_RESAMPLE_FS = 4.0
RR_MIN_PEAKS_REQUIRED = 10
RR_MIN_DURATION_SEC = 20.0

# Position-change / transition handling (motion-segment mode only).
POSITION_CHANGE_THRESHOLD = 0.18      # normalized center shift vs face size
POSITION_SIZE_THRESHOLD = 0.22        # relative face-size change
POSITION_CONFIRM_FRAMES = 12          # frames before declaring a new position
RECENT_MOTION_FRAMES = 30             # ~1 s at 30 FPS (fixed, like the script)
# Classify motion over only the recent window so labels do not get stuck
# after a transition (zigzag -> stable, zigzag -> left_right, ...).
MOTION_CLASS_WINDOW_SECONDS = 8.0
# Updates a new motion class must persist (each update = LIVE_UPDATE_EVERY_SECONDS
# = 2 s) before it splits off a new segment. The script uses 2 (~4 s minimum
# segment), which spawns a fresh segment for every brief transition between real
# motion phases, so a 2-min run fills with short 2-sample segments. Raised to 4
# (~8 s of *sustained* motion) so transitions are absorbed into the surrounding
# phase and the list stays close to one row per real motion phase. Because of
# the pending-class reset, alternating noise (left_right/zigzag/left_right/...)
# never reaches the threshold and is fully absorbed. Higher = fewer segments.
# This is a deliberate divergence from the script for a cleaner segment list.
MOTION_SEGMENT_CONFIRM_UPDATES = 4
MIN_HR_VALUES_PER_MOTION_SEGMENT = 2  # min HR samples for a reliable segment summary
TRANSITION_MOTION_THRESHOLD = 0.030   # normalized frame-to-frame center motion
REACQUIRE_SECONDS_AFTER_CHANGE = 2.0  # visual reacquire period after a position change

# Preview streaming
PREVIEW_WIDTH = 480
JPEG_QUALITY = 65

# Face detection is the dominant per-frame cost. The script runs Haar on the
# full 640x480 gray every frame, which caps a single-purpose loop near 30 FPS
# but leaves a browser-streaming process (this one) stuck around ~23 FPS.
# Running the detector on a downscaled gray is ~1/scale^2 cheaper (0.5 -> ~4x),
# which restores the 30 FPS the rest of the pipeline is tuned for.
#
# Parity note: this is the one intentional divergence from the script. The
# effective minimum face size is preserved (minSize is scaled with the image),
# and the returned box is only quantized by 1/scale px (~2 px at 0.5), so the
# central RGB ROI — and therefore HR — is effectively unchanged. Set to 1.0 for
# byte-exact script detection (at the lower frame rate).
FACE_DETECT_SCALE = 0.5


# ============================================================
# Live face ROI helpers (single visible box)
# ============================================================

def get_largest_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if FACE_DETECT_SCALE != 1.0:
        det_gray = cv2.resize(gray, None, fx=FACE_DETECT_SCALE, fy=FACE_DETECT_SCALE)
        min_side = max(20, int(90 * FACE_DETECT_SCALE))
    else:
        det_gray = gray
        min_side = 90
    faces = face_cascade.detectMultiScale(
        det_gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side)
    )
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]
    if FACE_DETECT_SCALE != 1.0:
        inv = 1.0 / FACE_DETECT_SCALE
        x, y, w, h = int(x * inv), int(y * inv), int(w * inv), int(h * inv)
    return P.keep_box_inside_frame((x, y, w, h), frame.shape)


def mean_rgb_from_face_box(frame, face_box):
    """One RGB sample from a central skin-oriented crop of the face box."""
    if face_box is None:
        return None
    x, y, w, h = face_box
    cx = x + int(0.20 * w)
    cy = y + int(0.20 * h)
    cw = int(0.60 * w)
    ch = int(0.55 * h)
    cx, cy, cw, ch = P.keep_box_inside_frame((cx, cy, cw, ch), frame.shape)
    roi = frame[cy:cy + ch, cx:cx + cw, :]
    if roi.size == 0:
        return None
    mean_bgr = np.mean(roi.reshape(-1, 3), axis=0)
    return mean_bgr[::-1].astype(float)


def estimate_live_motion_class(center_buffer, window_seconds=None, fps=TARGET_FPS):
    """Lightweight rolling motion class from face-box centers.

    When window_seconds is given, only the most recent
    window_seconds * fps samples are used (the 2-min motion-segment
    behaviour) so the label is not dominated by older motion and can adapt
    after a transition. When None, the full buffer is used (the original
    30s behaviour).

    fps must be the script's fixed TARGET_FPS (30), NOT the measured
    processing rate. The classifier's direction-change thresholds
    (y_changes >= 5, total_changes >= 7, ...) are tuned for a 240-sample
    window (TARGET_FPS * 8 s). Scaling the window by the *actual* fps feeds
    far fewer samples on a slow machine, so a real zigzag drops below the
    thresholds and the label flickers between classes. The script always
    keeps the last 240 frames, which is what makes its label stable.
    """
    if len(center_buffer) < 20:
        return "warming_up", {}

    if window_seconds is not None:
        max_samples = max(20, int(fps * window_seconds))
        if len(center_buffer) > max_samples:
            center_buffer = center_buffer[-max_samples:]

    centers = np.asarray(center_buffer, dtype=float)
    cx = centers[:, 0]
    cy = centers[:, 1]
    size = np.median(centers[:, 2])
    if size <= 1:
        size = 1.0

    x = (cx - np.median(cx)) / size
    y = (cy - np.median(cy)) / size

    x_amp = P.robust_range(x)
    y_amp = P.robust_range(y)
    total_amp = float(np.sqrt(x_amp ** 2 + y_amp ** 2))
    y_to_x_ratio = float(y_amp / (x_amp + 1e-8))

    x_changes = P.count_direction_changes(x)
    y_changes = P.count_direction_changes(y)
    total_changes = x_changes + y_changes

    features = {
        "x_amp": x_amp,
        "y_amp": y_amp,
        "total_amp": total_amp,
        "y_to_x_ratio": y_to_x_ratio,
        "x_direction_changes": x_changes,
        "y_direction_changes": y_changes,
        "total_direction_changes": total_changes,
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


def compute_recent_motion_level(center_buffer, recent_frames=RECENT_MOTION_FRAMES):
    """Recent frame-to-frame face-center motion, normalized by face size.
    Higher = the subject is moving/repositioning (rPPG less reliable)."""
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
    return float(np.mean(np.sqrt(dx ** 2 + dy ** 2)))


def compute_position_shift(current_center, reference_center):
    """Normalized spatial shift and relative face-size change vs a reference."""
    if current_center is None or reference_center is None:
        return 0.0, 0.0
    current = np.asarray(current_center, dtype=float)
    reference = np.asarray(reference_center, dtype=float)
    ref_size = max(reference[2], 1.0)
    cur_size = max(current[2], 1.0)
    spatial_shift = float(np.sqrt(
        ((current[0] - reference[0]) / ref_size) ** 2
        + ((current[1] - reference[1]) / ref_size) ** 2
    ))
    size_shift = float(abs(cur_size - ref_size) / ref_size)
    return spatial_shift, size_shift


# ============================================================
# CMS50D waveform HR (oximeter, when connected)
# ============================================================

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
    peaks, _ = P.find_peaks(waveform, distance=min_distance, prominence=prominence)
    if len(peaks) < 2:
        return None
    intervals = np.diff(peaks) / sampling_rate
    bpm_values = 60.0 / intervals
    bpm_values = bpm_values[(bpm_values >= min_bpm) & (bpm_values <= max_bpm)]
    if len(bpm_values) == 0:
        return None
    return float(np.median(bpm_values))


# ============================================================
# Live rPPG HR + RR from rolling buffer (reuses pipeline DSP)
# ============================================================

def estimate_live_rr_from_pos(pos_filt, times, sampling_rate):
    if pos_filt is None or len(pos_filt) < int(sampling_rate * RR_MIN_DURATION_SEC):
        return None

    peak_distance = max(1, int(sampling_rate * 60.0 / P.MAX_BPM))
    peak_prominence = max(0.35 * np.std(pos_filt), 0.15)
    pos_peaks, _ = P.find_peaks(
        pos_filt - np.mean(pos_filt),
        distance=peak_distance,
        prominence=peak_prominence,
    )
    if len(pos_peaks) < RR_MIN_PEAKS_REQUIRED:
        return None

    peak_times = times[pos_peaks]
    rr_intervals = np.diff(peak_times)
    rr_times = 0.5 * (peak_times[:-1] + peak_times[1:])
    if len(rr_times) < 5:
        return None
    if rr_times[-1] - rr_times[0] < RR_MIN_DURATION_SEC:
        return None

    try:
        t_uniform = np.arange(rr_times[0], rr_times[-1], 1.0 / RR_RESAMPLE_FS)
        cs = CubicSpline(rr_times, rr_intervals)
        rr_resampled = P.detrend(cs(t_uniform))

        nperseg = max(min(len(rr_resampled), int(RR_RESAMPLE_FS * 32)), 16)
        freqs, psd = welch(
            rr_resampled, fs=RR_RESAMPLE_FS, window="hann",
            nperseg=nperseg, noverlap=nperseg // 2, scaling="density",
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
        delta = P.parabolic_interpolation(resp_psd, peak_idx)
        rr_freq = float(resp_freqs[peak_idx]) + delta * freq_step
        return float(np.clip(rr_freq * 60.0, RR_MIN_BPM, RR_MAX_BPM))
    except Exception:
        return None


def estimate_live_rppg_from_buffer(rgb_times, rgb_values, selected_pipeline,
                                   min_seconds_for_hr=LIVE_MIN_SECONDS_FOR_HR):
    if len(rgb_values) < 30:
        return None

    times = np.asarray(rgb_times, dtype=float)
    rgb_signal = np.asarray(rgb_values, dtype=float)
    duration = times[-1] - times[0]
    if duration < min_seconds_for_hr:
        return None

    sampling_rate = (len(times) - 1) / max(duration, 1e-8)

    green_norm = P.normalize_signal(rgb_signal[:, 1])
    green_filt = P.bandpass_filter(green_norm, fs=sampling_rate,
                                   min_bpm=P.MIN_BPM, max_bpm=P.MAX_BPM)

    pos_raw = P.pos_rppg(rgb_signal, fs=sampling_rate,
                         window_seconds=P.POS_WINDOW_SECONDS)
    try:
        pos_raw = P.detrend(pos_raw)
    except Exception:
        pos_raw = pos_raw - np.mean(pos_raw)
    pos_filt = P.bandpass_filter(P.normalize_signal(pos_raw), fs=sampling_rate,
                                 min_bpm=P.MIN_BPM, max_bpm=P.MAX_BPM)

    hr_green_fft, hr_pos_fft, hr_consensus, green_info, pos_info = \
        P.estimate_smart_fft_pair(green_filt, pos_filt, sampling_rate,
                                  min_bpm=P.MIN_BPM, max_bpm=P.MAX_BPM,
                                  trim_start_seconds=P.TRIM_START_SECONDS)

    hr_green_peak = P.estimate_hr_peaks(green_filt, sampling_rate,
                                        min_bpm=P.MIN_BPM, max_bpm=P.MAX_BPM)
    hr_pos_peak = P.estimate_hr_peaks(pos_filt, sampling_rate,
                                      min_bpm=P.MIN_BPM, max_bpm=P.MAX_BPM)

    hr_green_fft, hr_pos_fft, hr_consensus, peak_guidance_hr = \
        P.adaptive_final_correction(green_info, pos_info, hr_green_fft,
                                    hr_pos_fft, hr_consensus,
                                    hr_green_peak, hr_pos_peak)

    if selected_pipeline in ("zigzag", "left_right"):
        hr_green_fft, hr_pos_fft, hr_consensus, _ = \
            P.motion_peak_guidance_correction(hr_green_fft, hr_pos_fft,
                                              hr_consensus, peak_guidance_hr)

    if selected_pipeline == "left_right":
        hr_green_fft, hr_pos_fft, hr_consensus, _ = \
            P.left_right_candidate_guidance_correction(
                green_info, pos_info, hr_green_fft, hr_pos_fft,
                hr_consensus, peak_guidance_hr)
        hr_green_fft, hr_pos_fft, hr_consensus, _ = \
            P.left_right_low_lock_rescue(
                green_info, pos_info, hr_green_fft, hr_pos_fft,
                hr_consensus, peak_guidance_hr)
        hr_green_fft, hr_pos_fft, hr_consensus, _ = \
            P.left_right_high_hr_rescue(
                green_info, pos_info, hr_green_fft, hr_pos_fft,
                hr_consensus, peak_guidance_hr)

    rr_final = estimate_live_rr_from_pos(pos_filt, times, sampling_rate)

    return {
        "sampling_rate": sampling_rate,
        "consensus_hr": hr_consensus,
        "hr_green_fft": hr_green_fft,
        "hr_pos_fft": hr_pos_fft,
        "hr_green_peak": hr_green_peak,
        "hr_pos_peak": hr_pos_peak,
        "green_reliability": green_info["reliability"],
        "pos_reliability": pos_info["reliability"],
        "rr_final": rr_final,
    }


def _f(v):
    """JSON-safe float (handles numpy + None)."""
    if v is None:
        return None
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


# ============================================================
# Live session
# ============================================================

class LiveSession:
    """Drives a webcam (and optional CMS50D) and produces per-frame state.

    Call process_frame() repeatedly; each call grabs one frame, updates the
    rolling buffers, periodically re-estimates HR/RR/motion, optionally polls
    the oximeter, and returns a JSON-serialisable state dict (including a
    base64 JPEG preview). Call close() when done.
    """

    def __init__(self, camera_index=0, serial_port=None, record_path=None,
                 segment_mode=False):
        self.camera_index = camera_index
        self.serial_port = serial_port
        self.record_path = record_path
        # segment_mode = the 2-min "motion segment" behaviour: position-change
        # detection + per-segment HR summaries. Off = original 30s behaviour.
        self.segment_mode = segment_mode
        # Per-mode constants that differ between the two scripts.
        self.min_seconds_for_hr = (
            LIVE_MIN_SECONDS_FOR_HR_SEGMENT if segment_mode
            else LIVE_MIN_SECONDS_FOR_HR_ORIGINAL
        )

        self.cap = None
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.monitor = None
        self.cms_connected = False   # serial port opened + acquisition started
        self.cms_streaming = False   # at least one data packet received
        self.cms_error = None        # human-readable reason connection failed

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.frame_count = 0
        self.start_time = None
        # Measured processing rate; motion windows use this so they stay a true
        # N seconds in time even when the machine can't sustain 30 FPS.
        self.actual_fps = float(TARGET_FPS)

        # Latest frame/state for on-demand preview encoding (streaming layer).
        self._last_frame = None
        self._last_face_box = None
        self._last_state = None
        self._capture_done = False
        self._capture_error = None

        self.rgb_times = []
        self.rgb_values = []
        self.center_buffer = []

        # CMS50D waveform buffer (for CMS_HR_FFT / CMS_HR_Peak columns).
        self.cms_times = []
        self.cms_waveform = []
        # Most recent valid device pulse rate, sampled into each motion segment
        # so its median is comparable to the rPPG HR median (segment mode).
        self.last_hardware_hr = None

        self.live_result = None
        self.live_motion_class = "warming_up"
        self.last_live_update = 0.0

        # Motion-segment / position-tracking state (segment_mode only).
        self.position_reference = None
        self.position_change_counter = 0
        self.last_position_change_time = -1e9
        self.position_segment_id = 0
        self.position_state = "warming_up"
        self.recent_motion_level = 0.0
        self.position_shift = 0.0
        self.current_motion_segment = None
        self.motion_segments = []
        self.motion_segment_id = 0
        self.pending_motion_class = None
        self.pending_motion_count = 0

    # -- lifecycle ------------------------------------------------------

    def open(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Webcam index {self.camera_index} could not be opened. "
                "On macOS, grant the host app Camera permission in "
                "System Settings > Privacy & Security > Camera."
            )

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.record_path:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(
                str(self.record_path), fourcc, TARGET_FPS, (width, height)
            )
            self._open_csv()

        if self.serial_port:
            try:
                self.monitor = CMS50D(port=self.serial_port)
                self.monitor.connect()
                self.monitor.start_live_acquisition()
                self.cms_connected = True
                # Verify the device is actually streaming, not just that the
                # port opened. The CMS50D must be powered on with the finger
                # probe inserted, or it sends nothing.
                self.cms_streaming = self._wait_for_first_packet(timeout=3.0)
                if not self.cms_streaming:
                    self.cms_error = (
                        f"Port {self.serial_port} opened but no data was received. "
                        "Check the CMS50D is powered on, the finger probe is "
                        "inserted, and this is the correct port."
                    )
            except Exception as exc:  # oximeter optional; degrade gracefully
                self.monitor = None
                self.cms_connected = False
                self.cms_streaming = False
                self.cms_error = f"Could not open {self.serial_port}: {exc}"

        self.start_time = time.time()
        return self

    def _wait_for_first_packet(self, timeout=3.0):
        """Poll the oximeter queue until a packet arrives or timeout. The
        first packet seeds the waveform buffer used for CMS_HR columns."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            pkt = self.monitor.get_latest_data()
            if pkt is not None:
                if pkt["waveform"] is not None:
                    self.cms_times.append(pkt["timestamp"])
                    self.cms_waveform.append(pkt["waveform"])
                return True
            time.sleep(0.05)
        return False

    def _open_csv(self):
        csv_path = str(self.record_path).rsplit(".", 1)[0] + "_sync.csv"
        self.csv_file = open(csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # Columns match the original live scripts exactly: the 30s script in
        # original mode, the 2-min script (extra position/segment columns) in
        # segment mode. The recording evaluator consumes either directly.
        header = [
            "Timestamp", "Frame_Index",
            "CMS_Waveform", "CMS_SpO2", "CMS_Pulse_Rate_Hardware",
            "CMS_HR_FFT", "CMS_HR_Peak",
            "CMS_Signal_Strength", "CMS_Probe_Error",
            "Live_Motion_Class",
        ]
        if self.segment_mode:
            header += [
                "Motion_Segment_ID", "Position_Segment", "Position_State",
                "Position_Shift", "Recent_Motion_Level", "RGB_Buffer_Seconds",
            ]
        header += [
            "Live_rPPG_HR", "Live_RR",
            "Live_Green_FFT", "Live_POS_FFT", "Live_Green_Peak", "Live_POS_Peak",
            "Live_Green_Reliability", "Live_POS_Reliability", "Live_FS",
        ]
        self.csv_writer.writerow(header)

    def close(self):
        # Idempotent: the backend calls this once after capture to flush the
        # recording (CSV + video) before reading it back, and again in its
        # finally block as a safety net. Null out each handle after releasing
        # so the second call is a no-op.
        if self.monitor is not None:
            try:
                self.monitor.stop_live_acquisition()
                self.monitor.disconnect()
            except Exception:
                pass
            self.monitor = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # -- per-frame ------------------------------------------------------

    def _poll_cms(self, now):
        """Drain the oximeter queue; accumulate a 10 s waveform buffer.
        Returns the latest packet dict, or None."""
        if self.monitor is None:
            return None
        latest = None
        while True:
            pkt = self.monitor.get_latest_data()
            if pkt is None:
                break
            latest = pkt
            if pkt["waveform"] is not None:
                self.cms_times.append(pkt["timestamp"])
                self.cms_waveform.append(pkt["waveform"])
        if self.cms_times:
            cutoff = now - datetime.timedelta(seconds=10)
            recent = [(t, w) for t, w in zip(self.cms_times, self.cms_waveform)
                      if t >= cutoff]
            if recent:
                self.cms_times, self.cms_waveform = map(list, zip(*recent))
            else:
                self.cms_times, self.cms_waveform = [], []
        return latest

    def _cms_hr(self):
        """CMS HR from the waveform buffer (FFT, peak), or (None, None)."""
        if len(self.cms_times) <= 8:
            return None, None
        dt = (self.cms_times[-1] - self.cms_times[0]).total_seconds()
        cms_fs = len(self.cms_times) / dt if dt > 0 else 60.0
        return (
            estimate_cms_hr_with_fft(self.cms_waveform, cms_fs),
            estimate_cms_hr_with_peaks(self.cms_waveform, cms_fs),
        )

    def process_frame(self):
        """Grab + process one frame. Returns state dict, or None if the
        camera stops delivering frames."""
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.frame_count += 1
        now = datetime.datetime.now()
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.actual_fps = min(TARGET_FPS, self.frame_count / elapsed)

        if self.writer is not None:
            self.writer.write(frame)

        face_box = get_largest_face(frame, self.face_cascade)
        rgb_sample = mean_rgb_from_face_box(frame, face_box)
        current_center = None

        if face_box is not None:
            x, y, w, h = face_box
            current_center = [x + w / 2.0, y + h / 2.0, max(w, h)]
            self.center_buffer.append(current_center)
            max_len = int(TARGET_FPS * LIVE_RGB_WINDOW_SECONDS)
            if len(self.center_buffer) > max_len:
                self.center_buffer = self.center_buffer[-max_len:]

        if self.segment_mode:
            self._track_position(current_center, elapsed)

        if rgb_sample is not None:
            self.rgb_times.append(elapsed)
            self.rgb_values.append(rgb_sample)
            while (len(self.rgb_times) > 2
                   and self.rgb_times[-1] - self.rgb_times[0] > LIVE_RGB_WINDOW_SECONDS):
                self.rgb_times.pop(0)
                self.rgb_values.pop(0)

        packet = self._poll_cms(now)
        if packet is not None and packet.get("pulse_rate"):
            self.last_hardware_hr = float(packet["pulse_rate"])

        # Periodic re-estimation.
        if elapsed - self.last_live_update >= LIVE_UPDATE_EVERY_SECONDS:
            # Segment mode classifies over the recent window (2-min script
            # behaviour) so labels adapt after a transition; original mode
            # uses the full buffer.
            motion_window = MOTION_CLASS_WINDOW_SECONDS if self.segment_mode else None
            # Use the fixed TARGET_FPS window (last 240 frames), exactly like
            # the script — NOT actual_fps. A measured-fps window feeds too few
            # samples below 30 FPS and makes the motion label flicker.
            self.live_motion_class, _ = estimate_live_motion_class(
                self.center_buffer, window_seconds=motion_window, fps=TARGET_FPS
            )

            if self.segment_mode:
                self._detect_segment_switch(elapsed)

            correction_class = (
                "stable" if self.live_motion_class == "warming_up"
                else self.live_motion_class
            )
            new_result = estimate_live_rppg_from_buffer(
                self.rgb_times, self.rgb_values, correction_class,
                min_seconds_for_hr=self.min_seconds_for_hr,
            )
            if new_result is not None:
                self.live_result = new_result
                if self.segment_mode and self.current_motion_segment is not None:
                    self._accumulate_segment(new_result)
            self.last_live_update = elapsed

        state = self._build_state(frame, face_box, packet, now, elapsed)

        if self.csv_writer is not None:
            self._write_csv_row(packet, now, state)

        # Keep the latest frame for on-demand preview encoding (the WS layer
        # encodes/streams at a throttled rate, decoupled from this loop so
        # processing stays at the camera's ~30 FPS like the scripts).
        self._last_frame = frame
        self._last_face_box = face_box
        self._last_state = state

        return state

    def run_capture(self, stop_event, duration=None):
        """Process frames in a tight loop paced to TARGET_FPS, exactly like the
        scripts (frame_interval sleep), until stop_event is set, the duration
        elapses, or the camera stops. Runs in its own thread so the buffers
        fill at ~30 FPS regardless of the preview-streaming rate."""
        frame_interval = 1.0 / TARGET_FPS
        try:
            while not stop_event.is_set():
                loop_start = time.time()
                state = self.process_frame()
                if state is None:
                    self._capture_error = "Camera stopped delivering frames."
                    break
                if duration and state["elapsed"] >= float(duration):
                    break
                sleep_time = frame_interval - (time.time() - loop_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except Exception as exc:
            self._capture_error = str(exc)
        finally:
            self._capture_done = True

    # -- motion-segment mode -------------------------------------------

    def _track_position(self, current_center, elapsed):
        """Detect position changes and set position_state (segment mode)."""
        if current_center is not None:
            if self.position_reference is None:
                self.position_reference = current_center
                self.position_segment_id = 1
                self.position_state = "stable"

            self.position_shift, size_shift = compute_position_shift(
                current_center, self.position_reference
            )
            # Fixed 30-frame window like the script (RECENT_MOTION_FRAMES), not
            # an actual_fps-scaled one, so the moving/stable state is stable.
            self.recent_motion_level = compute_recent_motion_level(
                self.center_buffer, RECENT_MOTION_FRAMES
            )

            possible_new = (
                self.position_shift >= POSITION_CHANGE_THRESHOLD
                or size_shift >= POSITION_SIZE_THRESHOLD
            )
            if possible_new:
                self.position_change_counter += 1
            else:
                self.position_change_counter = max(0, self.position_change_counter - 1)

            if self.position_change_counter >= POSITION_CONFIRM_FRAMES:
                # New position confirmed; the short rolling RGB buffer naturally
                # flushes old-position samples, so HR adapts without a hard wipe.
                self.position_segment_id += 1
                self.position_reference = current_center
                self.position_change_counter = 0
                self.last_position_change_time = elapsed
                self.position_state = "reacquiring"
        else:
            self.recent_motion_level = 0.0
            self.position_state = "face_lost"

        is_recently_changed = (
            elapsed - self.last_position_change_time
        ) < REACQUIRE_SECONDS_AFTER_CHANGE
        is_currently_moving = self.recent_motion_level >= TRANSITION_MOTION_THRESHOLD

        if current_center is not None:
            if is_recently_changed:
                self.position_state = "reacquiring"
            elif is_currently_moving:
                self.position_state = "moving"
            else:
                self.position_state = "stable"

    def _new_segment(self, elapsed):
        self.motion_segment_id += 1
        seg = {
            "id": self.motion_segment_id,
            "motion_class": self.live_motion_class,
            "start_sec": elapsed,
            "end_sec": None,
            "hr_values": [],
            "rr_values": [],
            "device_hr_values": [],
            "green_reliability": [],
            "pos_reliability": [],
        }
        self.motion_segments.append(seg)
        self.current_motion_segment = seg
        self.pending_motion_class = None
        self.pending_motion_count = 0

    def _detect_segment_switch(self, elapsed):
        """Start/switch the active motion segment with persistence guarding."""
        if self.live_motion_class == "warming_up":
            return

        if self.current_motion_segment is None:
            self._new_segment(elapsed)
        elif self.live_motion_class != self.current_motion_segment["motion_class"]:
            if self.pending_motion_class == self.live_motion_class:
                self.pending_motion_count += 1
            else:
                self.pending_motion_class = self.live_motion_class
                self.pending_motion_count = 1

            if self.pending_motion_count >= MOTION_SEGMENT_CONFIRM_UPDATES:
                self.current_motion_segment["end_sec"] = elapsed
                self._new_segment(elapsed)
        else:
            self.pending_motion_class = None
            self.pending_motion_count = 0

    def _accumulate_segment(self, result):
        seg = self.current_motion_segment
        if result.get("consensus_hr") is not None:
            seg["hr_values"].append(float(result["consensus_hr"]))
        if result.get("rr_final") is not None:
            seg["rr_values"].append(float(result["rr_final"]))
        # Device (CMS50D) HR sampled at the same cadence as the rPPG HR above,
        # so the per-segment device median lines up with the rPPG median.
        if self.last_hardware_hr is not None:
            seg["device_hr_values"].append(float(self.last_hardware_hr))
        if result.get("green_reliability") is not None:
            seg["green_reliability"].append(float(result["green_reliability"]))
        if result.get("pos_reliability") is not None:
            seg["pos_reliability"].append(float(result["pos_reliability"]))

    def segment_summaries(self):
        """Per-segment HR/RR summaries (segment mode). One row per segment."""
        total_time = (time.time() - self.start_time) if self.start_time else 0.0
        out = []
        for seg in self.motion_segments:
            hr = np.asarray(seg["hr_values"], dtype=float)
            rr = np.asarray(seg["rr_values"], dtype=float)
            dev = np.asarray(seg["device_hr_values"], dtype=float)
            gr = np.asarray(seg["green_reliability"], dtype=float)
            pr = np.asarray(seg["pos_reliability"], dtype=float)
            end_sec = seg["end_sec"] if seg["end_sec"] is not None else total_time
            out.append({
                "id": seg["id"],
                "motion_class": seg["motion_class"],
                "start_sec": round(float(seg["start_sec"]), 1),
                "end_sec": round(float(end_sec), 1),
                "duration_sec": round(float(end_sec) - float(seg["start_sec"]), 1),
                "hr_median": _f(np.median(hr)) if len(hr) else None,
                "hr_mean": _f(np.mean(hr)) if len(hr) else None,
                "hr_count": int(len(hr)),
                "device_hr_median": _f(np.median(dev)) if len(dev) else None,
                "device_hr_count": int(len(dev)),
                "rr_median": _f(np.median(rr)) if len(rr) else None,
                "rr_count": int(len(rr)),
                "green_reliability_median": _f(np.median(gr)) if len(gr) else None,
                "pos_reliability_median": _f(np.median(pr)) if len(pr) else None,
                "low_samples": len(hr) < MIN_HR_VALUES_PER_MOTION_SEGMENT,
            })
        return out

    def write_segment_summary_csv(self, path):
        """Write the in-memory motion-segment HR summary, with the same
        columns the 2-min script's MOTION_SUMMARY_FILENAME uses."""
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "Motion_Segment_ID", "Motion_Class", "Start_sec", "End_sec",
                "Duration_sec", "Final_HR_Median", "Final_HR_Mean",
                "HR_Sample_Count", "Device_HR_Median", "Device_HR_Sample_Count",
                "Final_RR_Median", "RR_Sample_Count",
                "Green_Reliability_Median", "POS_Reliability_Median",
            ])
            for s in self.segment_summaries():
                w.writerow([
                    s["id"], s["motion_class"], s["start_sec"], s["end_sec"],
                    s["duration_sec"], s["hr_median"], s["hr_mean"],
                    s["hr_count"], s["device_hr_median"], s["device_hr_count"],
                    s["rr_median"], s["rr_count"],
                    s["green_reliability_median"], s["pos_reliability_median"],
                ])

    def _build_state(self, frame, face_box, packet, now, elapsed):
        r = self.live_result or {}
        spo2 = packet["spO2"] if packet else None
        hardware_hr = packet["pulse_rate"] if packet else None

        return {
            "type": "frame",
            # Preview JPEG is attached on demand by encode_last_preview() so the
            # processing loop is not slowed by encoding every frame.
            "frame": None,
            "frame_index": self.frame_count,
            "elapsed": round(elapsed, 1),
            # Sustained processing rate. The whole live pipeline is tuned for
            # the script's steady 30 FPS; if this sits well below 30 the motion
            # buffer is sparse and HR/motion quality degrades.
            "fps": _f(self.actual_fps),
            "face_detected": face_box is not None,
            "buffer_seconds": round(
                (self.rgb_times[-1] - self.rgb_times[0]) if len(self.rgb_times) > 1 else 0.0,
                1,
            ),
            "warming_up": self.live_result is None,
            # rPPG (camera)
            "motion_class": self.live_motion_class,
            "consensus_hr": _f(r.get("consensus_hr")),
            "hr_green_fft": _f(r.get("hr_green_fft")),
            "hr_pos_fft": _f(r.get("hr_pos_fft")),
            "hr_green_peak": _f(r.get("hr_green_peak")),
            "hr_pos_peak": _f(r.get("hr_pos_peak")),
            "green_reliability": _f(r.get("green_reliability")),
            "pos_reliability": _f(r.get("pos_reliability")),
            "rr_final": _f(r.get("rr_final")),
            "sampling_rate": _f(r.get("sampling_rate")),
            # CMS50D (oximeter) — null unless connected
            "cms_connected": self.cms_connected,
            "spo2": spo2,
            "hardware_hr": hardware_hr,
            # Motion-segment mode extras (defaults when not in segment mode)
            "segment_mode": self.segment_mode,
            # Stable label for the UI: the *committed* segment's motion class,
            # which only changes when a new segment is confirmed (after
            # MOTION_SEGMENT_CONFIRM_UPDATES). Unlike the raw per-update
            # motion_class above (kept verbatim for the CSV / script parity),
            # this does not flicker on brief transient classifications. Null
            # while warming up or in original mode, where the UI falls back to
            # motion_class (original mode's full-buffer class is already stable).
            "segment_motion_class": (
                self.current_motion_segment["motion_class"]
                if self.segment_mode and self.current_motion_segment else None
            ),
            "position_state": self.position_state if self.segment_mode else None,
            "position_segment_id": self.position_segment_id if self.segment_mode else None,
            "motion_segment_id": (
                self.current_motion_segment["id"]
                if self.segment_mode and self.current_motion_segment else None
            ),
            "position_shift": round(self.position_shift, 2) if self.segment_mode else None,
            "recent_motion_level": round(self.recent_motion_level, 3) if self.segment_mode else None,
            "n_motion_segments": len(self.motion_segments) if self.segment_mode else None,
        }

    def encode_last_preview(self):
        """Encode the most recent frame as a base64 JPEG data URI, or None.
        Called by the streaming layer (in a thread-pool worker) at a throttled
        rate. Resize first, then draw, so the common 640px path skips a full
        640x480x3 copy and the rectangle is drawn over far fewer pixels."""
        frame = self._last_frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = PREVIEW_WIDTH / w if w > PREVIEW_WIDTH else 1.0
        if scale != 1.0:
            preview = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale)))
        else:
            # Copy only when we are not resizing, so we never draw the box onto
            # the shared raw frame.
            preview = frame.copy()

        face_box = self._last_face_box
        if face_box is not None:
            x, y, bw, bh = face_box
            x, y, bw, bh = int(x * scale), int(y * scale), int(bw * scale), int(bh * scale)
            cv2.rectangle(preview, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        ok, buf = cv2.imencode(
            ".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")

    def _write_csv_row(self, packet, now, state):
        cms_hr_fft, cms_hr_peak = self._cms_hr()
        row = [
            now.strftime("%Y-%m-%d %H:%M:%S.%f"),
            self.frame_count,
            packet["waveform"] if packet else None,
            packet["spO2"] if packet else None,
            packet["pulse_rate"] if packet else None,
            cms_hr_fft,
            cms_hr_peak,
            packet["signal_strength"] if packet else None,
            packet["probe_error"] if packet else None,
            state["motion_class"],
        ]
        if self.segment_mode:
            row += [
                state["motion_segment_id"],
                self.position_segment_id,
                self.position_state,
                round(self.position_shift, 4),
                round(self.recent_motion_level, 4),
                state["buffer_seconds"],
            ]
        row += [
            state["consensus_hr"],
            state["rr_final"],
            state["hr_green_fft"],
            state["hr_pos_fft"],
            state["hr_green_peak"],
            state["hr_pos_peak"],
            state["green_reliability"],
            state["pos_reliability"],
            state["sampling_rate"],
        ]
        self.csv_writer.writerow(row)
