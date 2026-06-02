# NEARABLE-TECHNOLOGIES-FOR-HEALTH-DATA-SCIENCE

This project develops a non-invasive **remote photoplethysmography (rPPG)** system for estimating heart rate from standard RGB video. The system focuses on improving robustness against motion artifacts using head pose estimation, dense optical flow motion analysis, and frequency-domain motion artifact cancellation.

The estimated rPPG heart rate is validated against a **CMS50D pulse oximeter**, which provides wearable PPG ground-truth heart rate, SpO2, and pulse waveform data.

## Project Goal

The main goal is to build a nearable physiological monitoring system that can estimate heart rate from subtle skin color variations in RGB video while reducing errors caused by head movement and motion artifacts.

The project includes:

- rPPG signal extraction from RGB video
- Forehead ROI detection and tracking
- Green-channel physiological signal extraction
- Head pose estimation using yaw, pitch, and roll
- Dense optical flow motion magnitude estimation
- Motion-aware frame selection
- Frequency-domain motion artifact cancellation
- Heart rate estimation using FFT and peak detection
- Validation using CMS50D wearable PPG data

## System Pipeline

```text
RGB Video
↓
Face / Forehead ROI Detection
↓
Green Channel Signal Extraction
↓
Signal Normalization
↓
Bandpass Filtering
↓
Head Pose Estimation
↓
Dense Optical Flow Motion Analysis
↓
Motion Artifact Cancellation
↓
Heart Rate Estimation
↓
CMS50D Ground Truth Validation
