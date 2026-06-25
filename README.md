# NEARABLE TECHNOLOGIES FOR HEALTH DATA SCIENCE

# Robust rPPG Monitoring and Motion Artifact Cancellation via Dual-Modality
### Contactless Heart Rate and Respiratory Rate Monitoring from RGB Videos

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

This repository contains the final implementation of a nearable rPPG monitoring system developed for the Laboratory of Nearable Technologies for Health Data Science. The system estimates heart rate from RGB facial videos, validates the results against synchronized CMS50D wearable pulse oximeter measurements, and adapts the live analysis according to detected head-motion patterns.

---

# Overview

The proposed framework estimates physiological parameters directly from facial videos while maintaining robustness under realistic acquisition conditions, including different head movements.

The system automatically classifies head motion patterns and dynamically adapts the signal-processing pipeline according to the detected motion scenario.

The proposed framework estimates:

- Heart Rate (HR)
- Respiratory Rate (RR)
- Blood Volume Pulse (BVP)
- Motion Classification Confidence
- Signal Quality Metrics

---
# Main Scripts

| Purpose | File |
|---|---|
| Live webcam acquisition, CMS50D synchronization, motion-aware rPPG estimation, motion-segment HR output, and post-recording metrics | `live codes/live_rppg_cms50d_2min_motion_segment_with_metrics.py` |
| Offline rPPG processing, respiratory-rate estimation, and metric evaluation from recorded videos | `offline codes/final_with_RR_n_metrics.py` |
| Development notebooks and earlier experimental versions | `archive/` |
| GUI/demo material | `gui/` |
| Presentation material | `Presentation/` |

The recommended script for the final live demonstration is:

```bash
python live codes/live_rppg_cms50d_2min_motion_segment_with_metrics.py
````

# Project Objectives

The main objectives of this project are:

- Develop a robust contactless physiological monitoring system using RGB videos.
- Automatically classify head motion patterns.
- Adapt signal conditioning according to the detected motion type.
- Estimate heart rate under static and motion conditions.
- Recover respiratory information from heart-rate variability.
- Validate physiological estimates against wearable measurements.
- Build both offline and real-time implementations.

---

# Motivation

Traditional physiological monitoring systems require physical contact with the body and may become uncomfortable during long monitoring sessions. Remote photoplethysmography (rPPG) provides a non-contact alternative by extracting subtle skin color variations produced by blood circulation.

However, rPPG signals are highly sensitive to:

- Head motion
- Illumination variations
- Face tracking errors
- ROI displacement
- Motion artifacts

The goal of this project is to design a robust nearable monitoring system capable of operating reliably under realistic acquisition conditions

---

# Experimental Protocol

Three motion paradigms were considered.

## 1. Stable Baseline

The subject remains stationary and directly faces the camera.

**Purpose**
- Establish ideal acquisition conditions
- Determine upper-bound performance
- Measure baseline signal quality

---

## 2. Left-to-Right Rotation

The subject continuously performs horizontal head rotations.

**Purpose**
- Introduce ROI displacement
- Generate moderate motion artifacts
- Evaluate tracking robustness

---

## 3. Zigzag Motion

The subject performs combined horizontal and vertical head movements.

**Purpose**
- Create a challenging motion scenario
- Stress-test signal robustness
- Evaluate adaptive routing performance

# System Architecture

```text
RGB Video
      ↓
Face Detection
      ↓
Multi-ROI Extraction
      ↓
RGB Signal Acquisition
      ↓
Green + POS rPPG Extraction
      ↓
Signal Conditioning
      ↓
FFT + Peak Detection
      ↓
Consensus Heart Rate
      ↓
Tachogram Construction
      ↓
Respiratory Rate Estimation
      ↓
CMS50D Validation
```

# Repository Structure

```text
NEARABLE-TECHNOLOGIES-FOR-HEALTH-DATA-SCIENCE/
│
├── live_codes/
│   ├── live_rppg_cms50d_2min_motion_segment_with_metrics.py
│   └── live_rppg_cms50d_original_settings_with_metrics.py
│
├── offline_codes/
│   ├── final_with_RR_n_metrics.py
│   └── final_with_RR_n_metrics.ipynb
│
├── GUI/
│   └── GUI-related files and prototypes
│
├── Presentation/
│   └── presentation material
│
├── archive/
│   └── development notebooks, early scripts, and experimental versions
│
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── .gitignore
```

# How to Run

## Live Pipeline

Before running the live script, connect the CMS50D pulse oximeter and check the active COM port.

Open the live script and update the configuration section if needed:

```python
CMS50D_PORT = "COM5"
WEBCAM_INDEX = 0
```

# Outputs

The live pipeline generates several output files during and after recording:

| Output | Description |
|---|---|
| `.avi` video | Raw webcam video saved without overlay graphics to preserve rPPG pixels |
| synchronized `.csv` file | Frame-level CMS50D and live rPPG measurements |
| motion-segment summary `.csv` | Final HR value for each detected motion segment |
| metric summary `.csv` | Overall HR error metrics and validation results |
| evaluation dashboard `.png` | Visual comparison between CMS50D reference and rPPG estimates |

Important CSV columns include:

- `CMS_Pulse_Rate_Hardware`
- `CMS_SpO2`
- `CMS_Waveform`
- `Live_rPPG_HR`
- `Live_Motion_Class`
- `Motion_Segment_ID`
- `Live_Green_FFT`
- `Live_POS_FFT`
- `Live_FS`

For the final live demo, the most important output is the motion-segment summary, which provides one final heart-rate estimate for each detected motion condition.

# Head Motion Classification

The framework automatically classifies head motion before physiological estimation.

The facial center is tracked over time and several motion descriptors are extracted:

- Horizontal amplitude
- Vertical amplitude
- Total displacement
- Direction changes
- Path length
- Motion ratios

The recording is then classified into one of the following categories:

- Stable
- Left-Right
- Zigzag

The selected motion class determines which adaptive correction rules are activated during physiological estimation.

<p align="center">
<img width="1400" height="250" alt="photo_2026-06-25_21-16-46" src="https://github.com/user-attachments/assets/c5f4a41b-aad3-4111-b8db-7919a6b77400" />
</p>

<p align="center">
.</b> Representative zigzag motion classification.
</p>

---

# ROI Extraction

The framework uses facial regions to extract RGB traces related to blood-volume changes.

In the offline pipeline, multiple facial skin ROIs are used:

- Forehead
- Left cheek
- Right cheek

The average RGB values from these regions are combined to improve robustness against local illumination changes, partial occlusion, and tracking inaccuracies.

In the live implementation, only one green face box is displayed in the preview to keep the interface simple and readable. Internally, a central skin-oriented crop inside the detected face box is used for RGB extraction, while the saved video remains raw and does not include the overlay box.

This design prevents graphical annotations from corrupting the pixels used later for rPPG analysis.

---

# Blood Volume Pulse (BVP) Extraction

Two complementary rPPG approaches are employed.

## Green Channel Method

The green channel exhibits the strongest pulsatile modulation because hemoglobin absorbs green wavelengths more strongly than red and blue wavelengths.

```math
s_{green}(t)=G(t)
```

Advantages:

- Computationally efficient
- Strong pulsatile response
- Simple implementation

---

## POS Method

The Plane-Orthogonal-to-Skin (POS) algorithm projects normalized RGB traces onto a plane orthogonal to the skin-tone direction.

```math
S_1 = G-B
```

```math
S_2 = G+B-2R
```

```math
H=S_1+\alpha S_2
```

where

```math
\alpha=\frac{\sigma(S_1)}{\sigma(S_2)}
```

Advantages:

- Robust under illumination changes
- Improved motion resilience
- Better pulse separation

---

# Signal Conditioning

The extracted signals are:

1. Detrended
2. Normalized
3. Bandpass filtered

Heart-rate search range:

```text
60–90 BPM
```

Filtering suppresses:

- Illumination drift
- Sensor noise
- Motion artifacts
- High-frequency interference

---

# Heart Rate Estimation

Heart-rate estimation consists of several stages.

## Windowed FFT Analysis

The framework performs:

- Windowed FFT
- Zero-padding
- Peak interpolation
- Reliability estimation
- Multi-candidate ranking

Heart rate is estimated as:

```math
HR = f_{peak}\times60
```

<p align="center">
 <img width="1472" height="595" alt="image" src="https://github.com/user-attachments/assets/89c76c27-8a60-46e7-9e27-e1cf8a006561" />
</p>

<p align="center">
<b>Figure 4.</b> Smart FFT spectrum comparison between Green and POS estimators.
</p>

Representative Zigzag example:

| Metric | Value |
|--------|--------|
| Green FFT HR | 74.70 BPM |
| POS FFT HR | 75.00 BPM |
| Final Consensus HR | 74.85 BPM |
| CMS50D Reference HR | 77.00 BPM |
| Signed Error | -2.15 BPM |
| MAE | 2.15 BPM |
| MAPE | 2.80 % |

---

## Peak Detection

Systolic peaks are detected in both Green and POS signals.

Inter-beat intervals are computed as:

```math
IBI_n=t_n-t_{n-1}
```

Peak-derived estimates are used to validate FFT estimates.

---

## Consensus Heart Rate

The final HR estimate is computed by combining:

- Green FFT estimate
- POS FFT estimate
- Window reliability
- Peak guidance information

The system automatically rejects inconsistent candidates and applies adaptive correction rules.

---

# 🛠 Motion Artifact Correction

Several adaptive correction modules were developed.

## Motion Peak Guidance Correction

Corrects underestimated FFT estimates using peak information.

---

## Left-Right Candidate Guidance

Selects FFT candidates that agree with peak guidance.

---

## Low-Lock Rescue

Recovers estimates trapped in low-frequency artifacts.

---

## High-HR Rescue

Restores physiological frequencies suppressed by motion contamination.

---

# Respiratory Rate Estimation

Respiratory information is recovered from heart-rate variability.

Pipeline:

```text
POS Peaks
      ↓
R-R Intervals
      ↓
Tachogram
      ↓
Cubic Spline Interpolation
      ↓
Welch PSD
      ↓
Hilbert Validation
      ↓
Final Respiratory Rate
```

Respiratory search band:

```text
10–40 breaths/min
```

The final respiratory estimate combines:

- Welch dominant frequency
- Hilbert instantaneous frequency
- Cross-method agreement

<p align="center">
<img width="1678" height="457" alt="image" src="https://github.com/user-attachments/assets/9815e120-a4ea-434f-8926-cd8f9494a9e8" />
</p>

<p align="center">
<b>Figure 5.</b> Tachogram generated from beat-to-beat intervals.
</p>

<p align="center">
<img width="495" height="176" alt="image" src="https://github.com/user-attachments/assets/840af977-38d4-451a-8f1b-9b9fac91a118" />
</p>

<p align="center">
<b>Figure 6.</b> Respiratory PSD showing the dominant breathing frequency.
</p>

Representative Zigzag example:

| Metric | Value |
|--------|--------|
| Detected POS Peaks | 35 |
| R-R Intervals | 34 |
| Mean R-R Interval | 825.5 ms |
| Welch Dominant Frequency | 0.1818 Hz |
| Final Respiratory Rate | 10.9 breaths/min |
| Confidence | HIGH |

---

# Respiratory Sinus Arrhythmia (RSA)

The respiratory component of heart-rate variability is isolated from the tachogram.

RSA analysis provides insight into:

- Autonomic nervous system activity
- Respiratory modulation of heart rate
- Cardiorespiratory coupling

---

# Offline Pipeline

The offline implementation processes previously recorded videos.

### Inputs

- RGB facial video
- CMS50D CSV file

### Outputs

- Motion classification
- Green rPPG signal
- POS rPPG signal
- Heart Rate
- Respiratory Rate
- Evaluation metrics
- Visualization dashboard

---

# Live Pipeline

The live implementation performs real-time physiological monitoring using a webcam and the CMS50D pulse oximeter.

### Inputs

- Webcam video stream
- CMS50D pulse oximeter connected through a serial COM port

### Outputs

- Live rPPG heart rate
- Live motion classification
- Motion-segment heart-rate summary
- CMS50D heart rate and SpO2 logging
- Synchronized CSV file
- Post-recording evaluation metrics

Pipeline:

```text
Webcam
    ↓
Face Detection
    ↓
One-Box Face Preview
    ↓
Central RGB Signal Extraction
    ↓
Rolling Buffer
    ↓
Green + POS Signals
    ↓
Live HR Estimation
    ↓
Motion-Segment Summary
    ↓
CMS50D Validation
---
```

# Evaluation Metrics

The framework computes several evaluation metrics.

## Error Metrics

- Signed Error
- MAE
- RMSE
- MAPE

## Statistical Analysis

- Pearson Correlation
- Spearman Correlation
- Bland-Altman Bias
- Limits of Agreement

## Clinical Metrics

- Within ±3 BPM
- Within ±5 BPM
- Clinical Pass/Fail

<p align="center">
  <img width="522" height="445" alt="image" src="https://github.com/user-attachments/assets/48116834-dcd5-4f8c-9480-ed35def1e75e" />
  <img width="512" height="462" alt="image" src="https://github.com/user-attachments/assets/225a787c-c8f9-4208-9bc2-c04c77360010" />
</p>

<p align="center">
<b>Figure 8.</b> Comparison between CMS50D reference measurements and estimated heart rate.
</p>

---

# Key Findings

- Motion classification enables adaptive signal conditioning.
- Multi-ROI extraction improves robustness.
- Consensus fusion stabilizes heart-rate estimation.
- Tachogram analysis enables respiratory-rate estimation.
- The framework maintains clinically meaningful physiological estimates even under Zigzag motion.
- The representative Zigzag recording achieved an MAE of 2.15 BPM and an RR estimate of 10.9 breaths/min.
---

# Limitations

- The live pipeline requires local execution because webcam and CMS50D serial-port access are not available in standard cloud runtimes.
- rPPG performance depends on lighting conditions, face visibility, camera quality, and motion intensity.
- The respiratory-rate estimate is derived from the POS tachogram and requires a sufficient number of detected pulse peaks.
- The current implementation is optimized for controlled academic demonstrations and short experimental recordings.
- The heart-rate search range is currently limited to the configured physiological band used in the experiments.

  
# Technologies

- Python
- OpenCV
- NumPy
- SciPy
- Pandas
- Matplotlib
- PySerial
- CMS50D Pulse Oximeter

Optional / development tools:
- Jupyter Notebook
- Gradio or Streamlit for GUI prototyping

  # Disclaimer

This project is developed for academic and research demonstration purposes only.

It is not intended for clinical diagnosis, treatment, patient monitoring, or medical decision-making.

The physiological estimates generated by this software should not be used as a substitute for certified medical devices or professional healthcare evaluation.

# Citation

If you use this repository or refer to this project, please cite it using the information provided in `CITATION.cff`.
