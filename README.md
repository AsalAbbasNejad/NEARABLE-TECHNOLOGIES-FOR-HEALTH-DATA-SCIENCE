# NEARABLE TECHNOLOGIES FOR HEALTH DATA SCIENCE

# Robust rPPG Monitoring and Motion Artifact Cancellation via Dual-Modality
### Contactless Heart Rate and Respiratory Rate Monitoring from RGB Videos

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-orange)
![SciPy](https://img.shields.io/badge/SciPy-Signal%20Processing-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

# 📖 Overview

This project presents a robust remote photoplethysmography (rPPG) framework for contactless physiological monitoring using conventional RGB cameras and synchronized CMS50D pulse oximeter measurements.

The proposed framework estimates physiological parameters directly from facial videos while maintaining robustness under realistic acquisition conditions, including different head movements and motion artifacts.

The system automatically classifies head motion patterns and dynamically adapts the signal-processing pipeline according to the detected motion scenario.

The proposed framework estimates:

- ❤️ Heart Rate (HR)
- 🌬️ Respiratory Rate (RR)
- 📈 Blood Volume Pulse (BVP)
- 🎯 Motion Classification Confidence
- 📊 Signal Quality Metrics

---

# 🎯 Project Objectives

The main objectives of this project are:

- Develop a robust contactless physiological monitoring system using RGB videos.
- Automatically classify head motion patterns.
- Adapt signal conditioning according to the detected motion type.
- Estimate heart rate under static and motion conditions.
- Recover respiratory information from heart-rate variability.
- Validate physiological estimates against wearable measurements.
- Build both offline and real-time implementations.

---

# 🧠 Motivation

Traditional physiological monitoring systems require physical contact with the body and may become uncomfortable during long monitoring sessions.

Remote photoplethysmography (rPPG) provides a non-contact alternative by extracting subtle skin color variations produced by blood circulation.

However, rPPG signals are highly sensitive to:

- Head motion
- Illumination variations
- Face tracking errors
- ROI displacement
- Motion artifacts

The goal of this project is to design a robust nearable monitoring system capable of operating reliably under realistic acquisition conditions.

---

# 🧪 Experimental Protocol

Three motion paradigms were considered.

<p align="center">
  <img src="figures/motion_protocols.png" width="900">
</p>

<p align="center">
<b>Figure 1.</b> Experimental motion paradigms used during data acquisition.
</p>

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

---

# ⚙️ System Architecture

<p align="center">
  <img src="figures/system_architecture.png" width="850">
</p>

<p align="center">
<b>Figure 2.</b> Overall architecture of the proposed adaptive rPPG framework.
</p>

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

---

# 📂 Repository Structure

```text
project/

├── offline_pipeline/
│   ├── motion_classifier.py
│   ├── roi_extraction.py
│   ├── pos_algorithm.py
│   ├── hr_estimation.py
│   ├── rr_estimation.py
│   └── evaluation.py
│
├── live_pipeline/
│   ├── webcam_acquisition.py
│   ├── cms50d_reader.py
│   ├── rolling_buffer.py
│   ├── live_hr_estimation.py
│   └── live_rr_estimation.py
│
├── figures/
├── results/
├── requirements.txt
└── README.md
```

---

# 🧠 Head Motion Classification

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
<img width="272" height="60" alt="image" src="https://github.com/user-attachments/assets/71c3a7d6-0e32-4901-a36a-04c510575a00" />
</p>

<p align="center">
<b>Figure 3.</b> Representative zigzag motion classification.
</p>

---

# 🎭 Multi-ROI Extraction

Three facial skin regions are extracted:

- Forehead
- Left Cheek
- Right Cheek

The average RGB value from all ROIs is used as the physiological signal.

Using multiple skin regions improves robustness against:

- Illumination changes
- Motion corruption
- Partial occlusions
- Tracking inaccuracies

<p align="center">
 <img width="522" height="472" alt="image" src="https://github.com/user-attachments/assets/68f95d1f-1618-4196-bd7c-53b5d7ec83ae" />
</p>

<p align="center">
<b>Figure 4.</b> Forehead and cheek regions used for RGB signal extraction.
</p>

---

# 📈 Blood Volume Pulse (BVP) Extraction

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

# 🔧 Signal Conditioning

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

# ❤️ Heart Rate Estimation

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
<b>Figure 5.</b> Smart FFT spectrum comparison between Green and POS estimators.
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

# 🌬️ Respiratory Rate Estimation

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
<b>Figure 6.</b> Tachogram generated from beat-to-beat intervals.
</p>

<p align="center">
<img width="495" height="176" alt="image" src="https://github.com/user-attachments/assets/840af977-38d4-451a-8f1b-9b9fac91a118" />
</p>

<p align="center">
<b>Figure 7.</b> Respiratory PSD showing the dominant breathing frequency.
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

# 🌬️ Respiratory Sinus Arrhythmia (RSA)

The respiratory component of heart-rate variability is isolated from the tachogram.

RSA analysis provides insight into:

- Autonomic nervous system activity
- Respiratory modulation of heart rate
- Cardiorespiratory coupling

---

# 💻 Offline Pipeline

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

# 🎥 Live Pipeline

The live implementation performs real-time physiological monitoring.

### Inputs

- Webcam
- CMS50D Pulse Oximeter

### Outputs

- Live Heart Rate
- Live Respiratory Rate
- Motion Classification
- Signal Quality Indicators
- Synchronized CSV Logging

Pipeline:

```text
Webcam
    ↓
Face Detection
    ↓
Multi-ROI Extraction
    ↓
Rolling Buffer
    ↓
Green + POS Signals
    ↓
Live HR Estimation
    ↓
Live RR Estimation
    ↓
CMS50D Validation
```

<p align="center">
  <img src="figures/live_demo.png" width="750">
</p>

<p align="center">
<b>Figure 8.</b> Real-time physiological monitoring interface.
</p>

---

# 📊 Evaluation Metrics

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
<b>Figure 9.</b> Comparison between CMS50D reference measurements and estimated heart rate.
</p>

---

# 🔍 Key Findings

- Motion classification enables adaptive signal conditioning.
- Multi-ROI extraction improves robustness.
- Consensus fusion stabilizes heart-rate estimation.
- Tachogram analysis enables respiratory-rate estimation.
- The framework maintains clinically meaningful physiological estimates even under Zigzag motion.
- The representative Zigzag recording achieved an MAE of 2.15 BPM and an RR estimate of 10.9 breaths/min.

---

# 🛠 Technologies

- Python
- OpenCV
- NumPy
- SciPy
- Pandas
- Matplotlib
- MediaPipe
- Google Colab
- CMS50D Pulse Oximeter

---

# 🚀 Future Work

Potential future extensions include:

- Deep-learning-based ROI selection
- Optical-flow motion compensation
- SpO₂ estimation from RGB videos
- Stress estimation using HRV and RSA
- Transformer-based rPPG models
- Edge-device deployment
- Continuous remote health monitoring

---

# 👥 Authors

- Asal Abbas Nejad Fard
- Mohammadreza Zamani
- Mohanesh Ravi

---

# 🎓 Course Project

**Robust rPPG Monitoring and Motion Artifact Cancellation via Dual-Modality**

Politecnico di Milano  
Academic Year 2025–2026

---
