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

This project presents a robust remote photoplethysmography (rPPG) framework for contactless physiological monitoring using conventional RGB cameras.

The proposed system estimates physiological parameters from facial videos and validates all measurements using a synchronized CMS50D pulse oximeter.

The framework is designed to operate under realistic conditions, including different head movements and motion artifacts. To improve robustness, the system automatically classifies head motion and dynamically adapts the signal-processing pipeline according to the detected motion pattern.

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

## Stable

Minimal movement in both directions.

## Left-Right

Dominant horizontal head rotation.

## Zigzag

Combined horizontal and vertical movements.

The selected motion class determines which adaptive correction rules are activated during physiological estimation.

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

---

# 📈 Blood Volume Pulse Extraction

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
HR = f_{peak} \times 60
```

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

---

# 📊 Evaluation Metrics

The framework computes several evaluation metrics.

## Error Metrics

- Signed Error
- MAE
- RMSE
- MAPE

---

## Statistical Analysis

- Pearson Correlation
- Spearman Correlation
- Bland-Altman Bias
- Limits of Agreement

---

## Clinical Metrics

- Within ±3 BPM
- Within ±5 BPM
- Clinical Pass/Fail

---

## Signal Metrics

- Green Reliability
- POS Reliability
- Perfusion Index
- Pulse Amplitude Variability

---

## Physiological Metrics

- Heart Rate
- Respiratory Rate
- HR/RR Ratio
- SpO₂ Statistics

---

# 📈 Generated Outputs

The framework automatically generates:

- Head motion trajectories
- ROI preview frames
- Green and POS signals
- FFT spectra
- Peak detection plots
- Tachograms
- Respiratory PSD plots
- RSA visualizations
- Evaluation dashboards
- Bland-Altman plots
- Correlation plots
- Per-video error analysis

---

# 🛠 Technologies

- Python
- OpenCV
- NumPy
- SciPy
- Pandas
- Matplotlib
- Google Colab

---

# 🚀 Future Work

Potential future extensions include:

- Deep-learning-based ROI selection
- MediaPipe Face Mesh integration
- Optical-flow motion compensation
- SpO₂ estimation from RGB videos
- Stress estimation using HRV and RSA
- Transformer-based rPPG models
- Edge-device deployment
- Mobile implementation
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

# 📜 License

This project is released for educational and research purposes.

---

# 🙏 Acknowledgments

We would like to thank the instructors of the course and the Politecnico di Milano community for their guidance and support throughout the development of this project.
