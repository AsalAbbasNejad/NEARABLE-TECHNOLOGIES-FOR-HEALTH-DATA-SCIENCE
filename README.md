# NEARABLE TECHNOLOGIES FOR HEALTH DATA SCIENCE

# Adaptive rPPG Signal Conditioning Router
### Robust Contactless Heart Rate and Respiratory Rate Monitoring from RGB Videos

---

## 📖 Overview

Remote photoplethysmography (rPPG) enables contactless physiological monitoring by analyzing subtle color variations of the skin captured by conventional RGB cameras. However, head motion and illumination changes significantly degrade signal quality and estimation accuracy.

This project proposes an **Adaptive rPPG Signal Conditioning Router** that automatically identifies head-motion patterns and dynamically selects an appropriate signal-conditioning strategy to improve robustness under realistic acquisition scenarios.

The proposed framework estimates:

- ❤️ Heart Rate (HR)
- 🌬️ Respiratory Rate (RR)
- 📈 Blood Volume Pulse (BVP)
- 🎯 Motion Classification Confidence
- 📊 Signal Quality Indicators

All estimates are validated against synchronized measurements obtained from a **CMS50D pulse oximeter**.

---

## 🎯 Project Objectives

The objectives of this project are:

- Develop a robust offline rPPG pipeline using RGB facial videos.
- Automatically classify head motion patterns.
- Adapt signal conditioning according to motion type.
- Estimate heart rate under static and motion conditions.
- Recover respiratory information from beat-to-beat variability.
- Validate physiological estimates against wearable measurements.

---

# 🧪 Experimental Protocol

Three behavioral paradigms were designed to evaluate the robustness of the proposed framework.

## 1. Stable Baseline

The subject remains stationary while facing the camera.

**Purpose**
- Establish ideal recording conditions
- Determine upper-bound algorithm performance
- Measure baseline signal quality

---

## 2. Left-to-Right Rotation (Yaw)

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

# ⚙️ System Pipeline

```text
RGB Video
      ↓
Motion Classification
      ↓
Face Detection + Multi-ROI Tracking
      ↓
RGB Signal Extraction
      ↓
Green / POS / CHROM Processing
      ↓
Heart Rate Estimation
      ↓
Tachogram Construction
      ↓
Respiratory Rate Estimation
      ↓
CMS50D Validation
```

---

# 🧠 Motion Classification

The facial center is tracked throughout the recording and horizontal and vertical motion trajectories are extracted.

Motion descriptors include:

- Horizontal amplitude
- Vertical amplitude
- Motion path length
- Direction changes
- Motion ratios
- Tracking quality

The recording is then classified into one of the predefined motion categories.

---

# 🎭 Multi-ROI Signal Extraction

The framework uses three skin regions:

- Forehead
- Left cheek
- Right cheek

Using multiple ROIs improves robustness against:

- Illumination changes
- Local motion corruption
- Partial occlusions
- Tracking inaccuracies

Spatial averaging generates temporal color traces:

```text
R(t), G(t), B(t)
```

which are subsequently processed for rPPG extraction.

---

# 📈 Blood Volume Pulse (BVP) Extraction

The framework employs three complementary methods.

## Green Channel

The green channel exhibits the strongest pulsatile modulation because hemoglobin absorbs green wavelengths more strongly than red and blue wavelengths.

```text
s_green(t) = G(t)
```

Advantages:

- Simple implementation
- Computational efficiency
- Strong pulsatile response under ideal conditions

---

## CHROM Method

CHROM reduces luminance variations by constructing chrominance projections:

```text
X = 3R − 2G
Y = 1.5R + G − 1.5B
```

Advantages:

- Illumination robustness
- Improved color separation

---

## POS Method

The Plane-Orthogonal-to-Skin (POS) method projects normalized RGB traces onto a plane orthogonal to the skin-tone direction.

Advantages:

- Reduced sensitivity to illumination changes
- Improved robustness under motion
- Enhanced pulsatile signal separation

---

# ❤️ Heart Rate Estimation

After signal conditioning:

1. Filtered BVP signals are generated.
2. Welch Power Spectral Density (PSD) is computed.
3. The dominant spectral peak is searched within the physiological range.

Heart-rate search band:

```text
45 – 150 BPM
```

Heart rate estimation:

```text
HR = f_peak × 60
```

To improve robustness, Green and POS estimates are combined using a consensus strategy.

---

# 🌬️ Tachogram Construction

Respiration information is recovered from beat-to-beat variability.

Processing steps:

1. Detect systolic peaks.
2. Compute Inter-Beat Intervals (IBI):

```text
IBI_n = t_n − t_(n−1)
```

3. Construct the tachogram.

The tachogram represents respiratory sinus arrhythmia (RSA) and acts as an indirect respiratory signal.

---

# 🌬️ Respiratory Rate Estimation

Since IBI samples are irregularly spaced in time, cubic spline interpolation is first applied.

The resampled tachogram is then analyzed using Welch PSD.

Respiratory search band:

```text
0.15 – 0.60 Hz
```

Equivalent to:

```text
9 – 36 breaths/min
```

The dominant spectral peak provides the final respiratory rate estimate.

---

# 📊 Representative Example: Zigzag Motion

### Motion Classification

| Metric | Value |
|--------|--------|
| Predicted Motion | ZIGZAG |
| Confidence | 0.95 |
| Tracking Success | 100% |

### Heart Rate Estimation

| Metric | Value |
|--------|--------|
| Green FFT HR | 74.70 BPM |
| POS FFT HR | 75.00 BPM |
| Consensus HR | 74.85 BPM |
| CMS50D Ground Truth | 77.00 BPM |
| Signed Error | -2.15 BPM |
| MAE | 2.15 BPM |
| MAPE | 2.80 % |

### Respiratory Rate Estimation

| Metric | Value |
|--------|--------|
| Welch Peak Frequency | 0.1818 Hz |
| Respiratory Rate | 10.9 breaths/min |
| Confidence | HIGH |

---

# ✅ Key Contributions

- Motion-aware adaptive rPPG routing
- Automatic head-motion classification
- Multi-ROI facial signal extraction
- Green, POS, and CHROM processing
- Consensus-based heart-rate estimation
- Tachogram-based respiratory estimation
- Validation against synchronized CMS50D measurements

---

# 🛠️ Technologies

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib
- MediaPipe
- Gradio
- CMS50D Pulse Oximeter

---

# 🚀 Future Work

Potential future extensions include:

- Real-time deployment
- Motion-artifact cancellation
- Signal quality assessment modules
- SpO₂ estimation
- Larger multi-subject datasets
- Machine-learning-based adaptive routing
- Continuous physiological monitoring in unconstrained environments

---

# 👥 Authors

- Asal Abbas Nejad Fard
- Mohammadreza Zamani
- Mohanesh Ravi

---

# 🎓 Course

**Adaptive rPPG Signal Conditioning Router for Contactless Vital Sign Monitoring**

Politecnico di Milano  
Academic Year 2025–2026
