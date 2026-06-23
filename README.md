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

<p align="center">
  <img src="figures/motion_protocols.png" width="900">
</p>

<p align="center">
<b>Figure 1.</b> Experimental motion paradigms: stable baseline, left-right rotation, and zigzag motion.
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
