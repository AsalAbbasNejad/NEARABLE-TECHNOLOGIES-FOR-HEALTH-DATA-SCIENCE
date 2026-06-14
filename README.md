# NEARABLE TECHNOLOGIES FOR HEALTH DATA SCIENCE

## Project Overview

This project implements a nearable remote photoplethysmography (rPPG) system for estimating heart rate from RGB video and validating the result against a CMS50D pulse oximeter. The system records webcam video, extracts facial color variations, estimates live rPPG heart rate, reads synchronized CMS50D measurements, and stores all results in a single CSV file for later analysis.

The main goal is to build a non-invasive monitoring pipeline where heart rate can be estimated from a normal camera without physical contact, while using the CMS50D sensor as reference ground truth.

## Main Features

* Live webcam acquisition
* CMS50D serial data acquisition
* Synchronized video and sensor logging
* One-face-box preview interface
* Raw video saving for offline rPPG processing
* Green-channel and POS-based rPPG estimation
* FFT-based heart rate estimation
* Peak-based heart rate estimation
* Rolling live rPPG estimation
* Respiratory rate estimation from POS tachogram
* Lightweight motion classification
* Motion-aware correction rules
* CSV output containing CMS50D and rPPG values

## System Pipeline

```text
Webcam Video
    ↓
Face Detection
    ↓
RGB Signal Extraction from Face ROI
    ↓
Green Channel + POS rPPG Processing
    ↓
Bandpass Filtering
    ↓
FFT and Peak-Based HR Estimation
    ↓
Consensus HR Estimation
    ↓
CMS50D Sensor Validation
    ↓
CSV Logging + Raw Video Saving
```

## Hardware and Software Requirements

### Hardware

* Webcam or laptop camera
* CMS50D pulse oximeter
* USB/serial connection for CMS50D
* Local machine capable of accessing webcam and COM port

### Software

* Python 3.x
* OpenCV
* NumPy
* SciPy
* PySerial

Install dependencies:

```bash
pip install opencv-python numpy scipy pyserial
```

## Important Note About Runtime

This code must be run on a local machine or Google Colab local runtime. Standard Google Colab cloud runtime cannot directly access a local webcam or COM port.

## Configuration

Before running the code, update these values according to your system:

```python
CMS50D_PORT = "COM7"
WEBCAM_INDEX = 0
TARGET_FPS = 30
DURATION_SECONDS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

`CMS50D_PORT` should match the active serial port of the CMS50D device. On Windows it may be something like `COM7`. On Linux or macOS it may be something like `/dev/ttyUSB0`.

## Output Files

The program automatically creates two output files:

```text
live_rppg_input_<timestamp>.avi
live_sync_results_<timestamp>.csv
```

### Raw Video Output

The video file stores the raw webcam recording. The face box is shown only in the preview window and is not drawn on the saved video. This is important because drawing on the saved frames would modify the pixel values and could corrupt later rPPG processing.

### CSV Output

The CSV file contains synchronized data from the CMS50D sensor and the live rPPG system.

Main columns include:

```text
Timestamp
Frame_Index
CMS_Waveform
CMS_SpO2
CMS_Pulse_Rate_Hardware
CMS_HR_FFT
CMS_HR_Peak
CMS_Signal_Strength
CMS_Probe_Error
Live_Motion_Class
Live_rPPG_HR
Live_RR
Live_Green_FFT
Live_POS_FFT
Live_Green_Peak
Live_POS_Peak
Live_Green_Reliability
Live_POS_Reliability
Live_FS
```

## rPPG Methodology

The rPPG signal is extracted from the face region detected in each frame. A single green box is displayed in the preview to keep the interface clean. Internally, the system uses a central face crop to reduce background, hair, and edge effects.

Two main rPPG signals are used:

### 1. Green Channel Signal

The green channel is commonly used in rPPG because blood volume changes are often more visible in the green wavelength range. The extracted green signal is normalized and filtered within the expected heart rate band.

### 2. POS rPPG Signal

The Plane-Orthogonal-to-Skin (POS) method combines RGB channels to reduce illumination noise and improve pulse signal extraction. The POS signal is computed over a moving temporal window and then filtered for heart rate estimation.

## Heart Rate Estimation

Heart rate is estimated using two approaches:

### FFT-Based Estimation

The signal is transformed into the frequency domain. The dominant frequency inside the heart rate band is selected and converted to beats per minute.

### Peak-Based Estimation

Peaks are detected in the filtered rPPG waveform. The time interval between peaks is used to estimate heart rate.

The system computes:

```text
Green FFT HR
POS FFT HR
Green Peak HR
POS Peak HR
Consensus HR
```

The final live rPPG heart rate is reported as the consensus HR.

## Motion Handling

The code includes a lightweight motion classifier based on the movement of the detected face box. It classifies recent motion into:

```text
stable
left_right
zigzag
warming_up
```

This motion class is used to decide whether correction rules should be applied. For example, if the face is moving left-right, the algorithm applies additional checks to avoid locking onto motion-related frequency artifacts.

## Respiratory Rate Estimation

The respiratory rate is estimated from the POS-derived tachogram when enough pulse peaks are available. The time intervals between pulse peaks are resampled and analyzed in the respiratory frequency range.

The respiratory rate output is reported as:

```text
Live_RR
```

This value may be unavailable when the recording is too short or when too few reliable peaks are detected.

## Validation with CMS50D

The CMS50D pulse oximeter provides:

```text
Pulse rate
SpO2
Pulse waveform
Signal strength
Probe error flag
```

These values are logged together with the rPPG outputs. The CMS50D pulse rate can be used as the reference value for evaluating the rPPG heart rate estimation.

## How to Run

1. Connect the CMS50D device to the computer.
2. Check the correct serial port.
3. Update `CMS50D_PORT` in the code.
4. Connect or enable the webcam.
5. Run the script.
6. Press Enter when prompted to start acquisition.
7. Keep the face visible and reasonably well lit.
8. Press `q` in the preview window to stop early, or wait until the recording duration finishes.

## Example Output

During acquisition, the preview window shows:

```text
CMS HR
SpO2
CMS FFT
Live rPPG HR
Live RR
Green FFT
POS FFT
POS Peak
Reliability values
Motion class
```

At the end, the program prints a report:

```text
LIVE ACQUISITION REPORT
Video saved
CSV saved
Frames logged
Run time
Average FPS
Last live HR
Last live RR
```

## Current Limitations

* The system requires good lighting conditions.
* Large head movements can reduce rPPG accuracy.
* Webcam exposure changes may introduce artifacts.
* CMS50D serial port configuration may differ between computers.
* The respiratory rate estimate requires enough stable pulse peaks.
* The live rPPG estimate is less stable during the first few seconds because the rolling buffer needs enough data.

## Recommended Acquisition Conditions

For best results:

* Keep the face clearly visible.
* Avoid strong shadows or flickering light.
* Keep the camera stable.
* Avoid excessive head movement.
* Record at least 25–30 seconds.
* Make sure the CMS50D probe is properly attached.
* Use the raw saved video for offline validation if needed.

## Project Significance

This project demonstrates a low-cost, non-contact physiological monitoring pipeline. By combining camera-based rPPG with wearable PPG validation, the system provides a practical framework for comparing nearable and contact-based heart rate monitoring methods.

The project can be extended toward:

* Offline batch analysis of multiple videos
* More robust motion artifact cancellation
* Better ROI selection
* Binary classification of normal and abnormal HR
* Stress or respiratory pattern analysis
* Gradio-based user interface
* Dataset-level performance evaluation using MAE, RMSE, and classification accuracy

## Repository Structure

```text
.
├── live_rppg_cms50d_one_box_integrated.py
├── README.md
├── outputs/
│   ├── live_rppg_input_<timestamp>.avi
│   └── live_sync_results_<timestamp>.csv
└── docs/
```

