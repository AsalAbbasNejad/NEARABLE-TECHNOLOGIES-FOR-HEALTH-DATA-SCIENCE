import cv2
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import find_peaks
import csv

from cms50d import CMS50D

# Check the line 52 first

# --- CONFIGURATION ---
video_filename = f"rppg_input_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
csv_filename = f"cms50d_sync_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

target_fps = 30
frame_interval = 1.0 / target_fps  # 0.033 seconds (33.3ms)
duration = 30  # Total recording time in seconds
frame_width = 640
frame_height = 480

# --- SIGNAL PROCESSING FUNCTIONS ---
def estimate_hr_with_fft(waveform, sampling_rate):
    if len(waveform) < 2: return 0.0
    fft_result = np.fft.fft(waveform)
    fft_freqs = np.fft.fftfreq(len(waveform), 1 / sampling_rate)
    fft_magnitude = np.abs(fft_result)
    
    # Ignore DC component (0 Hz)
    peak_index = np.argmax(fft_magnitude[1:]) + 1
    peak_frequency = abs(fft_freqs[peak_index])
    return peak_frequency * 60  # Hz to BPM

def estimate_hr_with_peak_detection(waveform, sampling_rate):
    waveform = np.array(waveform)
    peaks, _ = find_peaks(waveform, distance=int(sampling_rate * 0.4)) # Min 400ms between beats
    if len(peaks) < 2: return 0.0
    
    peak_times = np.diff(peaks) / sampling_rate
    avg_peak_interval = np.mean(peak_times)
    return 60 / avg_peak_interval if avg_peak_interval > 0 else 0.0

# --- HARDWARE SETUP ---
print("Initializing webcam and pulse oximeter...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_filename, fourcc, target_fps, (frame_width, frame_height))

monitor = CMS50D(port="COM7")  # Adjust to your active COM port
monitor.connect()

# --- CSV & PLOT SETUP ---
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Frame_Index', 'Waveform', 'SpO2', 'Pulse_Rate_Hardware', 'HR_FFT', 'HR_Peak'])

plt.ion()
fig, ax = plt.subplots(figsize=(10, 4))
xdata, ydata = [], []
line, = ax.plot_date([], [], fmt='-m', label='Pulse Waveform')
text_info = ax.text(0.02, 0.80, '', transform=ax.transAxes, color='black', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_ylim(0, 128)
ax.set_ylabel("Waveform Amplitude")
ax.set_xlabel("Time")
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
plt.title("Synchronized Data Acquisition (Recording Live)")

print("\n--- READY ---")
print("Focus on the terminal and press ENTER to start the 30-second capture...")
input()

# --- MAIN ACQUISITION LOOP ---
monitor.start_live_acquisition()
start_time = time.time()
frame_count = 0

print("Recording and calculating real-time metrics... Press 'Ctrl+C' in terminal to abort.")

try:
    while time.time() - start_time < duration:
        loop_start = time.time()
        current_datetime = datetime.datetime.now()

        # 1. Grab Webcam Frame (Maintains 20 FPS constraint)
        ret, frame = cap.read()
        if not ret:
            print("Webcam frame drop encountered.")
            break
        
        out.write(frame)
        frame_count += 1

        # 2. Collect Hardware Data Packets 
        # Flush the serial queue accumulated during this 50ms frame interval
        latest_packet = None
        while True:
            packet = monitor.get_latest_data()
            if not packet:
                break
            latest_packet = packet
            
            # Keep historical logging trace updated
            xdata.append(packet['timestamp'])
            ydata.append(packet['waveform'])

        # 3. Trim Memory Arrays to Last 10-Second Window
        if xdata:
            cutoff = current_datetime - datetime.timedelta(seconds=10)
            # Filter lists using list comprehensions to maintain perfect alignment
            combined = [(t, w) for t, w in zip(xdata, ydata) if t >= cutoff]
            if combined:
                xdata, ydata = map(list, zip(*combined))
            else:
                xdata, ydata = [], []

        # 4. Process Signals if we have an active sample batch
        if len(xdata) > 5:
            dt_window = (xdata[-1] - xdata[0]).total_seconds()
            sampling_rate = len(xdata) / dt_window if dt_window > 0 else 60.0
            
            hr_fft = estimate_hr_with_fft(ydata, sampling_rate)
            hr_peak = estimate_hr_with_peak_detection(sampling_rate=sampling_rate, waveform=ydata)
            
            # Setup localized snapshot data variable for logging
            hw_spo2 = latest_packet['spO2'] if latest_packet else 98
            hw_hr = latest_packet['pulse_rate'] if latest_packet else 70
            w_val = latest_packet['waveform'] if latest_packet else 64
        else:
            sampling_rate, hr_fft, hr_peak, hw_spo2, hw_hr, w_val = 60.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # 5. Write Synchronized Row to Data Log
        csv_writer.writerow([
            current_datetime.strftime('%Y-%m-%d %H:%M:%S.%f'),
            frame_count,
            w_val,
            hw_spo2,
            hw_hr,
            hr_fft,
            hr_peak
        ])
        csv_file.flush()

        # 6. Fast OpenCV Visual UI alternative (Replace Matplotlib entirely)
        cv2.putText(frame, f"Frame: {frame_count} | HW HR: {hw_hr} BPM", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FFT HR: {hr_fft:.1f} | Peak HR: {hr_peak:.1f}", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Real-time Synchronized Recording", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 7. Regulate Execution Speed to Target Frame Rate (50ms cycles)
        elapsed = time.time() - loop_start
        time_to_wait = frame_interval - elapsed
        if time_to_wait > 0:
            time.sleep(time_to_wait)

except KeyboardInterrupt:
    print("\nRecording cut short by user manual interrupt.")

finally:
    # --- GRACEFUL HARDWARE TEARDOWN ---
    print("\nShutting down hardware and flushing buffers...")
    monitor.stop_live_acquisition()
    monitor.disconnect()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    csv_file.close()
    plt.ioff()
    plt.close('all')
    
    total_run_time = time.time() - start_time
    print("\n--- ACQUISITION REPORT ---")
    print(f"Video Saved: {video_filename}")
    print(f"Data CSV Saved: {csv_filename}")
    print(f"Total Frames Logged: {frame_count} frames across {total_run_time:.2f} seconds.")
    print(f"Calculated Average Recording Speed: {frame_count / total_run_time:.2f} FPS")
    print("System offline.")