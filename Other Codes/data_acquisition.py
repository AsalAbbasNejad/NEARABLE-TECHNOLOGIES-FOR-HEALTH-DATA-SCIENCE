# this cell is for defining the function for acquiring the data

import threading # imports python's threading module, used to run data collection in the background without blocking the main program.
import queue # Imports the thread-safe queue module, used as a buffer between the background data collection thread and the main porgram
import time # Imports the utilites
import datetime # Imports the datetimestamping data packets and measuring keepalive intervals
import serial # imports pyserial, the library that handles the physical USB/serial communication with the device

class CMS50D:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        self.realtime_streaming = False
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        #self.data_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory overload
        self.data_queue = queue.Queue(maxsize=10)  # Limit queue size to avoid memory overload
        self.thread = None  # Thread for background data collection

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

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()

    def send_command(self, command):
        def encode_package(cmd):
            package_type = 0x7D
            data = [cmd] + [0x00] * 6 # builds a 7- byte data array: the command byte followed by 6 zero padding bytes
            high_byte = 0x80
            for i in range(len(data)):
                high_byte |= (data[i] & 0x80) >> (7 - i)
                data[i] |= 0x80  # Set sync bit
            package_type &= 0x7F  # Clear sync bit
            return [package_type, high_byte] + data

        package = encode_package(command)
        self.connection.write(bytes(package))
        self.connection.flush()


    def send_keepalive(self):
        now = datetime.datetime.now()
        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xaf)  # keepalive
            self.keepalive_timestamp = now

    def start_live_acquisition(self):
        self.connection.reset_input_buffer()
        self.send_command(0xA1)  # Start real-time data
        self.realtime_streaming = True

        # Start a background thread for data collection
        self.thread = threading.Thread(target=self._collect_data)
        self.thread.daemon = True  # Daemonize the thread so it exits with the program
        self.thread.start()

    def stop_live_acquisition(self):
        self.send_command(0xA2)  # Stop real-time data
        self.realtime_streaming = False

    def _collect_data(self):
        while self.realtime_streaming:
            #print("Collecting data...")
            packet = self._read_packet()
            if packet:
                package_type, data = self._decode_packet(packet)
                if package_type == 0x01 and len(data) == 7:
                    # Parse fields and put the data in the queue
                    signal_strength = data[0] & 0x0F
                    pulse_beep = (data[0] & 0x40) >> 6
                    probe_error = (data[0] & 0x80) >> 7
                    pulse_waveform = data[1] & 0x7F
                    pulse_rate = data[3]
                    spO2 = data[4]

                    # Only put the latest data into the queue
                    if not self.data_queue.full():
                        self.data_queue.put({
                            "timestamp": datetime.datetime.now(),
                            "pulse_rate": None if pulse_rate == 0xFF else pulse_rate,
                            "spO2": None if spO2 == 0x7F else spO2,
                            "waveform": pulse_waveform,
                            "signal_strength": signal_strength,
                            "pulse_beep": pulse_beep,
                            "probe_error": probe_error
                        })
                    #else:
                        #print("Queue is full, discarding data.")
                #else:
                    #print("No data received. Check device connection or timeouts.")
            #time.sleep(0.01)  # Add small delay to help stabilize the connection

    def _read_packet(self):
        while True:
            self.send_keepalive()
            byte = self.connection.read()
            if not byte:
                print("Serial timeout or no data received.")
                return None
            #print(f"Reading byte: {byte}")
            if not byte:
                return None
            if not (byte[0] & 0x80):
                packet = byte + self.connection.read(8)
                if len(packet) == 9:
                    return list(packet)

    def _decode_packet(self, packet):
        package_type = packet[0]
        high_byte = packet[1]
        data = list(packet[2:])
        for i in range(len(data)):
            data[i] = (data[i] & 0x7F) | ((high_byte << (7 - i)) & 0x80)
        return package_type, data

    def get_latest_data(self):
        """
        Fetch the latest data from the queue.
        Returns None if no data is available.
        """
        try:
            return self.data_queue.get_nowait()  # Non-blocking, fetch data from queue
        except queue.Empty:
            return None
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