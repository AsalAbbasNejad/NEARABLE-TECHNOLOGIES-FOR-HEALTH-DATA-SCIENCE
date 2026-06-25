"""CMS50D pulse-oximeter serial interface + port detection.

The CMS50D class is lifted verbatim from
live_rppg_cms50d_original_settings_with_metrics.py (serial packet
protocol unchanged). list_serial_ports() helps the UI surface candidate
ports so the oximeter features auto-enable when the device is connected.
"""

import datetime
import queue
import threading

import serial
import serial.tools.list_ports


class CMS50D:
    def __init__(self, port, baudrate=115200, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.connection = None
        self.realtime_streaming = False
        self.keepalive_interval = datetime.timedelta(seconds=5)
        self.keepalive_timestamp = datetime.datetime.now()
        self.data_queue = queue.Queue(maxsize=10)
        self.thread = None

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
        print(f"[CMS50D] Connected on {self.port}")

    def disconnect(self):
        if self.connection and self.connection.is_open:
            self.connection.close()
            print("[CMS50D] Disconnected")

    def send_command(self, command):
        def encode_package(cmd):
            package_type = 0x7D
            data = [cmd] + [0x00] * 6
            high_byte = 0x80

            for i in range(len(data)):
                high_byte |= (data[i] & 0x80) >> (7 - i)
                data[i] |= 0x80

            package_type &= 0x7F
            return [package_type, high_byte] + data

        package = encode_package(command)
        self.connection.write(bytes(package))
        self.connection.flush()

    def send_keepalive(self):
        now = datetime.datetime.now()

        if now - self.keepalive_timestamp > self.keepalive_interval:
            self.send_command(0xAF)
            self.keepalive_timestamp = now

    def start_live_acquisition(self):
        if self.connection is None or not self.connection.is_open:
            raise RuntimeError("CMS50D is not connected.")

        self.connection.reset_input_buffer()
        self.send_command(0xA1)
        self.realtime_streaming = True

        self.thread = threading.Thread(target=self._collect_data)
        self.thread.daemon = True
        self.thread.start()

        print("[CMS50D] Live acquisition started")

    def stop_live_acquisition(self):
        if self.connection and self.connection.is_open:
            try:
                self.send_command(0xA2)
            except Exception:
                pass

        self.realtime_streaming = False
        print("[CMS50D] Live acquisition stopped")

    def _read_packet(self):
        while self.realtime_streaming:
            self.send_keepalive()

            byte = self.connection.read()

            if not byte:
                return None

            if not (byte[0] & 0x80):
                packet = byte + self.connection.read(8)

                if len(packet) == 9:
                    return list(packet)

        return None

    def _decode_packet(self, packet):
        package_type = packet[0]
        high_byte = packet[1]
        data = list(packet[2:])

        for i in range(len(data)):
            data[i] = (data[i] & 0x7F) | ((high_byte << (7 - i)) & 0x80)

        return package_type, data

    def _collect_data(self):
        while self.realtime_streaming:
            packet = self._read_packet()

            if packet is None:
                continue

            package_type, data = self._decode_packet(packet)

            if package_type == 0x01 and len(data) == 7:
                signal_strength = data[0] & 0x0F
                pulse_beep = (data[0] & 0x40) >> 6
                probe_error = (data[0] & 0x80) >> 7
                pulse_waveform = data[1] & 0x7F
                pulse_rate = data[3]
                spo2 = data[4]

                packet_dict = {
                    "timestamp": datetime.datetime.now(),
                    "pulse_rate": None if pulse_rate == 0xFF else pulse_rate,
                    "spO2": None if spo2 == 0x7F else spo2,
                    "waveform": pulse_waveform,
                    "signal_strength": signal_strength,
                    "pulse_beep": pulse_beep,
                    "probe_error": probe_error
                }

                # Keep only recent data. If queue is full, discard the oldest one.
                if self.data_queue.full():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.data_queue.put(packet_dict)

    def get_latest_data(self):
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None


# ============================================================
# Port detection
# ============================================================

# Built-in macOS ports that are never a CMS50D; hidden from the picker.
_IGNORE_SUBSTRINGS = (
    "bluetooth",
    "debug-console",
    "wlan-debug",
)

# Hints that a serial port is a USB device worth trying as the oximeter.
_CANDIDATE_HINTS = (
    "usb",
    "cms",
    "spo2",
    "contec",
    "ch340",
    "cp210",
    "pl2303",
    "silicon labs",
    "wch",
)


def _is_ignored(device: str) -> bool:
    low = device.lower()
    return any(s in low for s in _IGNORE_SUBSTRINGS)


def _looks_like_candidate(port) -> bool:
    blob = " ".join(
        str(x).lower() for x in (port.device, port.description, port.hwid)
    )
    return any(h in blob for h in _CANDIDATE_HINTS)


def list_serial_ports():
    """Return selectable serial ports with a 'candidate' flag.

    Built-in Bluetooth/debug ports are filtered out. A port is flagged as
    a candidate when it looks like a USB-serial device (the CMS50D uses a
    USB bridge), so the UI can pre-select it.
    """
    ports = []
    for p in serial.tools.list_ports.comports():
        if _is_ignored(p.device):
            continue
        ports.append(
            {
                "device": p.device,
                "description": p.description or "",
                "hwid": p.hwid or "",
                "candidate": _looks_like_candidate(p),
            }
        )
    return ports


def detect_cms50d_port():
    """Best-guess CMS50D port, or None. Picks the first candidate."""
    for p in list_serial_ports():
        if p["candidate"]:
            return p["device"]
    return None
