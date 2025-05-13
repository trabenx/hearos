import sounddevice as sd
# import soundfile as sf # Only needed if saving locally, wavio handles WAV format
import numpy as np
import requests
import time
import datetime
import threading
import os
import wavio # For saving WAV files easily
import traceback

# --- Configuration ---
# CRITICAL: Adjust SAMPLE_RATE based on `arecord --dump-hw-params -D hw:X,Y`
SAMPLE_RATE = 48000  # Hz (e.g., 48000 or 44100 - MUST BE SUPPORTED BY MIC)
DURATION_SECONDS_PER_CHUNK = 4  # Process audio in 2-second chunks (Increased to reduce overflow)
RECORD_SECONDS_ON_TRIGGER = 10 # Record 10 seconds when triggered
RMS_THRESHOLD = 0.035  # Adjust this based on microphone and environment (Slightly increased)
SERVER_AUDIO_ENDPOINT = "YOUR_SERVER_AUDIO_ENDPOINT_HERE/upload_audio"
SERVER_HEARTBEAT_ENDPOINT = "YOUR_SERVER_HEARTBEAT_ENDPOINT_HERE/heartbeat"
HEARTBEAT_INTERVAL_SECONDS = 60
RECORDING_PATH = os.path.expanduser("~/recordings") # For local saving if server fails

# --- Get Unique ID ---
def get_rpi_serial():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Serial'):
                    serial = line.split(':')[1].strip()
                    return serial[-8:] if len(serial) > 8 else serial # More robust
        print("audio_monitor.py: Warning - Could not read serial from /proc/cpuinfo.")
        return "cpuinfo_read_error"
    except Exception as e:
        print(f"audio_monitor.py: Warning - Exception reading serial: {e}")
        return "unknown_pi_serial_exception"

RPI_UNIQUE_ID = get_rpi_serial()

# --- Ensure recording path exists ---
try:
    if not os.path.exists(RECORDING_PATH):
        print(f"audio_monitor.py: Recordings directory {RECORDING_PATH} not found. Creating it.")
        os.makedirs(RECORDING_PATH, exist_ok=True)
except Exception as e_mkdir_rec:
    print(f"audio_monitor.py: CRITICAL - Error creating RECORDING_PATH {RECORDING_PATH}: {e_mkdir_rec}")
    # This could be a fatal error for the recording functionality

# --- Global state for cooldown ---
last_trigger_time = 0
COOLDOWN_PERIOD_SECONDS = 30

# --- Audio Processing & Server Communication ---
def send_audio_to_server(filename):
    print(f"audio_monitor.py: Attempting to send {filename} to server...")
    if SERVER_AUDIO_ENDPOINT == "YOUR_SERVER_AUDIO_ENDPOINT_HERE/upload_audio":
        print("audio_monitor.py: ERROR - Server audio endpoint not configured. Cannot send audio.")
        return

    try:
        with open(filename, 'rb') as f:
            files = {'audio_file': (os.path.basename(filename), f, 'audio/wav')}
            headers = {'X-Device-ID': RPI_UNIQUE_ID}
            response = requests.post(SERVER_AUDIO_ENDPOINT, files=files, headers=headers, timeout=30)
            if response.status_code == 200:
                print(f"audio_monitor.py: Successfully sent {filename}. Server response: {response.text}")
                # Consider deleting file after successful upload if desired: os.remove(filename)
            else:
                print(f"audio_monitor.py: Failed to send {filename}. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"audio_monitor.py: Network error sending {filename}: {e}")
    except Exception as e:
        print(f"audio_monitor.py: Unexpected error sending {filename}: {e}")
        traceback.print_exc()

def record_and_send_audio():
    global last_trigger_time, COOLDOWN_PERIOD_SECONDS # Ensure globals are used
    print("audio_monitor.py: !!! Potential Event Detected - Checking Cooldown !!!") # Added for clarity
    current_time = time.time()
    if current_time - last_trigger_time < COOLDOWN_PERIOD_SECONDS:
        print(f"audio_monitor.py: Cooldown active. Time since last trigger: {current_time - last_trigger_time:.2f}s. Skipping.")
        return # This IS the debounce/cooldown
    last_trigger_time = current_time # Update last_trigger_time ONLY if not in cooldown
    print("audio_monitor.py: Cooldown passed. Starting Recording Process.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORDING_PATH, f"event_{RPI_UNIQUE_ID}_{timestamp}.wav")

    try:
        print(f"audio_monitor.py: Recording {RECORD_SECONDS_ON_TRIGGER}s of audio to {filename}...")
        # Record as float32 first to match InputStream, then convert
        recording_float32 = sd.rec(int(RECORD_SECONDS_ON_TRIGGER * SAMPLE_RATE),
                                   samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print(f"audio_monitor.py: Recording finished. Shape: {recording_float32.shape}, Dtype: {recording_float32.dtype}")

        if recording_float32 is None or recording_float32.size == 0:
            print("audio_monitor.py: ERROR - sd.rec() returned empty data!")
            return
        if np.all(recording_float32 == 0):
            print("audio_monitor.py: WARNING - Recorded data appears to be all zeros (silence).")

        # Convert to int16 for WAV
        recording_clipped = np.clip(recording_float32, -1.0, 1.0)
        recording_int16 = (recording_clipped * 32767).astype(np.int16)
        
        wavio.write(filename, recording_int16, SAMPLE_RATE, sampwidth=2) # sampwidth=2 for 16-bit
        file_size = os.path.getsize(filename)
        print(f"audio_monitor.py: Audio saved to {filename}. File size: {file_size} bytes.")
        if file_size < 44:
             print("audio_monitor.py: ERROR - Saved WAV file size is too small, likely empty or corrupt!")

        threading.Thread(target=send_audio_to_server, args=(filename,)).start()

    except sd.PortAudioError as pae:
        print(f"audio_monitor.py: CRITICAL PortAudioError during sd.rec(): {pae}")
        traceback.print_exc()
    except Exception as e:
        print(f"audio_monitor.py: Error during recording/saving: {e}")
        traceback.print_exc()

def audio_callback(indata, frames, time_info, status):
    if status:
        # Be cautious with printing in callback; can cause its own overflows if too frequent
        # print(f"audio_monitor.py: Audio callback status: {status}") # Potentially verbose
        if status.input_overflow:
             print("audio_monitor.py: WARNING - Input overflow detected in audio_callback!")
        # Add other status checks if needed (output_underflow, etc.)

    rms = np.sqrt(np.mean(indata**2))
    # print(f"RMS: {rms:.4f}") # For debugging threshold, very verbose

    if rms > RMS_THRESHOLD:
        # Launching a thread directly from callback can be resource-intensive if triggers are rapid.
        # For now, keep it simple. If still overflow, might need a queue or debounce.
        print(f"audio_monitor.py: RMS {rms:.4f} > threshold {RMS_THRESHOLD}. Triggering record_and_send_audio.") # Log before threading
        threading.Thread(target=record_and_send_audio).start()

def start_listening():
    print("--- audio_monitor.py: start_listening() ENTERED ---")
    global SAMPLE_RATE, DURATION_SECONDS_PER_CHUNK, RPI_UNIQUE_ID, RECORDING_PATH, SERVER_AUDIO_ENDPOINT, SERVER_HEARTBEAT_ENDPOINT, RMS_THRESHOLD
    print(f"Audio_monitor: Using SAMPLE_RATE={SAMPLE_RATE}, DURATION_SECONDS_PER_CHUNK={DURATION_SECONDS_PER_CHUNK}")
    print(f"Audio_monitor: Device ID: {RPI_UNIQUE_ID}")
    print(f"Audio_monitor: Recordings will be saved to: {RECORDING_PATH}")
    print(f"Audio_monitor: Server audio endpoint: {SERVER_AUDIO_ENDPOINT}")
    print(f"Audio_monitor: Server heartbeat endpoint: {SERVER_HEARTBEAT_ENDPOINT}")
    print(f"Audio_monitor: RMS_THRESHOLD={RMS_THRESHOLD}")

    try:
        print("Audio_monitor: Querying audio devices with sd.query_devices()...")
        devices = sd.query_devices()
        print(f"Audio_monitor: Found audio devices:\n{devices}")
        # You might want to log sd.default.device if debugging specific device selection
        # print(f"Audio_monitor: Default input device index: {sd.default.device[0]}, Default output: {sd.default.device[1]}")


        print("Audio_monitor: Attempting to open sd.InputStream...")
        with sd.InputStream(callback=audio_callback,
                             samplerate=SAMPLE_RATE,
                             channels=1,
                             blocksize=int(SAMPLE_RATE * DURATION_SECONDS_PER_CHUNK), # Blocksize based on duration
                             dtype='float32'): # Use float32 for internal processing by sounddevice
            print("--- audio_monitor.py: sd.InputStream OPENED. Monitoring Audio. ---")
            print("Audio_monitor: Press Ctrl+C (in main app) to stop.")
            while True: # Keep this thread's main function alive
                time.sleep(3600) # The callback operates in its own thread managed by sounddevice
    except sd.PortAudioError as pae:
        print(f"--- audio_monitor.py: CRITICAL PortAudioError in start_listening ---")
        print(f"Error details: {pae}")
        traceback.print_exc()
    except Exception as e:
        print(f"--- audio_monitor.py: CRITICAL UNEXPECTED ERROR in start_listening ---")
        print(f"Error details: {e}")
        traceback.print_exc()
    print("--- audio_monitor.py: start_listening() EXITED (likely due to error or manual stop if not in loop) ---")

# --- Heartbeat Mechanism ---
def send_heartbeat():
    # ... (send_heartbeat function as previously defined, with print statements prefixed by "audio_monitor.py:") ...
    print(f"audio_monitor.py: Attempting to send heartbeat...")
    if SERVER_HEARTBEAT_ENDPOINT == "YOUR_SERVER_HEARTBEAT_ENDPOINT_HERE/heartbeat":
        # print("audio_monitor.py: Heartbeat endpoint not configured. Skipping.") # Can be verbose
        return
    headers = {'X-Device-ID': RPI_UNIQUE_ID}
    payload = {"timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z"}
    try:
        response = requests.post(SERVER_HEARTBEAT_ENDPOINT, headers=headers, json=payload, timeout=15)
        if response.status_code == 200:
            # print(f"audio_monitor.py: Heartbeat sent successfully at {payload['timestamp_utc']}.") # Verbose
            pass
        else:
            print(f"audio_monitor.py: Failed to send heartbeat. Status: {response.status_code}, Response: {response.text}")
    except requests.exceptions.Timeout:
        print(f"audio_monitor.py: Heartbeat request timed out to {SERVER_HEARTBEAT_ENDPOINT}.")
    except requests.exceptions.ConnectionError:
        print(f"audio_monitor.py: Heartbeat connection error to {SERVER_HEARTBEAT_ENDPOINT}.")
    except requests.exceptions.RequestException as e:
        print(f"audio_monitor.py: Error sending heartbeat: {e}")
    except Exception as e:
        print(f"audio_monitor.py: An unexpected error occurred during heartbeat: {e}")
        traceback.print_exc()


def heartbeat_loop():
    print("--- audio_monitor.py: Heartbeat_loop() ENTERED. ---")
    while True:
        send_heartbeat()
        time.sleep(HEARTBEAT_INTERVAL_SECONDS)
    # This loop will run until the thread is killed
    # print("--- audio_monitor.py: Heartbeat_loop() EXITED. ---") # Unlikely to be reached

# --- For direct testing of audio_monitor.py ---
if __name__ == "__main__":
    print("audio_monitor.py: Running directly for testing...")
    # Start heartbeat in a separate thread if testing audio_monitor.py directly
    if SERVER_HEARTBEAT_ENDPOINT != "YOUR_SERVER_HEARTBEAT_ENDPOINT_HERE/heartbeat":
        print("audio_monitor.py: Starting test heartbeat_loop thread...")
        heartbeat_test_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        heartbeat_test_thread.start()
    
    print("audio_monitor.py: Calling start_listening() for direct test...")
    start_listening()
    print("audio_monitor.py: Direct test finished (start_listening completed or was interrupted).")
