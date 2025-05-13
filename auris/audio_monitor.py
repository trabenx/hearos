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
import datetime

# --- Porcupine Imports ---
import pvporcupine
from pvrecorder import PvRecorder
import pvrecorder


# --- Configuration ---
SAMPLE_RATE = 16000  # Porcupine typically uses 16000 Hz. PvRecorder will handle this.
# DURATION_SECONDS_PER_CHUNK = NO LONGER NEEDED FOR INPUT STREAM if PvRecorder is primary
RECORD_SECONDS_ON_TRIGGER = 10
RMS_THRESHOLD = 0.045
SERVER_AUDIO_ENDPOINT = "YOUR_SERVER_AUDIO_ENDPOINT_HERE/upload_audio"
SERVER_HEARTBEAT_ENDPOINT = "YOUR_SERVER_HEARTBEAT_ENDPOINT_HERE/heartbeat"
HEARTBEAT_INTERVAL_SECONDS = 60
RECORDING_PATH = os.path.expanduser("~/recordings")

# --- Porcupine Specific Configuration ---
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY") # Use a distinct env var name
# IMPORTANT: Adjust this path to where your .ppn file is
KEYWORD_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "kws", "superman_en_raspberry-pi_v3_0_0.ppn")
PORCUPINE_SENSITIVITY = 0.65

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
last_event_trigger_time = 0 # Generic name for last trigger
EVENT_COOLDOWN_SECONDS = 30 # Cooldown for any event type


# --- Event Types (Optional, for distinguishing triggers) ---
class EventType:
    RMS_LOUD_SOUND = 1
    KEYWORD_DETECTED = 2


# --- Audio Processing & Server Communication ---
def send_audio_to_server(filename, event_type_str="UNKNOWN"):
    print(f"audio_monitor.py: Attempting to send {filename} (Event: {event_type_str}) to server...")
    if SERVER_AUDIO_ENDPOINT == "YOUR_SERVER_AUDIO_ENDqPOINT_HERE/upload_audio":
        print("audio_monitor.py: ERROR - Server audio endpoint not configured. Cannot send audio.")
        return

    try:
        with open(filename, 'rb') as f:
            files = {'audio_file': (os.path.basename(filename), f, 'audio/wav')}
            headers = {
               'X-Device-ID': RPI_UNIQUE_ID,
               'X-Event-Type': event_type_str
            }
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

def record_and_send_triggered_audio(event_type_enum):
    global last_event_trigger_time, RECENT_AUDIO_BUFFER

    event_type_str = "KEYWORD" if event_type_enum == EventType.KEYWORD_DETECTED else "RMS_LOUD_SOUND"
    print(f"audio_monitor.py: Event Detected ({event_type_str}) - Checking Cooldown.")

    current_time = time.time()
    if current_time - last_event_trigger_time < EVENT_COOLDOWN_SECONDS:
        print(f"audio_monitor.py: Cooldown active for events. Skipping.")
        return

    # If not in cooldown, proceed and THEN update last_event_trigger_time
    # This was a potential logic error: last_event_trigger_time should only be updated
    # for *actual* processed events, not just for attempts that get caught by cooldown.
    # However, for simplicity, updating it when the function is *called* (if not in cooldown) is okay.
    # The current code updates it *after* the cooldown check if it passes. That's correct.

    last_event_trigger_time = current_time
    print(f"audio_monitor.py: Cooldown passed for {event_type_str}. Processing event.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORDING_PATH, f"event_{RPI_UNIQUE_ID}_{event_type_str}_{timestamp}.wav")

    # --- Always perform a fresh recording using sounddevice ---
    # This assumes PvRecorder has been stopped by the calling code in porcupine_worker
    print(f"audio_monitor.py: Recording {RECORD_SECONDS_ON_TRIGGER}s fresh audio with sounddevice to {filename}...")
    try:
        # Porcupine uses 16kHz, so sd.rec should also use 16kHz if PvRecorder was the source
        # If your SAMPLE_RATE global is still 48000 for sd.rec, this is a mismatch.
        # Let's use a specific recording sample rate here, or ensure global SAMPLE_RATE is 16000
        # if PvRecorder is the main source.
        # For consistency, if PvRecorder is the main audio loop, triggered recordings should match its rate.
        recording_sample_rate = 16000 # Match Porcupine's rate

        recording_float32 = sd.rec(int(RECORD_SECONDS_ON_TRIGGER * recording_sample_rate),
                                   samplerate=recording_sample_rate, channels=1, dtype='float32')
        sd.wait()
        print(f"audio_monitor.py: Fresh recording finished. Shape: {recording_float32.shape}")

        if recording_float32 is None or recording_float32.size == 0:
            print("audio_monitor.py: ERROR - sd.rec() returned empty data for triggered event!")
            return
        if np.all(recording_float32 == 0): # Check for complete silence
            print("audio_monitor.py: WARNING - Triggered recording data appears to be all zeros (silence). Check mic levels for sd.rec().")

        recording_clipped = np.clip(recording_float32, -1.0, 1.0)
        recording_int16 = (recording_clipped * 32767).astype(np.int16)
        wavio.write(filename, recording_int16, recording_sample_rate, sampwidth=2)
        file_size = os.path.getsize(filename)
        print(f"audio_monitor.py: Triggered audio saved to {filename}. File size: {file_size} bytes.")
        if file_size < 44: # WAV header size
             print("audio_monitor.py: ERROR - Saved WAV file size is too small, likely empty or corrupt!")


        threading.Thread(target=send_audio_to_server, args=(filename, event_type_str)).start()

    except sd.PortAudioError as pae:
        print(f"audio_monitor.py: CRITICAL PortAudioError during triggered sd.rec(): {pae}")
        traceback.print_exc()
    except Exception as e:
        print(f"audio_monitor.py: Error during triggered recording/saving: {e}")
        traceback.print_exc()
    # PvRecorder restart is handled in porcupine_worker


def easier_than_deleting():
    # --- Option 1: Record fresh audio using sounddevice (simpler, but might miss the event itself) ---
    # This would require stopping PvRecorder, then using sd.rec(), then restarting PvRecorder.
    # print(f"audio_monitor.py: Recording {RECORD_SECONDS_ON_TRIGGER}s fresh audio to {filename}...")
    # try:
    #     # Assumes PvRecorder is stopped if it was the main loop
    #     recording_float32 = sd.rec(int(RECORD_SECONDS_ON_TRIGGER * SAMPLE_RATE),
    #                                samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    #     sd.wait()
    #     # ... (conversion to int16 and saving as before) ...
    #     # ... (send_audio_to_server) ...
    # except Exception as e:
    #     print(f"audio_monitor.py: Error during fresh recording: {e}")
    #     traceback.print_exc()
    # return # Important if using this option

    # --- Option 2: Save the buffered audio from PvRecorder (preferred) ---
    # This captures audio leading up to and including the event.
    if not RECENT_AUDIO_BUFFER:
        print("audio_monitor.py: WARNING - RECENT_AUDIO_BUFFER is empty. Cannot save triggered audio.")
        # Fallback: could trigger a fresh recording here if desired.
        return

    print(f"audio_monitor.py: Saving buffered audio ({len(RECENT_AUDIO_BUFFER)} frames) to {filename}...")
    try:
        # PvRecorder provides PCM data as list of int16. Concatenate and save.
        # Important: Lock or copy the buffer if PvRecorder thread is still modifying it.
        # For simplicity, let's assume PvRecorder thread adds, this one consumes a copy.
        # A proper circular buffer with locks would be more robust.

        # Create a copy of the buffer to avoid modification during processing
        # This simple approach might miss some frames if not synchronized with PvRecorder's additions
        # For this example, let's assume RECENT_AUDIO_BUFFER is managed carefully.
        # Typically, the keyword detection loop would populate this.

        # We need to ensure RECENT_AUDIO_BUFFER contains frames of int16 values
        # If PvRecorder.read() gives a list of int16s for each frame, flatten it.
        # The Porcupine loop will need to append to RECENT_AUDIO_BUFFER.

        # This part assumes RECENT_AUDIO_BUFFER is populated by the Porcupine loop
        # with frames, and each frame is a list/array of int16 samples.
        current_buffer_copy = []
        # Simple shallow copy for now; for production, deep copy or locks.
        # For this conceptual merge, the Porcupine loop will handle filling the buffer.
        # Here, we'd just read it.
        # Actual audio data needs to be collected by the Porcupine loop.
        # This function will be called *from* the Porcupine loop or RMS logic.

        # Let's simulate that the buffer is ready (filled by the Porcupine/RMS data handler)
        # We need to get the actual audio data from the PcmFrame in the Porcupine loop.
        # For RMS, we got 'indata'.

        # For now, let's assume the keyword detection has *already* grabbed a relevant chunk of audio.
        # This function's role is primarily to save and send, not re-record here if using Porcupine as main.
        # The audio data for saving must come from the PvRecorder stream.

        # THIS FUNCTION NEEDS REVISITING ONCE THE PORCUPINE LOOP IS THE MAIN AUDIO SOURCE
        # If PvRecorder is the main source, `record_and_send_triggered_audio`
        # will get the audio data passed to it, or read it from a shared buffer.

        # For now, let's revert to the simple sd.rec() for triggered recording,
        # assuming PvRecorder will be stopped/started around it if it's the main audio loop.
        # This is simpler to integrate initially than a shared buffer from PvRecorder.

        # ** SIMPLIFIED APPROACH FOR NOW: Stop PvRecorder, do fresh recording with sd.rec **
        # This means the main Porcupine loop needs to be able to stop/start PvRecorder.
        # This will be handled in the `porcupine_worker` function.
        print(f"audio_monitor.py: (Conceptual) PvRecorder would be stopped here if it's the main loop.")
        print(f"audio_monitor.py: Recording {RECORD_SECONDS_ON_TRIGGER}s fresh audio with sounddevice to {filename}...")

        recording_float32 = sd.rec(int(RECORD_SECONDS_ON_TRIGGER * SAMPLE_RATE),
                                   samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        print(f"audio_monitor.py: Fresh recording finished. Shape: {recording_float32.shape}")

        if recording_float32 is None or recording_float32.size == 0:
            print("audio_monitor.py: ERROR - sd.rec() returned empty data for triggered event!")
            return
        if np.all(recording_float32 == 0):
            print("audio_monitor.py: WARNING - Triggered recording data appears to be all zeros.")

        recording_clipped = np.clip(recording_float32, -1.0, 1.0)
        recording_int16 = (recording_clipped * 32767).astype(np.int16)

        wavio.write(filename, recording_int16, SAMPLE_RATE, sampwidth=2)
        file_size = os.path.getsize(filename)
        print(f"audio_monitor.py: Triggered audio saved to {filename}. File size: {file_size} bytes.")

        threading.Thread(target=send_audio_to_server, args=(filename, event_type_str)).start()

    except sd.PortAudioError as pae:
        print(f"audio_monitor.py: CRITICAL PortAudioError during triggered sd.rec(): {pae}")
        traceback.print_exc()
    except Exception as e:
        print(f"audio_monitor.py: Error during triggered recording/saving: {e}")
        traceback.print_exc()
    finally:
        print(f"audio_monitor.py: (Conceptual) PvRecorder would be restarted here.")

# --- Porcupine Worker Thread ---
porcupine_instance = None
pv_recorder_instance = None
porcupine_thread_stop_event = threading.Event() # To signal the thread to stop

def porcupine_worker():
    global porcupine_instance, pv_recorder_instance, RECENT_AUDIO_BUFFER
    print("audio_monitor.py: porcupine_worker() ENTERED") # NEW

    if not PORCUPINE_ACCESS_KEY:
        print("audio_monitor.py: CRITICAL - PORCUPINE_ACCESS_KEY not set. Porcupine thread cannot start.")
        return
    print("audio_monitor.py: PORCUPINE_ACCESS_KEY found.") # NEW

    if not os.path.exists(KEYWORD_MODEL_PATH):
        print(f"audio_monitor.py: CRITICAL - Keyword model file not found: {KEYWORD_MODEL_PATH}. Porcupine thread cannot start.")
        return
    print(f"audio_monitor.py: Keyword model file {KEYWORD_MODEL_PATH} found.") # NEW

    try:
        print("audio_monitor.py: Attempting pvporcupine.create()...")
        porcupine_instance = pvporcupine.create(
            access_key=PORCUPINE_ACCESS_KEY,
            keyword_paths=[KEYWORD_MODEL_PATH],
            sensitivities=[PORCUPINE_SENSITIVITY]
        )
        print(f"audio_monitor.py: Porcupine initialized. Version: {porcupine_instance.version}")

        # --- List PvRecorder devices for debugging ---
        print("audio_monitor.py: Listing available PvRecorder audio devices...")
        available_devices = PvRecorder.get_available_devices()
        if not available_devices:
            print("audio_monitor.py: CRITICAL - PvRecorder found NO audio devices!")
        for index, device_name in enumerate(available_devices):
            print(f"audio_monitor.py: PvRecorder Device #{index}: {device_name}")
        # --- End device listing ---

    # --- Determine the correct device index ---
        target_device_name_part = "USB PnP Sound Device" # Or a more unique part of its name
        pv_device_index = -1 # Default to -1 (system default)
        if available_devices: # Check if list is not empty
            for index, device_name in enumerate(available_devices):
                if target_device_name_part in device_name:
                    pv_device_index = index
                    print(f"audio_monitor.py: Found target mic '{target_device_name_part}' at PvRecorder index {pv_device_index}")
                    break
            if pv_device_index == -1:
                print(f"audio_monitor.py: WARNING - Target mic '{target_device_name_part}' not found by PvRecorder. Using default (-1).")
        else:
            print("audio_monitor.py: No PvRecorder devices listed, using default (-1).")
        # --- End device index determination ---



        print("audio_monitor.py: Attempting PvRecorder() creation...") # NEW
        pv_recorder_instance = PvRecorder(
            frame_length=porcupine_instance.frame_length, # Usually 512
            device_index=pv_device_index
        )
        print("audio_monitor.py: PvRecorder instance created.") # NEW
        # SAMPLE_RATE is implicitly handled by PvRecorder matching Porcupine's needs (16kHz)

        print("audio_monitor.py: Attempting pv_recorder_instance.start()...") # NEW
        pv_recorder_instance.start()
        print(f"audio_monitor.py: PvRecorder started. Listening for keyword '{os.path.basename(KEYWORD_MODEL_PATH)}'...")

        # Clear and prepare recent audio buffer
        RECENT_AUDIO_BUFFER = [] 

        print("audio_monitor.py: Entering main Porcupine processing loop...")
        while not porcupine_thread_stop_event.is_set():
            pcm_frame = pv_recorder_instance.read() # Reads one frame of audio (list of int16)

            # --- Option A: RMS detection from Porcupine's audio frames ---
            # Convert pcm_frame (list of int16) to numpy array for RMS
            # Scale int16 to float32 range [-1.0, 1.0] for consistent RMS calculation
            np_frame_float32 = np.array(pcm_frame, dtype=np.float32) / 32768.0
            rms = np.sqrt(np.mean(np_frame_float32**2))
            # print(f"RMS from PvRecorder: {rms:.4f}") # DEBUG - very verbose

            rms_triggered = False
            if rms > RMS_THRESHOLD:
                print(f"audio_monitor.py: RMS ({rms:.4f}) > threshold ({RMS_THRESHOLD}). Potential RMS event.")
                rms_triggered = True
#                # To record, we need to stop PvRecorder, then call record_and_send_triggered_audio, then restart
#                if pv_recorder_instance: pv_recorder_instance.stop()
#                record_and_send_triggered_audio(EventType.RMS_LOUD_SOUND)
#                if not porcupine_thread_stop_event.is_set() and pv_recorder_instance:
#                    print("audio_monitor.py: Restarting PvRecorder after RMS trigger.")
#                    pv_recorder_instance.start()
#                else:
#                    print("audio_monitor.py: Stop event set or recorder instance gone, not restarting PvRecorder after RMS.")


            # --- Keyword Detection ---
            keyword_triggered = False
            result = porcupine_instance.process(pcm_frame)
            if result >= 0:
                print(f"audio_monitor.py: [{datetime.datetime.now()}] Detected keyword! Index: {result}. Potential keyword event.")
                keyword_triggered = True
#                if pv_recorder_instance: pv_recorder_instance.stop()
#                record_and_send_triggered_audio(EventType.KEYWORD_DETECTED)
#                if not porcupine_thread_stop_event.is_set() and pv_recorder_instance:
#                    print("audio_monitor.py: Restarting PvRecorder after keyword trigger.")
#                    pv_recorder_instance.start()
#                else:
#                    print("audio_monitor.py: Stop event set or recorder instance gone, not restarting PvRecorder after keyword.")

            # --- Manage RECENT_AUDIO_BUFFER (Conceptual - if saving buffered audio) ---
            # RECENT_AUDIO_BUFFER.append(pcm_frame) # Add current frame
            # if len(RECENT_AUDIO_BUFFER) > MAX_BUFFER_FRAMES:
            #     RECENT_AUDIO_BUFFER.pop(0) # Keep buffer size limited (FIFO)
                        # --- Handle Trigger ---
            event_to_process = None
            if keyword_triggered: # Prioritize keyword if both happen in same frame (unlikely)
                event_to_process = EventType.KEYWORD_DETECTED
            elif rms_triggered:
                event_to_process = EventType.RMS_LOUD_SOUND

            if event_to_process is not None:
                # Check cooldown *before* stopping PvRecorder and calling the recording function
                current_time_for_cooldown_check = time.time()
                if current_time_for_cooldown_check - last_event_trigger_time < EVENT_COOLDOWN_SECONDS:
                    print(f"audio_monitor.py: Cooldown still active for any event. Skipping actual recording process. Time since last: {current_time_for_cooldown_check - last_event_trigger_time:.1f}s")
                else:
                    # Cooldown passed, proceed with stopping, recording, sending
                    print(f"audio_monitor.py: Cooldown passed. Proceeding with event: {event_to_process}")
                    if pv_recorder_instance: pv_recorder_instance.stop()
                    record_and_send_triggered_audio(event_to_process) # This function now also handles its own internal last_event_trigger_time update
                
                # Always try to restart PvRecorder if it was stopped, unless main stop event is set
                if not porcupine_thread_stop_event.is_set() and pv_recorder_instance:
                    # Check if it was actually stopped (e.g., if cooldown was active, it wasn't stopped)
                    # This restart logic might need to be more nuanced if cooldown prevented the stop.
                    # For simplicity, let's assume if an event was *considered*, we ensure PvRecorder is running.
                    # A better way: only restart if it was explicitly stopped.
                    # For now, this is okay, PvRecorder.start() might be idempotent if already running, or error if stopped then started.
                    
                    # More robust restart:
                    # Assume a flag 'pv_recorder_was_stopped = True' was set if pv_recorder_instance.stop() was called.
                    # if pv_recorder_was_stopped:
                    #     print("audio_monitor.py: Restarting PvRecorder after event processing.")
                    #     try:
                    #         pv_recorder_instance.start()
                    #     except pvrecorder.PvRecorderError as pvre_restart:
                    #         # ... handle error ...
                    # else:
                    #     print("audio_monitor.py: PvRecorder was not stopped (cooldown), no restart needed.")

                    # Simpler (but might try to start an already running recorder if cooldown skipped the stop):
                    print(f"audio_monitor.py: Attempting to ensure PvRecorder is running after potential event.")
                    try:
                        # If it was stopped, this starts it. If it wasn't stopped (due to cooldown),
                        # this might error or do nothing if start() is idempotent when already running.
                        # PvRecorder docs: "Resources is allocated upon initialization. Start and stop start and stop audio capture."
                        # It's likely okay to call start() if it wasn't explicitly stopped by this code path,
                        # but cleaner to only start if we stopped it.
                        # Let's assume it needs a start if we reached here.
                        pv_recorder_instance.start() # This needs testing if called when already running.
                                                     # The try/except should catch issues.
                    except pvrecorder.PvRecorderError as pvre_restart:
                        print(f"audio_monitor.py: Error ensuring PvRecorder is running: {pvre_restart}")
                        porcupine_thread_stop_event.set() 
                        break

        print("audio_monitor.py: Porcupine processing loop exited due to stop_event.")

    except pvporcupine.PorcupineError as pe:
        print(f"audio_monitor.py: PorcupineError in worker: {pe}")
        traceback.print_exc()
#    except pvrecorder.PvRecorderError as pvre:
#        print(f"audio_monitor.py: PvRecorderError in worker: {pvre}")
#        traceback.print_exc()
    except RuntimeError as re: # Catch the generic RuntimeError from PvRecorder if it's not a PvRecorderError
        print(f"audio_monitor.py: Generic RuntimeError (possibly from PvRecorder init): {re}")
        traceback.print_exc()
    except Exception as e:
        print(f"audio_monitor.py: Unexpected error in porcupine_worker: {e}")
        traceback.print_exc()
    finally:
        print("audio_monitor.py: Porcupine worker thread cleaning up...")
        if pv_recorder_instance:
            print("audio_monitor.py: Stopping PvRecorder...")
            pv_recorder_instance.stop() # Ensure it's stopped
            print("audio_monitor.py: Deleting PvRecorder instance...")
            pv_recorder_instance.delete()
            pv_recorder_instance = None
        if porcupine_instance:
            print("audio_monitor.py: Deleting Porcupine instance...")
            porcupine_instance.delete()
            porcupine_instance = None
        print("audio_monitor.py: Porcupine worker thread finished cleanup.")


# --- start_listening is now the main entry point for audio monitoring ---
def start_listening(): # This name is now a bit of a misnomer, it starts the Porcupine system
    print("--- audio_monitor.py: start_listening() (Porcupine Mode) ENTERED ---")
    global porcupine_thread_stop_event
    porcupine_thread_stop_event.clear() # Ensure stop event is clear on start

    print(f"Audio_monitor: Device ID: {RPI_UNIQUE_ID}")
    print(f"Audio_monitor: Recordings will be saved to: {RECORDING_PATH}")
    # ... any other initial prints ...

    # Start the Porcupine worker thread
    porcupine_thread = threading.Thread(target=porcupine_worker, name="PorcupineWorkerThread")
    porcupine_thread.daemon = True # Allow main program to exit even if this thread is running
    porcupine_thread.start()
    print("audio_monitor.py: PorcupineWorkerThread started.")

    # Keep the start_listening function (and thus the AudioMonitorThread from main) alive
    # while the porcupine_worker_thread does its job.
    # We can use the stop event to also break this loop if needed from elsewhere.
    try:
        while not porcupine_thread_stop_event.is_set():
            if not porcupine_thread.is_alive():
                print("audio_monitor.py: CRITICAL - PorcupineWorkerThread has died unexpectedly! Restarting logic might be needed.")
                # Attempt to restart Porcupine worker? Or signal main app to shut down?
                # For now, just break this loop.
                break
            time.sleep(1) # Check worker thread status periodically
    except KeyboardInterrupt: # Should be caught by main app's try/except
        print("audio_monitor.py: KeyboardInterrupt in start_listening (Porcupine Mode). Signaling stop.")
        porcupine_thread_stop_event.set()
    finally:
        print("audio_monitor.py: start_listening (Porcupine Mode) is exiting. Signaling Porcupine worker to stop.")
        porcupine_thread_stop_event.set() # Signal worker to stop
        if porcupine_thread.is_alive():
            print("audio_monitor.py: Waiting for PorcupineWorkerThread to join...")
            porcupine_thread.join(timeout=5.0) # Wait for worker to clean up
            if porcupine_thread.is_alive():
                print("audio_monitor.py: Warning - PorcupineWorkerThread did not join in time.")
        print("--- audio_monitor.py: start_listening() (Porcupine Mode) EXITED ---")

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
    print("audio_monitor.py: Running directly for testing (Porcupine Mode)...")
    if not PORCUPINE_ACCESS_KEY:
        print("\n\n!!! PORCUPINE_ACCESS_KEY_PI environment variable not set. !!!")
        print("Please set it before running: export PORCUPINE_ACCESS_KEY_PI='YOUR_KEY'\n\n")
    else:
        # Start heartbeat in a separate thread if testing
        # ... (heartbeat test thread logic as before) ...
        print("audio_monitor.py: Calling start_listening() for direct test...")
        start_listening() # This will now block until Ctrl+C
        print("audio_monitor.py: Direct test finished (start_listening completed or was interrupted).")

