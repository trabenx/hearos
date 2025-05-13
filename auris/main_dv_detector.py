import threading
import time
import subprocess
import os
import sys
# Force stdout and stderr to be line-buffered (flush on newline)
# Note: This might not be necessary for all Python versions or environments
# but can help with journalctl visibility for services.
# Python 3.7+ allows unbuffered mode via -u or PYTHONUNBUFFERED=1
# For now, let's try explicit flushing.
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1) # Line buffering
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1) # Line buffering
# A simpler way that often works is just to flush after important prints:

import traceback

# --- Attempt to import local modules ---
# This block helps catch import errors early if files are missing or there are circular dependencies
try:
    print("main_dv_detector.py: Attempting to import photo_frame...")
    from photo_frame import show_slideshow
    print("main_dv_detector.py: Imported photo_frame successfully.")

    print("main_dv_detector.py: Attempting to import audio_monitor...")
    from audio_monitor import (
        start_listening,
        heartbeat_loop,
        RPI_UNIQUE_ID, # Comes from audio_monitor
        SERVER_AUDIO_ENDPOINT, # Comes from audio_monitor
        SERVER_HEARTBEAT_ENDPOINT # Comes from audio_monitor
    )
    print("main_dv_detector.py: Imported audio_monitor successfully.")
except ImportError as e:
    print(f"main_dv_detector.py: CRITICAL ImportError during initial module loading: {e}")
    print("Ensure photo_frame.py and audio_monitor.py are in the same directory and have no internal import errors.")
    traceback.print_exc()
    sys.exit(1) # Exit if essential modules can't be imported
except Exception as e:
    print(f"main_dv_detector.py: CRITICAL Exception during initial module loading: {e}")
    traceback.print_exc()
    sys.exit(1)

# --- Configuration (mostly from imported audio_monitor) ---
PICTURES_DIR = os.path.expanduser("~/pictures")

# --- Helper to kill display process ---
def kill_display_process():
    try:
        print("main_dv_detector.py: kill_display_process() called - Attempting to stop fbi slideshow...")
        # Using pkill to find fbi by name. -f matches against full command line.
        subprocess.run("sudo pkill -f /usr/bin/fbi", shell=True, check=False)
        print("main_dv_detector.py: pkill fbi command executed.")
    except Exception as e:
        print(f"main_dv_detector.py: Error in kill_display_process: {e}")

# --- Main Application Logic ---
if __name__ == "__main__":
    # Initialize thread variables to None outside the try block
    # so they are defined for the finally block even if startup fails
    photo_thread_for_check = None
    audio_thread_for_check = None
    heartbeat_thread_for_check = None
    photo_thread_has_finished_log_printed = False # Flag for logging

    # This outer try/except/finally is for the whole application lifecycle
    try:
        print("--- main_dv_detector.py: Domestic Violence Detector - RPi Initializing ---")
        print(f"main_dv_detector.py: Device ID: {RPI_UNIQUE_ID}")
        print(f"main_dv_detector.py: Server Audio Endpoint: {SERVER_AUDIO_ENDPOINT}")
        print(f"main_dv_detector.py: Server Heartbeat Endpoint: {SERVER_HEARTBEAT_ENDPOINT}")
        print(f"main_dv_detector.py: Picture directory: {PICTURES_DIR}")
        sys.stdout.flush() # Add this


        if not os.path.isdir(PICTURES_DIR):
            print(f"main_dv_detector.py: Warning - Pictures directory {PICTURES_DIR} not found. Creating it.")
            try:
                os.makedirs(PICTURES_DIR, exist_ok=True)
            except Exception as e_mkdir:
                print(f"main_dv_detector.py: Error creating PICTURES_DIR: {e_mkdir}")
                # Decide if this is fatal, for now, continue

        all_threads = []

        # 1. Start the photo frame slideshow (Simplified version for now)
        print("main_dv_detector.py: Starting PhotoFrameThread...")
        photo_thread = threading.Thread(target=show_slideshow, name="PhotoFrameThread", daemon=True)
        photo_thread.start()
        photo_thread_for_check = photo_thread # Assign for checking

        # 2. Start the audio listener
        print("main_dv_detector.py: Starting AudioMonitorThread...")
        audio_thread = threading.Thread(target=start_listening, name="AudioMonitorThread", daemon=True)
        audio_thread.start()
        audio_thread_for_check = audio_thread # Assign for checking

        # 3. Start the heartbeat mechanism
        if SERVER_HEARTBEAT_ENDPOINT != "YOUR_SERVER_HEARTBEAT_ENDPOINT_HERE/heartbeat":
            print("main_dv_detector.py: Starting HeartbeatThread...")
            heartbeat_thread_obj = threading.Thread(target=heartbeat_loop, name="HeartbeatThread", daemon=True)
            heartbeat_thread_obj.start()
            heartbeat_thread_for_check = heartbeat_thread_obj # Assign for checking
        else:
            print("main_dv_detector.py: Heartbeat endpoint not configured, skipping HeartbeatThread.")

        print("main_dv_detector.py: Giving threads a moment to initialize (2 seconds)...")
        time.sleep(2)

        # --- Main Monitoring Loop ---
        print("main_dv_detector.py: Entering main monitoring loop...")
        sys.stdout.flush() # Add this
        while True:
            # Check audio thread (most critical)
            if audio_thread_for_check and not audio_thread_for_check.is_alive():
                print("main_dv_detector.py: CRITICAL - AudioMonitorThread has died. Exiting main loop.")
                break # Exit the while loop

            # Check heartbeat thread if it was started
            if heartbeat_thread_for_check and not heartbeat_thread_for_check.is_alive():
                print("main_dv_detector.py: Warning - HeartbeatThread has died. Audio monitoring continues.")
                heartbeat_thread_for_check = None # Stop re-checking

            # Check PhotoFrameThread (simplified sleep version)
            if photo_thread_for_check and not photo_thread_for_check.is_alive():
                if not photo_thread_has_finished_log_printed:
                    print("main_dv_detector.py: Info - PhotoFrameThread (simplified sleep version) has finished/died as expected.")
                    photo_thread_has_finished_log_printed = True
                # The thread is expected to die after its sleep, so we don't break the loop.
                # We can set photo_thread_for_check = None to stop checking it, or just let the flag handle it.
                # photo_thread_for_check = None # Optional: stop checking it further
            
            # If only essential threads remain to check and one has died, this loop might exit if not careful.
            # The primary condition for continuing is audio_thread_for_check.is_alive().

            time.sleep(10) # Check thread health

    except KeyboardInterrupt:
        print("\nmain_dv_detector.py: User initiated shutdown (Ctrl+C).")
    except Exception as e:
        print("\nmain_dv_detector.py: CRITICAL UNHANDLED EXCEPTION IN MAIN SCRIPT ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        traceback.print_exc()
        sys.exit(1) # Exit with error code to signal failure to systemd
    finally:
        print("\nmain_dv_detector.py: Script terminating, performing cleanup...")
        kill_display_process()
        # Wait for daemon threads to potentially finish their current task if not abrupt
        # This is a bit conceptual as daemon threads usually exit with main
        # print("main_dv_detector.py: Allowing daemon threads a moment to close...")
        # time.sleep(1)
        print("main_dv_detector.py: Application finished.")
