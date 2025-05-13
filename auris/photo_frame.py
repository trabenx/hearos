import os
import time
import subprocess # Keep for kill_fbi, even if show_slideshow is simple
import traceback
import glob # Not needed for simplified version
import shlex # Not needed for simplified version

IMAGE_DIR = os.path.expanduser("~/pictures")
DISPLAY_DURATION = 10 # seconds per image

def kill_fbi():
    """Helper function to kill any running fbi processes."""
    print("photo_frame.py: kill_fbi() called")
    try:
        subprocess.run("sudo pkill -f /usr/bin/fbi", shell=True, check=False)
    except Exception as e:
        print(f"photo_frame.py: Error killing fbi: {e}")

def show_slideshow():
    """Starts the fbi slideshow in the background using nohup."""
    print("--- photo_frame.py: show_slideshow() ENTERED ---") # Added log
    # Kill any previous instance first
    kill_fbi()
    time.sleep(1) # Brief pause after killing

    if not os.path.isdir(IMAGE_DIR):
        print(f"photo_frame.py: Error - Image directory {IMAGE_DIR} not found.")
        return

    # Use glob to find image files (fbi often works better with explicit file list)
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.JPG', '*.JPEG', '*.PNG', '*.GIF']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, pattern)))
    # Add recursive search if needed:
    # for pattern in image_patterns:
    #     image_files.extend(glob.glob(os.path.join(IMAGE_DIR, '**', pattern), recursive=True))

    if not image_files:
        print(f"photo_frame.py: No valid image files found in {IMAGE_DIR}.")
        return

    # --- Command string for fbi ---
    # We will pass this whole string to "nohup ... &" via shell
    fbi_base_cmd = [
        '/usr/bin/fbi',
        '--noverbose',
        '-a',
        '-t', str(DISPLAY_DURATION),
        '-T', '1',
        '--random',
    ] + image_files

    # Construct the full shell command with sudo, nohup, and backgrounding (&)
    # Need to escape arguments properly for the shell
    # Using shlex.join is safer if available (Python 3.8+)
    # Manual quoting for wider compatibility:
    quoted_fbi_args = " ".join(shlex.quote(arg) for arg in fbi_base_cmd)
    full_shell_cmd = f"nohup sudo {quoted_fbi_args} > /dev/null 2>&1 &"

    print(f"photo_frame.py: Executing shell command: {full_shell_cmd}") # This is the key line

    try:
        # Use Popen with shell=True to execute the full nohup command
        # We don't wait for this process - nohup handles backgrounding
        process = subprocess.Popen(full_shell_cmd, shell=True,
                                 stdout=subprocess.DEVNULL, # Redirect stdout/stderr
                                 stderr=subprocess.DEVNULL)
        print(f"photo_frame.py: fbi process launched in background via nohup (PID likely different from Popen object: {process.pid}).")
        # Since we launched in background, this function now finishes quickly.
        # The thread running this function WILL exit, which is *expected* now.
        # The main loop will need to know not to warn about this specific thread dying.

    except FileNotFoundError:
        print("photo_frame.py: CRITICAL FileNotFoundError (nohup/sudo/fbi missing?)")
        traceback.print_exc()
    except Exception as e:
        print(f"photo_frame.py: An unexpected Python error occurred while trying to launch fbi via nohup: {e}")
        traceback.print_exc()

    print("photo_frame.py: show_slideshow function finished (fbi launched in background).")


if __name__ == "__main__":
    # For direct testing of this simplified version
    print("photo_frame.py: Running simplified show_slideshow directly for testing...")
    show_slideshow()
    print("photo_frame.py: Direct test of simplified show_slideshow finished.")
