import sounddevice as sd
import numpy as np
import time
import librosa
import threading
from collections import deque
import csv
import os # For clearing screen (optional, can be disruptive)
import sys # For sys.stdout.write and flush

# Try to import tflite_runtime, fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Using tensorflow.lite")
    except ImportError:
        print("Neither tflite_runtime nor tensorflow.lite found. Please install one.")
        exit()


import soundfile as sf # For saving to cloud
import io             # For saving to cloud

# --- Constants ---
SAMPLE_RATE = 16000 # YAMNet expects 16kHz
CHUNK_DURATION = 0.1 # seconds - Shorter chunk for faster callback, more responsive buffering
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

CONTEXT_BUFFER_DURATION = 5
CONTEXT_BUFFER_MAX_CHUNKS = int(CONTEXT_BUFFER_DURATION / CHUNK_DURATION)

ANALYSIS_WINDOW_DURATION = 2.0 # Analyze last 2s
ANALYSIS_WINDOW_CHUNKS = int(ANALYSIS_WINDOW_DURATION / CHUNK_DURATION)
ANALYSIS_WINDOW_SAMPLES = int(SAMPLE_RATE * ANALYSIS_WINDOW_DURATION)

MIN_SAMPLES_FOR_LIBROSA = 2048

# --- YAMNet Specific Constants ---
YAMNET_MODEL_PATH = '../models/classification-tflite/1.tflite'
YAMNET_CLASS_MAP_PATH = '../models/classification-tflite/yamnet_class_map.csv'
YAMNET_EXPECTED_WAVEFORM_LENGTH = int(SAMPLE_RATE * 0.975) # YAMNet input length is 15600 samples (0.975s)
YAMNET_AGGRESSION_CLASSES = [
    "Speech",
    "Shout",
    "Bellow",
    "Yell",
    "Children shouting",
    "Screaming",
    "Slam",        # (Sound of a door slamming)
    "Glass",       # (Sound of breaking glass - check if 'Shatter' is better or also needed)
    "Shatter",
    "Slap, smack", # (Sound of a slap)
    "Whack, thwack",
    "Smash, crash",
    "Breaking",    # (Generic breaking sounds)
    "Gunshot, gunfire", # If relevant to your definition of "violent argument" context
    "Explosion",        # If relevant
    "Cacophony",        # General loud, chaotic noise
    # "Alarm", # Could be many types of alarms, might create false positives for arguments
    # "Crying, sobbing", # Can be part of an argument, but also just sadness.
                       # Its presence alongside shouting might be a stronger indicator.
    "Hubbub, speech noise, speech babble",
    "Grunt",
    "Groan",
    "Wail, moan",
    "Whimper",
    "Crying, sobbing",
    "Whispering",
    "Silence"
    
]
# Store the last known scores for these monitored classes
current_monitored_scores = {cls_name: 0.0 for cls_name in YAMNET_AGGRESSION_CLASSES}
# Keep track of the number of lines printed for the table for clearing
TABLE_LINES_PRINTED = 0

# --- Thresholds ---
RMS_THRESHOLD_MODERATE = 0.04 # Example
RMS_THRESHOLD_LOUD = 0.08     # Example
PITCH_VARIANCE_THRESHOLD_MODERATE = 80 # Example
PITCH_VARIANCE_THRESHOLD_HIGH = 150    # Example
LOCAL_AGGRESSION_SCORE_THRESHOLD = 0.7 # Increased due to YAMNet contribution
YAMNET_AGGRESSION_THRESHOLD = 0.15 # Min probability for a class to be considered

# --- Globals ---
audio_buffer_lock = threading.Lock()
live_audio_buffer = deque(maxlen=CONTEXT_BUFFER_MAX_CHUNKS)

cloud_upload_pending = False
cloud_upload_lock = threading.Lock()

yamnet_interpreter = None
yamnet_input_details = None
yamnet_output_details = None
yamnet_class_names = []

# --- Function to print the updating table ---
def print_status_table(librosa_features, yamnet_scores_dict, local_aggression_score):
    global TABLE_LINES_PRINTED, current_monitored_scores

    # ANSI escape codes
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[2K'

    # Move cursor up by the number of lines previously printed for the table
    if TABLE_LINES_PRINTED > 0:
        sys.stdout.write(CURSOR_UP_ONE * TABLE_LINES_PRINTED)

    lines_to_print = []

    # --- Librosa Features ---
    lines_to_print.append(f"{ERASE_LINE}--- Librosa Features ---")
    lines_to_print.append(f"{ERASE_LINE}RMS        : {librosa_features.get('rms', 0):.4f}")
    lines_to_print.append(f"{ERASE_LINE}Pitch Std  : {librosa_features.get('pitch_std', 0):.2f}")
    lines_to_print.append(f"{ERASE_LINE}") # Blank line

    # --- YAMNet Monitored Scores ---
    lines_to_print.append(f"{ERASE_LINE}--- YAMNet Scores (Monitored) ---")
    # Update current_monitored_scores with new scores, keep old if not present in current frame
    for cls_name in YAMNET_AGGRESSION_CLASSES:
        current_monitored_scores[cls_name] = yamnet_scores_dict.get(cls_name, current_monitored_scores[cls_name]) # Keep last if not updated
        # Or, to reset if not detected in current frame:
        # current_monitored_scores[cls_name] = yamnet_scores_dict.get(cls_name, 0.0)

    for cls_name in YAMNET_AGGRESSION_CLASSES:
        score = current_monitored_scores[cls_name]
        lines_to_print.append(f"{ERASE_LINE}{cls_name:<25}: {score:.3f}")
    lines_to_print.append(f"{ERASE_LINE}") # Blank line

    # --- Aggression Score & Status ---
    lines_to_print.append(f"{ERASE_LINE}--- Overall Status ---")
    lines_to_print.append(f"{ERASE_LINE}Local Aggression Score: {local_aggression_score:.2f}")
    if local_aggression_score >= LOCAL_AGGRESSION_SCORE_THRESHOLD:
        lines_to_print.append(f"{ERASE_LINE}Status: POTENTIAL AGGRESSION DETECTED")
    else:
        lines_to_print.append(f"{ERASE_LINE}Status: Monitoring...")

    with cloud_upload_lock: # Access cloud_upload_pending safely
        if cloud_upload_pending:
            lines_to_print.append(f"{ERASE_LINE}Cloud Upload: PENDING...")
        else:
            lines_to_print.append(f"{ERASE_LINE}Cloud Upload: Idle")
    lines_to_print.append(f"{ERASE_LINE}")


    # Print all lines
    for line in lines_to_print:
        sys.stdout.write(line + "\n")

    sys.stdout.flush()
    TABLE_LINES_PRINTED = len(lines_to_print)

def send_to_cloud_dummy(audio_data_full_context):
    global cloud_upload_pending
    print(f"--- DUMMY: Preparing to upload {len(audio_data_full_context)/SAMPLE_RATE:.1f}s of audio...")
    time.sleep(2)
    print("--- DUMMY: Cloud analysis complete. ---")
    with cloud_upload_lock:
        cloud_upload_pending = False


# --- YAMNet Functions ---
def load_yamnet_model():
    global yamnet_interpreter, yamnet_input_details, yamnet_output_details, yamnet_class_names
    try:
        print(f"Loading YAMNet model from: {YAMNET_MODEL_PATH}")
        yamnet_interpreter = tflite.Interpreter(model_path=YAMNET_MODEL_PATH)
        yamnet_interpreter.allocate_tensors()
        yamnet_input_details = yamnet_interpreter.get_input_details()
        yamnet_output_details = yamnet_interpreter.get_output_details()
        print("YAMNet model loaded successfully.")

        # Load class names
        print(f"Loading YAMNet class map from: {YAMNET_CLASS_MAP_PATH}")
        with open(YAMNET_CLASS_MAP_PATH, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            next(reader) # Skip header
            for row in reader:
                # Assuming class map has 'index,mid,display_name'
                yamnet_class_names.append(row[2]) # Or adjust index based on your CSV
        print(f"Loaded {len(yamnet_class_names)} YAMNet class names.")
        if len(yamnet_class_names) != yamnet_output_details[0]['shape'][1]:
             print(f"Warning: Number of class names ({len(yamnet_class_names)}) does not match model output size ({yamnet_output_details[0]['shape'][1]})")

    except Exception as e:
        print(f"Error loading YAMNet model or class map: {e}")
        yamnet_interpreter = None # Ensure it's None if loading failed

def run_yamnet_inference(waveform):
    if yamnet_interpreter is None:
        return {} # Return an empty dict if model not loaded

    # YAMNet expects audio as a 1D float32 numpy array in range [-1.0, 1.0]
    # and a specific length.
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32)

    # Normalize if not already (sounddevice typically gives -1 to 1)
    # if np.max(np.abs(waveform)) > 1.0:
    #     waveform = waveform / np.max(np.abs(waveform))

    # Pad or truncate to YAMNet's expected input size
    if len(waveform) < YAMNET_EXPECTED_WAVEFORM_LENGTH:
        # Pad with zeros
        padding = np.zeros(YAMNET_EXPECTED_WAVEFORM_LENGTH - len(waveform), dtype=np.float32)
        waveform = np.concatenate((waveform, padding))
    elif len(waveform) > YAMNET_EXPECTED_WAVEFORM_LENGTH:
        # Truncate (take the most recent part)
        waveform = waveform[-YAMNET_EXPECTED_WAVEFORM_LENGTH:]

    input_shape = yamnet_input_details[0]['shape']
    if len(input_shape) == 2 and input_shape[0] == 1 and input_shape[1] == YAMNET_EXPECTED_WAVEFORM_LENGTH:
        waveform_for_model = np.expand_dims(waveform, axis=0)
    elif len(input_shape) == 1 and input_shape[0] == YAMNET_EXPECTED_WAVEFORM_LENGTH:
        waveform_for_model = waveform
    else:
        # This print should ideally only happen once if there's a persistent issue
        print(f"Error: YAMNet input shape mismatch. Model expects {input_shape}, prepared {waveform.shape}")
        return {}

    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], waveform_for_model)
    yamnet_interpreter.invoke()
    scores = yamnet_interpreter.get_tensor(yamnet_output_details[0]['index'])[0] # Get first (and only) batch

    # --- DEBUGGING: Print top N YAMNet classes and scores ---
    # print("--- Top YAMNet Scores (Raw Debug) ---")
    # top_n_debug = 15 # Print top 15 raw scores
    # top_indices = np.argsort(scores)[-top_n_debug:][::-1]
    # for i in top_indices:
    #     if i < len(yamnet_class_names): # Safety check
    #         print(f"  Raw YAMNet: {yamnet_class_names[i]} ({scores[i]:.3f})")
    # print("----------------------------------")
    # --- End Debugging ---

    # Instead of returning a list of tuples, return a dictionary of scores for ALL classes
    # This allows the table to pick the ones it cares about.
    # Or, more efficiently, just the scores for monitored classes that exceed a base threshold.
    detected_scores_dict = {}
    for i, score in enumerate(scores):
        if i < len(yamnet_class_names): # Safety check
            class_name = yamnet_class_names[i]
            # We can send all scores for monitored classes, or only if above a display threshold
            if class_name in YAMNET_AGGRESSION_CLASSES: # Only populate for monitored ones for the table
                 detected_scores_dict[class_name] = float(score)
            # Or, if you want to apply aggression threshold here for logic, but table might want raw scores
            # if score > YAMNET_AGGRESSION_THRESHOLD and class_name in YAMNET_AGGRESSION_CLASSES:
            #    detected_scores_dict[class_name] = float(score)


    # --- NO DEBUGGING PRINTS HERE to keep table clean ---
    return detected_scores_dict # Return a dictionary of class_name: score

def extract_features(audio_segment):
    if audio_segment is None or len(audio_segment) < MIN_SAMPLES_FOR_LIBROSA:
        return {'rms': 0, 'pitch_mean': 0, 'pitch_std': 0, 'valid': False}
    try:
        rms = np.mean(librosa.feature.rms(y=audio_segment, frame_length=MIN_SAMPLES_FOR_LIBROSA, hop_length=MIN_SAMPLES_FOR_LIBROSA//2)[0])
        pitches, magnitudes = librosa.piptrack(y=audio_segment, sr=SAMPLE_RATE, n_fft=MIN_SAMPLES_FOR_LIBROSA, hop_length=MIN_SAMPLES_FOR_LIBROSA//2)
        valid_pitches_indices = magnitudes > np.median(magnitudes[magnitudes > 0])
        pitch_values = pitches[valid_pitches_indices]
        pitch_values = pitch_values[pitch_values > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 1 else 0
        return {'rms': rms, 'pitch_mean': pitch_mean, 'pitch_std': pitch_std, 'valid': True}
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return {'rms': 0, 'pitch_mean': 0, 'pitch_std': 0, 'valid': False}


# --- Local Analysis ---
def analyze_locally(audio_segment_for_analysis):
    global current_monitored_scores # To update it for the table display

    # 1. Librosa Feature Extraction
    # -----------------------------
    librosa_features = extract_features(audio_segment_for_analysis) # Returns a dict
    librosa_score_contribution = 0.0
    is_loud_sound = False
    is_agitated_pitch = False

    if librosa_features.get('valid', False):
        current_rms = librosa_features['rms']
        current_pitch_std = librosa_features['pitch_std']

        # Evaluate RMS (Loudness)
        if current_rms > RMS_THRESHOLD_LOUD: # Defined higher up
            librosa_score_contribution += 0.25 # Significant contribution for loud sound
            is_loud_sound = True
        elif current_rms > RMS_THRESHOLD_MODERATE: # Defined higher up
            librosa_score_contribution += 0.10 # Moderate contribution
            is_loud_sound = True # Still considered loud enough for further checks

        # Evaluate Pitch Standard Deviation (Agitation)
        if current_pitch_std > PITCH_VARIANCE_THRESHOLD_HIGH: # Defined higher up
            librosa_score_contribution += 0.25 # Significant contribution for agitated pitch
            is_agitated_pitch = True
        elif current_pitch_std > PITCH_VARIANCE_THRESHOLD_MODERATE: # Defined higher up
            librosa_score_contribution += 0.10 # Moderate contribution
            is_agitated_pitch = True
    # else:
        # Librosa features not valid, librosa_score_contribution remains 0

    # 2. YAMNet Inference
    # --------------------
    yamnet_input_segment = audio_segment_for_analysis[-YAMNET_EXPECTED_WAVEFORM_LENGTH:]
    # Gets a dictionary of scores for (at least) YAMNET_AGGRESSION_CLASSES
    current_yamnet_scores_for_monitored = run_yamnet_inference(yamnet_input_segment)

    yamnet_score_contribution = 0.0
    yamnet_confirms_vocal_aggression = False
    yamnet_detects_speech_confidently = False
    yamnet_detects_strong_non_vocal_indicator = False

    if current_yamnet_scores_for_monitored: # If YAMNet produced some scores
        # Check for confident "Speech" first (for context)
        if "Speech" in current_yamnet_scores_for_monitored and \
           current_yamnet_scores_for_monitored["Speech"] > 0.5: # High confidence in speech
            yamnet_detects_speech_confidently = True

        # Iterate through YAMNet detections to find aggression indicators
        for class_name, score in current_yamnet_scores_for_monitored.items():
            if class_name not in YAMNET_AGGRESSION_CLASSES or score < YAMNET_AGGRESSION_THRESHOLD:
                # Skip if not an aggression class we care about for logic, or score too low
                continue

            # Class is in YAMNET_AGGRESSION_CLASSES and score is above YAMNET_AGGRESSION_THRESHOLD
            if class_name in ["Shout", "Yell", "Screaming"]:
                yamnet_score_contribution += score * 0.6 # Strong weight for these direct vocal aggressors
                yamnet_confirms_vocal_aggression = True
            elif class_name in ["Slap, smack", "Shatter", "Smash, crash", "Breaking", "Gunshot, gunfire", "Explosion"]:
                yamnet_score_contribution += score * 0.7 # Very strong weight for these physical indicators
                yamnet_detects_strong_non_vocal_indicator = True
            elif class_name == "Speech" and is_loud_sound and is_agitated_pitch:
                # If Librosa already suggests arousal AND YAMNet confirms "Speech" confidently
                # (even if it didn't classify as "Shout" etc.)
                # this adds a bit more confidence.
                # The YAMNET_AGGRESSION_THRESHOLD for "Speech" might be higher in this context.
                if score > 0.6: # Higher confidence for Speech to contribute here
                    yamnet_score_contribution += score * 0.1 # Small contribution if Librosa already high
            elif class_name == "Cacophony" and is_loud_sound: # General loud chaos
                 yamnet_score_contribution += score * 0.2


    # 3. Combine Scores and Apply Logic
    # ---------------------------------
    # Cap individual contributions if necessary (already done by score multiplication)
    # Cap YAMNet's total contribution to avoid it dominating too much if Librosa is low
    yamnet_score_contribution = min(yamnet_score_contribution, 0.5) # Max 0.5 from YAMNet

    # Combine Librosa and YAMNet contributions
    local_score = librosa_score_contribution + yamnet_score_contribution

    # --- Additional Heuristics/Bonuses ---
    # Bonus if Librosa features are high AND YAMNet confirms vocal aggression or strong non-vocal
    if is_loud_sound and is_agitated_pitch and (yamnet_confirms_vocal_aggression or yamnet_detects_strong_non_vocal_indicator):
        local_score += 0.15 # Synergy bonus
    # Bonus if strong non-vocal indicator from YAMNet, even if Librosa vocal features are not extreme
    elif yamnet_detects_strong_non_vocal_indicator and is_loud_sound: # e.g. loud smash
        local_score += 0.1

    # Ensure score doesn't exceed 1.0 (or your defined max)
    local_score = min(local_score, 1.0)


    # 4. Update Display Table
    # -----------------------
    # Pass the Librosa features dict and the YAMNet scores for monitored classes
    print_status_table(librosa_features, current_yamnet_scores_for_monitored, local_score)


    # 5. Decision
    # -----------
    if local_score >= LOCAL_AGGRESSION_SCORE_THRESHOLD:
        return True
    return False
# --- Audio Callback (Keep this VERY LIGHTWEIGHT) ---
def audio_callback(indata, frames, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        # This will print "input overflow" if processing_thread is too slow
        # OR if this callback itself becomes slow.
        print(status, flush=True)

    with audio_buffer_lock:
        live_audio_buffer.append(indata[:, 0].copy()) # Assuming mono, take first channel

# --- Dedicated Processing Thread ---
def processing_thread_function():
    global cloud_upload_pending
    print("Processing thread started.")
    while True:
        time.sleep(CHUNK_DURATION) # Process at roughly the rate new chunks arrive

        current_buffer_list = []
        with audio_buffer_lock:
            if len(live_audio_buffer) > 0:
                # Create a snapshot for processing
                current_buffer_list = list(live_audio_buffer)

        if not current_buffer_list:
            continue # Nothing to process yet

        # Prepare the segment for local analysis (latest ANALYSIS_WINDOW_DURATION)
        # Ensure we have enough chunks for the analysis window
        if len(current_buffer_list) >= ANALYSIS_WINDOW_CHUNKS:
            # Get the last ANALYSIS_WINDOW_CHUNKS
            analysis_chunks = current_buffer_list[-ANALYSIS_WINDOW_CHUNKS:]
            audio_segment_for_analysis = np.concatenate(analysis_chunks)

            # Ensure the concatenated segment is long enough
            if len(audio_segment_for_analysis) >= ANALYSIS_WINDOW_SAMPLES:
                potential_aggression = analyze_locally(audio_segment_for_analysis)

                with cloud_upload_lock:
                    if potential_aggression and not cloud_upload_pending:
                        print("--- Triggering Cloud Analysis ---")
                        cloud_upload_pending = True
                        # Get the full context buffer for cloud upload
                        full_context_audio = np.concatenate(current_buffer_list) # Send all we have
                        # Start cloud upload in a separate thread
                        # Replace send_to_cloud_dummy with your actual cloud function later
                        upload_thread = threading.Thread(target=send_to_cloud_dummy, args=(full_context_audio,))
                        upload_thread.daemon = True # Allow main program to exit even if thread is running
                        upload_thread.start()
            # else:
            #     print(f"Debug: Not enough samples for analysis window: {len(audio_segment_for_analysis)}/{ANALYSIS_WINDOW_SAMPLES}")
        # else:
        #     print(f"Debug: Not enough chunks in buffer for analysis: {len(current_buffer_list)}/{ANALYSIS_WINDOW_CHUNKS}")


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting audio processing system...")

    # Load YAMNet model at startup
    load_yamnet_model()
    if yamnet_interpreter is None:
        print("Failed to load YAMNet. YAMNet features will be disabled.")
        # You might choose to exit or continue without YAMNet
    # Initial clear screen and print empty table structure once (optional)
    # os.system('cls' if os.name == 'nt' else 'clear') # Can be jarring
    # print_status_table({}, {}, 0.0) # Print an empty table structure initially

    proc_thread = threading.Thread(target=processing_thread_function)
    proc_thread.daemon = True # So it exits when the main thread exits
    proc_thread.start()

    try:
        # Keep CHUNK_SAMPLES relatively small for sd.InputStream blocksize
        # to ensure audio_callback returns quickly.
        with sd.InputStream(callback=audio_callback,
                            samplerate=SAMPLE_RATE, # Ensure this matches YAMNet's expectation
                            channels=1, # Request mono
                            dtype='float32',
                            blocksize=CHUNK_SAMPLES):
            # print(f"Audio stream started...") # One-time print is OK
            # Print a blank header for the table area so subsequent updates don't jump
            initial_lines = len(YAMNET_AGGRESSION_CLASSES) + 10 # Estimate lines
            for _ in range(initial_lines):
                 print("") # Print blank lines to reserve space
            TABLE_LINES_PRINTED = initial_lines


            while True:
                time.sleep(10) # Keep main thread alive, print a heartbeat or status
                # print("Main thread alive...")

    except KeyboardInterrupt:
        # Before exiting, move cursor below the table to print final messages cleanly
        sys.stdout.write("\n" * (TABLE_LINES_PRINTED + 1))
        print("\nStopping system.")
    except Exception as e:
        sys.stdout.write("\n" * (TABLE_LINES_PRINTED + 1))
        print(f"An error occurred in main: {e}")
        import traceback
        traceback.print_exc()