# hearos
main repo for the hearos proejct to use DL to detect domestic violence


## 🔊 Audio Classification Interface – Usage Instructions

simple GUI for manually classifying audio clips as **"violent"** or **"non-violent"**, using Firebase Firestore and Storage.

---

### 📁 Prerequisites

- Make sure Firebase credentials are available at:
  `audio_exchange/firebase_credentials.json`
- Firestore and Storage must be enabled in your Firebase project.
- Required Firestore collection: `audio_clips`

---

### ▶️ How to Use

1. Place a WAV file named `output.wav` inside the `audio_exchange/` directory.

2. In `upload_audio.py`, set a unique ID for each clip to avoid overwriting existing data:

   ```python
   clip_id = "clip001"  # Change to a unique value each time
3. Open a terminal and run the controller (GUI):
    ```python
    python audio_controller.py
4. In a separate terminal, upload the audio file:
    ```python
    python upload_audio.py

### ✅ Notes
* Only un-classified clips will be shown
* Once classified, the Firestore document is updated with a classification field
* The interface listens to Firestore changes in real time – no manual refresh is needed