from firebase_utils import init_firebase, get_firestore, get_storage_bucket
from audio_utils import decompress_gzip_to_wav
from config import CREDENTIALS_PATH, BUCKET_NAME, COLLECTION_NAME


class AudioClip:
    def __init__(self, doc_id, storage_path, created_at, classification=None):
        self.doc_id = doc_id
        self.storage_path = storage_path
        self.created_at = created_at
        self.classification = classification

    def __str__(self):
        return f"{self.doc_id} ({self.created_at})"


class AudioModel:
    def __init__(self):
        try:
            init_firebase(CREDENTIALS_PATH, BUCKET_NAME)
            self.db = get_firestore()
            self.bucket = get_storage_bucket()
        except Exception as e:
            print(f"❌ Fail to initialize Firebase: {e}")
            raise

    def listen_for_unclassified(self, callback):
        def on_snapshot(col_snapshot, changes, read_time):
            print(f"[DEBUG] Snapshot triggered: {len(col_snapshot)} docs")
            clips = []
            for doc in col_snapshot:
                data = doc.to_dict()
                classification = data.get("classification")
                if classification in ["violent", "non-violent"]:
                    continue
                clip = AudioClip(
                    doc_id=doc.id,
                    storage_path=data.get("storage_path"),
                    created_at=data.get("created_at"),
                    classification=classification
                )
                clips.append(clip)
            callback(clips)

        try:
            self.db.collection(COLLECTION_NAME).order_by("created_at").on_snapshot(on_snapshot)
        except Exception as e:
            print(f"Error setting up snapshot listener: {e}")

    def download_clip_audio(self, clip: AudioClip) -> bytes:
        try:
            blob = self.bucket.blob(clip.storage_path)
            compressed_data = blob.download_as_bytes()
            wav_data = decompress_gzip_to_wav(compressed_data)
            return wav_data
        except Exception as e:
            print(f"❌ Failed retrieving or decompressing clip: {clip.doc_id}: {e}")
            return b''

    def classify_clip(self, clip: AudioClip, label: str):
        if label not in ["violent", "non-violent"]:
            print("⚠️ Illegal classification")
            return
        try:
            doc_ref = self.db.collection(COLLECTION_NAME).document(clip.doc_id)
            doc_ref.update({"classification": label})
            print(f"✅ Clip {clip.doc_id} classified as {label}")
        except Exception as e:
            print(f"❌ Failed updating clip classification {clip.doc_id}: {e}")
