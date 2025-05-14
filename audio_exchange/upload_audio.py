from config import CREDENTIALS_PATH, BUCKET_NAME, COLLECTION_NAME
from firebase_admin import firestore
from audio_exchange.firebase_utils import init_firebase, get_firestore, get_storage_bucket
from audio_exchange.audio_utils import compress_wav


def upload_compressed_audio(local_wav_path: str, audio_id: str):
    init_firebase(CREDENTIALS_PATH, BUCKET_NAME)
    db = get_firestore()
    bucket = get_storage_bucket()

    compressed_data = compress_wav(local_wav_path)
    storage_path = f"audio/{audio_id}.wav.gz"

    blob = bucket.blob(storage_path)
    blob.upload_from_string(compressed_data, content_type='application/gzip')

    doc_ref = db.collection(COLLECTION_NAME).document(audio_id)
    doc_ref.set({
        'id': audio_id,
        'storage_path': storage_path,
        'created_at': firestore.SERVER_TIMESTAMP,
        'classification': None
    })

    print(f"Audio '{audio_id}' uploaded successfully.")


if __name__ == "__main__":
    wav_file = "output.wav"
    clip_id = "clip001"
    upload_compressed_audio(wav_file, clip_id)
