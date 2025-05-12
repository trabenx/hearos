import firebase_admin
from firebase_admin import credentials, firestore, storage

_db = None
_bucket = None


def init_firebase(cred_path: str, bucket_name: str):
    global _db, _bucket

    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': bucket_name
        })

    _db = firestore.client()
    _bucket = storage.bucket()


def get_firestore():
    if _db is None:
        raise RuntimeError("Firebase not initialized. Call init_firebase() first.")
    return _db


def get_storage_bucket():
    if _bucket is None:
        raise RuntimeError("Firebase not initialized. Call init_firebase() first.")
    return _bucket
