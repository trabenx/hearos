import gzip
import io
import pyaudio
import wave


def compress_wav(input_wav_path: str) -> bytes:
    with open(input_wav_path, 'rb') as f:
        raw_data = f.read()
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
        gz_file.write(raw_data)
    return buffer.getvalue()


def decompress_gzip_to_wav(gz_bytes: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(gz_bytes)) as gz_file:
        return gz_file.read()


def play_audio(wav_bytes: bytes):
    try:
        with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            print(f"🔊 Playing: {wf.getframerate()}Hz, {wf.getnchannels()}ch, {wf.getsampwidth()*8}bit")
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()
    except Exception as e:
        print(f"❌ Audio playback error: {e}")
