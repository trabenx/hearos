import threading
from audio_model import AudioModel
from audio_view import AudioView
from audio_utils import play_audio


class AudioController:
    def __init__(self):
        self.model = AudioModel()
        self.view = AudioView(self)
        self.clips = []
        self.model.listen_for_unclassified(self.on_clips_updated)

    def run(self):
        self.view.start()

    def on_clips_updated(self, updated_clips):
        print(f"[DEBUG] Updating clips: {len(updated_clips)} items")
        self.clips = updated_clips
        self.view.populate_list(self.clips)
        self.view.set_status(f"{len(self.clips)} unclassified clips available")

    def get_selected_clip(self):
        index = self.view.get_selected_index()
        if index is None:
            self.view.show_error("שגיאה", "לא נבחר קטע אודיו.")
            return None
        return self.clips[index]

    def play_selected_clip(self):
        clip = self.get_selected_clip()
        if not clip:
            return
        self.view.set_status(f"Playing {clip.doc_id}...")

        def playback():
            wav_data = self.model.download_clip_audio(clip)
            if not wav_data:
                self.view.show_error("שגיאה", "שגיאה בנגינה.")
                return
            play_audio(wav_data)
        threading.Thread(target=playback, daemon=True).start()

    def classify_selected_clip(self, label):
        clip = self.get_selected_clip()
        if not clip:
            return
        self.model.classify_clip(clip, label)
        self.view.show_message("עודכן", f"הקטע '{clip.doc_id}' סווג כ־{'אלים' if label == 'violent' else 'לא אלים'}")
        self.load_clips()  # refresh the list


if __name__ == "__main__":
    controller = AudioController()
    controller.run()
