import os
from playsound import playsound
from gtts import gTTS
import tempfile

VOICE_DIR = os.path.join("backend", "voice", "voice_feedback_clips")

def speak(tag_or_message):
    # Check if it's a saved audio tag
    mp3_file = os.path.join(VOICE_DIR, f"{tag_or_message}.mp3")

    if os.path.exists(mp3_file):
        try:
            playsound(mp3_file)
        except Exception as e:
            print(f"[Voice Error] Could not play pre-saved sound: {e}")
    else:
        try:
            # Fallback: dynamically generate TTS
            tts = gTTS(text=tag_or_message, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                playsound(fp.name)
                os.remove(fp.name)
        except Exception as e:
            print(f"[Voice Error] Failed to generate TTS: {e}")
