import whisper

def transcribe(audio_file_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file_path)
    return result['text']
