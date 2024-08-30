import whisper

# Load whisper model
model = whisper.load_model("large")

# Load audio file
audio_file = r'data\records\output.wav'
audio = whisper.load_audio(audio_file)
audio = whisper.pad_or_trim(audio) # 30 seconds length

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# save transcripte
with open(r"data\transcripts\transcript_large.txt", "w") as file:
    file.write(result.text)


print("Transcription completed and saved to data\transcripts\transcript_large.txt")