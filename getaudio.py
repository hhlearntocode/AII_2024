import pyaudio
import wave
from pydub import AudioSegment

def record_audio(filename, duration=5, sample_rate=44100, channels=2, chunk=1024):
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    print("Recording...")
    
    frames = []
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Save as WAV first
    with wave.open(filename + ".wav", 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    
    # Convert WAV to MP3
    audio = AudioSegment.from_wav(filename + ".wav")
    audio.export(filename + ".mp3", format="mp3")
    
    print(f"Audio saved as {filename}.mp3")

# Usage
record_audio("my_recording", duration=10)