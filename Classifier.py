import deepspeech
import numpy as np
import wave
import warnings
import time
import io
import torch
import torchaudio
import os

class DeepSpeechTranscriber:
    def __init__(self, model_path, scorer_path):
        self.model = deepspeech.Model(model_path)
        self.model.enableExternalScorer(scorer_path)

    def transcribe_audio_tensor(self, audio_file):
        # Specify the file path to your WAV file
        return self.model.stt(audio_file.to(torch.int16).detach().cpu().numpy())
    
    def transcribe_audio_file(self, audio_file):
       buffer, rate = self._read_wav_file(audio_file)
       data16 = np.frombuffer(buffer, dtype=np.int16)
       return self.model.stt(data16)
   
    @staticmethod
    def _read_wav_file(filename):
        with wave.open(filename, 'rb') as w:
            rate = w.getframerate()
            frames = w.getnframes()
            buffer = w.readframes(frames)
        return buffer, rate

# Example
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    t0 = time.time()

    model_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.pbmm'
    scorer_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.scorer'
    transcriber = DeepSpeechTranscriber(model_path, scorer_path)
    to_pop = "yes"
    words = ["down", "go", "left", "no", "off", "on", "right", "stop", "up", "yes"]
    words.remove(to_pop)
    # Assuming you have audio data as a tensor and sample rate (modify this part accordingly)
    # audio_file = torch.tensor([0.1, 0.2, 0.3, 0.4])  # Replace with your actual audio data tensor
    folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\output_10_500_20\data" + "//" + to_pop
    fdir = os.listdir(folder) # All files in current directory
    
    # All files
    fs = [f for f in fdir if f.endswith('wav')] # Find soundfiles
    
    # Sort
    fs = np.sort(fs) # Numpy uses a faster sorting algorithm
    n = 0
   
    for i in fs:
        audio_file = folder + "\\" + i
        output_file_path = r"C:\Users\kyhne\Downloads\output_audio.wav"
    
        transcription = transcriber.transcribe_audio_file(audio_file)
        if transcription == to_pop:
            print(f'{transcription}: Correct')
            n += 1
        else:
            print(f'{transcription}: Wrong')    
            
    print(n/20)

    print(time.time() - t0)
