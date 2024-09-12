# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 00:02:11 2023

@author: kyhne
"""

from Classifier import DeepSpeechTranscriber
import time
from torchmetrics import CharErrorRate
import pandas as pd
import re
import os

t0 = time.time()


class Experiment():
    
    def __init__(self):
        self.model_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.pbmm'
        self.scorer_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.scorer'
    
    def dataloader(self, data):
        """
        Load data

        Parameters
        ----------
        data : String
            Folder with csv. file with correct strings and filenames.

        Returns
        -------
        None.

        """
        self.data = data
        # Read the lines from the file
        with open(data, 'r') as file:
            lines = file.readlines()

        # Process each line and split it into columns
        data = [line.strip().split(',') for line in lines]

        # Create a DataFrame from the processed data
        df = pd.DataFrame(data)

        # Select the first two columns
        df = df.iloc[:, :2]
        df.columns = ["File Name", "Sentences"]
        self.files, self.sentences = df.iloc[:, 0], df.iloc[:, 1]

    # @torch.no_grad()
    def run(self):

            # Classifier:
        
        transcriber = DeepSpeechTranscriber(self.model_path, self.scorer_path)
        
       
        cer = CharErrorRate()
        
        # With diffusion:
        folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Diffusion_Output"
        
        
        # Without diffusion:
        
        # Short sentences:
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_10_5000"
        
        # Medium sentences:
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_10_5000_medium_internet"
        
        # Long sentences:
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_10_5000_long"

        fdir = os.listdir(folder) # All files in current directory
            
        # All files
        fs = [f for f in fdir if f.endswith('wav')] # Find soundfiles
        # Sort
        def extract_number(s):
            return int(re.search(r'\d+', s).group())
         
        fs = sorted(fs, key=extract_number)
        
        original = []
        attacked = []
        n = 0
        target_num = 0
        for i in fs:
            audio_file = folder + "\\" + i   
            transcription = transcriber.transcribe_audio_file(audio_file)
            
            if transcription == self.sentences[n]:
                print(f'{transcription}: Correct')
            # elif transcription == "open all doors": #short target
            elif transcription == "switch off internet connection": #medium target
            # elif transcription == "i need a reservation for sixteen people at the seafood restaurant down the street": #long target
                # print(f'{transcription}: Wrong')
                target_num += 1
            else:
                print(f'{transcription}: Wrong')   
                
            if len(transcription) <= len(self.sentences[n]):
                original.append(transcription)
                attacked.append(self.sentences[n])
            else:
                original.append(self.sentences[n])  
                attacked.append(transcription)

            n += 1
            print(n)
        return 1 - cer(original, attacked), target_num / len(fs)
    
exp = Experiment()

exp.dataloader(r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/adversarial_dataset-A/Normal-Examples/medium-signals/list-medium.csv')

res = exp.run()

print(res)
print(time.time() - t0)
