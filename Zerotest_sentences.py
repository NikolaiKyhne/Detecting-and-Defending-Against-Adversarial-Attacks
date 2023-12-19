
import argparse
import os
import pandas as pd
import time
from torchmetrics import CharErrorRate

data = r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/adversarial_dataset-A/Normal-Examples/long-signals/list-long.csv'
# Read the lines from the file
with open(data, 'r') as file:
    lines = file.readlines()

# Process each line and split it into columns
data = [line.strip().split(',') for line in lines]
cer = CharErrorRate()
# Create a DataFrame from the processed data
df = pd.DataFrame(data)

# Select the first two columns
df = df.iloc[:, :2]
df.columns = ["File Name", "Sentences"]
files, sentences = df.iloc[:, 0], df.iloc[:, 1]

attacked = []
original = []
transcription = "i need a reservation for sixteen people at the seafood restaurant down the street"

for n in range(len(sentences)):
    if len(transcription) <= len(sentences[n]):
        original.append(transcription)
        attacked.append(sentences[n])
    else:
        original.append(sentences[n])
        attacked.append(transcription)

print(1 - cer(original, attacked))