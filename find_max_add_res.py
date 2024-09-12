# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:07:34 2023

@author: kyhne
"""

import pandas as pd
import numpy as np

df = pd.read_csv(r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\adv_detection\5step\long\threshold.csv')

print(np.max(df["accuracy"]))