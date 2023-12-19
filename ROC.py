import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
plt.rcParams.update({'font.size': 19})
plt.rcParams.update({'font.family': "Times New Roman"})

# Assuming df is your DataFrame with columns TP, FP, TN, FN
# You can replace df with your actual DataFrame
df1 = pd.read_csv(r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/ROC/shortROC.csv')
df2 = pd.read_csv(r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/ROC/mediumROC.csv')
df3 = pd.read_csv(r'C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/ROC/longROC.csv')

# Calculate true positive rate (Sensitivity) and false positive rate (1 - Specificity)
tpr1 = df1['TP'] / (df1['TP'] + df1['FN'])
fpr1 = df1['FP'] / (df1['FP'] + df1['TN'])


tpr2 = df2['TP'] / (df2['TP'] + df2['FN'])
fpr2 = df2['FP'] / (df2['FP'] + df2['TN'])


tpr3 = df3['TP'] / (df3['TP'] + df3['FN'])
fpr3 = df3['FP'] / (df3['FP'] + df3['TN'])


# Calculate AUC (Area Under the Curve)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)
roc_auc3 = auc(fpr3, tpr3)

# Plot ROC Curve
plt.figure(figsize=(10, 6), dpi = 200)
# plt.scatter([1-0.73333], [0.9926])
# plt.plot([1-0.73333], [0.9926])


plt.plot(fpr1, tpr1, lw=2, label=f'ROC short sentences, AUC = {roc_auc1:.2f}')
plt.plot(fpr2, tpr2, lw=2, label=f'ROC medium sentences, AUC = {roc_auc2:.2f}')
plt.plot(fpr3, tpr3, lw=2, label=f'ROC long sentences, AUC = {roc_auc3:.2f}')
# plt.scatter([1-0.73333], [0.9926], s = 60)
# plt.scatter([1-0.7667], [0.95926], s = 60)
# plt.scatter([1-0.74444], [0.95926], s = 60)

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
# plt.title('ROC Curve')
plt.legend(loc="lower right")

# plt.show()
