import torch
import pandas as pd
from torchmetrics import CharErrorRate, WordErrorRate
import time
import numpy as np

t0 = time.time()

class Experiment():
    
    def __init__(self, folder):
        self.folder = folder
        
    
    def dataloader(self):
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
        data = self.folder
        df = pd.read_csv(data)
        df = df.fillna("")
        self.clean, self.diffusion = df["Clean"], df["Diffusion"]

    @torch.no_grad()
    def run(self):
        clean_list = self.clean
        diffusion_list = self.diffusion
        cer = CharErrorRate()
        # i = 0
        i_list = []
        accuracy_list = []
        precision_list = []
        sensitivity_recall_list = []
        specificity_list = []
        TN_list = []
        FP_list = []
        FN_list = []
        TP_list = []
        
        liste = np.linspace(0,1,101) - 0.00000001 # due to numeric errors
        for i in liste:
            TN = 0
            TP = 0
            FN = 0
            FP = 0
            for n in range(270): #First 30 only (testing) 260 for training
                # print(f'Clean: {clean_list[n]}')
                # print(f'Diffusion: {diffusion_list[n]}')
                if len(clean_list[n]) <= len(diffusion_list[n]):
                    CER = cer(clean_list[n], diffusion_list[n])
                else:
                    CER = cer(diffusion_list[n], clean_list[n])     
                
                if CER <= i:
                    TN += 1
                else:
                    FP += 1
                # print(n)
                # print(f'{n}, {CER}\n')
                # print(i)
                
            
            for n in range(270,540):
                # print(f'Clean: {clean_list[n]}')
                # print(f'Diffusion: {diffusion_list[n]}')
                if len(clean_list[n]) <= len(diffusion_list[n]):
                    CER = cer(clean_list[n], diffusion_list[n])
                else:
                    CER = cer(diffusion_list[n], clean_list[n])
                
                if CER <= i:
                    FN += 1
                else:
                    TP += 1
                n += 1
                # print(n)
                # print(f'{n}, {CER}\n')
                # print(i)
            
           
            # https://scaryscientist.blogspot.com/2016/03/confusion-matrix.html
            accuracy = (TN + TP) / (TN + FP + FN + TP)
            try:
                precision =  TP / (FP + TP)
            except ZeroDivisionError:
                precision = 1            
            sensitivity_recall = TP / (TP + FN)
            specificity = TN / (TN + FP)
            
            print(f'TP = {TP}')
            print(f'TN = {TN}')
            print(f'FP = {FP}')
            print(f'FN = {FN}')
    
            
            print(f'accuracy = {accuracy}')
            print(f'precision = {precision}')
            print(f'sensitivity_recall = {sensitivity_recall}')
            print(f'specificity = {specificity}')
            accuracy_list.append(accuracy)
            precision_list.append(accuracy)
            sensitivity_recall_list.append(sensitivity_recall)
            specificity_list.append(specificity)
            TN_list.append(TN)
            FP_list.append(FP)
            FN_list.append(FN)
            TP_list.append(TP)
            i_list.append(i)
            print(i)
            i += 0.01
        # Create a DataFrame
        data = {
            'accuracy': accuracy_list,
            'precision': precision_list,
            'sensitivity_recall': sensitivity_recall_list,
            'specificity': specificity_list,
            'TN': TN_list,
            'FP': FP_list,
            'FN': FN_list,
            'TP': TP_list,
            'i': i_list,
        }
        
        df = pd.DataFrame(data)
        
        # Save the DataFrame to a CSV file
        df.to_csv(r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\ROC\shortROC.csv', index=False)

    
exp = Experiment(r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\ROC\short2.csv')

exp.dataloader()

res = exp.run()

print(res)
print(time.time() - t0)