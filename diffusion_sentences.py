import torch
from Classifier import *
from diffusion_models.diffwave_ddpm import create_diffwave_model
import argparse
import os
from robustness_eval.certified_robust import *
import pandas as pd
from torchmetrics import CharErrorRate
import time
from diffusion_models.diffwave_sde import *
from acoustic_system import AcousticSystem

t0 = time.time()

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

'''SC09 classifier arguments'''
# # parser.add_argument("--data_path", help='sc09 dataset folder', default = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\datasets\speech_commands")
# parser.add_argument("--classifier_path", help='dir of saved classifier model', default=r'C:/Users/kyhne/Downloads/vgg19_bn-c79401a0.pth')
# parser.add_argument("--classifier_input", choices=['mel32'], default='mel32', help='input of NN')
# parser.add_argument("--num_per_class", type=int, default=1)

'''DiffWave arguments'''
parser.add_argument('--config', type=str, default=r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\diffusion_models\DiffWave_Unconditional\config.json', help='JSON file for configuration')
parser.add_argument('--defender_path', type=str, help='dir of diffusion model checkpoint', default = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\1000000.pkl')


# '''certified robustness arguments'''
# parser.add_argument('--defense_method', type=str, default='diffusion', choices=['diffusion', 'randsmooth'])
# parser.add_argument('--sigma', type=float, default=0.25)
# parser.add_argument('--num_sampling', type=int, default=10)

'''device arguments'''
# parser.add_argument("--dataload_workers_nums", type=int, default=4, help='number of workers for dataloader')
# parser.add_argument("--batch_size", type=int, default=5, help='batch size')
parser.add_argument('--gpu', type=int, default=0)

'''DiffWave-VPSDE arguments'''
parser.add_argument('--ddpm_config', type=str, default=r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\diffusion_models\DiffWave_Unconditional\config.json', help='JSON file for configuration')
parser.add_argument('--ddpm_path', type=str, help='dir vbhfyr of diffusion model checkpoint', default = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\1000000.pkl')
parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
parser.add_argument('--t', type=int, default=4, help='diffusion steps, control the sampling noise scale')
parser.add_argument('--t_delta', type=int, default=0, help='perturbation range of sampling noise scale; set to 0 by default')
parser.add_argument('--rand_t', action='store_true', default=False, help='decide if randomize sampling noise scale')
parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde, ddpm]')
parser.add_argument('--use_bm', action='store_true', default=False, help='whether to use brownian motion')

# '''file saving arguments'''
# parser.add_argument('--save_path', type=str, default='_Experiments/certified_robustness/records')

args = parser.parse_args()


'''device setting'''
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
use_gpu = torch.cuda.is_available()
print('use_gpu', use_gpu)
print('gpu id: {}'.format(args.gpu))





class Experiment():
    
    def __init__(self):
        pass

    @torch.no_grad()
    def run(self):
        # Classifier:
        model_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.pbmm'
        scorer_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.scorer'
        Classifier = DeepSpeechTranscriber(model_path, scorer_path)
        
        # Diffusion model        
        Defender = RevDiffWave(args)
        defense_type = 'wave'
        AS_MODEL = AcousticSystem(classifier=Classifier, defender=Defender, defense_type=defense_type)
        AS_MODEL.eval().cuda()

        index = 0   
        
        # Short sentences:
        # Clean
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Input_Common_10_5000"
        
        # WB
        # folder = r"C:/Users/kyhne/OneDrive - Aalborg Universitet/Uni/7. semester/P7 - Informationsbehandling i teknologiske systemer/AudioPure-master/Output_100_1000"
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_10_5000"
        
        # Medium sentences:
        # Clean
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Input_Common_10_5000_medium"
        
        # WB
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_100_1000_medium"
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_100_1000_medium_internet"
        folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_10_5000_medium_internet"
        
        # Long sentences:
        # Clean
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Input_Common_10_5000_long"
        
        # WB
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_100_1000_long\Output_Common_100_1000_long"
        # folder = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Output_Common_10_5000_long"

        
        fdir = os.listdir(folder) # All files in current directory
        
        # All files
        fs = [f for f in fdir if f.endswith('wav')] # Find soundfiles
        
        for i in fs:
            audio_file = folder + "\\" + i
            waveform, sample_rate = torchaudio.load(audio_file)

            waveform = torch.unsqueeze(waveform, 1)
            AS_MODEL.defender.rev_vpsde.audio_shape = (1, waveform.shape[-1])
 
            output_file_path = r"C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\Diffusion_Output" + "\\"  + f'{index}.wav'

            predicted = AS_MODEL.defender(waveform)
  
            # Save the PyTorch tensor as a WAV file
            torchaudio.save(output_file_path, predicted.detach().cpu().view(1, predicted.size(dim=2)), sample_rate, encoding='PCM_S', bits_per_sample=16)
            
            index += 1
            print(index)
    
exp = Experiment()

exp.run()

print(time.time() - t0)