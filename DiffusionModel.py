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
parser.add_argument('--t', type=int, default=1, help='diffusion steps, control the sampling noise scale')
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


class Diffusion():

    def __init__(self, classifier: torch.nn.Module, transform=None, denoiser=None, one_shot_rev: bool=False, num_classes=10) -> None:

        self.classifier = classifier
        # self.transform = transform
        self.denoiser = denoiser
        self.num_classes = num_classes
        self.one_shot_rev = one_shot_rev
        # self.model = torch.nn.Sequential(self.denoiser, self.transform, self.classifier)
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor):

        x_in = x

        # if self.denoiser is not None:
            # x_in = self.denoiser.denoise(x_in)
        x_in = self.denoiser.one_shot_denoise(x_in.cuda())
        # x_in = self.denoiser.two_shot_denoise(x_in.cuda())

        output = x_in
        return output
    
    def compute_t_star(self, alpha_bar_star):
        
        T = self.denoiser.diffusion_hyperparams['T']
        Alpha_bar = self.denoiser.diffusion_hyperparams['Alpha_bar']

        t_star = torch.abs(Alpha_bar- alpha_bar_star).min(0, keepdim=True)[1].item() + 1
        # t_star = (Alpha_bar[0] - alpha_star) / (Alpha_bar[0] - Alpha_bar[-1]) * (T - 1) + 1 # in [1, T]

        return t_star
    
    @torch.no_grad()
    def diffuse(self, x: torch.Tensor, sigma=0.0):

        assert(x.shape[0] == 1)

        x_in = x
        delta = torch.normal(0,sigma,size=x_in.shape).cuda()
        x_in = x_in.cuda() + delta

        # if self.denoiser is not None:
        alpha_bar_star = 1 / (1 + sigma**2)
        t_star = self.compute_t_star(alpha_bar_star)
        self.denoiser.reverse_timestep = t_star
        x_in = alpha_bar_star**0.5 * x_in

        output = self.forward(x_in)

        return output
@torch.no_grad()
def example():

    # Diffusion Model:
    # DiffWave_Denoiser = create_diffwave_model(model_path=args.defender_path, config_path=args.config, reverse_timestep=5)
    # DiffWave_Denoiser.eval().cuda()
    # predict = Diffusion(classifier=Classifier, denoiser=DiffWave_Denoiser)
    
    # # Classifier:
    model_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.pbmm'
    scorer_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.scorer'
    Classifier = DeepSpeechTranscriber(model_path, scorer_path)
    
    Defender = RevDiffWave(args)
    defense_type = 'wave'
    AS_MODEL = AcousticSystem(classifier=Classifier, defender=Defender, defense_type=defense_type)
    AS_MODEL.eval().cuda()
    
    # audio_file = r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\adversarial_dataset-A\Adversarial-Examples\medium-signals\adv-short-target\adv-medium2short-000957.wav'
    audio_file = r'C:\Users\kyhne\Downloads\output_10_500_20\output_10_500_20\result\right\yes\1b63157b_nohash_3.wav'
    
    
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = torch.unsqueeze(waveform, 1)
    AS_MODEL.defender.rev_vpsde.audio_shape = (1, waveform.shape[-1])
    

    predicted = AS_MODEL.defender(waveform)
    # print(predicted)
    
    output_file_path = r'C:\Users\kyhne\Downloads\output_audio.wav'
    
    # # Save the PyTorch tensor as a WAV file
    torchaudio.save(output_file_path, predicted.detach().cpu().view(1, predicted.size(dim=2)), sample_rate, encoding='PCM_S', bits_per_sample=16)
    
    transcription1 = Classifier.transcribe_audio_file(audio_file)
    print(transcription1)
    transcription2 = Classifier.transcribe_audio_file(output_file_path)
    print(transcription2)
# example()

class Experiment():
    
    def __init__(self):
        pass
    
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

    @torch.no_grad()
    def run(self, folder):
        fdir = os.listdir(folder) # All files in current directory
        
        # All files
        fs = [f for f in fdir if f.endswith('wav')] # Find soundfiles
        
        # Sort
        fs = np.sort(fs) # Numpy uses a faster sorting algorithm
        
        # Classifier:
        model_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.pbmm'
        scorer_path = r'C:\Users\kyhne\Downloads\deepspeech-0.9.3-models.scorer'
        Classifier = DeepSpeechTranscriber(model_path, scorer_path)
        
        # Diffusion model        
        Defender = RevDiffWave(args)
        defense_type = 'wave'
        AS_MODEL = AcousticSystem(classifier=Classifier, defender=Defender, defense_type=defense_type)
        AS_MODEL.eval().cuda()

        # Diffusion Model:
        DiffWave_Denoiser = create_diffwave_model(model_path=args.defender_path, config_path=args.config, reverse_timestep=5)
        DiffWave_Denoiser.eval().cuda()
        
        # # Model (only if Diffusion model defence is used!!!)
        predict = Diffusion(classifier=Classifier, denoiser=DiffWave_Denoiser)
        
        # Error label rate
        cer = CharErrorRate()
        output_file_path = r'C:\Users\kyhne\Downloads\output_audio.wav'
        n = 0
        index = 0   
        original = []
        attacked = []
        for i in fs:
            audio_file = folder + "\\" + i
            waveform, sample_rate = torchaudio.load(audio_file)

            waveform = torch.unsqueeze(waveform, 1)
            AS_MODEL.defender.rev_vpsde.audio_shape = (1, waveform.shape[-1])
            # sigma = 0.001
            # delta = torch.normal(0,sigma,size=waveform.shape).cuda()
            # waveform = waveform.cuda() + delta
            # predicted = predict.diffuse(x=waveform)
            predicted = AS_MODEL.defender(waveform)

            # Save the PyTorch tensor as a WAV file
            torchaudio.save(output_file_path, predicted.detach().cpu().view(1, predicted.size(dim=2)), sample_rate, encoding='PCM_S', bits_per_sample=16)
            
            transcription = Classifier.transcribe_audio_file(output_file_path)
            # n += 1 - cer(transcription, self.sentences[index])
            # n += 1 - cer(transcription, "zero")
            
            if transcription == self.sentences[index]:
                print(f'{transcription}: Correct')
            else:
                print(f'{transcription}: Wrong')    
            original.append(self.sentences[index])
            attacked.append(transcription)
            index += 1
            print(index)
        return 1 - cer(original, attacked)
    
exp = Experiment()

exp.dataloader(r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\adversarial_dataset-A\Adversarial-Examples\medium-signals\list-medium.csv')

res = exp.run(r'C:\Users\kyhne\OneDrive - Aalborg Universitet\Uni\7. semester\P7 - Informationsbehandling i teknologiske systemer\AudioPure-master\adversarial_dataset-A\Adversarial-Examples\medium-signals\adv-long-target')

print(res)

print(time.time() - t0)
