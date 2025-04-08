
# Detecting and Defending against Adversarial Attacks on Automatic Speech Recognition via Diffusion Models (Accepted to ICASSP 2025)

![Alt text](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/blob/main/system.png)

This the official implementation of the paper: Detecting and Defending against Adversarial Attacks on Automatic Speech Recognition via Diffusion Models. We defend against adversarial attacks on sentence and propose a novel approach for detecting adversarial attacks. The first draft of our paper is available on arXiv: "link to be added"  

For the white-box attack (C&W) used in this project see [here](https://github.com/carlini/audio_adversarial_examples ) 

When running the code use DeepSpeech 0.9.3, Tensorflow-gpu 1.15.4 (technically not necessary), PyTorch 1.13.1+cu116, CUDA 10.1 and cuDNN 7.6.5.  

For the pre-trained diffusion model DiffWave, the officially provided checkpoints is used and linked [here](https://github.com/philsyn/DiffWave-unconditional/tree/master/exp/ch256_T200_betaT0.02/logs/checkpoint?fbclid=IwAR3MX0AMM7h8e-FIyF1EXJhVPI64AJAej61FVL_CicVCNABxJKx1MxRKUN8)  

**For guide on how to reproduce results, see [guide.docx](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/blob/main/Guide.md)**

If you find this work useful, please consider citing our paper:
```
@inproceedings{kuhne2025detecting,
  title={Detecting and Defending Against Adversarial Attacks on Automatic Speech Recognition via Diffusion Models},
  author={K{\"u}hne, Nikolai L and Kitchena, Astrid HF and Jensen, Marie S and Br{\o}ndt, Mikkel SL and Gonzalez, Martin and Biscio, Christophe and Tan, Zheng-Hua},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```
Please note that this repository also contains some code originally from AudioPure. To replicate their results, please see: https://github.com/cychomatica/AudioPure. Additionally, we would like to thank the authors of AudioPure for their work and codebase. So, please consider citing them as well
```
@inproceedings{wu2023defending,
  title={Defending against Adversarial Audio via Diffusion Model},
  author={Wu, Shutong and Wang, Jiongxiao and Ping, Wei and Nie, Weili and Xiao, Chaowei},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/pdf?id=5-Df3tljit7}
}
```
Examples:  

**Clean:**  

[n = 0](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/68fe9637-e795-4d01-83e4-22acd0045da0)

[n = 1](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/e34ebcb0-4435-460a-b7e4-4a8c75ea52a0)

[n = 3](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/d0e230cb-b528-4c66-bcf6-d21d6f8984a5)

[n = 5](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/31c30f26-befd-41c8-9799-3db4c5b781a1)


**White-box attack:**  

[n = 0](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/b983b786-dc6d-4839-8d2a-24ef18825571)

[n = 1](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/29d21e5f-bbd8-47e4-adaa-88c3fe7f19cf)

[n = 3](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/5cc2dd12-df24-45fd-a529-b47f0becb1b3)

[n = 5](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/4cc22f70-495b-46a8-b740-c8d77cbf02c9)


**More noisy White-box attack:**  

[n = 0](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/bdac8a8c-9d7c-4209-bba1-0d078b841ed9)

[n = 1](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/3a88045c-3126-4981-a676-609968d49c2e)

[n = 3]( https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/eda6f5fe-523b-4089-9ea8-c3ae7344cc75)

[n = 5](https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/99d059cf-5321-47ea-8d14-12e8bc2c17f3)
