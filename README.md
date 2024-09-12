# Detecting and Defending Against Adversarial Attacks
This is an extension of previous research extending defence against adversarial attacks via diffusion models to sentences as well as a novel approach in regards to detecting adversarial attacks. The first draft of our paper is available in this repository.   

For the white-box attack used in this project see https://github.com/carlini/audio_adversarial_examples  

When running the code use DeepSpeech 0.9.3, Tensorflow-gpu 1.15.4, PyTorch 1.13.1+cu116, CUDA 10.1 and cuDNN 7.6.5.  

For the pre-trained diffusion model DiffWave the officially provided checkpoints is used and linked here: https://github.com/philsyn/DiffWave-unconditional/tree/master/exp/ch256_T200_betaT0.02/logs/checkpoint?fbclid=IwAR3MX0AMM7h8e-FIyF1EXJhVPI64AJAej61FVL_CicVCNABxJKx1MxRKUN8  
Examples:  

**Clean:**  
n = 0: https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/68fe9637-e795-4d01-83e4-22acd0045da0  

n = 1: https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/e34ebcb0-4435-460a-b7e4-4a8c75ea52a0  

n = 3: https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/d0e230cb-b528-4c66-bcf6-d21d6f8984a5  

n = 5: https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/31c30f26-befd-41c8-9799-3db4c5b781a1  


**White-box attack:**  

n = 0:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/b983b786-dc6d-4839-8d2a-24ef18825571  

n = 1:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/29d21e5f-bbd8-47e4-adaa-88c3fe7f19cf  

n = 3:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/5cc2dd12-df24-45fd-a529-b47f0becb1b3  

n = 5:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/4cc22f70-495b-46a8-b740-c8d77cbf02c9  


**More noisy White-box attack:**  

n = 0:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/bdac8a8c-9d7c-4209-bba1-0d078b841ed9  

n = 1:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/3a88045c-3126-4981-a676-609968d49c2e  

n = 3:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/eda6f5fe-523b-4089-9ea8-c3ae7344cc75  

n = 5:  https://github.com/Kyhne/Detecting-and-Defending-Against-Adversarial-Attacks/assets/70662482/99d059cf-5321-47ea-8d14-12e8bc2c17f3  








