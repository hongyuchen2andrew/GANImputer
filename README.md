# GANImputer
## Introduction of GANImputer
This is the code of a novel missing data imputation algorithm based on Generative Adversarial Network(GAN), which consists of a discriminator and a generator. In this setup, the discriminator is formulated as a multi-class classifier, different from the binary classifier in the original GAN. The discriminator is trained to maximize the multi-class classification error and the generator is trained to minimize the multi-class classification error. After the adversarial training, the generator is fixed and the latent variables are optimized. Finally, the generator and latent variables are fine-tuned as a whole to impute the missing values.

## Overall Algorithms
The algorithm contains three step:\
Step 1: Optimization of generator. ![image](https://github.com/hongyuchen2andrew/GANImputer/blob/main/optimization/stage1.png)
Step 2: Optimization of latent space while keeping the optimized generator in step 1 fixed. ![image](https://github.com/hongyuchen2andrew/GANImputer/blob/main/optimization/stage2.png)
Step 3: Fine-tuning of the genrator and the latent space at the same time. ![image](https://github.com/hongyuchen2andrew/GANImputer/blob/main/optimization/stage3.png)
Details of the algorithm can be found in the 'GANImputer.pdf' file.

Our Algorithms are compared with other state-of-the-art(SOTA) algorithms including [GAIN](https://arxiv.org/pdf/1806.02920.pdf), [MIWAE](https://arxiv.org/pdf/1812.02633.pdf), [Singhorn](https://arxiv.org/pdf/2002.03860.pdf), and etc.. Code of these models can be found in the folder 'model'.
