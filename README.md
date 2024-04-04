<script type="text/javascript"
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# GANImputer
## Introduction of GANImputer
GANImputer is a novel missing data imputation algorithm based on Generative Adversarial Network(GAN), which consists of a discriminator and a generator. In this setup, the discriminator is formulated as a multi-class classifier, different from the binary classifier in the original GAN. The discriminator is trained to maximize the multi-class classification error and the generator is trained to minimize the multi-class classification error. After the adversarial training, the generator is fixed and the latent variables are optimized. Finally, the generator and latent variables are fine-tuned as a whole to impute the missing values. Code of our model can be found [here](https://github.com/hongyuchen2andrew/GANImputer/blob/main/GANImputer). Code of other related state-of-the-art (SOTA) algorithms can be found [here](https://github.com/hongyuchen2andrew/GANImputer/blob/main/models).

## Overall Algorithms
### Stage 1: Adversarial Learning for Generator
![image](https://github.com/hongyuchen2andrew/GANImputer/blob/main/optimization/stage1.png)
### Stage 2: Latent Variable Optimization
![image](https://github.com/hongyuchen2andrew/GANImputer/blob/main/optimization/stage2.png)
### Step 3: Fine-Tuning of Generator and Latent Variable
![image](https://github.com/hongyuchen2andrew/GANImputer/blob/main/optimization/stage3.png)

## Computational Complexity Analysis
Assume for each epoch, $B$ samples are randomly selected and latent variables are randomly generated from the standard normal distribution. The architecture of the discriminator and generator both follow a shape of low-dimension to high-dimension to low-dimension. In this study, we consider 2 hidden layers in both the discriminator and generator. Particularly, the network structures of the discriminator and generator are $[d_x, 2d_x, 2d_x, d_x]$ and $[d_z, d_x, 2d_x, d_x]$ respectively. Then when optimizing the discriminator, the time complexity and space complexities per iteration are $\mathcal{O}(Bd_x^2)$ and $\mathcal{O}(Bd_x)$ respectively. When optimizing the generator, the time complexity and space complexities per iteration are $\mathcal{O}(Bd_x^2+Bd_xd_z)$ and $\mathcal{O}(Bd_x+Bd_z)$ respectively.

## Numerical Results
The experimental results on five benchmark datasets at different missing rates, followed by downstream tasks, show that our GANImputer can outperform strong baselines such as [GAIN](https://arxiv.org/pdf/1806.02920.pdf), [MIWAE](https://arxiv.org/pdf/1812.02633.pdf), and [Singhorn](https://arxiv.org/pdf/2002.03860.pdf). We also apply GANImputer to the Tennessee Eastman process [24], which further demonstrates the effectiveness of our method.

Details of the model structure and results of the experiments can be found in our paper [GANImputer](https://github.com/hongyuchen2andrew/GANImputer/blob/main/GANImputer.pdf).

