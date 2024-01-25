# Latent Diffusion Model for Conditional Reservoir Facies Generation

This is an official GitHub repository for the PyTorch implementation of [Latent Diffusion Model for Conditional Reservoir Facies Generation](https://arxiv.org/abs/2311.01968#:~:text=Latent%20Diffusion%20Model%20for%20Conditional%20Reservoir%20Facies%20Generation,-Daesoo%20Lee%2C%20Oscar&text=Creating%20accurate%20and%20geologically%20realistic,the%20oil%20and%20gas%20sector.).
The paper proposes conditional reservoir facies generation via diffusion modeling. 
The following figure illustrates the gradual generation process (_i.e.,_ noise to sample).

<p align="center">
<img src=".fig/gradual_sampling_process.png" alt="" width=80% height=80%>
</p>


## Usage

#### LDM for Conditional Reservoir Facies Generation
To train the LDM, run
```commandline
python train.py --method ldm_stage1  # stage 1 training
python train.py --method ldm_stage2  # stage 2 training
```
To sample with the trained LDM, run
```commandline

```


#### U-Net GAN
The [U-Net GAN](https://link.springer.com/article/10.1007/s10596-020-10027-w) paper proposed to utilize the [pix2pix-style GAN](https://arxiv.org/abs/1611.07004) for conditional facies generation.
To train the U-Net GAN, run
```commandline
python train.py --method unet_gan
```
To sample with the trained U-Net GAN, run
```commandline
python sample.py --method unet_gan
```
Its parameters can be adjusted in `configs/unet_gan.yaml`.


