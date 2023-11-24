# Latent Diffusion Model for Conditional Reservoir Facies Generation


## Usage

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


