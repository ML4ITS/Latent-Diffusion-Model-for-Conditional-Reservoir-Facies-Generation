import argparse

# import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from preprocessing.preprocess import DatasetImporter, GeoDataset
from methods.unet_gan.models import GeneratorUNet, Discriminator
from methods.utils import get_root_dir, load_yaml_param_settings
from methods.unet_gan.utils import preprocess_data


class GANSampler(object):
    def __init__(self) -> None:
        # config
        config_fname = get_root_dir().joinpath('configs', 'unet_gan.yaml')
        self.config = load_yaml_param_settings(config_fname)
        self.device = self.config['device']
        self.n_categories = self.config['n_categories']

        # Initialize generator
        self.generator = GeneratorUNet(1, 1, self.config['img_size'])

        # load
        epoch_of_pretrained_model = self.config['epoch_of_pretrained_model']
        ckpt_fname = get_root_dir().joinpath('methods', 'unet_gan', 'saved_models', f'generator_{epoch_of_pretrained_model}.ckpt')
        checkpoint = torch.load(ckpt_fname)
        self.generator.load_state_dict(checkpoint)
        self.generator.to(self.device)

        # inference mode
        self.generator.eval()  
    
    @torch.no_grad()
    def unconditional_sampling(self, n_samples: int) -> torch.Tensor:
        """
        Generate unconditional samples using the generator network.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated unconditional samples of shape (n_samples, h, w).
        """
        
        h, w = self.config['unet_in_size'], self.config['unet_in_size']
        x_uncond = self.n_categories * torch.ones((n_samples, 1, h, w)).to(self.device)  # (b 1 h w); no valid pixel left (i.e., all pixels are masked)
        
        # forward
        fake_x = self.generator(x_uncond, return_noise_vector=False)  # (b 1 h w);
        
        # postprocess
        fake_x = fake_x[:, 0, :, :]  # (b h w)
        fake_x = torch.clip(torch.round(fake_x), min=0, max=self.n_categories-1)  # (b h w)
        return fake_x

    @torch.no_grad()
    def conditional_sampling(self, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Generate a synthetic sample based on the given conditional input.

        Args:
            x_cond (torch.Tensor): The conditional input tensor of shape (b, c, h, w); one-hot encoded.

        Returns:
            torch.Tensor: The generated synthetic sample tensor; (b h w); consisting of facies indices (integeres).

        """
        x_cond = preprocess_data(x_cond, self.config['unet_in_size'], self.config['unet_in_size'], self.device)  # (b 1 h' w'); h', w' <- `unet_in_size``
        
        # forward
        fake_x = self.generator(x_cond, return_noise_vector=False)  # (b 1 h w);

        # postprocess
        fake_x = fake_x[:, 0, :, :]  # (b h w)
        fake_x = torch.clip(torch.round(fake_x), min=0, max=self.n_categories-1)  # discretize to facies indices; (b h w)
        return fake_x
    