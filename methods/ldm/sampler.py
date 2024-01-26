import torch

from methods.utils import load_yaml_param_settings, get_root_dir
from methods.ldm.utils import diffusion_trainer_fn
from methods.ldm.modules.module_vqvae import quantize


class LDMSampler(object):
    def __init__(self):
        # setup the trainer
        self.config = load_yaml_param_settings(get_root_dir().joinpath('configs', 'ldm.yaml'))
        self.trainer = diffusion_trainer_fn(initialize_wandb=False)

        # load the pretrained LDM
        fname = self.config['sampling']['trained_stage2_module_fname']
        self.trainer.load(ckpt_fname=fname)

        # models
        self.encoder_cond = self.trainer.pretrained_encoder_cond
        self.vq = self.trainer.pretrained_vq
        self.decoder = self.trainer.pretrained_decoder
        self.generator = self.trainer.ema.ema_model

        # inference mode
        self.encoder_cond.eval()
        self.vq.eval()
        self.decoder.eval()
        self.generator.eval()
        
    @torch.no_grad()
    def unconditional_sampling(self, n_samples: int) -> torch.Tensor:
        """
        Generate a synthetic sample, unconditionally.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            torch.Tensor: The generated synthetic sample; (b c h w)
        """
        z_gen = self.generator.sample(z_cond=None, batch_size=n_samples)  # (n_samples c h w) = (b d h w)

        # apply vq
        zq_gen, _, _, _ = quantize(z_gen, self.vq)  # (b d h w)

        # apply decoder
        x_gen = self.decoder(zq_gen)  # (b c h w)
        return x_gen
    
    @torch.no_grad()
    def conditional_sampling(self, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Generate a synthetic sample based on the given conditional input.

        Args:
            x_cond (torch.Tensor): The conditional input tensor of shape (b c h w); one-hot encoded.

        Returns:
            torch.Tensor: The generated synthetic sample; (b c h w)
        """
        # encode x_cond
        z_cond = self.encoder_cond(x_cond)  # (b d h w)

        # sample
        z_gen = self.generator.sample(z_cond, batch_size=z_cond.shape[0])  # (b d h w)

        # apply vq
        zq_gen, _, _, _ = quantize(z_gen, self.vq)  # (b d h w)

        # apply decoder
        x_gen = self.decoder(zq_gen)  # (b c h w)
        return x_gen
