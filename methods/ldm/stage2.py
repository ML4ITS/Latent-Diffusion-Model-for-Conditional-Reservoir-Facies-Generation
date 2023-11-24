import os
import wandb
import torch

from preprocessing.preprocess import DatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from methods.ldm.modules.module_vqvae import ModuleVQVAE
from methods.utils import load_yaml_param_settings, get_root_dir, freeze, unfreeze
from methods.ldm.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


# def load_pretrained_encoder_decoder_vq(config: dict, dirname, freeze_models: bool = True, load_cond_models=False):
#     dim = config['encoder']['dim']
#     bottleneck_dim = config['encoder']['bottleneck_dim']
#     in_channels = config['dataset']['in_channels']
#     downsampling_rate = config['encoder']['downsampling_rate']
#     img_size = config['dataset']['img_size']
#
#     encoder = VQVAEEncoder(dim, bottleneck_dim, in_channels, downsampling_rate, config['encoder']['n_resnet_blocks'], config['encoder']['output_norm'])
#     decoder = VQVAEDecoder(dim, bottleneck_dim, in_channels, downsampling_rate, config['decoder']['n_resnet_blocks'], img_size)
#     vq_model = VectorQuantize(bottleneck_dim, **config['VQ-VAE'])
#
#     if not load_cond_models:
#         encoder_fname = 'encoder.ckpt'
#         decoder_fname = 'decoder.ckpt'
#         vq_model_fname = 'vq_model.ckpt'
#     else:
#         encoder_fname = 'encoder_cond.ckpt'
#         decoder_fname = 'decoder_cond.ckpt'
#         vq_model_fname = 'vq_model_cond.ckpt'
#
#     encoder.load_state_dict(torch.load(os.path.join(dirname, encoder_fname)))
#     decoder.load_state_dict(torch.load(os.path.join(dirname, decoder_fname)))
#     vq_model.load_state_dict(torch.load(os.path.join(dirname, vq_model_fname)))
#
#     encoder.eval()
#     decoder.eval()
#     vq_model.eval()
#
#     if freeze_models:
#         freeze(encoder)
#         freeze(decoder)
#         freeze(vq_model)
#
#     return encoder, decoder, vq_model


# def load_args():
#     parser = ArgumentParser()
#     parser.add_argument('--config', type=str, help="Path to the config data  file.",
#                         default=get_root_dir().joinpath('configs', 'config.yaml'))
#     return parser.parse_args()


# ========== RUN ==========

# load config
config = load_yaml_param_settings(get_root_dir().joinpath('configs', 'ldm.yaml'))

# data pipeline
dataset_importer = DatasetImporter(**config['dataset'])
batch_size = config['dataset']['batch_sizes']['stage2']
train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

# load the stage 1 module
stage1_ckpt_fname = get_root_dir().joinpath('methods', 'ldm', 'saved_models', config['diffusion']['stage1_ckpt_fname'])
img_size = train_data_loader.dataset.X.shape[-1]
module_vqvae = ModuleVQVAE.load_from_checkpoint(stage1_ckpt_fname,
                                                img_size=img_size,
                                                config=config,
                                                n_train_samples=len(train_data_loader.dataset))
module_vqvae.to(config['trainer_params']['gpu_idx'])
module_vqvae.eval()
freeze(module_vqvae.encoder)
freeze(module_vqvae.encoder_cond)

# diffusion model
model = Unet(
    in_channels=config['encoder']['bottleneck_dim'],
    dim=config['diffusion']['unet']['dim'],
    dim_mults=(1, 2, 4, 8),
    resnet_block_groups=4,
    self_condition=config['diffusion']['unet']['self_condition'],
    z_size=module_vqvae.encoder.H_prime[0].item(),  # width or height of z
    p_unconditional=config['diffusion']['p_unconditional'],
).to(config['trainer_params']['gpu_idx'])

diffusion = GaussianDiffusion(
    model,
    in_size=module_vqvae.encoder.H_prime[0].item(),  # width or height of z
    timesteps=1000,  # number of steps
    sampling_timesteps=1000,
    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type='l1',  # L1 or L2
    objective='pred_x0',
    auto_normalize=False,
    classifier_free_guidance_scale=config['diffusion']['classifier_free_guidance_scale'],
).to(config['trainer_params']['gpu_idx'])

# train
wandb.init(project='LDM for facies generation; stage2', config=config)
trainer = Trainer(
    train_data_loader,
    test_data_loader,
    diffusion,
    config,
    module_vqvae,
    train_batch_size=config['dataset']['batch_sizes']['stage2'],
    train_lr=config['trainer_params']['LR']['stage2'],
    train_num_steps=config['trainer_params']['max_num_steps']['stage2'],  # total training steps
    gradient_accumulate_every=2,  # gradient accumulation steps
    ema_decay=0.995,  # exponential moving average decay
    amp=False,  # turn on mixed precision
    fp16=False,
    save_and_sample_every=config['diffusion']['save_and_sample_every'],
    num_samples=config['diffusion']['num_samples'],
    augment_horizontal_flip=False,
    preserv_loss_weight=config['diffusion']['preserv_loss_weight'],
    saved_model_folder=get_root_dir().joinpath('methods', 'ldm', 'saved_models'),
    results_folder=get_root_dir().joinpath('methods', 'ldm', 'results'),
)

trainer.train()
