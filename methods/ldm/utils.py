import wandb

from preprocessing.preprocess import DatasetImporter
from preprocessing.data_pipeline import build_data_pipeline
from methods.ldm.modules.module_vqvae import ModuleVQVAE
from methods.utils import load_yaml_param_settings, get_root_dir, freeze
from methods.ldm.denoising_diffusion_pytorch.denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer


def diffusion_trainer_fn(initialize_wandb: bool) -> Trainer:
    """
    This function is used to initialize a trainer object for the diffusion model.
    :param initialize_wandb: whether to initialize wandb or not
    :return: a trainer object
    """

    # load config
    config = load_yaml_param_settings(get_root_dir().joinpath('configs', 'ldm.yaml'))

    # data pipeline
    dataset_importer = DatasetImporter(**config['dataset'])
    batch_size = config['dataset']['batch_sizes']['stage2']
    train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

    # load the stage 1 module
    stage1_ckpt_fname = get_root_dir().joinpath('methods', 'ldm', 'saved_models', config['diffusion']['stage1_ckpt_fname'])
    module_vqvae = ModuleVQVAE.load_from_checkpoint(stage1_ckpt_fname,
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
        z_size=module_vqvae.encoder.H_prime.item(),  # width or height of z
        p_unconditional=config['diffusion']['p_unconditional'],
    ).to(config['trainer_params']['gpu_idx'])

    diffusion = GaussianDiffusion(
        model,
        in_size=module_vqvae.encoder.H_prime.item(),  # width or height of z
        timesteps=config['diffusion']['timesteps'],  # number of steps
        sampling_timesteps=None,  # if `sampling_timesteps < timesteps`, DDIM is used.
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
        loss_type='l1',  # L1 or L2
        objective='pred_x0',
        auto_normalize=False,
        classifier_free_guidance_scale=config['diffusion']['classifier_free_guidance_scale'],
    ).to(config['trainer_params']['gpu_idx'])

    # train
    if initialize_wandb:
        wandb.init(project='LDM for facies generation; stage2', config=config)

    trainer = Trainer(
        train_data_loader,
        test_data_loader,
        diffusion,
        config,
        module_vqvae,
        train_batch_size=config['dataset']['batch_sizes']['stage2'],
        train_lr=config['trainer_params']['stage2']['lr'],
        train_num_steps=config['trainer_params']['stage2']['max_num_steps']['train'],
        val_num_steps_for_loss=config['trainer_params']['stage2']['max_num_steps']['val_loss'],
        val_num_steps_for_sampling=config['trainer_params']['stage2']['max_num_steps']['val_sampling'],
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=False,  # turn on mixed precision
        fp16=False,
        save_model_every=config['trainer_params']['stage2']['save_model_every'],
        num_samples=config['diffusion']['num_val_samples'],
        augment_horizontal_flip=False,
        preserv_loss_weight=config['diffusion']['preserv_loss_weight'],
        saved_model_folder=get_root_dir().joinpath('methods', 'ldm', 'saved_models'),
        results_folder=get_root_dir().joinpath('methods', 'ldm', 'results'),
    )
    return trainer
