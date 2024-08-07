"""
Stage 1: VQ training

run `python stage1.py`
"""
from argparse import ArgumentParser

import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from methods.utils import get_root_dir, load_yaml_param_settings
from methods.ldm.modules.module_vqvae import ModuleVQVAE
from preprocessing.preprocess import DatasetImporter
from preprocessing.data_pipeline import build_data_pipeline


# def load_args():
#     parser = ArgumentParser()
#     parser.add_argument('--config', type=str, help="Path to the config data file.",
#                         default=get_root_dir().joinpath('configs', 'ldm.yaml'))
#     return parser.parse_args()


def train_stage1(config: dict,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpu_idx: int,
                 ):
    # fit
    img_size = train_data_loader.dataset.X.shape[-1]
    module_vqvae = ModuleVQVAE(img_size, config, len(train_data_loader.dataset))

    wandb_logger = WandbLogger(project='LDM for facies generation; stage1', name=None, config=config,
                               save_dir=get_root_dir().joinpath('methods', 'ldm', 'saved_models'))
    checkpoint_callback = ModelCheckpoint(dirpath=get_root_dir().joinpath('methods', 'ldm', 'saved_models'),
                                          filename=f'stage1',
                                          every_n_epochs=config['trainer_params']['stage1']['save_period_in_epoch'])
    trainer = pl.Trainer(logger=wandb_logger,
                         callbacks=[LearningRateMonitor(logging_interval='epoch'), checkpoint_callback],
                         max_epochs=config['trainer_params']['stage1']['max_epochs'],
                         devices=[gpu_idx,],
                         accelerator='gpu',
                         check_val_every_n_epoch=config['trainer_params']['stage1']['check_val_every_n_epoch'])
    trainer.fit(module_vqvae,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    # additional log
    n_trainable_params = sum(p.numel() for p in module_vqvae.parameters() if p.requires_grad)
    wandb.log({'n_trainable_params:': n_trainable_params})

    # test
    print('closing...')
    wandb.finish()


# ========== RUN ==========

# load config
config = load_yaml_param_settings(get_root_dir().joinpath('configs', 'ldm.yaml'))

# data pipeline
dataset_importer = DatasetImporter(**config['dataset'])
batch_size = config['dataset']['batch_sizes']['stage1']
train_data_loader, test_data_loader = [build_data_pipeline(batch_size, dataset_importer, config, kind) for kind in ['train', 'test']]

# train
train_stage1(config,
             train_data_loader, test_data_loader,
             config['trainer_params']['gpu_idx'])
