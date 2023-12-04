"""
Implementation of the model from "U-net generative adversarial network for subsurface facies modeling"
GAN implementation from [https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py]
"""

import argparse
import os
import numpy as np
import time
import datetime
import sys

import matplotlib.pyplot as plt
import wandb

import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from einops import rearrange

from methods.unet_gan.models import GeneratorUNet, Discriminator, weights_init_normal
from methods.unet_gan.utils import preprocess_data
from preprocessing.preprocess import DatasetImporter, GeoDataset
from methods.utils import get_root_dir, load_yaml_param_settings


# config
config_fname = get_root_dir().joinpath('configs', 'unet_gan.yaml')
config = load_yaml_param_settings(config_fname)


save_model_dirname = get_root_dir().joinpath('methods', 'unet_gan', "saved_models")
os.makedirs(save_model_dirname, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
# criterion_GAN = torch.nn.MSELoss()
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_pixelwise = torch.nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, config['unet_in_size'] // 2 ** 4, config['unet_in_size'] // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet(1, 1, config['img_size'])
discriminator = Discriminator(1)

if cuda:
    generator = generator.to(config['device'])
    discriminator = discriminator.to(config['device'])
    criterion_GAN.to(config['device'])
    criterion_pixelwise.to(config['device'])


# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['lr'], betas=(config['b1'], config['b2']))

# Configure dataloaders
fname = get_root_dir().joinpath('dataset', 'facies_5000.npy')
dataset_importer = DatasetImporter(fname,
                                   train_ratio=config['train_ratio'],
                                   data_scaling=True,
                                   n_categories=4)
train_dataloader = DataLoader(GeoDataset("train", dataset_importer), batch_size=config['batch_size'], num_workers=config['n_cpu'], shuffle=True)
test_dataloader = DataLoader(GeoDataset("test", dataset_importer), batch_size=config['batch_size'], num_workers=0, shuffle=True)


# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




# ----------
#  Training
# ----------

wandb.init(project='GeoLDM; unet_gan', config=config)
prev_time = time.time()
for epoch in range(0, config['n_epochs']+1):
    for i, batch in enumerate(train_dataloader):
        x, x_cond = batch  # x: (b c h w); x_cond: (b c+1 h w)

        x = preprocess_data(x, config['unet_in_size'], config['unet_in_size'], config['device'])
        x_cond = preprocess_data(x_cond, config['unet_in_size'], config['unet_in_size'], config['device'])

        # Adversarial ground truths
        batch_size = x.shape[0]
        valid = torch.FloatTensor(np.ones((batch_size, *patch))).to(config['device'])  # (b 1 p p)
        fake = torch.FloatTensor(np.zeros((batch_size, *patch))).to(config['device'])  # (b 1 p p)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        fake_x, noise_vec = generator(x_cond, return_noise_vector=True)  # (b 1 h w);
        pred_fake = discriminator(fake_x, x_cond)  # (b 1 p p)
        loss_G = criterion_GAN(pred_fake.flatten(), valid.flatten())

        # content loss, L_c
        cond_loc = (x_cond != config['n_categories'])  # (b 1 h w)
        x_with_fake_cond = torch.where(condition=cond_loc, input=fake_x, other=x)  # (b 1 h w)
        content_loss = F.l1_loss(input=x_with_fake_cond, target=x)

        # diversity loss
        d_x = torch.cdist(rearrange(fake_x, 'b 1 h w -> b (h w)'), rearrange(fake_x, 'b 1 h w -> b (h w)'))
        d_z = torch.cdist(rearrange(noise_vec, 'b c h w -> b (c h w)'), rearrange(noise_vec, 'b c h w -> b (c h w)'))
        diversity_loss = - d_x.mean() / d_z.mean()

        # Total loss
        content_loss_weight = config['content_loss_weight']
        diversity_loss_weight = config['diversity_loss_weight']
        loss_gen = loss_G + (content_loss_weight * content_loss) + (diversity_loss_weight*diversity_loss)

        loss_gen.backward()
        optimizer_G.step()

        wandb.log({'epoch': epoch,
                   'train/loss_G': loss_G.item(),
                   'train/content_loss': content_loss.item(),
                   'train/diversity_loss': diversity_loss.item(),
                   })

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        pred_real = discriminator(x, x_cond)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(fake_x.detach(), x_cond)
        loss_fake = criterion_GAN(pred_fake, fake)

        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)

        loss_D.backward()
        optimizer_D.step()

        wandb.log({'epoch': epoch,
                   'train/loss_D': loss_D.item()})

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = config['n_epochs'] * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (
                epoch,
                config['n_epochs'],
                i,
                len(train_dataloader),
                loss_D.item(),
                loss_G.item(),
                time_left,
            )
        )

        # If at sample interval save image
        if epoch % config['sample_interval'] == 0:
            x, x_cond = next(iter(test_dataloader))
            x = preprocess_data(x, config['unet_in_size'], config['unet_in_size'], config['device'])
            x_cond = preprocess_data(x_cond, config['unet_in_size'], config['unet_in_size'], config['device'])

            generator.eval()
            fake_x = generator(x_cond)  # (b 1 h w)
            generator.train()
            x, x_cond, fake_x = x.cpu(), x_cond.cpu(), fake_x.detach().cpu()
            fake_x = torch.clip(fake_x, min=0, max=config['n_categories']-1)  # to prevent sampling a mask token

            discrete_fake_x = torch.round(fake_x)
            cond_loc = (x_cond != config['n_categories'])  # (b 1 h w)
            x_with_fake_cond = torch.where(condition=cond_loc, input=discrete_fake_x, other=x)  # (b 1 h w)
            preservation_error_map = torch.clip(torch.abs(x_with_fake_cond - x), 0, 1)  # (b 1 h w)
            preservation_error_map = preservation_error_map.detach().cpu()

            # log
            content_loss = F.l1_loss(input=x_with_fake_cond, target=x)
            wandb.log({'epoch': epoch, 'test/content_loss': content_loss})

            # plot
            b = 0
            n_subfigs = 5
            fig, axes = plt.subplots(1, n_subfigs, figsize=(3*n_subfigs, 3))
            axes[0].imshow(x[b,0], vmin=0, vmax=config['n_categories'], cmap='Accent', interpolation='nearest')
            axes[1].imshow(x_cond[b, 0], vmin=0, vmax=config['n_categories'], cmap='Accent', interpolation='nearest')
            axes[2].imshow(fake_x[b, 0], vmin=0, vmax=config['n_categories'], cmap='Accent', interpolation='nearest')
            axes[3].imshow(discrete_fake_x[b, 0], vmin=0, vmax=config['n_categories'], cmap='Accent', interpolation='nearest')
            axes[4].imshow(preservation_error_map[b, 0], vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
            for ax in axes:
                ax.invert_yaxis()
            plt.suptitle(f'epoch:{epoch}')
            wandb.log({'sample on x_cond_test': wandb.Image(plt)})
            plt.close()

    if config['checkpoint_interval'] != -1 and epoch % config['checkpoint_interval'] == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), save_model_dirname.joinpath("generator_%d.ckpt" % (epoch,)))
        torch.save(discriminator.state_dict(), save_model_dirname.joinpath("discriminator_%d.ckpt" % (epoch,)))
