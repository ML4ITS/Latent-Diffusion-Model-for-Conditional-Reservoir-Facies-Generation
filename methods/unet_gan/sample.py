import argparse

# import wandb
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from einops import repeat
from preprocessing.preprocess import DatasetImporter, GeoDataset
from methods.unet_gan.models import GeneratorUNet, Discriminator
from utils import get_root_dir, load_yaml_param_settings
from methods.unet_gan.utils import preprocess_data


# config
config_fname = get_root_dir().joinpath('configs', 'unet_gan.yaml')
config = load_yaml_param_settings(config_fname)

# load the test data loader
# Configure dataloaders
dirname = get_root_dir().joinpath('dataset', 'facies_200')
dataset_importer = DatasetImporter(dirname,
                                   train_ratio=0.8,
                                   data_scaling=True,
                                   n_categories=4)
test_dataloader = DataLoader(GeoDataset("test", dataset_importer), batch_size=config['batch_size'], num_workers=0, shuffle=True)

# load the pretrained GAN model
# do not forget to set .eval()
generator = GeneratorUNet(1, 1).to(config['device'])
discriminator = Discriminator(1).to(config['device'])

save_model_dirname = get_root_dir().joinpath('methods', 'unet_gan', "saved_models")

generator.load_state_dict(torch.load(save_model_dirname.joinpath('generator_%d.ckpt' % (config['epoch_of_pretrained_model'],))))
discriminator.load_state_dict(torch.load(save_model_dirname.joinpath('discriminator_%d.ckpt' % (config['epoch_of_pretrained_model'],))))

generator.eval()
discriminator.eval()

# sample: many-to_many
x, x_cond = next(iter(test_dataloader))
x = preprocess_data(x, config['img_height'], config['img_width'], config['device'])
x_cond = preprocess_data(x_cond, config['img_height'], config['img_width'], config['device'])

fake_x = generator(x_cond)  # (b 1 h w)
x, x_cond, fake_x = x.cpu(), x_cond.cpu(), fake_x.detach().cpu()
discrete_fake_x = torch.round(fake_x)
cond_loc = (x_cond != config['n_categories'])  # (b 1 h w)
x_with_fake_cond = torch.where(condition=cond_loc, input=discrete_fake_x, other=x)  # (b 1 h w)
preservation_error_map = torch.clip(torch.abs(x_with_fake_cond - x), 0, 1)  # (b 1 h w)
preservation_error_map = preservation_error_map.detach().cpu()

b = 0
n_subfigs = 5
fig, axes = plt.subplots(1, n_subfigs, figsize=(3 * n_subfigs, 3))
axes[0].imshow(x[b, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
axes[1].imshow(x_cond[b, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
axes[2].imshow(fake_x[b, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
axes[3].imshow(discrete_fake_x[b, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
axes[4].imshow(preservation_error_map[b, 0], vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
for ax in axes:
    ax.invert_yaxis()
plt.tight_layout()
plt.show()


# sample: one-to_many
x, x_cond = x.cuda(), x_cond.cuda()
x = repeat(x[[0]], '1 1 h w -> b 1 h w', b=32)
x_cond = repeat(x_cond[[0]], '1 1 h w -> b 1 h w', b=32)

fake_x = generator(x_cond)  # (b 1 h w)
x, x_cond, fake_x = x.cpu(), x_cond.cpu(), fake_x.detach().cpu()
discrete_fake_x = torch.round(fake_x)
cond_loc = (x_cond != config['n_categories'])  # (b 1 h w)
x_with_fake_cond = torch.where(condition=cond_loc, input=discrete_fake_x, other=x)  # (b 1 h w)
preservation_error_map = torch.clip(torch.abs(x_with_fake_cond - x), 0, 1)  # (b 1 h w)
preservation_error_map = preservation_error_map.detach().cpu()

b = 0
n_subfigs = 5
fig, axes = plt.subplots(n_subfigs+1, 2, figsize=(4, 3*(n_subfigs+1)))
axes[0,0].imshow(x[b, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
axes[0,1].imshow(x_cond[b, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
for i in range(1, n_subfigs+1):
    axes[i, 0].imshow(discrete_fake_x[i, 0], vmin=0, vmax=config['n_categories'], cmap='gist_ncar', interpolation='nearest')
    axes[i, 1].imshow(preservation_error_map[i, 0], vmin=0, vmax=1, cmap='Greys', interpolation='nearest')
for ax in axes.flatten():
    ax.invert_yaxis()
plt.tight_layout()
plt.show()
