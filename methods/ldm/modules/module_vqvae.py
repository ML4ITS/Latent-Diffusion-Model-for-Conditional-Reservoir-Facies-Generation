import torch
import numpy as np
import matplotlib.pyplot as plt
import wandb
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from matplotlib.colors import ListedColormap

from methods.ldm.models.vq_enc_dec import VQVAEEncoder, VQVAEDecoder
from methods.ldm.models.vq import VectorQuantize
from methods.ldm.modules.module_base import ModuleBase, detach_the_unnecessary


def quantize(z, vq_model, transpose_channel_length_axes=False, **kwargs):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, 'b c h w -> b (h w) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, **kwargs)
        z_q = rearrange(z_q, 'b (h w) c -> b c h w', h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, 'b c l -> b (l) c')
        z_q, indices, vq_loss, perplexity = vq_model(z, **kwargs
                                                     )
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, 'b (l) c -> b c l')
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity


class ModuleVQVAE(pl.LightningModule):
    def __init__(self,
                 img_size: int,
                 config: dict,
                 n_train_samples: int):
        """
        :param config: configs/config.yaml
        :param n_train_samples: number of training samples
        """
        super().__init__()
        self.config = config
        self.T_max = int(config['trainer_params']['max_epochs']['stage1'] * (np.ceil(n_train_samples / config['dataset']['batch_sizes']['stage1']) + 1))

        in_channels = config['dataset']['in_channels']

        # encoder
        self.encoder = VQVAEEncoder(in_channels, **config['encoder'])
        self.decoder = VQVAEDecoder(in_channels, **config['encoder'])
        self.vq_model = VectorQuantize(config['encoder']['hid_dim'], **config['VQ-VAE'])

        self.encoder_cond = VQVAEEncoder(in_channels+1, **config['encoder'])
        self.decoder_cond = VQVAEDecoder(in_channels+1, **config['encoder'])
        self.vq_model_cond = VectorQuantize(config['encoder']['hid_dim'], **config['VQ-VAE'])

    def forward(self, x, kind):
        assert kind in ['x', 'x_cond']

        if kind == 'x':
            encoder = self.encoder
            decoder = self.decoder
            vq_model = self.vq_model
        elif kind == 'x_cond':
            encoder = self.encoder_cond
            decoder = self.decoder_cond
            vq_model = self.vq_model_cond
        else:
            raise ValueError

        # forward
        z = encoder(x)  # (b d h' w')
        z_q, indices, vq_loss, perplexity = quantize(z, vq_model)  # z_q: (b d h' w')
        vq_loss = vq_loss['loss']
        xhat = decoder(z_q)  # (b c h w)

        y_true = x.argmax(dim=1)  # (b h w)
        y_true = y_true.flatten()  # (b*h*w)
        y_pred = rearrange(xhat, 'b c h w -> (b h w) c')
        categorical_recons_loss = torch.nn.functional.cross_entropy(y_pred, y_true)

        # plot `x` and `xhat`
        r = np.random.rand()
        if self.training and r <= 0.05:
            x = x.cpu()
            xhat = xhat.detach().cpu()
            b = np.random.randint(0, x.shape[0])

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            custom_colors = ['C3', 'C2', 'C1', 'C0', '#D3D3D3']
            cmap = ListedColormap(custom_colors)
            plt.suptitle(f'ep_{self.current_epoch}')
            axes[0].imshow(x[b].argmax(dim=0), vmin=0, vmax=self.config['dataset']['n_categories'], cmap=cmap, interpolation='nearest')
            axes[0].invert_yaxis()
            axes[0].set_xticks([])
            axes[0].set_yticks([])

            axes[1].imshow(xhat[b].argmax(dim=0), vmin=0, vmax=self.config['dataset']['n_categories'], cmap=cmap, interpolation='nearest')
            axes[1].invert_yaxis()
            axes[1].set_xticks([])
            axes[1].set_yticks([])

            plt.tight_layout()
            if kind == 'x':
                wandb.log({"x vs xhat (training)": wandb.Image(plt)})
            elif kind == 'x_cond':
                wandb.log({"x_cond vs xhat_cond (training)": wandb.Image(plt)})
            plt.close()

        # plot histogram of z
        # r = np.random.rand()
        # if self.training and r <= 0.05:
        #     z = z.detach().cpu().flatten().numpy()
        #
        #     fig, ax = plt.subplots(1, 1, figsize=(5, 2))
        #     ax.hist(z, bins='auto')
        #     plt.tight_layout()
        #     if kind == 'x':
        #         wandb.log({"hist(z)": wandb.Image(plt)})
        #     elif kind == 'x_cond':
        #         wandb.log({"hist(z_cond)": wandb.Image(plt)})
        #     plt.close()

        return categorical_recons_loss, vq_loss, perplexity

    def training_step(self, batch, batch_idx):
        x, x_cond = batch
        categorical_recons_loss, vq_loss, perplexity = self.forward(x, kind='x')
        categorical_recons_loss_cond, vq_loss_cond, perplexity_cond = self.forward(x_cond, kind='x_cond')
        loss = categorical_recons_loss + vq_loss
        loss += categorical_recons_loss_cond + vq_loss_cond

        # lr scheduler
        sch = self.lr_schedulers()
        sch.step()

        # log
        loss_hist = {'categorical_recons_loss': categorical_recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     'categorical_recons_loss_cond': categorical_recons_loss_cond,
                     'vq_loss_cond': vq_loss_cond,
                     'perplexity_cond': perplexity_cond,
                     }
        loss_hist = {f'train/{k}': v for k, v in loss_hist.items()}
        self.log_dict(loss_hist)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, x_cond = batch
        categorical_recons_loss, vq_loss, perplexity = self.forward(x, kind='x')
        categorical_recons_loss_cond, vq_loss_cond, perplexity_cond = self.forward(x_cond, kind='x_cond')
        loss = categorical_recons_loss + vq_loss
        loss += categorical_recons_loss_cond + vq_loss_cond

        # log
        loss_hist = {'categorical_recons_loss': categorical_recons_loss,
                     'vq_loss': vq_loss,
                     'perplexity': perplexity,
                     'categorical_recons_loss_cond': categorical_recons_loss_cond,
                     'vq_loss_cond': vq_loss_cond,
                     'perplexity_cond': perplexity_cond,
                     }
        loss_hist = {f'val/{k}': v for k, v in loss_hist.items()}
        self.log_dict(loss_hist)
        return {'loss': loss}

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params': self.parameters(),
                                  'lr': self.config['trainer_params']['LR']['stage1']},
                                 ])
        return {'optimizer': opt, 'lr_scheduler': CosineAnnealingLR(opt, self.T_max)}
