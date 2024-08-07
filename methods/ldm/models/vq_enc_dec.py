"""
reference: https://github.com/nadavbh12/VQ-VAE/blob/master/vq_vae/auto_encoder.py
"""
import numpy as np
import torch
import torch.nn as nn
import einops


# class ResBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
#         super(ResBlock, self).__init__()

#         if mid_channels is None:
#             mid_channels = out_channels

#         layers = [
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels, mid_channels,
#                       kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(),
#             nn.Conv2d(mid_channels, out_channels,
#                       kernel_size=1, stride=1, padding=0)
#         ]
#         if bn:
#             layers.insert(2, nn.BatchNorm2d(out_channels))
#         self.convs = nn.Sequential(*layers)

#     def forward(self, x):
#         return x + self.convs(x)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout:float=0.):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels
        
        kernel_size = (3, 3)
        padding = (1, 1)

        layers = [
            nn.GELU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=(1, 1), padding=padding),
        ]
        self.convs = nn.Sequential(*layers)
        self.proj = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.proj(x) + self.convs(x)
        return out
    

# class VQVAEEncBlock(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  ):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
#                       padding_mode='replicate'),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True))

#     def forward(self, x):
#         out = self.block(x)
#         return out
class VQVAEEncBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout:float=0.
                 ):
        super().__init__()
        
        kernel_size = (4, 4)
        padding = (1, 1)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=(2, 2), padding=padding,
                      padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out
    

def Upsample(dim_in, dim_out):
    """
    Better Deconvolution without the checkerboard problem [1].
    [1] https://distill.pub/2016/deconv-checkerboard/
    """
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim_in, dim_out, 3, padding = 1)
    )


# class VQVAEDecBlock(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  ):
#         super().__init__()
#         self.block = nn.Sequential(
#             # nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
#             Upsample(in_channels, out_channels),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True))

#     def forward(self, x):
#         out = self.block(x)
#         return out
class VQVAEDecBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout:float=0.
                 ):
        super().__init__()
        
        self.block = nn.Sequential(
            Upsample(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x):
        out = self.block(x)
        return out
    

# class VQVAEEncoder(nn.Module):
#     """
#     following the same implementation from the VQ-VAE paper.
#     """

#     def __init__(self,
#                  d: int,
#                  bottleneck_d: int,
#                  num_channels: int,
#                  downsample_rate: int,
#                  n_resnet_blocks: int,
#                  output_norm: bool,
#                  bn: bool = True,
#                  **kwargs):
#         """
#         :param d: hidden dimension size
#         :param num_channels: channel size of input
#         :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
#         :param n_resnet_blocks: number of ResNet blocks
#         :param bn: use of BatchNorm
#         :param kwargs:
#         """
#         super().__init__()
#         self.output_norm = output_norm
#         self.encoder = nn.Sequential(
#             VQVAEEncBlock(num_channels, d),
#             *[VQVAEEncBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
#             *[nn.Sequential(ResBlock(d, d, bn=bn), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
#             nn.Conv2d(d, bottleneck_d, kernel_size=1)
#         )

#         self.is_num_tokens_updated = False
#         self.register_buffer('num_tokens', torch.zeros(1).int())
#         self.register_buffer('H_prime', torch.zeros(1).int())
#         self.register_buffer('W_prime', torch.zeros(1).int())

#     def forward(self, x):
#         """
#         :param x: (B, C, H, W)
#         :return (B, C, H, W') where W' <= W
#         """
#         z = self.encoder(x)

#         if self.output_norm:
#             z = z / z.abs().max(dim=1, keepdim=True).values  # (b c h w'); normalize `z` to be within [-1, 1]

#         if not self.is_num_tokens_updated:
#             self.H_prime += z.shape[2]
#             self.W_prime += z.shape[3]
#             self.num_tokens += self.H_prime * self.W_prime
#             self.is_num_tokens_updated = True
#         return z

class VQVAEEncoder(nn.Module):
    def __init__(self,
                 num_channels: int,
                 init_dim:int,
                 hid_dim: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 output_norm: bool,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param bn: use of BatchNorm
        :param kwargs:
        """
        super().__init__()
        self.output_norm = output_norm

        d = init_dim
        enc_layers = [VQVAEEncBlock(num_channels, d, dropout=dropout),]
        d *= 2
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            enc_layers.append(VQVAEEncBlock(d//2, d, dropout=dropout))
            for _ in range(n_resnet_blocks):
                enc_layers.append(ResBlock(d, d, dropout=dropout))
            d *= 2
        enc_layers.append(ResBlock(d//2, hid_dim, dropout=dropout))
        self.encoder = nn.Sequential(*enc_layers)

        self.is_num_tokens_updated = False
        self.register_buffer('num_tokens', torch.tensor(0))
        self.register_buffer('H_prime', torch.tensor(0))
        self.register_buffer('W_prime', torch.tensor(0))
    
    def forward(self, x):
        """
        x: (b c h w)        
        """
        z = self.encoder(x)  # (b c h w)

        if self.output_norm:
            z = z / z.abs().max(dim=1, keepdim=True).values  # (b c h w'); normalize `z` to be within [-1, 1]

        if not self.is_num_tokens_updated:
            self.H_prime = torch.tensor(z.shape[2])
            self.W_prime = torch.tensor(z.shape[3])
            self.num_tokens = self.H_prime * self.W_prime
            self.is_num_tokens_updated = True
        return z
    

# class VQVAEDecoder(nn.Module):
#     """
#     following the same implementation from the VQ-VAE paper.
#     """

#     def __init__(self,
#                  d: int,
#                  bottleneck_d: int,
#                  out_channels: int,
#                  downsample_rate: int,
#                  n_resnet_blocks: int,
#                  img_size: int,
#                  **kwargs):
#         """
#         :param d: hidden dimension size
#         :param out_channels: channel size of output
#         :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
#         :param n_resnet_blocks: number of ResNet blocks
#         :param kwargs:
#         """
#         super().__init__()
#         pos_emb_dim = bottleneck_d
#         self.pos_emb = torch.nn.parameter.Parameter(torch.randn(pos_emb_dim, img_size // downsample_rate, img_size // downsample_rate))

#         self.decoder = nn.Sequential(
#             nn.Conv2d(bottleneck_d+pos_emb_dim, d, kernel_size=1),
#             # nn.Conv2d(bottleneck_d, d, kernel_size=1),
#             nn.Conv2d(d, d, kernel_size=3, padding=1),
#             *[nn.Sequential(ResBlock(d, d), nn.BatchNorm2d(d)) for _ in range(n_resnet_blocks)],
#             *[VQVAEDecBlock(d, d) for _ in range(int(np.log2(downsample_rate)) - 1)],
#             Upsample(d, out_channels),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#         )

#     def forward(self, x):
#         """
#         :param x: output from the encoder (B, C, H, W')
#         :return  (B, C, H, W)
#         """
#         pos_emb = einops.repeat(self.pos_emb, 'd h w -> b d h w', b=x.shape[0])
#         x = torch.cat((x, pos_emb), dim=1)

#         out = self.decoder(x)
#         return out

class VQVAEDecoder(nn.Module):
    """
    following the same implementation from the VQ-VAE paper.
    """

    def __init__(self,
                 num_channels: int,
                 init_dim:int,
                 hid_dim: int,
                 downsample_rate: int,
                 n_resnet_blocks: int,
                 dropout:float=0.3,
                 **kwargs):
        """
        :param d: hidden dimension size
        :param num_channels: channel size of input
        :param downsample_rate: should be a factor of 2; e.g., 2, 4, 8, 16, ...
        :param n_resnet_blocks: number of ResNet blocks
        :param kwargs:
        """
        super().__init__()
        
        d = int(init_dim * 2**(int(round(np.log2(downsample_rate))) - 1))  # enc_out_dim == dec_in_dim
        if round(np.log2(downsample_rate)) == 0:
            d = int(init_dim * 2**(int(round(np.log2(downsample_rate)))))

        dec_layers = [ResBlock(hid_dim, d, dropout=dropout)]
        for _ in range(int(round(np.log2(downsample_rate))) - 1):
            for _ in range(n_resnet_blocks):
                dec_layers.append(ResBlock(d, d, dropout=dropout))
            d //= 2
            dec_layers.append(VQVAEDecBlock(2*d, d, dropout=dropout))
        dec_layers.append(VQVAEDecBlock(d, num_channels))
        dec_layers.append(ResBlock(num_channels, num_channels))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        """
        x: (b c h` w`)
        """
        out = self.decoder(x)  # (b c h w)
        return out
