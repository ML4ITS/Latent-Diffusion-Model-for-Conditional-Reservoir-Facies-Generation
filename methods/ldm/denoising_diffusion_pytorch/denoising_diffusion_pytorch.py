import os
import math
import copy
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import einops
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, einsum
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from torchvision.utils import _log_api_usage_once, make_grid

from torch.optim import Adam
from lion_pytorch import Lion

from torchvision import transforms as T, utils

from einops import repeat, rearrange, reduce
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.version import __version__

from preprocessing.preprocess import DatasetImporter
from preprocessing.preprocess import GeoDataset as Dataset
from methods.ldm.modules.module_vqvae import quantize
from methods.ldm.modules.module_vqvae import ModuleVQVAE

# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model
class UnetCond(nn.Module):
    def __init__(
            self,
            in_channels,
            dim,
            init_dim=None,
            dim_mults=(1, 2, 4, 8),
            self_condition=False,
            resnet_block_groups=8,
    ):
        super(UnetCond, self).__init__()
        # determine dimensions

        self.channels = in_channels
        self.self_condition = self_condition
        input_channels = in_channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        m = 1
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(m * dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(m * dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(m * mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(m * mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(m * dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(m * dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))



class Unet(nn.Module):
    def __init__(
        self,
        in_channels,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        p_unconditional=0.1,
        z_size: int = None,
    ):
        super().__init__()
        self.net_cond = UnetCond(in_channels, dim, init_dim, dim_mults, self_condition, resnet_block_groups)

        # determine dimensions
        self.channels = in_channels
        self.self_condition = self_condition
        input_channels = in_channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        # time embeddings
        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        m = 2
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(m*dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(m*dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(m*mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(m*mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(m*dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(m*dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = in_channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        # mask token
        self.p_unconditional = p_unconditional
        self.mask_token = nn.Parameter(torch.randn((in_channels, z_size, z_size)))  # (d h' w')

    def forward(self, x, time, x_self_cond = None, x_cond=None):
        """
        Input:
            - x: == z_q for LDM with dim of (b d h' w')
            - cond_emb: (b d h' w')
        """
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        if (self.training and np.random.rand() < self.p_unconditional) or (x_cond == None):
            mask_token = einops.repeat(self.mask_token, 'd h w -> b d h w', b=x.shape[0])
            x_cond = mask_token

        # initial convolution
        x = self.init_conv(x)
        xc = self.net_cond.init_conv(x_cond)
        r = x.clone()

        # time conditioning
        t = self.time_mlp(time)

        # go through the layers of the unet, down and up
        h = []
        h_cond = []
        for (block1, block2, attn, downsample), (block1_c, block2_c, attn_c, downsample_c) in zip(self.downs, self.net_cond.downs):
            xc = block1_c(xc, t)
            x = torch.cat((x, xc), dim=1)
            x = block1(x, t)
            h.append(x)
            h_cond.append(xc)

            xc = block2_c(xc, t)
            x = torch.cat((x, xc), dim=1)
            x = block2(x, t)

            xc = attn_c(xc)
            x = attn(x)
            h.append(x)
            h_cond.append(xc)

            xc = downsample_c(xc)
            x = downsample(x)

        xc = self.net_cond.mid_block1(xc, t)
        x = torch.cat((x, xc), dim=1)
        x = self.mid_block1(x, t)

        xc = self.net_cond.mid_attn(xc)
        # x = torch.cat((x, xc), dim=1)
        x = self.mid_attn(x)

        xc = self.net_cond.mid_block2(xc, t)
        x = torch.cat((x, xc), dim=1)
        x = self.mid_block2(x, t)

        for (block1, block2, attn, upsample), (block1_c, block2_c, attn_c, upsample_c) in zip(self.ups, self.net_cond.ups):
            xc = torch.cat((xc, h_cond.pop()), dim=1)
            xc = block1_c(xc, t)
            x = torch.cat((x, h.pop(), xc), dim = 1)
            x = block1(x, t)

            xc = torch.cat((xc, h_cond.pop()), dim=1)
            xc = block2_c(xc, t)
            x = torch.cat((x, h.pop(), xc), dim = 1)
            x = block2(x, t)

            xc = attn(xc)
            x = attn(x)

            xc = upsample_c(xc)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        in_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_x0',
        beta_schedule = 'sigmoid',
        schedule_fn_kwargs = dict(),
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        auto_normalize = False,
        classifier_free_guidance_scale = 1.0,
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.in_size = in_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

        self.classifier_free_guidance_scale = classifier_free_guidance_scale

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        """
        q_posterior is defined by
            q(x_{t-1} | x_t, x_0) = N(x_{t-1}; \tilde{\mu}_t(x_t,x_0), \tilde{\beta}_t I)  ... Eq.(6)
        where
            \tilde{\mu}_t(x_t,x_0) := (..)x_0 + (..)x_t  ... Eq.(7); posterior_mean
        NB!
            x_{t-1} = E[q(x_{t-1} | x_t, x_0)]
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, z_cond=None, clip_x_start = False):
        if z_cond == None:
            # unconditional sampling
            model_output = self.model(x, t, x_self_cond, None)
        else:
            # conditional sampling
            if self.classifier_free_guidance_scale > 1.0:
                model_output_uncond = self.model(x, t, x_self_cond, None)
                model_output_cond = self.model(x, t, x_self_cond, z_cond)
                model_output = self.classifier_free_guidance_scale * (model_output_cond - model_output_uncond) + model_output_uncond
            else:
                model_output = self.model(x, t, x_self_cond, z_cond)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, z_cond=None, clip_denoised = True, dynamic_thresholding = False):
        assert np.sum([clip_denoised, dynamic_thresholding]) <= 1, "Only one of `clip_denoised` and `dynamic_thresholding` must be used, not both."

        preds = self.model_predictions(x, t, x_self_cond, z_cond)
        x_start = preds.pred_x_start  # x_0; (b d h' w')

        if clip_denoised:
            x_start.clamp_(-1., 1.)
        elif dynamic_thresholding:  # from the Imagen paper
            x_start_flat = einops.rearrange(x_start, 'b d h w -> b (d h w)')
            percentile = 0.95  # hyper-parameter
            s = torch.quantile(x_start_flat, percentile, dim=1)  # (b,)
            s = torch.nn.functional.relu(s - 1) + 1  # same as max(s, 1); (b,)
            s = s[:,None,None,None]  # (b 1 1 1)
            x_start = torch.clip(x_start, min=-s, max=s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, z_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x,
                                                                          t = batched_times,
                                                                          x_self_cond = x_self_cond,
                                                                          z_cond=z_cond)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, z_cond=None, return_all_timesteps = False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, self_cond, z_cond)
            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def ddim_sample(self, shape, return_all_timesteps = False):
        assert False, "not ready yet for GeoDiffusion."
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device = device)
        imgs = [img]

        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(img, time_cond, self_cond, clip_x_start = True)

            if time_next < 0:
                img = x_start
                imgs.append(img)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            imgs.append(img)

        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    @torch.no_grad()
    def sample(self, z_cond=None, batch_size = 16, return_all_timesteps = False):
        in_size, channels = self.in_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(shape=(batch_size, channels, in_size, in_size), z_cond=z_cond, return_all_timesteps = return_all_timesteps)

    @torch.no_grad()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, z_cond, noise = None, return_pred=False):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, t, x_self_cond, z_cond)
        # model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # diffusion loss
        diff_loss = self.loss_fn(model_out, target, reduction = 'none')
        diff_loss = reduce(diff_loss, 'b ... -> b (...)', 'mean')
        diff_loss = diff_loss * extract(self.p2_loss_weight, t, diff_loss.shape)
        diff_loss = diff_loss.mean()

        # # preservation loss
        # assert self.objective == 'pred_x0', "preservation loss is only available when `self.objective == 'pred_x0'`."
        # model_out, _, _, _ = quantize(model_out, )
        # preserv_loss = F.l1_loss(input=..., target=...)
        #
        # # loss
        # loss = ...

        if return_pred:
            return diff_loss, model_out
        else:
            return diff_loss

    def forward(self, z_q, z_cond, *args, **kwargs):
        b, c, h, w, device, img_size, = *z_q.shape, z_q.device, self.in_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        z_q = self.normalize(z_q)
        z_cond = self.normalize(z_cond)
        return self.p_losses(z_q, t, z_cond, *args, **kwargs)

# trainer class

@torch.no_grad()
def save_image(
    X_cond,
    Xhat: torch.FloatTensor,
    fp,
    in_channels: int,
    step: int,
    wandb_log: bool,
) -> None:
    """
    :param X_cond (b c+1 h w)
    :param Xhat (b 1 h w)
    :param fp: file name for the saved image.
    """
    n_samples = Xhat.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_rows * 3, figsize=(12*3, 12))
    # axes = axes.flatten()

    X_cond = torch.flip(X_cond, dims=(2,))  # (b c+1 h w)
    cond_loc = (X_cond[:, [-1], :, :] != 1).numpy().astype(int)  # (b 1 h w); the last channel corresponds to masking
    X_cond = X_cond.argmax(dim=1).float().numpy()  # (b h w)
    Xhat = np.flip(Xhat.numpy(), axis=2)  # (b 1 h w)
    Xhat = Xhat[:, 0, :, :]  # (b h w)

    # color range
    sample_idx = 0
    for i in range(n_rows):
        for j in range(n_rows):
            x_cond = X_cond[sample_idx]  # (h w)
            cond_loc_ = cond_loc[sample_idx, 0, :, :]  # (h w)

            axes[i, j].imshow(x_cond, interpolation='nearest', vmin=0, vmax=in_channels, cmap='Accent')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            # xhat
            xhat = Xhat[sample_idx]  # (h w)
            axes[i, n_rows+j].imshow(xhat, interpolation='nearest', vmin=0, vmax=in_channels, cmap='Accent')
            axes[i, n_rows+j].set_xticks([])
            axes[i, n_rows+j].set_yticks([])

            # diff(x, xhat)
            # cond_loc = (x_cond != in_channels).astype(int)  # (h w)
            # diff = np.abs(cond_loc - (xhat * cond_loc))  # (h w)
            diff = np.abs((x_cond * cond_loc_) - (xhat * cond_loc_))  # (h w)
            diff = np.clip(diff, a_min=0, a_max=1)
            axes[i, 2*n_rows + j].imshow(diff, interpolation='nearest', vmin=0, vmax=1, cmap='binary')
            axes[i, 2*n_rows + j].set_xticks([])
            axes[i, 2*n_rows + j].set_yticks([])

            sample_idx += 1
    plt.suptitle(f'step-{step}')
    plt.tight_layout()
    plt.savefig(fp)
    if wandb_log:
        wandb.log({'sample from x_cond_test': wandb.Image(plt)})
    plt.close()


@torch.no_grad()
def save_image_unconditional(
    Xhat: torch.FloatTensor,
    fp,
    in_channels: int,
) -> None:
    """
    :param X_cond (b 1 h w)
    :param Xhat (b 1 h w)
    :param fp: file name for the saved image.
    """
    n_samples = Xhat.shape[0]
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_rows * 1, figsize=(12*1, 12))
    # axes = axes.flatten()

    Xhat = np.flip(Xhat.numpy(), axis=2)  # (b 1 h w)
    Xhat = Xhat.squeeze()  # (b h w)

    # color range
    # n_colors = len(np.unique(Xhat))
    sample_idx = 0
    for i in range(n_rows):
        for j in range(n_rows):
            # xhat
            xhat = Xhat[sample_idx]  # (h w)
            axes[i, j].imshow(xhat, interpolation='nearest', vmin=0, vmax=in_channels, cmap='Accent')
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

            sample_idx += 1
    plt.tight_layout()
    plt.savefig(fp)
    plt.close()

class Trainer(object):
    def __init__(
        self,
        train_data_loader,
        test_data_loader,
        diffusion_model,
        config: dict,
        module_vqvae: ModuleVQVAE,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        saved_model_folder='./saved_models',
        results_folder = './results',
        amp = False,
        fp16 = False,
        use_lion = False,
        split_batches = True,
        convert_image_to = None,
        preserv_loss_weight=1,  # to strongly encourage the preservation loss
    ):
        super().__init__()
        self.config = config
        self.accelerator = Accelerator(
            device_placement=False,
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.state.device = torch.device(config['trainer_params']['gpu_idx'])

        self.accelerator.native_amp = amp

        self.pretrained_encoder = module_vqvae.encoder
        self.pretrained_encoder_cond = module_vqvae.encoder_cond
        self.pretrained_decoder = module_vqvae.decoder
        self.pretrained_vq = module_vqvae.vq_model

        self.pretrained_encoder.eval()
        self.pretrained_encoder_cond.eval()
        self.pretrained_decoder.train()
        self.pretrained_vq.train()

        self.model = diffusion_model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.in_size = diffusion_model.in_size

        # dataset and dataloader
        # dataset_importer = DatasetImporter(**config['dataset'])
        # self.ds = Dataset(dataset_type, dataset_importer)
        # self.dl = DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=config['dataset']['num_workers'])
        #
        # self.dl = self.accelerator.prepare(self.dl)
        # self.dl = cycle(self.dl)
        self.dl = train_data_loader
        self.ds = test_data_loader.dataset
        self.dl = self.accelerator.prepare(self.dl)
        self.dl = cycle(self.dl)

        # optimizer
        optim_klass = Lion if use_lion else Adam
        # self.opt = optim_klass(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.opt = optim_klass([{'params': diffusion_model.parameters(), 'lr': train_lr},
                                {'params': self.pretrained_vq.parameters(), 'lr': train_lr},
                                {'params': self.pretrained_decoder.parameters(), 'lr': train_lr},
                                ],
                               betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.saved_model_folder = Path(saved_model_folder)
        self.saved_model_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        self.preserv_loss_weight = preserv_loss_weight

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'encoder': self.pretrained_encoder.state_dict(),
            'encoder_cond': self.pretrained_encoder_cond.state_dict(),
            'vq': self.pretrained_vq.state_dict(),
            'decoder': self.pretrained_decoder.state_dict(),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__
        }

        torch.save(data, str(self.saved_model_folder / f'stage2-{milestone}.ckpt'))

    def load(self, ckpt_fname: str):
        accelerator = self.accelerator
        device = accelerator.device

        # data = torch.load(str(self.saved_model_folder / f'stage2-{milestone}.ckpt'), map_location=device)
        data = torch.load(str(self.saved_model_folder / ckpt_fname), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.pretrained_encoder.load_state_dict(data['encoder'])
        self.pretrained_encoder_cond.load_state_dict(data['encoder_cond'])
        self.pretrained_vq.load_state_dict(data['vq'])
        self.pretrained_decoder.load_state_dict(data['decoder'])

        self.pretrained_encoder.eval()
        self.pretrained_encoder_cond.eval()

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = torch.device(self.config['trainer_params']['gpu_idx'])

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, x_cond = next(self.dl)  # I assume (b c h w)
                    x, x_cond = x.to(device), x_cond.to(device)

                    z = self.pretrained_encoder(x)  # (b d h' w')
                    z_cond = self.pretrained_encoder_cond(x_cond)  # (b c h' w')

                    with self.accelerator.autocast():
                        # diffusion loss
                        diff_loss, z_hat = self.model(z, z_cond, return_pred=True)

                        # preservation loss
                        self.pretrained_vq.train()
                        self.pretrained_decoder.train()

                        zq_hat, _, vq_loss, _ = quantize(z_hat, self.pretrained_vq)
                        x_hat = self.pretrained_decoder(zq_hat)  # (b c h w)
                        cond_loc = (x_cond[:,[-1],:,:] != 1)  # (b 1 h w); the last channel corresponds to masking
                        c = x_hat.shape[1]
                        cond_loc_ = rearrange(repeat(cond_loc, 'b 1 h w -> b c h w', c=c), 'b c h w -> b h w c')
                        xhat_cond = rearrange(rearrange(x_hat, 'b c h w -> b h w c')[cond_loc_], '(n c) -> n c', c=c)
                        x_cond_argmax = x_cond.argmax(dim=1, keepdim=True)[cond_loc]
                        preserv_loss = F.cross_entropy(input=xhat_cond, target=x_cond_argmax)

                        # fig, axes = plt.subplots(1, 2)
                        # b = 0
                        # axes[0].imshow(x_cond_argmax.detach().cpu().numpy()[b])
                        # axes[1].imshow(xhat_cond.argmax(1).detach().cpu().numpy()[b])
                        # plt.show()

                        # reconstruction loss
                        y_true = x.argmax(dim=1)  # (b h w)
                        y_true = y_true.flatten()  # (bhw)
                        y_pred = rearrange(x_hat, 'b c h w -> (b h w) c')  # (bhw c)
                        categorical_recons_loss = F.cross_entropy(y_pred, y_true)

                        # (a), (c)
                        # `preserv_loss`: preserves the conditional information in generated samples.
                        # `(vq_loss['loss'] + categorical_recons_loss)` preserves the ability of the decoder.
                        loss = diff_loss + \
                               self.preserv_loss_weight * preserv_loss + \
                               (vq_loss['loss'] + categorical_recons_loss)

                        # (b)
                        # loss = diff_loss + \
                        #        (vq_loss['loss'] + categorical_recons_loss)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')
                wandb.log({'loss': loss,
                           'diff_loss': diff_loss,
                           'preserv_loss': preserv_loss,
                           'vq_loss': vq_loss['loss'],
                           'categorical_recons_loss': categorical_recons_loss,
                           })

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        # produce multiple synthetic samples
                        self.ema.ema_model.eval()
                        self.pretrained_vq.eval()
                        self.pretrained_decoder.eval()

                        with torch.no_grad():
                            # z_cond
                            X_cond = []
                            for _ in range(self.num_samples):
                                i = np.random.choice(len(self.ds))
                                x, x_cond = self.ds[i]  # (c h w)
                                X_cond.append(x_cond.numpy())
                            X_cond = torch.from_numpy(np.array(X_cond))  # (b c h w); b == num_samples
                            z_cond = self.pretrained_encoder_cond(X_cond.to(device))  # (b c h' w')

                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(self.num_samples, self.batch_size)
                            all_zs_list = list(map(lambda n: self.ema.ema_model.sample(z_cond, batch_size=n), batches))

                            all_zs = torch.cat(all_zs_list, dim = 0)  # (b d h' w')

                            # apply vq
                            all_zqs, _, _, _ = quantize(all_zs, self.pretrained_vq)  # z_q: (b d h' w')

                            # apply decoder
                            all_images = self.pretrained_decoder(all_zqs)  # (b c h w)
                            all_images = all_images.cpu().detach()
                            all_images = all_images.argmax(dim=1)[:, None, :, :].float()  # (b 1 h w)

                        # save
                        save_image(X_cond, all_images, str(self.results_folder / f'sample-{milestone}.png'), self.config['dataset']['in_channels'], self.step, True)
                        self.save(milestone)  # save model

                        # del X_cond, z_cond, all_zs, all_zqs, all_images
                        # torch.cuda.empty_cache()

                pbar.update(1)

        accelerator.print('training complete')
