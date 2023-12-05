import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class GaussianModel(nn.Module):

    def __init__(self, model, img_size, img_channel, betas, loss_type):
        super().__init__()

        self.seed = 1
        self.img_size = img_size
        self.img_channel = img_channel
        self.model = model

        self.step = 0

        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")
        self.loss_type = loss_type

        #self.latent_chanel = args.latent_channel
        self.num_timesteps = len(betas)

        #保存加噪去噪过程中所需要的参数
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)  # alphas累乘

        to_torch = partial(torch.tensor, dtype=torch.float32)  #

        self.register_buffer("beta", to_torch(betas))  # variable in buffer is fixed
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

    # 前向过程, 让denoiser去噪
    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device

        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[0]:
            raise ValueError("image width does not match diffusion parameters")

        t = torch.randint(0, self.num_timesteps, (b,), device=device)
        return self.get_losses(x, t, y)


    def get_losses(self, x, t, y):
        torch.manual_seed(self.seed)
        self.seed += 1

        noise = torch.randn_like(x)  # generateing a noise with latent‘s shape

        perturbed_x = self.add_noise(x, t, noise)  # add noise
        estimated_noise = self.model(perturbed_x, t, y)  # estimate noise

        # self.sample(estimated_noise.shape[0],estimated_noise.device)

        #这里也可以改成直接预测x0
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
            # loss = F.l1_loss(estimated_noise, x)
        elif self.loss_type == "l2":
            loss = F.mse_loss(estimated_noise, noise)
            # loss = F.mse_loss(estimated_noise, x)
        return loss

    def add_noise(self, x, t, noise):
        out = extract(self.sqrt_alphas_cumprod, t, x.shape) * x + extract(self.sqrt_one_minus_alphas_cumprod,
                                                                                    t, x.shape) * noise
        return out

    def remove_noise(self, x, t, y):
        output = ((x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape))

        return output


    @torch.no_grad()
    def sample(self, batch_size, device, y=None):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.latent_size, device=device)
        print('x.shape', x.shape)

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

        return x.cpu().detach()

    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")

        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach()]

        for t in range(self.num_timesteps - 1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)

            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)

            diffusion_sequence.append(x.cpu().detach())

        return diffusion_sequence


def generate_linear_schedule(T, low, high):
    """
    :param T: 一共的加噪步数
    :param low: 最低的t
    :param high: 最高的t
    :return: 给每个batch添加不同的t
    """
    print("generate_linear_schedule")
    beta = np.linspace(low * 1000 / T, high * 1000 / T, T)
    return beta

def extract(a, t, latent_shape):
    """
    extract element from a, and reshape to (b,1,1,1,1...) 1's number is len(x_shape)-1
    a : sqrt_alphas_cumprod [1000]
    t : time_step
    latent_shape : latent shape

    Example:
        extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
    """
    b, *_ = t.shape  # b : batch size
    out = a.gather(-1, t)

    return out.reshape(b, *((1,) * (len(latent_shape) - 1)))  # 第一个*是展开的操作

