from __future__ import print_function

import os
import sys

import numpy as np
import torch

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

import multiprocessing
multiprocessing.set_start_method('spawn', True)


from scipy.stats import truncnorm

def truncated_noise_sample(batch_size=1, dim_z=100, truncation=1., seed=None):
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def mean_latent(n_latent, style_dim, device):
    latent_in = torch.randn(
        n_latent, style_dim, device=device
    )
    latent = latent_in.mean(0, keepdim=True)
    return latent


def truncate(truncation, truncation_latent, noise):
    if truncation < 1:
        truncated_noise = truncation_latent + truncation * (noise - truncation_latent)
    return truncated_noise



