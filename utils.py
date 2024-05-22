from typing import Union
import torch
import numpy as np
from torch.distributions import Normal


def compute_likelihood_to_bpd(likelihood: Union[torch.Tensor, float], num_pixels: int):
    """
    Compute the likelihood to bits per dimension
    :param likelihood: the  log likelihood of the image
    :param num_pixels: the number of pixels in the image (e.g. for a 64X64 RGB image, this is 64x64x3 = 12,288)
    :return: the likelihood in bits per dimension
    """
    nll = -1 * likelihood
    return (nll / num_pixels) / np.log(2)


def latent_to_bpd(latent: torch.FloatTensor):
    """
    Compute the bits per dimension of a latent variable, w.r.t to the standard normal distribution
    :param latent: The latent varibale
    :return: the bits per dimension
    """
    standard_normal = Normal(0, 1)
    likelihood = standard_normal.log_prob(latent.detach().cpu()).sum()
    nll = -1 * likelihood
    return (nll / latent.nelement()) / np.log(2)

