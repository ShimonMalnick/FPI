import re

import math
from typing import Union, Dict, Callable
import torch
import numpy as np
from diffusers import DDIMScheduler
from matplotlib import pyplot as plt
from torch.distributions import Normal
from inversion import MyInvertFixedPoint


def compute_likelihood_to_bpd(likelihood: Union[torch.Tensor, float], num_pixels: int):
    """
    Compute the likelihood to bits per dimension
    :param likelihood: the  log likelihood of the image
    :param num_pixels: the number of pixels in the image (e.g. for a 64X64 RGB image, this is 64x64x3 = 12,288)
    :return: the likelihood in bits per dimension
    """
    nll = -1 * likelihood
    return (nll / num_pixels) / np.log(2)


def latent_to_bpd(latent: torch.FloatTensor, compute_batch: bool = True):
    """
    Compute the bits per dimension of a latent variable, w.r.t to the standard normal distribution
    :param compute_batch: whether to compute the bpd for a batch of latent variables
    :param latent: The latent varibale
    :return: the bits per dimension
    """
    standard_normal = Normal(0, 1)
    if compute_batch:
        assert latent.ndim > 1, "Latent must be a batch of latent variables"
        likelihood = standard_normal.log_prob(latent.cpu()).sum(dim=tuple(range(1, latent.dim())))
        n_element = latent[0].nelement()
    else:
        likelihood = standard_normal.log_prob(latent.detach().cpu()).sum()
        n_element = latent.nelement()
    nll = -1 * likelihood
    return (nll / n_element) / np.log(2)


def latent_to_image(latent, num_ddim_steps=10, pipe=None, save_path=None):
    GUIDANCE_SCALE = 2
    if pipe is None:
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = MyInvertFixedPoint.from_pretrained(
            model_id,
            scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
            safety_checker=None,
        ).to('cuda')
    image = pipe(prompt=[""], latents=latent, guidance_scale=GUIDANCE_SCALE,
                 num_inference_steps=num_ddim_steps, output_type='pil').images[0]
    if save_path:
        image.save(save_path)
    return image


def plot_bpd_histogram(data: Dict, save_path=None, show_fig=True, n_bins=None, **kwargs):
    """
    Plot the BPD of the given data. the data is expected as a dictionary of the form {label: array of bpd values}
    :param data: the data to plot
    :param save_path: the path to save the plot, leave None to not save
    :param show_fig: whether to show the figure
    :param n_bins: the number of bins to use in the histogram.
    """
    plt.clf()
    plt.figure(figsize=(10, 10))
    for label, values in data.items():
        plt.hist(values, label=label, bins=n_bins, alpha=0.5)
    plt.legend(loc='upper right')
    plt.xlabel("BPD")
    plt.ylabel("Count")
    if 'title' in kwargs and kwargs['title']:
        plt.title(kwargs['title'])
    if save_path:
        plt.savefig(save_path)
    if show_fig:
        plt.show()
    plt.close()


def sort_key_func_by_ordered_file_names(pattern_prefix: str) -> Callable[[str], int]:
    """
    Create a sorting key function that sorts by the number in the file name
    :param pattern_prefix:  the prefix of the pattern to match, right before the number. for example if the file name is
    "file_12.txt" and the pattern_prefix is "file_", the function will return 12
    :return: the sorting key function
    """
    return lambda s: int(re.match(f".*{pattern_prefix}(\d+).*", s).group(1))


