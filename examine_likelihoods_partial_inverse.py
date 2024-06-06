"""
This file will include the experiment to examine the likelihoods of the partial inverse images. this means that we will
use that for an FPI inversion that runs for X steps, we will examine the likelihood of latents along the process.
For example, the first experiment is to run FPI for 50 steps and store the latent achieved at step 25. Then we wish to
run the entire process again on this latent, and compare its difference from the result in the original run.
"""
import sys
from glob import glob
from typing import Union, List, Tuple

import math
from PIL import Image
from easydict import EasyDict

from datasets import CocoCaptions17, ChestXRay, NormalDistributedDataset, NoisedCocoCaptions17, AugmentedDataset
import os
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from inversion import MyInvertFixedPoint
from torch.distributions import Normal
import logging
from utils import latent_to_bpd, latent_to_image, plot_bpd_histogram, sort_key_func_by_ordered_file_names
from p2p.ptp_utils import plot_images
from time import time
from setup import setup_config
from datasets import ImageNetSubset
from torch.utils.data import DataLoader, Subset
from configs import BaseDatasetConfig, CocoDatasetConfig, ChestXRayDatasetConfig, ImageNetSubsetDatasetConfig, \
    NormalDistributedDatasetDatasetConfig, NoisedCocoCaptions17DatasetConfig, AugmentedDatasetConfig
from torchvision import transforms


@torch.no_grad()
def get_all_latents(guidance_scale, caption, images, middle_latent_step, num_ddim_steps, num_iter_fixed_point,
                    inversion_pipe, img2img_pipe):
    new_latents = img2img_pipe(prompt=caption, image=images, strength=0.05, guidance_scale=guidance_scale,
                               output_type="latent").images
    # RUN FP inversion on vae_latent
    inverse_results = inversion_pipe.invert(caption, latents=new_latents, num_inference_steps=num_ddim_steps,
                                            guidance_scale=guidance_scale, num_iter=num_iter_fixed_point,
                                            return_specific_latent=middle_latent_step)
    original_results_latent = inverse_results.latents
    halfway_latent = inverse_results.specific_latent
    halfway_result_latent = inversion_pipe.invert(caption, latents=halfway_latent, num_inference_steps=num_ddim_steps,
                                                  guidance_scale=guidance_scale, num_iter=num_iter_fixed_point).latents
    return halfway_latent, halfway_result_latent, original_results_latent


@torch.no_grad()
def run_experiment(config: BaseDatasetConfig, save_images=False, timeit=False):
    ds = get_ds(config)
    os.makedirs(config.save_dir, exist_ok=True)

    print(f"Saving results to {config.save_dir}")
    os.makedirs(config.save_dir, exist_ok=True)
    torch.manual_seed(8888)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inversion_pipe, img2img_pipe = get_pipelines(device)

    standard_normal = Normal(0, 1)

    dataloader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=16)
    for i, batch in enumerate(dataloader):
        if timeit:
            cur_time = time()
        images = batch[0].to(device)
        caption = [""] * images.shape[0]  # performing everything with null text to neutralize the effect of the prompt
        halfway_latent, halfway_result_latent, original_results_latent = get_all_latents(config.GUIDANCE_SCALE, caption,
                                                                                         images,
                                                                                         config.middle_latent_step,
                                                                                         config.num_ddim_steps,
                                                                                         config.num_iter_fixed_point,
                                                                                         inversion_pipe, img2img_pipe)

        output = create_output_dict(config, halfway_latent, halfway_result_latent,
                                    original_results_latent, standard_normal)

        if isinstance(config, ImageNetSubsetDatasetConfig):
            output['class'] = batch[1]

        if save_images:
            output['images'] = images
        print_str = f"finished batch {i + 1}/{len(dataloader)}"
        if timeit:
            print_str += f"in {(time() - cur_time)} seconds"
        print(print_str)

        torch.save(output, os.path.join(config.save_dir, f"batch_{i}.pt"))


def get_ds(config: BaseDatasetConfig):
    if isinstance(config, CocoDatasetConfig):
        ds = CocoCaptions17(transform=config.transform)
    elif isinstance(config, ChestXRayDatasetConfig):
        ds = ChestXRay(transform=config.transform)
    elif isinstance(config, NormalDistributedDatasetDatasetConfig):
        ds = NormalDistributedDataset(ds_size=config.ds_size, transform=config.transform)
    elif isinstance(config, ImageNetSubsetDatasetConfig):
        ds = ImageNetSubset(split=config.split, num_classes=config.num_classes,
                            num_images_per_class=config.num_images_per_class, transform=config.transform)
    elif isinstance(config, NoisedCocoCaptions17DatasetConfig):
        ds = NoisedCocoCaptions17(transform=config.transform, num_images_before_noise=config.num_images_before_noise,
                                  num_noise_levels=config.num_noise_levels, noise_multiplier=config.noise_multiplier)
    elif isinstance(config, AugmentedDatasetConfig):
        ds_config = config.dataset_config
        assert not isinstance(ds_config, AugmentedDatasetConfig)
        inner_ds = get_ds(ds_config)
        ds = AugmentedDataset(augmentations=config.augmentations, dataset=inner_ds, transform=config.transform)
    else:
        raise NotImplementedError(f"Dataset type is not supported")

    if config.dataset_indices is not None:
        ds = Subset(ds, range(*config.dataset_indices))
    return ds


def get_pipelines(device: torch.device):
    model_id = "CompVis/stable-diffusion-v1-4"
    inversion_pipe = MyInvertFixedPoint.from_pretrained(
        model_id,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", local_files_only=True),
        local_files_only=True,
        safety_checker=None,
    ).to(device)
    img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
                                                                  scheduler=DDIMScheduler.from_pretrained(model_id,
                                                                                                          subfolder="scheduler",
                                                                                                          local_files_only=True),
                                                                  local_files_only=True,
                                                                  ).to(device)
    return inversion_pipe, img2img_pipe


def create_output_dict(config: BaseDatasetConfig, halfway_latent, halfway_result_latent, original_results_latent,
                       standard_normal):
    # we examine the latents log-likelihood compared to standard normal distribution R.V,
    # as this is the distribution used for latent diffusion model
    # (see https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L166)
    output = {"original_likelihood": standard_normal.log_prob(original_results_latent).sum().item(),
              "original_bpd": latent_to_bpd(original_results_latent),
              "original_latent": original_results_latent,
              "halfway_bpd": latent_to_bpd(halfway_result_latent),
              "halfway_latent": halfway_latent,
              "halfway_latent_likelihood": standard_normal.log_prob(halfway_latent).sum().item(),
              "halfway_latent_run_again": halfway_result_latent,
              "halfway_latent_run_again_likelihood": standard_normal.log_prob(halfway_result_latent).sum().item(),
              "config": config.to_dict(),
              }
    return output


def plot_bpd_histograms_halfway_latents(experiment_dir, n_samples_per_data=50, save_path=None):
    files = os.listdir(experiment_dir)
    original_latents = [torch.load(os.path.join(experiment_dir, f))['original_latent'] for f in files][
                       :n_samples_per_data]
    halfway_latents = [torch.load(os.path.join(experiment_dir, f))['halfway_latent'] for f in files][
                      :n_samples_per_data]
    halfway_latents_run_again = [torch.load(os.path.join(experiment_dir, f))['halfway_latent_run_again'] for f in
                                 files][
                                :n_samples_per_data]
    latents_w_names = [(original_latents, "original_latent"), (halfway_latents, "halfway_latent"),
                       (halfway_latents_run_again, "halfway_latent_run_again")]
    data = []
    names = []
    for latents, name in latents_w_names:
        data.append([latent_to_bpd(latent) for latent in latents])
        names.append(name)
    assert all([len(a) == n_samples_per_data for a in data])
    plot_bpd_histogram({d: n for d, n in zip(data, names)}, save_path=save_path)


def save_latents_images(exp_dir, num_images, num_ddim_steps=10, save_path="results_partial_inverse/coco/images"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = MyInvertFixedPoint.from_pretrained(
        model_id,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
        safety_checker=None,
    ).to('cuda')

    for i in range(num_images):
        original_latent = torch.load(f"{exp_dir}/sample_{i}.pt")['original_latent']
        halfway_latent = torch.load(f"{exp_dir}/sample_{i}.pt")['halfway_latent']
        halfway_latent_again = torch.load(f"{exp_dir}/sample_{i}.pt")['halfway_latent_run_again']
        image_original_latent = latent_to_image(original_latent, num_ddim_steps, pipe=pipe,
                                                save_path=f"{save_path}/original_latent_{i}.png")
        image_halfway_latent = latent_to_image(halfway_latent, num_ddim_steps, pipe=pipe,
                                               save_path=f"{save_path}/halfway_latent_{i}.png")
        image_halfway_latent_again = latent_to_image(halfway_latent_again, num_ddim_steps, pipe=pipe,
                                                     save_path=f"{save_path}/halfway_latent_run_again_{i}.png")
        image_original = torch.load(f"{exp_dir}/sample_{i}.pt")['image']
        image_original.save(f"{save_path}/original_image_{i}.png")
        plot_images([image_original_latent, image_halfway_latent, image_halfway_latent_again, image_original],
                    num_rows=1, num_cols=4,
                    titles=["Full Process Latent", "Halfway Latent", "Halfway Latent Run Again", "Original Image"],
                    save_fig_path=f"{save_path}/compare_{i}", show_fig=False)


def run_experiment_on_multiple_datasets(dsets=None, exp_dir_base=None):
    if dsets is None:
        dsets = ['coco', 'chest_x_ray', 'random_normal']
    if exp_dir_base is None:
        exp_dir_base = f"{setup_config['REPO_BASE']}/results_fpi_5_iters"
    for ds_type in dsets:
        cur_exp_dir = f"{exp_dir_base}/{ds_type}_test"
        save_latents_images(cur_exp_dir, save_path=f"{cur_exp_dir}/images", num_images=5)
        run_experiment(num_images=5, num_ddim_steps=10, save_dir=cur_exp_dir, save_images=True, ds_type=ds_type,
                       num_iter_fixed_point=5, timeit=True)


def run_grayscale_duplicated_coco(output_path=None):
    if output_path is None:
        output_path = f"{setup_config['OUTPUT_ROOT']}/results_grayscale_coco"
    transform = transforms.Compose([lambda image: Image.open(image).convert('L').resize((512, 512)).convert('RGB'),
                                    transforms.ToTensor()])
    config = CocoDatasetConfig(save_dir=output_path, transform=transform)
    run_experiment(config)


def run_imagenet_on_different_devices(num_images=10000, num_devices=4):
    cur_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ds_indices = (cur_idx * num_images // num_devices, (cur_idx + 1) * num_images // num_devices)
    config = ImageNetSubsetDatasetConfig(
        save_dir=f"{setup_config['OUTPUT_ROOT']}/results_imagenet_subset_{ds_indices[0]}_to_{ds_indices[1]}",
        dataset_indices=ds_indices)
    run_experiment(config=config)


def plot_noisy_coco_images(num_images=10, images_per_row=5, save_dir=None, titles=None, show_fig=False,
                           num_noise_levels=None, save_path=None, coco_ds_images_indices=None):
    if save_dir is None and save_path is None:
        save_dir = f"{setup_config['OUTPUT_ROOT']}/results_coco_noised/images"
        os.makedirs(save_dir, exist_ok=True)
    if save_path:
        assert num_images == 1, "save_path is only supported for a single image, otherwise it will be overwritten"
    config = NoisedCocoCaptions17DatasetConfig()
    ds = get_ds(config)
    if num_noise_levels is None:
        num_noise_levels = config.num_noise_levels
    if titles is None:
        titles = [f"Var = {j + 1}" for j in range(num_noise_levels)]
    indices = coco_ds_images_indices if coco_ds_images_indices is not None else range(num_images)
    for i in indices:
        if not save_path:
            save_path = f"{save_dir}/grid_{i}.png" if save_dir else None
        plot_images([ds[(i * num_noise_levels) + j][0] for j in range(num_noise_levels)],
                    num_rows=num_noise_levels // images_per_row, num_cols=images_per_row,
                    titles=titles, show_fig=show_fig,
                    save_fig_path=save_path)


def run_augmented_dataset():
    inner_config = NoisedCocoCaptions17DatasetConfig(save_dir=None, dataset_indices=(0, 100))
    config = AugmentedDatasetConfig(dataset_config=inner_config,
                                    save_dir=f"{setup_config['OUTPUT_ROOT']}/results_augmented")
    run_experiment(config=config, save_images=True)


def create_imagenet_plot(base_dir="/home/shimon/research/diffusion_inversions/results/results_imagenet",
                         save_path='/home/shimon/research/diffusion_inversions/results/results_imagenet/bpd_hist.png',
                         num_class_to_show=None):
    cache_file_path = f"{base_dir}/labels2bpds.pt"
    if not os.path.isfile(cache_file_path):
        subsets_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        all_paths = [os.path.join(base_dir, sub, p) for sub in subsets_dirs for p in
                     os.listdir(os.path.join(base_dir, sub))]
        config = torch.load(all_paths[0])['config']
        ds = get_ds(ImageNetSubsetDatasetConfig(num_classes=config['num_classes'],
                                                num_images_per_class=config['num_images_per_class']))
        idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
        labels2bpds = {}
        for path in all_paths:
            data = torch.load(path)
            cur_batch_labels = data['class'].tolist()
            for label in cur_batch_labels:
                if label not in labels2bpds:
                    labels2bpds[label] = []
                # here we append the bpd of the original latent of each entry in the batch, one by one
                labels2bpds[label].append(latent_to_bpd(data['original_latent'], compute_batch=False))
        labels2bpds = {(k, idx_to_class[k]): v for k, v in labels2bpds.items()}
        torch.save(labels2bpds, cache_file_path)
    else:
        labels2bpds = torch.load(cache_file_path)

    labels2bpds = {k: v for k, v in list(labels2bpds.items())}
    if num_class_to_show is not None:
        labels2bpds = {k: v for k, v in list(labels2bpds.items())[:num_class_to_show]}

    # remove the label number from the dict keys
    labels2bpds = {k[1]: torch.stack(v) for k, v in labels2bpds.items()}
    plot_bpd_histogram(labels2bpds, save_path=save_path)

    return labels2bpds


def plot_multiple_sources_bpd_histogram(data_base_dirs: List[Union[str, Tuple[str, str]]], save_path: str = None,
                                        n_samples: int = 500, **kwargs):
    """
    Plot the BPD histograms of the given data.
    :param data_base_dirs: the data is expected as a list of either:
     1. directories, each containing the path to the directory containing results of the experiment
     2. tuples of (directories, labels), where directories is the same as in 1, and labels is the label to give to the
        data in the plot
    :param save_path: the path to save the plot, leave None to not save
    :param n_samples: the number of samples to take from each data directory. make sure that each directory contains
    at least n_samples samples
    """
    plot_data = {}
    assert data_base_dirs, "data_base_dirs is empty"
    if isinstance(data_base_dirs[0], str):
        names = data_base_dirs
    elif isinstance(data_base_dirs[0], tuple):
        names = [n for _, n in data_base_dirs]
        data_base_dirs = [d for d, _ in data_base_dirs]
    else:
        raise ValueError(f"Invalid data_base_dirs type: {type(data_base_dirs[0])}")
    for d, name in zip(data_base_dirs, names):
        cur_samples = [torch.load(p) for p in glob(f"{d}/batch_*.pt")]
        cur_bpds = torch.cat([latent_to_bpd(s['original_latent'], compute_batch=True) for s in cur_samples])
        assert cur_bpds.ndim == 1 and cur_bpds.nelement() >= n_samples, f"cur_bpds shape is {cur_bpds.shape} and " \
                                                                        f"n_samples is {n_samples}"
        cur_bpds = cur_bpds[:n_samples]
        plot_data[name] = cur_bpds
    plot_bpd_histogram(plot_data, save_path=save_path, **kwargs)


def plot_bpd_coco_noised_images(
        base_dir_noised="/home/shimon/research/diffusion_inversions/results/results_coco_noised",
        save_path="/home/shimon/research/diffusion_inversions/results/coco_noised_bpd_hist.png",
        use_only_every_nth_noise=None):
    all_batches_noised = glob(f"{base_dir_noised}/batch_*.pt")
    sorted_batches_paths = sorted(all_batches_noised, key=sort_key_func_by_ordered_file_names("batch_"))
    sorted_batches = [torch.load(p) for p in sorted_batches_paths]
    config = EasyDict(sorted_batches[0]['config'])

    # divide images according to noise_levels
    data = {i: [] for i in range(config.num_noise_levels)}

    global_idx = 0
    for batch in sorted_batches:
        cur_batch_bpds = latent_to_bpd(batch['original_latent'], compute_batch=True)
        for in_batch_idx in range(cur_batch_bpds.shape[0]):
            data[global_idx % config.num_noise_levels].append(cur_batch_bpds[in_batch_idx])
            global_idx += 1

    if use_only_every_nth_noise is not None:
        data = {k: v for k, v in data.items() if k % use_only_every_nth_noise == 0}
    data = {f"Var={(k + 1) * config.noise_multiplier}": v for k, v in data.items()}
    plot_bpd_histogram(data, title='BPD histogram of noised coco images', save_path=save_path)


def plot_noised_coco_images_with_bpd(num_images=5,
                                     base_dir_noised="/home/shimon/research/diffusion_inversions/results/results_coco_noised",
                                     save_dir="/home/shimon/research/diffusion_inversions/results/images_coco_noised/with_bpd"):
    config = torch.load(f"{base_dir_noised}/batch_0.pt")['config']
    num_noise_levels = config['num_noise_levels']
    batch_size = config['batch_size']
    num_batches = math.ceil((num_images * num_noise_levels) / batch_size)
    batches_data = [torch.load(f"{base_dir_noised}/batch_{i}.pt") for i in range(num_batches)]
    bpds = torch.cat([latent_to_bpd(b['original_latent'], compute_batch=True) for b in batches_data])
    bpds = bpds[:num_images * num_noise_levels]
    for i in range(num_images):
        titles = [f"Var = {j + 1}, BPD={round(bpds[j].item(), 3)}" for j in range(num_noise_levels)]
        bpds = bpds[num_noise_levels:]
        plot_noisy_coco_images(num_images=1, coco_ds_images_indices=[i], titles=titles, show_fig=False,
                               save_path=f"{save_dir}/grid_{i}.png")


if __name__ == '__main__':
    # set logging level to info
    torch.manual_seed(8888)
    logging.basicConfig(level=logging.INFO)
