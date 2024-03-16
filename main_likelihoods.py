import os
from glob import glob
from typing import List
import numpy as np
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from inversion import MyInvertFixedPoint
from torch.distributions import Normal
import logging
from datasets import CocoCaptions17, ChestXRay, NormalDistributedDataset, FolderDataset
from dataclasses import dataclass
import shutil


def run_coco(save_dir=None, num_ddim_step=50):
    coco_ds = CocoCaptions17()
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", "coco_inverse_latents")
    run(coco_ds, save_dir, num_ddim_step=num_ddim_step)


def run_chest_x_ray(save_dir=None, num_ddim_step=50):
    chest_x_ray_dataset = ChestXRay()
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", "chest_x_rays")
    run(chest_x_ray_dataset, save_dir, dataset_indices=(0, 50), num_ddim_step=num_ddim_step)


def run_normal_distributed_images(save_dir=None, num_ddim_step=50):
    normal_dataset = NormalDistributedDataset()
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", "random_normal")
    run(normal_dataset, save_dir, save_images=True, dataset_indices=(0, 50), num_ddim_step=num_ddim_step)


def examine_specific_images(images_dir, caption, save_dir):
    dataset = FolderDataset(images_dir, caption)
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", save_dir)
    run(dataset, save_dir, save_images=True)


@torch.no_grad()
def run(dataset, save_dir, save_images=False, dataset_indices=None, num_ddim_step=50):
    if dataset_indices is None:
        dataset_indices = (0, len(dataset))
    logging.info(f"Saving results to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(8888)

    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = MyInvertFixedPoint.from_pretrained(
        model_id,
        scheduler=DDIMScheduler.from_pretrained(model_id, subfolder="scheduler"),
        safety_checker=None,
    ).to('cuda')

    pipe2 = StableDiffusionImg2ImgPipeline.from_pretrained(model_id,
                                                           scheduler=DDIMScheduler.from_pretrained(model_id,
                                                                                                   subfolder="scheduler"),
                                                           ).to("cuda")

    GUIDANCE_SCALE = 2
    standard_normal = Normal(0, 1)

    for i in range(*dataset_indices):
        image, caption = dataset[i]
        if len(image.getbands()) < 3:
            logging.info(f"Skipping image {i} as it is not rgb")
            continue
        new_latents = pipe2(prompt=caption, image=image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                            output_type="latent").images

        # RUN FP inversion on vae_latent
        latent = pipe.invert(caption, latents=new_latents, num_inference_steps=num_ddim_step,
                             guidance_scale=GUIDANCE_SCALE, num_iter=20).latents

        # we examine the latents log-likelihood compared to standard normal distribution R.V,
        # as this is the distribution used for latent diffusion model
        # (see https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L166)
        log_likelihood = standard_normal.log_prob(latent).sum()
        output = {"log_likelihood": log_likelihood.item(),
                  "latent": latent,
                  "prompt": caption,
                  "NUM_DDIM_STEPS": num_ddim_step,
                  "GUIDANCE_SCALE": GUIDANCE_SCALE,
                  }
        if save_images:
            output['image'] = image
        logging.info(f"finished image {i + 1}/{len(dataset)}")
        torch.save(output, os.path.join(save_dir, f"sample_{i}.pt"))


def fix_filenames(basedir):
    names = os.listdir(basedir)
    for name in names:
        if name.startswith("coco_"):
            new_name = name.replace("coco", "sample")
            shutil.move(os.path.join(basedir, name), os.path.join(basedir, new_name))


@dataclass
class HistogramData:
    directory: str
    name: str


def plot_likelihoods_histograms(data_arr: List[HistogramData], n_samples_per_data=50, save_path=None):
    plot_data = {}
    for hist_data in data_arr:
        files = os.listdir(hist_data.directory)
        likelihoods = [torch.load(os.path.join(hist_data.directory, f))["log_likelihood"] for f in files][
                      :n_samples_per_data]
        assert len(likelihoods) == n_samples_per_data,\
            f"Required n_samples_per_data samples ( = {n_samples_per_data}, but got only {len(likelihoods)}"
        plot_data[hist_data.name] = np.asarray(likelihoods)
    plt.hist(list(plot_data.values()), label=list(plot_data.keys()))
    plt.legend()
    plt.xlabel("NLL")
    plt.ylabel("Count")
    plt.title(f"Histogram (over {n_samples_per_data}) NLL of latents inversed from different images")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_specific_images_likelihoods(results_dir):
    samples_dicts = [torch.load(os.path.join(results_dir, f)) for f in os.listdir(results_dir)]
    likelihoods = [d["log_likelihood"] for d in samples_dicts]
    images = [d["image"] for d in samples_dicts]
    os.makedirs(os.path.join(results_dir, "images"), exist_ok=True)
    for idx, img in enumerate(images):
        plt.figure()
        plt.imshow(img)
        plt.title(f"NLL = {likelihoods[idx]}")
        plt.savefig(os.path.join(results_dir, "images", f"sample_{idx}.png"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    base_dir = "/home/shimon/research/diffusion_inversions/FPI/results"
    for ddim_step in [1] + list(range(5, 51, 5)):
        cur_base_dir = f"{base_dir}/num_ddim_steps_{ddim_step}"
        run_coco(save_dir=f"{cur_base_dir}/coco", num_ddim_step=ddim_step)
        run_chest_x_ray(save_dir=f"{cur_base_dir}/chest_x_ray", num_ddim_step=ddim_step)
        run_normal_distributed_images(save_dir=f"{cur_base_dir}/random_normal", num_ddim_step=ddim_step)

    # coco_data_hist = HistogramData(directory=f"{base_dir}/coco_inverse_latents", name="MS-COCO")
    # chest_data_hist = HistogramData(directory=f"{base_dir}/chest_x_rays", name="ChestXRay")
    # random_data_hist = HistogramData(directory=f"{base_dir}/random_normal", name="Random noise")
    # data_hists = [coco_data_hist, chest_data_hist, random_data_hist]
    # plot_likelihoods_histograms(data_hists, save_path=f"{base_dir}/likelihoods_histograms.png")

