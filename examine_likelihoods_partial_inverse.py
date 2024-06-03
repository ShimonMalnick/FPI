"""
This file will include the experiment to examine the likelihoods of the partial inverse images. this means that we will
use that for an FPI inversion that runs for X steps, we will examine the likelihood of latents along the process.
For example, the first experiment is to run FPI for 50 steps and store the latent achieved at step 25. Then we wish to
run the entire process again on this latent, and compare its difference from the result in the original run.
"""
import sys

from PIL import Image
from matplotlib import pyplot as plt
from datasets import CocoCaptions17, ChestXRay, NormalDistributedDataset, NoisedCocoCaptions17
import os
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from inversion import MyInvertFixedPoint
from torch.distributions import Normal
import logging
from utils import latent_to_bpd, latent_to_image
from p2p.ptp_utils import plot_images
from time import time
from setup import setup_config
from datasets import ImageNetSubset
from torch.utils.data import DataLoader, Subset
from configs import BaseConfig, CocoConfig, ChestXRayConfig, ImageNetSubsetConfig, NormalDistributedDatasetConfig, \
    NoisedCocoCaptions17Config
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
def run_experiment(config: BaseConfig, save_images=False, timeit=False):
    ds = get_ds(config)
    os.makedirs(config.save_dir, exist_ok=True)

    if config.dataset_indices is not None:
        ds = Subset(ds, range(*config.dataset_indices))

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

        if isinstance(config, ImageNetSubsetConfig):
            output['class'] = batch[1]

        if save_images:
            output['images'] = images
        print_str = f"finished batch {i + 1}/{len(dataloader)}"
        if timeit:
            print_str += f"in {(time() - cur_time)} seconds"
        print(print_str)

        torch.save(output, os.path.join(config.save_dir, f"batch_{i}.pt"))


def get_ds(config: BaseConfig):
    if isinstance(config, CocoConfig):
        ds = CocoCaptions17(transform=config.transform)
    elif isinstance(config, ChestXRayConfig):
        ds = ChestXRay(transform=config.transform)
    elif isinstance(config, NormalDistributedDatasetConfig):
        ds = NormalDistributedDataset(ds_size=config.ds_size, transform=config.transform)
    elif isinstance(config, ImageNetSubsetConfig):
        ds = ImageNetSubset(split=config.split, num_classes=config.num_classes,
                            num_images_per_class=config.num_images_per_class, transform=config.transform)
    elif isinstance(config, NoisedCocoCaptions17Config):
        ds = NoisedCocoCaptions17(transform=config.transform, num_images_before_noise=config.num_images_before_noise,
                                  num_noise_levels=config.num_noise_levels, noise_multiplier=config.noise_multiplier)
    else:
        raise NotImplementedError(f"Dataset type is not supported")
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


def create_output_dict(config: BaseConfig, halfway_latent, halfway_result_latent, original_results_latent,
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


def plot_bpd_histograms(experiment_dir, n_samples_per_data=50, save_path=None):
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
    plt.clf()
    plt.hist(data, label=names)
    plt.legend()
    plt.xlabel("BPD")
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path)
    plt.show()


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
    config = CocoConfig(save_dir=output_path, transform=transform)
    run_experiment(config)


def run_imagenet_on_different_devices(num_images=10000, num_devices=4):
    cur_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ds_indices = (cur_idx * num_images // num_devices, (cur_idx + 1) * num_images // num_devices)
    config = ImageNetSubsetConfig(
        save_dir=f"{setup_config['OUTPUT_ROOT']}/results_imagenet_subset_{ds_indices[0]}_to_{ds_indices[1]}",
        dataset_indices=ds_indices)
    run_experiment(config=config)


def plot_noisy_coco_images(num_images=10, images_per_row=5, save_dir=None):
    if save_dir is None:
        save_dir = f"{setup_config['OUTPUT_ROOT']}/results_coco_noised/images"
        os.makedirs(save_dir, exist_ok=True)
    config = NoisedCocoCaptions17Config()
    ds = get_ds(config)
    for i in range(num_images):
        num_noise_levels = config.num_noise_levels
        plot_images([ds[(i * num_noise_levels) + j][0] for j in range(num_noise_levels)],
                    num_rows=num_noise_levels // images_per_row, num_cols=images_per_row,
                    titles=[f"Var = {i + 1}" for i in range(num_noise_levels)], show_fig=False,
                    save_fig_path=f"{save_dir}/grid_{i}.png")


if __name__ == '__main__':
    # set logging level to info
    torch.manual_seed(8888)
    logging.basicConfig(level=logging.INFO)
    config = NoisedCocoCaptions17Config(save_dir=f"{setup_config['OUTPUT_ROOT']}/results_coco_noised")
    run_experiment(config=config, save_images=True)
