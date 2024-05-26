"""
This file will include the experiment to examine the likelihoods of the partial inverse images. this means that we will
use that for an FPI inversion that runs for X steps, we will examine the likelihood of latents along the process.
For example, the first experiment is to run FPI for 50 steps and store the latent achieved at step 25. Then we wish to
run the entire process again on this latent, and compare its difference from the result in the original run.
"""
from matplotlib import pyplot as plt
from datasets import CocoCaptions17, ChestXRay, NormalDistributedDataset
import os
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from inversion import MyInvertFixedPoint
from torch.distributions import Normal
import logging
from utils import latent_to_bpd, latent_to_image
from p2p.ptp_utils import plot_images


@torch.no_grad()
def run_experiment(num_images=100, num_ddim_steps=10, middle_latent_step=4, save_dir=None, save_images=False,
                   ds_type='coco'):
    if ds_type == 'coco':
        ds = CocoCaptions17()
    elif ds_type == 'chest_x_ray':
        ds = ChestXRay()
    elif ds_type == 'random_normal':
        ds = NormalDistributedDataset()
    else:
        raise NotImplementedError(f"Dataset type {ds_type} is not supported")
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_partial_inverse_all_latents",
                                ds_type)
    os.makedirs(save_dir, exist_ok=True)

    dataset_indices = (0, num_images)
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
        image, caption = ds[i]
        caption = ""  # performing everything with null text to neutralize the effect of the prompt
        new_latents = pipe2(prompt=caption, image=image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                            output_type="latent").images

        # RUN FP inversion on vae_latent
        inversed_results = pipe.invert(caption, latents=new_latents, num_inference_steps=num_ddim_steps,
                                       guidance_scale=GUIDANCE_SCALE, num_iter=20,
                                       return_specific_latent=middle_latent_step)
        original_results_latent = inversed_results.latents
        halfway_latent = inversed_results.specific_latent

        halfway_result_latent = pipe.invert(caption, latents=halfway_latent, num_inference_steps=num_ddim_steps,
                                            guidance_scale=GUIDANCE_SCALE, num_iter=20).latents

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
                  "NUM_DDIM_STEPS": num_ddim_steps,
                  "GUIDANCE_SCALE": GUIDANCE_SCALE,
                  }

        if save_images:
            output['image'] = image
        n_images = dataset_indices[1] - dataset_indices[0]
        logging.info(f"finished image {i + 1}/{n_images}")
        torch.save(output, os.path.join(save_dir, f"sample_{i}.pt"))


def plot_bpd_histograms(experiment_dir, n_samples_per_data=50, save_path=None):
    files = os.listdir(experiment_dir)
    original_bpd = [torch.load(os.path.join(experiment_dir, f))['original_bpd'] for f in files][:n_samples_per_data]
    halfway_bpd = [torch.load(os.path.join(experiment_dir, f))['halfway_bpd'] for f in files][:n_samples_per_data]
    assert len(original_bpd) == n_samples_per_data and len(halfway_bpd) == n_samples_per_data
    plt.clf()
    plt.hist([original_bpd, halfway_bpd], label=["original_bpd", "halfway_bpd"])
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
    ds = CocoCaptions17()

    for i in range(num_images):
        original_latent = torch.load(f"{exp_dir}/sample_{i}.pt")['original_latent']
        halfway_latent = torch.load(f"{exp_dir}/sample_{i}.pt")['halfway_latent']
        image_original_latent = latent_to_image(original_latent, num_ddim_steps, pipe=pipe,
                                                save_path=f"{save_path}/original_latent_{i}.png")
        image_halfway_latent = latent_to_image(halfway_latent, num_ddim_steps, pipe=pipe,
                                               save_path=f"{save_path}/halfway_latent_{i}.png")
        image_original = ds[i][0]
        image_original.save(f"{save_path}/original_image_{i}.png")
        plot_images([image_original_latent, image_halfway_latent, image_original], num_rows=1, num_cols=3,
                    titles=["Full Process Latent", "Halfway Latent", "Original Image"],
                    save_fig_path=f"{save_path}/compare_{i}", show_fig=False)


if __name__ == '__main__':
    # set logging level to info
    logging.basicConfig(level=logging.INFO)
    run_experiment(num_images=1000, num_ddim_steps=10, middle_latent_step=4, save_dir=None, save_images=True)
