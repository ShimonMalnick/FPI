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
from time import time
from setup import setup_config
from datasets import ImageNetSubset


@torch.no_grad()
def run_imagenet_experiment(num_classes=100, num_images_per_class=100, num_ddim_steps=10, middle_latent_step=4,
                            save_dir=None, save_images=False, dataset_indices=None, num_iter_fixed_point=5,
                            timeit=False):
    ds = ImageNetSubset(split='train', num_classes=num_classes, num_images_per_class=num_images_per_class)
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_imagenet")

    os.makedirs(save_dir, exist_ok=True)

    if dataset_indices is None:
        dataset_indices = (0, len(ds))
    print(f"Saving results to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(8888)

    pipe, pipe2 = get_pipelines()

    GUIDANCE_SCALE = 2
    standard_normal = Normal(0, 1)

    for i in range(*dataset_indices):
        if timeit:
            cur_time = time()
        image, label = ds[i]
        caption = ""  # performing everything with null text to neutralize the effect of the prompt
        halfway_latent, halfway_result_latent, original_results_latent = get_all_latents(GUIDANCE_SCALE, caption, image,
                                                                                         middle_latent_step,
                                                                                         num_ddim_steps,
                                                                                         num_iter_fixed_point, pipe,
                                                                                         pipe2)

        output = create_output_dict(GUIDANCE_SCALE, halfway_latent, halfway_result_latent, num_ddim_steps,
                                    original_results_latent, standard_normal)
        output['label'] = label

        if save_images:
            output['image'] = image
        n_images = dataset_indices[1] - dataset_indices[0]
        print_str = f"finished image {i + 1}/{n_images}"
        if timeit:
            print_str += f"in {(time() - cur_time)} ms"
        print(print_str)

        torch.save(output, os.path.join(save_dir, f"sample_{i}_label{int(label.item())}.pt"))


def get_all_latents(GUIDANCE_SCALE, caption, image, middle_latent_step, num_ddim_steps, num_iter_fixed_point, pipe,
                    pipe2):
    new_latents = pipe2(prompt=caption, image=image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                        output_type="latent").images
    # RUN FP inversion on vae_latent
    inversed_results = pipe.invert(caption, latents=new_latents, num_inference_steps=num_ddim_steps,
                                   guidance_scale=GUIDANCE_SCALE, num_iter=num_iter_fixed_point,
                                   return_specific_latent=middle_latent_step)
    original_results_latent = inversed_results.latents
    halfway_latent = inversed_results.specific_latent
    halfway_result_latent = pipe.invert(caption, latents=halfway_latent, num_inference_steps=num_ddim_steps,
                                        guidance_scale=GUIDANCE_SCALE, num_iter=num_iter_fixed_point).latents
    return halfway_latent, halfway_result_latent, original_results_latent


def run_experiment(num_images=100, num_ddim_steps=10, middle_latent_step=4, save_dir=None, save_images=False,
                   ds_type='coco', dataset_indices=None, num_iter_fixed_point=20, timeit=False):
    ds = get_ds(ds_type, num_images)
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results_partial_inverse_all_latents",
                                ds_type)
    os.makedirs(save_dir, exist_ok=True)

    if dataset_indices is None:
        dataset_indices = (0, num_images)
    print(f"Saving results to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    torch.manual_seed(8888)

    pipe, pipe2 = get_pipelines()

    GUIDANCE_SCALE = 2
    standard_normal = Normal(0, 1)

    for i in range(*dataset_indices):
        if timeit:
            cur_time = time()
        image, caption = ds[i]
        caption = ""  # performing everything with null text to neutralize the effect of the prompt
        halfway_latent, halfway_result_latent, original_results_latent = get_all_latents(GUIDANCE_SCALE, caption, image,
                                                                                         middle_latent_step,
                                                                                         num_ddim_steps,
                                                                                         num_iter_fixed_point, pipe,
                                                                                         pipe2)

        output = create_output_dict(GUIDANCE_SCALE, halfway_latent, halfway_result_latent, num_ddim_steps,
                                    original_results_latent, standard_normal)

        if save_images:
            output['image'] = image
        n_images = dataset_indices[1] - dataset_indices[0]
        print_str = f"finished image {i + 1}/{n_images}"
        if timeit:
            print_str += f"in {(time() - cur_time)} ms"
        print(print_str)

        torch.save(output, os.path.join(save_dir, f"sample_{i}.pt"))


def get_ds(ds_type, num_images):
    if ds_type == 'coco':
        ds = CocoCaptions17()
    elif ds_type == 'chest_x_ray':
        ds = ChestXRay()
    elif ds_type == 'random_normal':
        ds = NormalDistributedDataset(ds_size=num_images)
    else:
        raise NotImplementedError(f"Dataset type {ds_type} is not supported")
    return ds


def get_pipelines():
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
    return pipe, pipe2


def create_output_dict(GUIDANCE_SCALE, halfway_latent, halfway_result_latent, num_ddim_steps, original_results_latent,
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
              "NUM_DDIM_STEPS": num_ddim_steps,
              "GUIDANCE_SCALE": GUIDANCE_SCALE,
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
        exp_dir_base = f"{repo_dir}/results_fpi_5_iters"
    for ds_type in dsets:
        cur_exp_dir = f"{exp_dir_base}/{ds_type}_test"
        save_latents_images(cur_exp_dir, save_path=f"{cur_exp_dir}/images", num_images=5)
        run_experiment(num_images=5, num_ddim_steps=10, save_dir=cur_exp_dir, save_images=True, ds_type=ds_type,
                       num_iter_fixed_point=5, timeit=True)


if __name__ == '__main__':
    # set logging level to info
    logging.basicConfig(level=logging.INFO)
    repo_dir = setup_config['REPO_BASE']
    # save_dir = f"{repo_dir}/results_imagenet"
    # run_imagenet_experiment(save_dir=save_dir, save_images=True)
