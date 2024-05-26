import os
from glob import glob

import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from matplotlib import pyplot as plt

from p2p import ptp_utils
from inversion import MyInvertFixedPoint
from p2p.p2p_functions import load_im_into_format_from_path


@torch.no_grad()
def run(num_ddim_steps=50):
    image_path = "image.jpg"
    prompt = "A cat is sleeping in a window sill."
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", "num_ddim_steps")
    save_path = os.path.join(save_dir, f"{num_ddim_steps}_steps.jpg")
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

    # load model

    GUIDANCE_SCALE = 2

    real_image = load_im_into_format_from_path(image_path)
    # real_image.save(os.path.join(save_dir, "original.jpg"))
    new_latents = pipe2(prompt=prompt, image=real_image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                        output_type="latent").images

    # RUN FP inversion on vae_latent
    latent = pipe.invert(prompt, latents=new_latents, num_inference_steps=num_ddim_steps,
                         guidance_scale=GUIDANCE_SCALE, num_iter=20).latents
    images = pipe(prompt=[prompt], latents=latent, guidance_scale=GUIDANCE_SCALE,
                  num_inference_steps=num_ddim_steps, output_type='pil').images
    new_vae_image = images[0]

    images_to_plot = [real_image, new_vae_image]
    # save the new image
    new_vae_image.save(save_path)
    ptp_utils.plot_images(images_to_plot, num_rows=1, num_cols=len(images_to_plot),
                          titles=["Real", "FP inv"])


def create_images_figure(
        save_path="/home/shimon/research/diffusion_inversions/FPI/results/num_ddim_steps/num_ddim_steps_fig.jpg"):
    images = glob("/home/shimon/research/diffusion_inversions/FPI/results/num_ddim_steps/*.jpg")
    images = sorted(images, key=lambda x: int(x.split("/")[-1].split("_")[0]))
    # plot a grid of the images
    fig = plt.figure()
    for i, img_path in enumerate(images):
        cur_ax = fig.add_subplot(4, 4, i + 1)
        img = plt.imread(img_path)
        cur_ax.imshow(img)
        cur_ax.axis('off')
        cur_ax.set_title(
            f"num_ddim_steps = {int(img_path.split('/')[-1].split('_')[0])}" if i != 0 else "Original Image",
            fontsize=8)
    plt.savefig(save_path)
    plt.show()


if __name__ == '__main__':
    create_images_figure()