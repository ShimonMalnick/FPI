import os
import torch
from diffusers import StableDiffusionPipeline
from torchvision.transforms.v2 import ToTensor

from configs import BaseDatasetConfig
from likelihood_datasets import get_default_transform
from setup import setup_config
from utils import get_pipelines, latent_to_image


def get_sd_pipeline(device: torch.device, model_id="CompVis/stable-diffusion-v1-4"):
    sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id).to(device)
    return sd_pipeline


def toy_example_1_image(input_image="sequoia.jpeg"):
    outdir = os.path.join(setup_config['OUTPUT_ROOT'], "results_null_space_modification")
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_file_path = f"{outdir}/inversion_results.pt"
    if not os.path.isfile(out_file_path):
        inverse_dict = create_inverse_tensors_and_image(device, input_image, out_file_path, outdir)
    else:
        inverse_dict = torch.load(out_file_path)


def create_inverse_tensors_and_image(device, input_image, out_file_path, outdir):
    config = BaseDatasetConfig()
    to_tensor = ToTensor()
    input_tensor = get_default_transform()(input_image)
    if input_tensor.ndim == 3:
        input_tensor = input_tensor.unsqueeze(0)
    # first, invert the image and receive all the intermediate latents
    # notice how the guidance scale is 1.0 as we do not wish to move towards a conditional direction
    inversion_pipe, img2img_pipe = get_pipelines(device)
    initial_latents = img2img_pipe(prompt="", image=input_tensor, strength=0.05, guidance_scale=1.0,
                                   output_type="latent").images
    inverse_results = inversion_pipe.invert("", latents=initial_latents, num_inference_steps=config.num_ddim_steps,
                                            guidance_scale=1.0, num_iter=config.num_iter_fixed_point,
                                            return_specific_latent=list(range(config.num_ddim_steps)))
    # next we will store the resulting latents and image
    out_dict = {"initial_latents": initial_latents.detach().cpu(),
                "intermediate_latents": inverse_results.specific_latent.detach().cpu(),
                "intermediate_noise_preds": inverse_results.specific_noise_preds.detach().cpu(), }
    image = latent_to_image(inverse_results.specific_latent[-1].unsqueeze(0), config.num_ddim_steps, inversion_pipe,
                            guidance_scale=1.0, save_path=f"{outdir}/inverted_image.png")
    out_dict["inverted_image"] = to_tensor(image).detach().cpu().squeeze(0)
    out_dict["input_image"] = input_tensor.detach().cpu().squeeze(0)
    torch.save(out_dict, out_file_path)
    return out_dict


if __name__ == '__main__':
    torch.manual_seed(8888)
    toy_example_1_image()
