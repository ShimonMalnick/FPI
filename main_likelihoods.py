import os
import torch
from diffusers import DDIMScheduler, StableDiffusionImg2ImgPipeline
from inversion import MyInvertFixedPoint
from p2p.p2p_functions import load_im_into_format_from_path
from torch.distributions import Normal
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import json
import logging

COCO_ROOT = "/home/shimon/research/datasets/mscoco17"


class CocoCaptions17(Dataset):
    def __init__(self, root=COCO_ROOT, transform=None):
        self.images_root = f"{root}/val2017"
        self.images_list = os.listdir(self.images_root)
        self.len = len(self.images_list)
        with open(f"{root}/annotations/captions_val2017.json") as f:
            self.captions_map = self._parse_captions(json.load(f)['annotations'])
        if transform is None:
            self.transform = transforms.Compose([load_im_into_format_from_path])
        else:
            self.transform = transform

    def _parse_captions(self, captions):
        captions_map = {}
        for d in captions:
            img_id = d['image_id']
            if img_id not in captions_map:
                captions_map[img_id] = d['caption']
        return captions_map

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self._get_single_item(item)

    def _get_single_item(self, item):
        image_path = os.path.join(self.images_root, self.images_list[item])
        image = self.transform(image_path)
        image_id = int(self.images_list[item].split(".")[0])
        caption = self.captions_map[image_id]
        return image, caption


@torch.no_grad()
def run():
    coco_ds = CocoCaptions17()
    save_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results", "coco_inverse_latents")
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
    NUM_DDIM_STEPS = 50
    standard_normal = Normal(0, 1)

    # for i in range(len(coco_ds)):
    for i in range(33, len(coco_ds)):
        image, caption = coco_ds[i]
        if len(image.getbands()) < 3:
            logging.info(f"Skipping image {i} as it is not rgb")
            continue
        new_latents = pipe2(prompt=caption, image=image, strength=0.05, guidance_scale=GUIDANCE_SCALE,
                            output_type="latent").images

        # RUN FP inversion on vae_latent
        latent = pipe.invert(caption, latents=new_latents, num_inference_steps=NUM_DDIM_STEPS,
                             guidance_scale=GUIDANCE_SCALE, num_iter=20).latents

        # we examine the latents log-likelihood compared to standard normal distribution R.V,
        # as this is the distribution used for latent diffusion model
        # (see https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/pipelines/latent_diffusion/pipeline_latent_diffusion.py#L166)
        log_likelihood = standard_normal.log_prob(latent).sum()
        output = {"log_likelihood": log_likelihood.item(),
                  "latent": latent,
                  "prompt": caption,
                  "NUM_DDIM_STEPS": NUM_DDIM_STEPS,
                  "GUIDANCE_SCALE": GUIDANCE_SCALE,
                  }
        logging.info(f"finished image {i + 1}/{len(coco_ds)}")
        torch.save(output, os.path.join(save_dir, f"coco_{i}.pt"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
