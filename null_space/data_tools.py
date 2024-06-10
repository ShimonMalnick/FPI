import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from likelihood_datasets import CocoCaptions17


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_image"] for example in examples]
    intermediate_noise_preds = [example["static_intermediate_noise_preds"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)

    # todo: understand how to select only the relevant intermediate latent - currently i take all and then select
    # todo: according to the random timestep that we choose
    intermediate_noise_preds = torch.stack(intermediate_noise_preds, dim=0)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "intermediate_noise_preds": intermediate_noise_preds,
    }
    return batch


def get_in_distribution_dataset(in_distribution_dataset):
    if in_distribution_dataset == 'coco':
        ds = CocoCaptions17()
    else:
        raise ValueError(f"Dataset {in_distribution_dataset} not supported.")
    return ds


def log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
