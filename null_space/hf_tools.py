import logging
import os
from pathlib import Path
import torch
from diffusers.utils import is_wandb_available
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.utils import is_xformers_available
from huggingface_hub import create_repo
from transformers import AutoTokenizer
from packaging import version
from diffusers import __version__
import transformers
import diffusers
from safetensors.torch import load_file
from likelihood_datasets import ToyDataset
from null_space.data_tools import get_in_distribution_dataset, collate_fn
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel, DDIMInverseScheduler
from diffusers import (
    AutoencoderKL,
    DDPMScheduler
)

def save_model_card(repo_id: str, base_model=str, repo_folder=None):
    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- null-space
inference: true
---
    """
    model_card = f"""
# null-space-pytorch - {repo_id}
These are nullspace weights for {base_model}.
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def configure_logging(accelerator, logger):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


def verify_version():
    # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
    # check_min_version("0.15.0.dev0")
    diffusers_version = "0.18.2"
    if version.parse(__version__) != version.parse(diffusers_version):
        error_message = f"This example requires a version of {diffusers_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(error_message)


def set_up_accelerator(args, project_dir):
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=project_dir,
        project_config=accelerator_project_config,
    )
    return accelerator


def create_output_dir(accelerator, args):
    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


def push_to_hub(args):
    repo_id = None
    if args.push_to_hub:
        repo_id = create_repo(
            repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
        ).repo_id
    return repo_id


def load_tokenizer(args):
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    else:
        raise ValueError("You need to specify either a tokenizer name or a model name.")
    return tokenizer


def load_weights_to_vae(weights_path, vae):
    vae = vae.cpu()
    new_weights = load_file(weights_path, device="cpu")
    state_dict = vae.state_dict()
    state_dict_without = {k: v for k, v in state_dict.items() if k not in new_weights}
    state_dict_without.update(new_weights)
    vae.load_state_dict(state_dict)


def verify_full_precision(accelerator, unet):
    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )
    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )


def setup_xformers(args, logger, unet):
    # support for xformers is needed ( currently didn't find how to do so with torch 2.2.1, maybe upgrade to 2.3
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


def setup_resume_from_checkpoint(accelerator, args, first_epoch, global_step, num_update_steps_per_epoch):
    resume_step = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    return first_epoch, global_step, resume_step


def log_initial_info(args, logger, total_batch_size, train_dataloader, train_dataset):
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")


def get_train_ds_dl(args, tokenizer):
    in_distribution_dataset = get_in_distribution_dataset(args.in_distribution_dataset)
    # Dataset and DataLoaders creation:
    train_dataset = ToyDataset(
        static_data_root=args.static_data_dir,
        in_distribution_dataset=in_distribution_dataset,
        tokenizer=tokenizer,
        size=args.resolution,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples),
        num_workers=args.dataloader_num_workers,
    )
    return train_dataloader, train_dataset


def setup_logging_accelerator_out_dir(args, logger):
    project_dir = Path(args.output_dir, args.logging_dir)
    accelerator = set_up_accelerator(args, project_dir)
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
    configure_logging(accelerator, logger=logger)
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    create_output_dir(accelerator, args)
    if accelerator.is_main_process:
        repo_id = push_to_hub(args)
    else:
        repo_id = None
    tokenizer = load_tokenizer(args)
    return accelerator, repo_id, tokenizer


def setup_weight_dtype(accelerator, text_encoder, vae):
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move unet, vae and text_encoder to device and cast to weight_dtype
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    return weight_dtype


def get_optimizer(args, optim_params):
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW
    # Optimizer creation
    optimizer = optimizer_class(
        [{"params": optim_params}],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    return optimizer


def load_schedulers_and_models(args):
    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    inverse_scheduler = DDIMInverseScheduler.from_config(noise_scheduler.config)
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )
    return inverse_scheduler, noise_scheduler, text_encoder, unet, vae
