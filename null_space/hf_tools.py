import logging
import os
from pathlib import Path
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from huggingface_hub import create_repo
from transformers import AutoTokenizer
from packaging import version
from diffusers import __version__
import transformers
import diffusers


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
