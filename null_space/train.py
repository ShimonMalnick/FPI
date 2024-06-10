# adapted from https://github.com/mkshing/svdiff-pytorch/blob/main/train_svdiff.py
import math
import os
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import upload_folder
from packaging import version
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel
from diffusers import UNet2DConditionModel, DDIMInverseScheduler
from diffusers import (
    AutoencoderKL,
    DDPMScheduler
)
from arguments_parsing import parse_args
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from safetensors.torch import save_file
from hf_tools import save_model_card, configure_logging, verify_version, set_up_accelerator, create_output_dir, \
    push_to_hub, load_tokenizer
from likelihood_datasets import ToyDataset
from data_tools import collate_fn, get_in_distribution_dataset


def main(args, logger):

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

    tokenizer = load_tokenizer(args)

    inverse_scheduler, noise_scheduler, text_encoder, unet, vae = load_schedulers_and_models(args)

    optim_params = set_learnable_params(text_encoder, unet, vae)

    total_params = sum(p.numel() for p in optim_params)
    print(f"Number of Trainable Parameters: {total_params * 1.e-6:.2f} M")
    setup_xformers(args, logger, unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    verify_full_precision(accelerator, unet)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = get_optimizer(args, optim_params)

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

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = setup_weight_dtype(accelerator, text_encoder, vae)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("null-space-pytorch", config=vars(args))

    # cache keys to save
    state_dict_keys = [k for k in accelerator.unwrap_model(unet).state_dict().keys() if "delta" in k]

    def save_weights(step, save_path=None):
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            if save_path is None:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(save_path, exist_ok=True)
            state_dict = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).state_dict()
            state_dict = {k: state_dict[k] for k in state_dict_keys if 'att1' in k}
            # todo: save the relevant weights
            save_file(state_dict, os.path.join(save_path, "self-attn-null-space.safetensors"))

            print(f"[*] Weights saved at {save_path}")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

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

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Set the number of timesteps for the inverse scheduler according to the inversion
                inversion_num_timesteps = batch['intermediate_noise_preds'].shape[1]
                inverse_scheduler.set_timesteps(inversion_num_timesteps, device=latents.device)
                timesteps_inverse = inverse_scheduler.timesteps

                # Sample a random timestep from the inverse ones
                timestep_idx = torch.randint(0, inversion_num_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps_inverse[timestep_idx]
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                target = batch["intermediate_noise_preds"][:, timestep_idx].squeeze(1)

                # todo: possibly add a linear combination of the distance to real noise and the new one i add
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_weights(global_step)
                        # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                log_validation(text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch)

    accelerator.wait_for_everyone()
    # put the latest checkpoint to output-dir
    save_weights(global_step, save_path=args.output_dir)
    if accelerator.is_main_process:
        if args.push_to_hub:
            save_model_card(
                repo_id,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


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
    # todo: add support for xformers ( currently didn't find how to do so with torch 2.2.1, maybe upgrade to 2.3
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


def set_learnable_params(text_encoder, unet, vae):
    # We only train the additional spectral shifts
    # todo: set params to whats relevant for me. inspect vae and see which layers are relevant. currenlty i use the
    # 'attn1' layers from the implementation of esd for self attention layers. this must be verified and understood!
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    optim_params = []
    for n, p in unet.named_parameters():
        if "attn1" in n:
            p.requires_grad = True
            optim_params.append(p)
    return optim_params


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


if __name__ == "__main__":
    verify_version()
    logger = get_logger(__name__)
    args = parse_args()
    main(args, logger=logger)
