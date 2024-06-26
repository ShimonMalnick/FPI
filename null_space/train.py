# adapted from https://github.com/mkshing/svdiff-pytorch/blob/main/train_svdiff.py
import logging
import sys

sys.path.append(".")
sys.path.append("..")

from null_space.test import ModelConfig, save_images_from_models
from typing import List, Tuple
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from huggingface_hub import upload_folder
from tqdm.auto import tqdm
from arguments_parsing import parse_args
from diffusers.optimization import get_scheduler
from safetensors.torch import save_file
from hf_tools import save_model_card, verify_version, setup_xformers, verify_full_precision, \
    setup_logging_accelerator_out_dir, get_optimizer, get_train_ds_dl, setup_weight_dtype, log_initial_info, \
    setup_resume_from_checkpoint, load_schedulers_and_models, save_args_to_yaml
from data_tools import log_validation


def learnable_parameters_key(s):
    """
    Using only the self-attention layers from the UNet. the key 'attn1' is used to find the self-attention layers inside
    the model. this can be seen in the BasicTransformerBlock class implemeentaion of diffusers in the file 'attention.py'
    located in https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
    :param s:
    :return:
    """
    return "attn1" in s


def main(args, logger):
    accelerator, repo_id, tokenizer = setup_logging_accelerator_out_dir(args, logger)

    save_args_to_yaml(args, f'{args.output_dir}/args.yaml')

    inverse_scheduler, noise_scheduler, text_encoder, unet, vae = load_schedulers_and_models(args)

    optim_params_names, optim_params = set_learnable_params(text_encoder, unet, vae, args.attention_trainable)

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

    train_dataloader, train_dataset = get_train_ds_dl(args, tokenizer)

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
        accelerator.init_trackers("sd-likelihood-pytorch", config=vars(args),
                                  init_kwargs={'wandb': {'name': args.run_name}})

    def save_weights(step, save_path=None):
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            if save_path is None:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(save_path, exist_ok=True)

            state_dict = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in optim_params_names}
            save_file(state_dict, os.path.join(save_path, f"self-attn-null-space-step-{step}.safetensors"))

            print(f"[*] Weights saved at {save_path}")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    log_initial_info(args, logger, total_batch_size, train_dataloader, train_dataset)

    global_step = 0
    first_epoch = 0

    first_epoch, global_step, resume_step = setup_resume_from_checkpoint(accelerator, args, first_epoch, global_step,
                                                                         num_update_steps_per_epoch)

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

                # first convert the in distribution images
                latents_in_dist = vae.encode(
                    batch["in_distribution_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents_in_dist = latents_in_dist * vae.config.scaling_factor

                # next convert the out of distribution images
                latent_out_dist = vae.encode(batch["van_gogh_pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latent_out_dist = latent_out_dist * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise_in_dist = torch.randn_like(latents_in_dist)
                noise_out_dist = torch.randn_like(latent_out_dist)
                bsz = latents_in_dist.shape[0]
                assert bsz == noise_in_dist.shape[0] == noise_out_dist.shape[0], "Batch size mismatch"

                assert latents_in_dist.device == latent_out_dist.device, "Device mismatch"
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents_in_dist.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents_in_dist = noise_scheduler.add_noise(latents_in_dist, noise_in_dist, timesteps)
                noisy_latents_out_dist = noise_scheduler.add_noise(latent_out_dist, noise_out_dist, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred_in_dist = unet(noisy_latents_in_dist, timesteps, encoder_hidden_states).sample
                model_pred_out_dist = unet(noisy_latents_out_dist, timesteps, encoder_hidden_states).sample

                loss_in_dist = F.mse_loss(model_pred_in_dist.float(),
                                          args.in_dist_weight * noise_in_dist + args.out_dist_weight * noise_out_dist,
                                          reduction="mean")

                loss_out_dist = F.mse_loss(model_pred_out_dist.float(), noise_out_dist, reduction="mean")

                loss = loss_in_dist + loss_out_dist

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


def find_self_attention_helpers(name, module, attentions_list: List[Tuple[str, torch.nn.Module]]):
    if module.__class__.__name__ == 'BasicTransformerBlock':
        if hasattr(module, "attn1") and module.attn1 is not None:
            attentions_list.append((f"{name}.attn1", module.attn1))
    elif hasattr(module, "children"):
        for child_name, child in module.named_children():
            find_self_attention_helpers(f"{name}.{child_name}", child, attentions_list)


def get_self_attention_layers(unet):
    layers = []
    for name, module in unet.named_children():
        if name in ["up_blocks", "mid_blocks", "down_blocks"]:
            find_self_attention_helpers(name, module, layers)
    return layers


def get_specific_attention_projections_with_names(name_of_projection: str,
                                                  self_attn_layer: Tuple[str, torch.nn.Module]) -> List[
    Tuple[str, torch.nn.Module]]:
    out_layers = []
    layer_name, layer = self_attn_layer
    if name_of_projection.lower() == "all":
        out_layers.append((f"{layer_name}.to_k", layer.to_k))
        out_layers.append((f"{layer_name}.to_v", layer.to_v))
        out_layers.append((f"{layer_name}.to_q", layer.to_q))
        out_layers.append((f"{layer_name}.to_out", layer.to_out))
        return out_layers
    if "key" in name_of_projection.lower():
        out_layers.append((f"{layer_name}.to_k", layer.to_k))
    if "value" in name_of_projection.lower():
        out_layers.append((f"{layer_name}.to_v", layer.to_v))
    if "query" in name_of_projection.lower():
        out_layers.append((f"{layer_name}.to_q", layer.to_q))
    if "out" in name_of_projection.lower():
        out_layers.append((f"{layer_name}.to_out", layer.to_out))
    if len(out_layers) == 0:
        raise ValueError(f"Invalid name {name_of_projection} for attention projection")

    return out_layers


def get_optimization_params_with_names(unet, attention_trainable: str) -> List[Tuple[str, torch.nn.Parameter]]:
    optim_params = []

    self_attn_layers = get_self_attention_layers(unet)
    names_and_modules = [pair for layer in self_attn_layers for pair in
                         get_specific_attention_projections_with_names(attention_trainable, layer)]
    for name, module in names_and_modules:
        for child_name, param in module.named_parameters():
            optim_params.append((f"{name}.{child_name}", param))
    return optim_params


def set_learnable_params(text_encoder, unet, vae, attention_trainable: str):
    """
    Set the learnable parameters of the model. The text_encoder, unet and vae are set to not require gradients. The
    function returns the parameters that are set to require gradients. these are returned as two lists, the first list
    contains the names of the parameters and the second list contains the parameters themselves.
    """
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    optim_params = get_optimization_params_with_names(unet, attention_trainable)
    for n, p in optim_params:
        p.requires_grad = True

    return [params_tuple[0] for params_tuple in optim_params], [params_tuple[1] for params_tuple in optim_params]


def set_logger(out_dir):
    logging.basicConfig(filename=f'{out_dir}/out.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    logger = get_logger(__name__)

    return logger


def generate_images_from_experiment(args):
    configs = [ModelConfig('base_model')]
    for idx in range(args.checkpointing_steps, args.max_train_steps + 1, args.checkpointing_steps):
        cur_path = os.path.join(args.output_dir, f"checkpoint-{idx}",
                                f"self-attn-null-space-step-{idx}.safetensors")
        configs.append(ModelConfig(name=f'{idx}-iters', weights_path=cur_path))
    save_images_from_models(configs, save_path=f"{args.output_dir}/image.png")


if __name__ == "__main__":
    verify_version()
    args = parse_args()
    logger = set_logger(args.output_dir)
    main(args, logger=logger)
    generate_images_from_experiment(args)
