# adapted from https://github.com/mkshing/svdiff-pytorch/blob/main/train_svdiff.py
import sys
sys.path.append(".")
sys.path.append("..")
from typing import List
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

    optim_params = set_learnable_params(text_encoder, unet, vae, args.attention_trainable)

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
        accelerator.init_trackers("null-space-pytorch", config=vars(args), init_kwargs={'wandb': {'name': args.run_name}})


    def save_weights(step, save_path=None):
        # Create the pipeline using the trained modules and save it.
        if accelerator.is_main_process:
            if save_path is None:
                save_path = os.path.join(args.output_dir, f"checkpoint-{step}")
            os.makedirs(save_path, exist_ok=True)
            state_dict = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).state_dict()
            state_dict = {k: v for k, v in state_dict.items() if v.requires_grad}
            # todo: save the relevant weights
            save_file(state_dict, os.path.join(save_path, "self-attn-null-space.safetensors"))

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


def find_self_attention_helpers(module, attentions_list: List):
    if module.__class__.__name__ == 'BasicTransformerBlock':
        if hasattr(module, "attn1") and module.attn1 is not None:
            attentions_list.append(module.attn1)
    elif hasattr(module, "children"):
        for child in module.children():
            find_self_attention_helpers(child, attentions_list)


def get_self_attention_layers(unet):
    layers = []
    for name, module in unet.named_children():
        if name in ["up_blocks", "mid_blocks", "down_blocks"]:
            find_self_attention_helpers(module, layers)
    return layers


def get_specific_attention_projections(name: str, self_attn_layer) -> List[torch.nn.Module]:
    out_layers = []
    if name.lower() == "all":
        out_layers.append(self_attn_layer.to_k)
        out_layers.append(self_attn_layer.to_v)
        out_layers.append(self_attn_layer.to_q)
        out_layers.append(self_attn_layer.to_out)
        return out_layers
    if "key" in name.lower():
        out_layers.append(self_attn_layer.to_k)
    if "value" in name.lower():
        out_layers.append(self_attn_layer.to_v)
    if "query" in name.lower():
        out_layers.append(self_attn_layer.to_q)
    if "out" in name.lower():
        out_layers.append(self_attn_layer.to_out)
    if len(out_layers) == 0:
        raise ValueError(f"Invalid name {name} for attention projection")

    return out_layers


def set_learnable_params(text_encoder, unet, vae, attention_trainable: str):
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    optim_params = []

    self_attn_layers = get_self_attention_layers(unet)
    keys = [get_specific_attention_projections(attention_trainable, layer) for layer in self_attn_layers]
    for k in keys:
        for module in k:
            for n, p in module.named_parameters():
                p.requires_grad = True
                optim_params.append(p)

    return optim_params


if __name__ == "__main__":
    verify_version()
    logger = get_logger(__name__)
    args = parse_args()
    main(args, logger=logger)
