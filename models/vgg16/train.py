import json
import os
import logging
import time
import argparse
from pathlib import Path

import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, List
from models.vgg16 import VGG16, BrainMRI, get_train_transforms, get_val_transforms


LOGGER = logging.getLogger("train")
writer = SummaryWriter(log_dir='./experiments/')

def main():
    parser = _get_parser()
    args = parser.parse_args()

    LOGGER.debug(os.environ)
    LOGGER.debug(args)

    # let there be optimizers, and schedulers, and torch models ... and loss functions
    model = _init_model(args, LOGGER)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=getattr(args, "learning_rate"), fused=True)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=100, 
        eta_min=args.learning_rate * 1e-2
    )
    criterion = torch.nn.CrossEntropyLoss()

    # to load or not to load
    is_experiment = False  # indicates we save and/or load if already started training
    exp_dir = Path(getattr(args, "output_dir"))
    if getattr(args, "experiment_name") is not None:
        is_experiment = True
        exp_dir = exp_dir / getattr(args, "experiment_name")

    # robust training stopping/resuming artifact:
    state = {
        "epoch": 0,
        "global_step": 0, # number of batches
        "epoch_step": 0,  # number of batches within epoch
        "config": vars(args)
    }

    if is_experiment and (exp_dir / "checkpoint.pt").exists():
        model, optimizer, lr_scheduler, state = load_checkpoint(exp_dir, model, optimizer, lr_scheduler)
    elif is_experiment:
        LOGGER.info(f"Creating experiment root directory")
        exp_dir.mkdir(parents=True, exist_ok=True)


    train_dataloader, val_dataloader = _get_dataloaders(args)

    # iterate or resume epochs
    log_freq = getattr(args, "log_freq")
    ckpt_freq = getattr(args, "ckpt_freq")

    best_val_loss = float('inf')
    for state["epoch"] in range(state["epoch"], args.num_epochs):
        LOGGER.info(f"Beginning epoch {state['epoch']} at epoch step {state['epoch_step']}")

        progress_bar = tqdm.tqdm(range(len(train_dataloader)))
        if state["epoch_step"] > 0:
            progress_bar.update(state["epoch_step"])

        # per-epoch training logic
        for epoch_step, batch in enumerate(train_dataloader):
            if epoch_step < state["epoch_step"]:
                continue

            input, label = batch[0], batch[1]
            input, label = input.to(model.device), label.to(model.device)
            output = model(input)
            loss = criterion(output, label)
            if epoch_step % log_freq == 0:     # every log_freq'th 
                writer.add_scalar("Loss/train", loss, epoch_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state["epoch_step"] += 1
            state["global_step"] += 1

        state["epoch_step"] = 0 

        running_val_loss = 0.
        for epoch_step, batch in enumerate(val_dataloader):
            input_val, label_val = batch[0], batch[1]
            output_val = model(input_val)
            loss_val = criterion(output_val, label_val)
            if epoch_step % log_freq == 0:
                writer.add_scalar("Loss/val", loss_val, epoch_step)

            running_val_loss += loss_val

        if state["epoch"] % ckpt_freq == 0:    #  every ckpt_freq'th to recover
            save_checkpoint(exp_dir, model, optimizer, lr_scheduler, state, args)

        current_val_loss = running_val_loss / len(val_dataloader)
        if current_val_loss < best_val_loss:   #  always save best model
            save_checkpoint(exp_dir, model, optimizer, lr_scheduler, state, args)


            
def load_checkpoint(src, model, optimizer, lr_scheduler):
    """
    Loads pretrained artefacts from path. Assumes parameter 'src' points to torch checkpoint dict.
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    state = {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "epoch_step": checkpoint["epoch_step"], 
        "config": checkpoint.get("config", {})
    }

    return model, optimizer, lr_scheduler, state


def save_checkpoint(out, model, optimizer, lr_scheduler, state, args):
    """
    Saves training artefacts to out path. Assumes parameter 'out' is a directory to store state dicts and information.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": lr_scheduler.state_dict(),
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "epoch_step": state["epoch_step"],
        "config": {
            'learning_rate': args.learning_rate,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'seed': args.seed,
            'dataset': args.dataset_dir
        }
    }

    output_dir = out / "checkpoint.pt"
    torch.save(checkpoint, output_dir)
    

def _init_model(args: object, logger: logging.Logger) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(getattr(args, "seed"))

    model = VGG16(num_classes=100).to(device)

    logger.info(
        f"Training {sum(p.numel() for p in model.parameters())} model parameters" 
        f"\nCompiled model using {_get_mem_stats(device)['curr_alloc_gb']} gb of memory"
    )
    return model


def _get_dataloaders(args) -> Tuple[DataLoader, DataLoader]:
    """
    Returns both the training and validating dataloader
    """
    root_dir = getattr(args, "dataset_dir")
    batch_size = getattr(args, "batch_size")

    train_transforms = get_train_transforms(do_augment=False)
    val_transforms = get_val_transforms(do_augment=False)

    train_dataset = BrainMRI(root_dir, split='train', transform=train_transforms)
    val_dataset = BrainMRI(root_dir, split='val', transform=val_transforms)

    print(
        f"Train dataset loaded with {len(train_dataset)} samples "
        f"\nVal dataset loaded with {len(val_dataset)} samples"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_dataloader, val_dataloader


def _get_mem_stats(device=None):
    mem = torch.cuda.memory_stats(device)
    props = torch.cuda.get_device_properties(device)
    return {
        "total_gb": 1e-9 * props.total_memory,
        "curr_alloc_gb": 1e-9 * mem["allocated_bytes.all.current"],
        "peak_alloc_gb": 1e-9 * mem["allocated_bytes.all.peak"],
        "curr_resv_gb": 1e-9 * mem["reserved_bytes.all.current"],
        "peak_resv_gb": 1e-9 * mem["reserved_bytes.all.peak"],
    }


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment_name", default=None)
    parser.add_argument("-d", "--dataset_dir", default=None, required=True)
    parser.add_argument("-o", "--output_dir", default="/models/vgg-16/experiments/")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-b", "--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=20, type=int)
    parser.add_argument("--log-freq", default=10, type=int)
    parser.add_argument("--ckpt-freq", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)

    return parser



if __name__ == "__main__":
    main()