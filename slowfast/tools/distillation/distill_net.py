#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Knowledge Distillation Training Script for SlowFast to X3D-M.
This script performs knowledge distillation from a larger teacher model (SlowFast)
to a smaller student model (X3D-M).
"""

import numpy as np
import pprint
import torch
import tqdm
import os
import sys
import argparse
import itertools
import copy
from fvcore.common.config import CfgNode

import slowfast.models.losses as losses
import slowfast.models.optimizer as optim
import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from slowfast.utils.multigrid import MultigridSchedule
from slowfast.config.defaults import get_cfg
import slowfast.visualization.tensorboard_vis as tb

logger = logging.get_logger(__name__)


def train_epoch(
    teacher_model,
    student_model,
    loader,
    distill_loss_fn,
    student_optimizer,
    train_meter,
    cur_epoch,
    cfg,
    writer=None,
):
    """
    Perform knowledge distillation training for one epoch.
    Args:
        teacher_model (model): the pre-trained teacher model.
        student_model (model): the student model to train.
        loader (loader): video loader.
        distill_loss_fn (nn.Module): distillation loss function.
        student_optimizer (optim): optimizer for student model parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """
    # Enable train mode.
    teacher_model.eval()  # Teacher is always in evaluation mode
    student_model.train()
    
    train_meter.iter_tic()
    data_size = len(loader)
    
    for cur_iter, (inputs, labels, _, meta) in enumerate(loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
        
        # Transfer the labels and meta data to the current GPU device.
        labels = labels.cuda()
        
        # Get teacher predictions (no grad needed)
        with torch.no_grad():
            teacher_preds = teacher_model(inputs)
        
        # Update the student model
        student_optimizer.zero_grad()
        student_preds = student_model(inputs)
        
        # Calculate distillation loss
        loss = distill_loss_fn(student_preds, teacher_preds, labels)
        
        # Check Nan Loss.
        misc.check_nan_losses(loss)
        
        # Perform the backward pass.
        loss.backward()
        # Update the parameters.
        student_optimizer.step()
        
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(student_preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / student_preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        
        # Gather all the predictions across all the devices.
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.all_reduce([loss, top1_err, top5_err])
            
        # Copy the stats from GPU to CPU (sync point).
        loss, top1_err, top5_err = (
            loss.item(),
            top1_err.item(),
            top5_err.item(),
        )
        
        train_meter.iter_toc()
        
        # Update and log stats.
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            inputs[0].size(0) if isinstance(inputs, (list,)) else inputs.size(0),
        )
        
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
        
    # Log epoch stats.
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def eval_epoch(
    student_model, val_loader, val_meter, cur_epoch, cfg, writer=None
):
    """
    Evaluate the student model on the validation set.
    Args:
        student_model (model): the student model to evaluate.
        val_loader (loader): data loader to provide validation data.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    student_model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)
            
        labels = labels.cuda()

        # Compute the predictions from student model
        preds = student_model(inputs)

        # Compute the errors
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])

        # Copy the stats from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()

        val_meter.iter_toc()
        
        # Update and log stats
        val_meter.update_stats(
            top1_err,
            top5_err,
            inputs[0].size(0) if isinstance(inputs, (list,)) else inputs.size(0),
        )
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    # Log epoch stats
    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the batch norm stats to make them more precise. During
    training both BN stats and the weight are changing after every
    iteration, so the running average can not precisely reflect the
    actual stats of the current model.
    In this function, the BN stats are recomputed with fixed weights, to make
    the running average more precise. Specifically, it computes the true average
    of per-batch mean/variance instead of the running average.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the BN stats.
        num_iters (int): number of iterations to compute and update the BN stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _get_num_samples_count(loader):
        """Get the number of samples in the data loader."""
        num_samples = 0
        for _, _, _, meta in loader:
            num_samples += meta["boxes"].shape[0] if "boxes" in meta else meta["img_size"].shape[0]
        return num_samples

    # Compute the number of mini-batches to use
    num_samples = _get_num_samples_count(loader)
    num_iters = min(num_samples // loader.batch_size, num_iters)

    # Retrieve the BN layers
    bn_layers = [
        m for m in model.modules() if any(
            isinstance(m, bn_type)
            for bn_type in [
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d, 
                torch.nn.BatchNorm3d
            ]
        )
    ]

    if len(bn_layers) == 0:
        return

    # Initialize variables to compute mean and variance
    running_mean = [torch.zeros_like(bn.running_mean) for bn in bn_layers]
    running_var = [torch.zeros_like(bn.running_var) for bn in bn_layers]
    
    # Remember the previous state
    previous_states = [
        (bn.running_mean.clone(), bn.running_var.clone(), bn.training)
        for bn in bn_layers
    ]

    # Set the BN layers to eval mode so they don't update themselves
    for bn in bn_layers:
        bn.eval()

    # Accumulate the statistics
    for inputs, _, _, _ in tqdm.tqdm(itertools.islice(loader, num_iters)):
        if use_gpu:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

        with torch.no_grad():
            model(inputs)

        for i, bn in enumerate(bn_layers):
            running_mean[i] += bn.running_mean
            running_var[i] += bn.running_var

    # Average over the iters
    for i, bn in enumerate(bn_layers):
        running_mean[i] /= num_iters
        running_var[i] /= num_iters
        bn.running_mean = running_mean[i]
        bn.running_var = running_var[i]
        bn.training = previous_states[i][2]


def build_teacher_model(cfg, teacher_cfg):
    """
    Build the pre-trained teacher model (SlowFast).
    Args:
        teacher_cfg (CfgNode): separate configs for teacher model
    Returns:
        model (nn.Module): the teacher model
    """
    # Build the teacher model architecture
    model = build_model(teacher_cfg)
    
    # Load pre-trained weights
    checkpoint_path = teacher_cfg.TRAIN.CHECKPOINT_FILE_PATH
    cu.load_checkpoint(checkpoint_path, model)
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU
    if cfg.NUM_GPUS > 0:
        model = model.cuda()
        
    return model


def build_student_model(cfg):
    """
    Build the student model (X3D-M).
    Args:
        cfg (CfgNode): configs. Details can be found in slowfast/config/defaults.py
    Returns:
        model (nn.Module): the student model
    """
    # Build the student model architecture
    model = build_model(cfg)
    
    # Initialize weights from scratch or load from checkpoint if provided
    if cfg.DISTILLATION.STUDENT_CHECKPOINT:
        cu.load_checkpoint(cfg.DISTILLATION.STUDENT_CHECKPOINT, model)
    
    # Move to GPU
    if cfg.NUM_GPUS > 0:
        model = model.cuda()
        
    return model


def distill_knowledge(cfg , teacher_cfg):
    """
    Main function for knowledge distillation from a teacher model (SlowFast)
    to a student model (X3D-M).
    Args:
        cfg (CfgNode): configs for distillation. Details can be found in slowfast/config/defaults.py
        teacher_cfg (CfgNode, optional): separate configs for teacher model if provided
    """
    # Set up logging format
    logging.setup_logging(cfg.OUTPUT_DIR)
    
    # Print config
    logger.info("Knowledge Distillation with config:")
    logger.info(pprint.pformat(cfg))
    if teacher_cfg is not None:
        logger.info("Teacher model config:")
        logger.info(pprint.pformat(teacher_cfg))
    
    
    # Build teacher and student models
    teacher_model = build_teacher_model(teacher_cfg)
    student_model = build_student_model(cfg)
    
    # Create distillation loss function
    distill_loss_fn = losses.DistillationLoss(
        alpha=cfg.DISTILLATION.ALPHA, 
        temperature=cfg.DISTILLATION.TEMPERATURE
    )
    
    # Create the video train and val loaders
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")
    
    # Create meters to metric the training and validation stats
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)
    
    # Create optimizer for student model
    student_optimizer = optim.construct_optimizer(student_model, cfg)
    
    # Create scheduler for student optimizer
    student_scheduler = optim.get_policy(student_optimizer, cfg)
    
    # Main distillation loop
    start_epoch = 0
    if cfg.DISTILLATION.STUDENT_CHECKPOINT:
        start_epoch = cu.load_checkpoint(
            cfg.DISTILLATION.STUDENT_CHECKPOINT, student_model, student_optimizer
        )[0]
        
    # Setup tensorboard if enabled
    if cfg.TENSORBOARD.ENABLE and du.is_master_proc(
        cfg.NUM_GPUS * cfg.NUM_SHARDS
    ):
        writer = tb.TensorboardWriter(cfg)
    else:
        writer = None
        
    # Perform the training loop
    logger.info("Start knowledge distillation training")
    epoch_timer = EpochTimer()
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset
        loader.shuffle_dataset(train_loader, cur_epoch)
        
        # Train for one epoch
        epoch_timer.epoch_tic()
        train_epoch(
            teacher_model,
            student_model,
            train_loader,
            distill_loss_fn,
            student_optimizer,
            train_meter,
            cur_epoch,
            cfg,
            writer,
        )
        epoch_timer.epoch_toc()
        
        # Update learning rate
        lr = student_scheduler.get_last_lr()[0]
        student_optimizer = student_scheduler.step()
        
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            bn_modules = [
                m for m in student_model.modules() if isinstance(m, (
                    torch.nn.BatchNorm1d, 
                    torch.nn.BatchNorm2d, 
                    torch.nn.BatchNorm3d
                ))
            ]
            if len(bn_modules) > 0:
                calculate_and_update_precise_bn(
                    train_loader, 
                    student_model, 
                    min(cfg.BN.NUM_BATCHES_PRECISE, len(train_loader)),
                    cfg.NUM_GPUS > 0,
                )
            
        # Save checkpoint
        if cu.is_checkpoint_epoch(
            cur_epoch, cfg.TRAIN.CHECKPOINT_PERIOD
        ):
            checkpoint_file_path = cu.save_checkpoint(
                cfg.OUTPUT_DIR, 
                student_model,
                student_optimizer,
                cur_epoch, 
                cfg,
            )
            
        # Evaluate the model on validation set
        if misc.is_eval_epoch(cfg, cur_epoch):
            eval_epoch(
                student_model,
                val_loader,
                val_meter,
                cur_epoch,
                cfg,
                writer,
            )
            
    # Final checkpoint
    if cu.is_checkpoint_epoch(
        cfg.SOLVER.MAX_EPOCH - 1, cfg.TRAIN.CHECKPOINT_PERIOD
    ):
        checkpoint_file_path = cu.save_checkpoint(
            cfg.OUTPUT_DIR, 
            student_model,
            student_optimizer,
            cfg.SOLVER.MAX_EPOCH - 1, 
            cfg,
        )
        
    if writer is not None:
        writer.close()


def parse_args():
    """
    Parse the following arguments for the knowledge distillation binary.
    Args:
        cfg (str): path to the config file.
        opts (argument): options to modify config from command line.
    """
    parser = argparse.ArgumentParser(
        description="Knowledge distillation from SlowFast to X3D-M"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the distillation config file",
        default="",
        type=str,
    )
    
    return parser.parse_args()


def load_config(args):
    """
    Given the arguments, load and initialize the configs.
    Args:
        args (argument): arguments includes `cfg_file` and `opts`.
    Returns:
        cfg (CfgNode): configs for distillation.
        teacher_cfg (CfgNode): configs for teacher model if provided.
    """
    # Setup main distillation cfg
    cfg = get_cfg()
    teacher_cfg = None
    
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)

    if args.DISTILLATION.TEACHER_CFG_FILE is not None:
        teacher_cfg = get_cfg()
        teacher_cfg.merge_from_file(args.DISTILLATION.TEACHER_CFG_FILE)   


    # Create output directory if not exists
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg, teacher_cfg


def main():
    """
    Main function to start knowledge distillation training.
    """
    # Setup environment
    args = parse_args()
    cfg, teacher_cfg = load_config(args)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    
    # Setup distributed training
    du.init_distributed_training(cfg)

    # Validate configuration
    if teacher_cfg is None:
        print(f"Error: Teacher config file path not provided.")
        return 
    
    if not os.path.exists(cfg.DISTILLATION.TEACHER_CHECKPOINT):
        print(f"Error: Teacher checkpoint not found: {cfg.DISTILLATION.TEACHER_CHECKPOINT}")
        return

    # Print config information
    print("Knowledge Distillation:")
    print(f"  Teacher: {cfg.DISTILLATION.TEACHER_ARCH} from {cfg.DISTILLATION.TEACHER_CHECKPOINT}")
    if cfg.DISTILLATION.TEACHER_CFG_FILE:
        print(f"  Teacher Config: {cfg.DISTILLATION.TEACHER_CFG_FILE}")
    print(f"  Student: {cfg.DISTILLATION.STUDENT_ARCH} (X3D-M)")
    print(f"  Temperature: {cfg.DISTILLATION.TEMPERATURE}")
    print(f"  Alpha: {cfg.DISTILLATION.ALPHA}")
    print(f"  Output Directory: {cfg.OUTPUT_DIR}")
    
    # Freeze the config
    cfg.freeze()
    if teacher_cfg is not None:
        teacher_cfg.freeze()

    # Initialize distillation process
    distill_knowledge(cfg, teacher_cfg)


if __name__ == "__main__":
    main()