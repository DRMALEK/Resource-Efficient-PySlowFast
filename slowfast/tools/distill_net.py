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
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats


# Add the path to the slowfast module or via export 'export PYTHONPATH=/path/to/SlowFast:$PYTHONPATH'
#sys.path.insert(0, '/workspace/Code/slowfast')
sys.path.insert(0, '/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast')


from slowfast.datasets.utils import pack_pathway_output
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
    teacher_cfg,
    writer=None,
):
    """
    Perform knowledge distillation training for one epoch.
    Args:
        teacher_model (model): the pre-trained teacher model (SlowFast).
        student_model (model): the student model to train (X3D-M).
        loader (loader): video training loader.
        distill_loss_fn (loss): distillation loss function.
        student_optimizer (optim): the optimizer to perform optimization on the student model's parameters.
        train_meter (TrainMeter): training meters to log the training performance.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs for the student model.
        teacher_cfg (CfgNode): configs for the teacher model.
        writer (TensorboardWriter, optional): TensorboardWriter object to writer Tensorboard log.
    """
    # Set teacher to eval mode and student to train mode
    teacher_model.eval()
    student_model.train()
    train_meter.iter_tic()
    data_size = len(loader)

    # Explicitly declare reduction to mean for standard classification loss
    cls_loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    
    # Create scaler for mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(loader):
        # Transfer the data to the current GPU device
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            
            labels = labels.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )
        
        # Update the learning rate
        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(student_optimizer, lr)

        train_meter.data_toc()
        
        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
            # Zero gradients
            student_optimizer.zero_grad()
            
            # Get teacher predictions (no grad)

            orginal_inputs = copy.deepcopy(inputs)

            with torch.no_grad():
                teacher_inputs = pack_pathway_output(teacher_cfg, inputs)
                teacher_preds = teacher_model(teacher_inputs)
                
            # Get student predictions
            student_inputs = pack_pathway_output(cfg, orginal_inputs)
            student_preds = student_model(student_inputs)
            
            # Compute distillation loss (combination of distillation and regular classification loss)
            loss = distill_loss_fn(student_preds, teacher_preds, labels)
            
        # Check for NaN losses
        misc.check_nan_losses(loss)
        
        # Backward pass with scaling for mixed precision training
        scaler.scale(loss).backward()
        
        # Unscale gradients for clipping
        scaler.unscale_(student_optimizer)
        
        # Clip gradients if necessary
        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                student_model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student_model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(student_model.parameters())
            
        # Update the parameters
        scaler.step(student_optimizer)
        scaler.update()
        
        # Compute accuracy metrics
        num_topks_correct = metrics.topks_correct(student_preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / student_preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        
        # Gather all the predictions across all the devices
        if cfg.NUM_GPUS > 1:
            loss, grad_norm, top1_err, top5_err = du.all_reduce(
                [loss.detach(), grad_norm, top1_err, top5_err]
            )
            
        # Copy the stats from GPU to CPU (sync point)
        loss, grad_norm, top1_err, top5_err = (
            loss.item(),
            grad_norm.item(),
            top1_err.item(),
            top5_err.item(),
        )
        
        # Update and log stats
        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            grad_norm,
            batch_size * max(cfg.NUM_GPUS, 1),
            None,  # No extra loss
        )
        
        # Write to tensorboard format if available
        if writer is not None:
            writer.add_scalars(
                {
                    "Train/loss": loss,
                    "Train/lr": lr,
                    "Train/Top1_err": top1_err,
                    "Train/Top5_err": top5_err,
                },
                global_step=data_size * cur_epoch + cur_iter,
            )
            
        train_meter.iter_toc()  # Measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        torch.cuda.synchronize()
        train_meter.iter_tic()
        
    # Clear memory
    del inputs
    torch.cuda.empty_cache()
    
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(
    student_model, val_loader, val_meter, cur_epoch, cfg, writer=None
):
    """
    Evaluate the student model on the validation set.
    Args:
        student_model (model): student model to evaluate.
        val_loader (loader): data loader to provide validation data.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): current epoch of training.
        cfg (CfgNode): configs for the student model.
        writer (TensorboardWriter, optional): TensorboardWriter object to writer Tensorboard log.
    """
    # Evaluation mode enabled. The running stats would not be updated.
    student_model.eval()
    val_meter.iter_tic()

    for cur_iter, (inputs, labels, index, time, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
            index = index.cuda()
            
        batch_size = (
            inputs[0][0].size(0) if isinstance(inputs[0], list) else inputs[0].size(0)
        )
        val_meter.data_toc()

        # Forward pass
        #student_inputs = pack_pathway_output(cfg, inputs)
        preds = student_model([inputs])
        
        # Compute the errors.
        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        
        # Combine the errors across the GPUs.
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.all_reduce([top1_err, top5_err])
            
        # Copy the errors from GPU to CPU (sync point).
        top1_err, top5_err = top1_err.item(), top5_err.item()
            
        val_meter.iter_toc()
        # Update and log stats.
        val_meter.update_stats(
            top1_err,
            top5_err,
            batch_size * max(cfg.NUM_GPUS, 1),
        )
        
        # Write to tensorboard format if available.
        if writer is not None:
            writer.add_scalars(
                {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                global_step=len(val_loader) * cur_epoch + cur_iter,
            )
            
        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
        
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    
    # Write to tensorboard format if available.
    if writer is not None:
        all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
        all_labels = [label.clone().detach() for label in val_meter.all_labels]
        if cfg.NUM_GPUS:
            all_preds = [pred.cpu() for pred in all_preds]
            all_labels = [label.cpu() for label in all_labels]
        writer.plot_eval(preds=all_preds, labels=all_labels, global_step=cur_epoch)
        
    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """
    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)

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
    checkpoint_path = teacher_cfg.TEST.CHECKPOINT_FILE_PATH
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
    teacher_model = build_teacher_model(cfg, teacher_cfg)
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
    
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        # Shuffle the dataset
        loader.shuffle_dataset(train_loader, cur_epoch)
        
        # Train for one epoch
        train_epoch(
            teacher_model,
            student_model,
            train_loader,
            distill_loss_fn,
            student_optimizer,
            train_meter,
            cur_epoch,
            cfg,
            teacher_cfg,
            writer,
        )
        
        # Update learning rate
        lr = optim.get_epoch_lr(cur_epoch + 1, cfg)
        optim.set_lr(student_optimizer, lr)
        
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
        default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/configs/meccano/distilled/SlowFast_to_X3D_M.yaml",
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

    if cfg.DISTILLATION.TEACHER_CFG_FILE is not None:
        teacher_cfg = get_cfg()
        teacher_cfg.merge_from_file(cfg.DISTILLATION.TEACHER_CFG_FILE)   


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


    logging.setup_logging(cfg.OUTPUT_DIR)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    
    # Setup distributed training
    du.init_distributed_training(cfg)

    # Validate configuration
    if teacher_cfg is None:
        logger.error("Error: Teacher config file path not provided.")
        return 
    
    if not os.path.exists(teacher_cfg.TEST.CHECKPOINT_FILE_PATH):
        logger.error(f"Error: Teacher checkpoint not found: {teacher_cfg.TEST.CHECKPOINT_FILE_PATH}")
        return

    # Print config information
    logger.info("Knowledge Distillation:")
    logger.info(f"  Teacher: {cfg.DISTILLATION.TEACHER_ARCH} from {teacher_cfg.TEST.CHECKPOINT_FILE_PATH}")
    if cfg.DISTILLATION.TEACHER_CFG_FILE:
        logger.info(f"  Teacher Config: {cfg.DISTILLATION.TEACHER_CFG_FILE}")
    logger.info(f"  Student: {cfg.DISTILLATION.STUDENT_ARCH} (X3D-M)")
    logger.info(f"  Temperature: {cfg.DISTILLATION.TEMPERATURE}")
    logger.info(f"  Alpha: {cfg.DISTILLATION.ALPHA}")
    logger.info(f"  Output Directory: {cfg.OUTPUT_DIR}")
    
    # Freeze the config
    cfg.freeze()
    if teacher_cfg is not None:
        teacher_cfg.freeze()

    # Initialize distillation process
    distill_knowledge(cfg, teacher_cfg)


if __name__ == "__main__":
    main()