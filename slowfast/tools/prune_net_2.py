#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Model pruning script using torch-pruning library."""

import argparse
import os
import torch
import torch_pruning as tp
import numpy as np
from functools import partial
import sys

# Add the path to the slowfast module or via export 'export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH'
sys.path.insert(0, '/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast')

from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
from slowfast.config.defaults import get_cfg
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.utils.misc import launch_job
from slowfast.utils.distributed import init_distributed_training
from slowfast.datasets.utils import pack_pathway_output
import slowfast.utils.logging as logging
from test_net import test
from train_net import train


# Set up proper logging
logger = logging.get_logger(__name__)


def parse_custom_args():
    parser = argparse.ArgumentParser(description="X3D Model Pruning Pipeline with torch-pruning")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", 
                      default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/configs/meccano/pruned/X3D_M_Pruned.yaml", type=str)
    
    # Pruning arguments
#    parser.add_argument("--pruning_method", help="Pruning method", 
#                      choices=["l1", "l2", "fpgm", "random"], default="l1")
#    parser.add_argument("--pruning_ratio", help="Target pruning ratio (0.0-1.0)", type=float, default=0.25)
#    parser.add_argument("--eval_after_finetune", help="Evaluate model after finetuning", default=True)
#    parser.add_argument("--global_pruning", help="Use global pruning strategy", default=False)
#    parser.add_argument("--max-epoch", help="Number of epochs for finetuning", type=int, default=1)

    return parser.parse_args()


def setup_cfg(args):
    """
    Set up the configuration for model training.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_file)
    
    # Create output folder
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
        
    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        
        
    return cfg


def prepare_dummy_input(cfg):
    """
    Prepare a dummy input tensor for the model.
    """
    # Create random input
    rgb_dimension = 3
    input_tensors = torch.rand(
        1,
        rgb_dimension,
        cfg.DATA.NUM_FRAMES,
        cfg.DATA.TEST_CROP_SIZE,
        cfg.DATA.TEST_CROP_SIZE,
    )
    
    inputs = pack_pathway_output(cfg, input_tensors)  # list of tensors (channels, time, height, width)

    if cfg.NUM_GPUS:
        # Transfer the data to the current GPU device
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                inputs[i] = inputs[i].cuda(non_blocking=True)
        else:
            inputs = inputs.cuda(non_blocking=True)

    return inputs


def prune_model(cfg, args):
    """
    Prune the model using torch-pruning
    """
    # Build the model
    model = build_model(cfg)
    
    checkpoint_path = cfg.PRUNING.CHECKPOINT_FILE_PATH

    # Load pretrained weights
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, cfg.NUM_GPUS > 1, None, weights_only=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.info(f"No checkpoint found at {checkpoint_path}, using random initialization")
    

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Prepare example inputs
    example_inputs = prepare_dummy_input(cfg)
    
    # Move inputs to the same device as the model
    if isinstance(example_inputs, (list,)):
        for i in range(len(example_inputs)):
            example_inputs[i] = example_inputs[i].to(device)
    else:
        example_inputs = example_inputs.to(device)
    
    # Configure importance criterion based on method
    if cfg.PRUNING.PRUNING_METHOD == "l1":
        importance = tp.importance.MagnitudeImportance(p=1)
    elif cfg.PRUNING.PRUNING_METHOD == "l2":
        importance = tp.importance.MagnitudeImportance(p=2)
    elif cfg.PRUNING.PRUNING_METHOD == "fpgm":
        importance = tp.importance.GroupNormImportance()
    elif cfg.PRUNING.PRUNING_METHOD == "random":
        importance = tp.importance.RandomImportance()
    
    # Ignore specific layers (like last linear layer)
    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == cfg.MODEL.NUM_CLASSES:
            ignored_layers.append(m)
    
    # Configure pruner
    iterative_steps = 1

    # Debug inputs
    if isinstance(example_inputs, (list,)):
        logger.info(f"Input is a list of length {len(example_inputs)}")
        for i, inp in enumerate(example_inputs):
            logger.info(f"Input {i} shape: {inp.shape}, dtype: {inp.dtype}, device: {inp.device}")
    else:
        logger.info(f"Input shape: {example_inputs.shape}, dtype: {example_inputs.dtype}, device: {example_inputs.device}")

    # Create a forward hook to wrap the model for torch_pruning
    # This addresses the pathway mismatch issue
    def model_forward_wrapper(model, model_input):
        #print(f"Model input type: {type(model_input)}")
        if not isinstance(model_input, (list, tuple)):
            if cfg.MODEL.ARCH in ["x3d", "mvit", "vision_transformer"]:
                # For X3D and other single pathway models, wrap in a list
                return model([model_input])
            else:
                return model(model_input)
        else:
            return model([model_input])


    pruner = tp.pruner.BasePruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=iterative_steps,
        pruning_ratio=cfg.PRUNING.PRUNING_RATE,
        global_pruning=cfg.PRUNING.GLOBAL,
        isomorphic=cfg.PRUNING.ISOMORPHIC,
        ignored_layers=ignored_layers,
        #forward_fn=model_forward_wrapper  # Use the wrapper to handle input formatting
    )
    
    logger.info(f"Using Global Pruning method: {cfg.PRUNING.GLOBAL}")

    # Print model statistics before pruning
    ori_size = tp.utils.count_params(model)
    logger.info("="*50)
    logger.info("Before pruning:")
    logger.info(f"Parameters: {ori_size}")
    
    # Perform pruning
    pruner.step()
    
    # Print model statistics after pruning
    pruned_size = tp.utils.count_params(model)
    logger.info("="*50)
    logger.info("After pruning:")
    logger.info(f"Parameters: {pruned_size}")
    logger.info(f"Pruned ratio: {1.0 - pruned_size/ori_size:.4f}")
    
    # Test forward pass after pruning
    with torch.no_grad():
        out = model(example_inputs)
        logger.info(f"Output shape: {out.shape}")
    
    # Save pruned model
    pruned_model_path = os.path.join(
        cfg.OUTPUT_DIR,
        f"pruned_model_{cfg.PRUNING.PRUNING_METHOD}_ratio{int(cfg.PRUNING.PRUNING_RATE*100)}.pyth"
    )
    
    # Save the whole model with wieghts, since it's stucture is changed    
    checkpoint = {
        "model": model,
    }
    torch.save(checkpoint, pruned_model_path)


    logger.info(f"Pruned model saved to {pruned_model_path}")
    
    return pruned_model_path


def finetune_model(cfg, pruned_model_path):
    """
    Finetune the pruned model
    """
    # Update config for finetuning
    cfg.TRAIN.CHECKPOINT_FILE_PATH = pruned_model_path
    cfg.TRAIN.ENABLE = True
    cfg.TEST.ENABLE = False

    # Adjust learning rate for finetuning
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * 0.1
    cfg.SOLVER.MAX_EPOCH = cfg.PRUNING.PRUNING_MAX_EPOCH  # Shorter training for finetuning
    
    # Launch finetuning job
    #launch_job(cfg=cfg, init_method="", func=train)
    train(cfg)



def evaluate_model(cfg):
    """
    Evaluate the model on validation set
    """
    # Update config for evaluation
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    

    # Launch evaluation job
    results = test(cfg)
    
    logger.info(f"Model evaluation results: {results}")
    return results


def run_pipeline(args):
    """
    Run the complete pruning pipeline based on the specified mode
    """
    cfg = setup_cfg(args)
    
    pruning_max_rate = cfg.PRUNING.PRUNING_MAX_RATE
    output_dir = cfg.OUTPUT_DIR
    starting_prune_rate = 0.05

    while starting_prune_rate <= pruning_max_rate:
        cfg.PRUNING.PRUNING_RATE = starting_prune_rate
        logger.info(f"Starting pruning with rate: {starting_prune_rate}")
        
        cfg.OUTPUT_DIR = os.path.join(output_dir, f"pruning_rate_{int(starting_prune_rate*100)}") # set output dir for each pruning rate
        logger.info(f"Output directory: {cfg.output_dir}")

        logging.setup_logging(cfg.OUTPUT_DIR)

        pruned_model_path = prune_model(cfg, args)
        logger.info("Model pruning completed.")

        finetune_model(cfg, pruned_model_path)
        logger.info("Model finetuning completed.")

        if cfg.PRUNING.EVALUATE_AFTER_FINE_TUNNING:
            evaluate_model(cfg)
            logger.info("Model evaluation completed.")

        starting_prune_rate += 0.05
        logger.info(f"Next pruning rate: {starting_prune_rate}")

def main():
    args = parse_custom_args()    
    run_pipeline(args)

if __name__ == "__main__":
    main()