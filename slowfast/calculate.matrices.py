#!/usr/bin/env python3

import os
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.models import build_model
from slowfast.datasets.utils import pack_pathway_output
from slowfast.config.defaults import get_cfg
from slowfast.utils.parser import load_config

logger = logging.get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure different matrices for video models"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/MobileNet.yaml",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_file",
        help="Path to the checkpoint file",
        default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/results/3DMobileNet_exp1/checkpoints/checkpoint_epoch_00040.pyth",
        type=str,
    )
    return parser.parse_args()


# This function measures the frames per second (FPS) of a model during inference.
def measure_fps(model, inputs, num_warmup=50, num_iterations=200):
    # Warm up
    for _ in range(num_warmup):
        model(inputs)
    
    # Measure inference time
    torch.cuda.synchronize() 
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        model(inputs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate FPS
    elapsed_time = end_time - start_time
    # Number of clips × number of frames per clip
    total_frames = num_iterations * inputs[0].shape[2]  # Assuming inputs[0] shape is [B, C, T, H, W]
    fps = total_frames / elapsed_time
    
    return fps, elapsed_time, total_frames

# This function Mesuare CPU Inference (By averaging the time taken for 1000 forward passes)
def measure_cpu_time(model, inputs, num_warmup=50, num_iterations=1000):
    # Warm up
    for _ in range(num_warmup):
        model(inputs)
    
    # Measure inference time
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in tqdm(range(num_iterations)):
        model(inputs)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate average time per forward pass
    elapsed_time = (end_time - start_time) / num_iterations
    
    return elapsed_time, num_iterations


def main():
    args = parse_args()
    
    # Load config
    cfg = get_cfg()
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    
    # Set up environment
    du.init_distributed_training(cfg)
    
    # Set random seed
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    
    # Build the model
    model = build_model(cfg)
    
    # Load checkpoint
    if args.checkpoint_file is not None and os.path.exists(args.checkpoint_file):
        cu.load_checkpoint(
            args.checkpoint_file, model, cfg.NUM_GPUS > 1
        )
    
    model.eval()
    
    # Create random input
    if cfg.MODEL.ARCH in cfg.MODEL.SINGLE_PATHWAY_ARCH:
        # For single pathway models like MobileNet
        rgb_dimension = 3
        input_tensors = torch.rand(
            1,  # batch size
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
    
    # Measure FPS for different batch sizes
    #batch_sizes = [1, 2, 4, 8, 16]
    batch_sizes = [1]

    print("\nMeasuring inference speed...")
    
    for batch_size in batch_sizes:
        # Duplicate the input to simulate a batch
        if isinstance(inputs, (list,)):
            batch_inputs = [x.repeat(batch_size, 1, 1, 1, 1) for x in inputs]
        else:
            batch_inputs = inputs.repeat(batch_size, 1, 1, 1, 1)
        
        # Measure FPS and CPU time
        with torch.no_grad():
            fps, elapsed_time, total_frames = measure_fps(model, batch_inputs)
            cpu_time = measure_cpu_time(model, batch_inputs)


        print(f"\nResults for batch size {batch_size}:")
        print(f"  - Total frames processed: {total_frames}")
        print(f"  - Elapsed time: {elapsed_time:.2f} seconds")
        print(f"  - Average FPS: {fps:.2f}")
        print(f"  - Time per frame: {1000/fps:.2f} ms")
        print(f"  - Average CPU time per forward pass: {cpu_time:.4f} seconds")
    
    # Model information
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,}")
    
    # Model size
    model_size = sum(p.element_size() * p.nelement() for p in model.parameters())
    print(f"Model size: {model_size / (1024 ** 2):.2f} MB")

    # Calculate model FLOPs
    inputs_flops = [inp[0:1].clone() for inp in inputs]  # Take only the first item in batch for FLOP calculation
    flops, _ = misc.log_model_info(model, cfg, inputs=inputs_flops)
    print(f"Model FLOPs: {flops:,}")

if __name__ == "__main__":
    main()