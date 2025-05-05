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
        default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/configs/meccano/quantized/X3D_M_QAT.yaml",
        type=str,
    )
    parser.add_argument(
        "--checkpoint_file",
        help="Path to the checkpoint file",
        default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/results/x3d_M_QAT_exp1/quantized_model.pth",
        type=str,
    )
    return parser.parse_args()


# This function measures CPU inference time using torch.autograd.profiler.profile.
def measure_cpu_time_and_fps(model, inputs, num_warmup=50, num_iterations=1000):
    # Warm up
    for _ in range(num_warmup):
        model(inputs)
    
    # Measure inference time using profiler
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        for _ in tqdm(range(num_iterations)):
            model(inputs)
    
    # Print detailed profiling information
    print("\nDetailed CPU Profiling Information:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    
    # Calculate average time per forward pass
    print(prof.self_cpu_time_total)
    elapsed_time = prof.self_cpu_time_total / 1000000  # Convert from microsecond to seconds
    cpu_time_for_one_iteration = elapsed_time  / num_iterations  # Convert from milliseconds to seconds
    
    # Calcaute fbs
    total_frames = num_iterations * inputs[0].shape[2]  # Assuming inputs[0] shape is [B, C, T, H, W]
    fps = total_frames / elapsed_time

    return cpu_time_for_one_iteration, fps, total_frames, num_iterations


def main():
    args = parse_args()
    
    # Load config
    cfg = get_cfg()


    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    #if args.opts is not None:
    #    cfg.merge_from_list(args.opts)
    
    cfg.NUM_GPUS = 0 # Test on CPU


    # Set up environment
    du.init_distributed_training(cfg)
    
    # Set random seed
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    
    # Build the model
    model = build_model(cfg)
    
    # Load checkpoint
    if args.checkpoint_file is not None and os.path.exists(args.checkpoint_file):
        cu.load_test_checkpoint(
            cfg, model
        )
    
    model.eval()
    
    # Create random input
    rgb_dimension = 3
    input_tensors = torch.rand(
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
            print(batch_inputs[0].shape)
            cpu_time, fps, _, _ = measure_cpu_time_and_fps(model, batch_inputs, num_warmup=50, num_iterations=1000)


        print(f"\nResults for batch size {batch_size}:")
        print(f"  - Average FPS: {fps:.2f}")
        print(f"  - Time per frame: {1000/fps:.2f} ms")
        print(f"  - Average CPU time per forward pass: {cpu_time} seconds")
    
    # Model information
    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {params:,}")
    
    # Model size
    model_size = sum(p.element_size() * p.nelement() for p in model.parameters())
    print(f"Model size: {model_size / (1024 ** 2):.2f} MB")

    # Calculate model FLOPs
    inputs_flops = [inp[0:1].clone() for inp in inputs]  # Take only the first item in batch for FLOP calculation
    flops, _ = misc.log_model_info(model, cfg, use_train_input=False)
    print(f"Model FLOPs: {flops:,}")

if __name__ == "__main__":
    main()