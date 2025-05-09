#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Apply post-training static quantization to a pre-trained X3D model."""

import numpy as np
import pprint
import torch
import os
import sys
import argparse
import time

# Add the path to the slowfast module
sys.path.insert(0, '/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast')

import torch.nn as nn
import torch.quantization as quantization
from torch.quantization import get_default_qconfig, prepare, convert

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import ValMeter
from slowfast.config.defaults import get_cfg

logger = logging.get_logger(__name__)

def dequantize_model(model):
    """
    Dequantize a quantized model to floating point for modules
    that don't support quantization
    """
    logger.info("Dequantizing problematic layers...")
    
    # Create state dict to hold dequantized parameters
    state_dict = {}
    
    # Dequantize state_dict
    for k, v in model.state_dict().items():
        if hasattr(v, 'is_quantized') and v.is_quantized:
            try:
                state_dict[k] = torch.dequantize(v)
            except Exception as e:
                logger.warning(f"Failed to dequantize {k}: {e}")
                state_dict[k] = v
        else:
            state_dict[k] = v
    
    # Load dequantized state dict
    model.load_state_dict(state_dict)
    
    return model

def apply_static_quantization(cfg):
    """
    Apply post-training static quantization to a model.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    """
    # Set up environment
    du.init_distributed_training(cfg)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Print config
    logger.info("Post-Training Static Quantization with config:")
    logger.info(pprint.pformat(cfg))
    
    # Set quantization backend
    backend = cfg.QUANTIZATION.BACKEND if cfg.QUANTIZATION.BACKEND else 'fbgemm'
    torch.backends.quantized.engine = backend
    logger.info(f"Using quantization backend: {backend}")
    
    # Create calibration data loader
    logger.info("Creating calibration data loader...")
    calib_loader = loader.construct_loader(cfg, "train")
    
    # Limit calibration batches for efficiency
    cfg.QUANTIZATION.CALIBRATION_NUM_BATCHES = min(
        cfg.QUANTIZATION.CALIBRATION_NUM_BATCHES 
        if hasattr(cfg.QUANTIZATION, "CALIBRATION_NUM_BATCHES") else 10, 
        len(calib_loader)
    )
    logger.info(f"Using {cfg.QUANTIZATION.CALIBRATION_NUM_BATCHES} batches for calibration")
    
    # Build floating point model
    logger.info("Building floating point model...")
    original_cfg = cfg.clone()
    original_cfg.QUANTIZATION.ENABLE = False
    model_fp = build_model(original_cfg)
    
    # Load pre-trained model weights
    logger.info(f"Loading weights from {cfg.TEST.CHECKPOINT_FILE_PATH}")
    if not os.path.exists(cfg.TEST.CHECKPOINT_FILE_PATH):
        logger.error(f"Checkpoint file not found: {cfg.TEST.CHECKPOINT_FILE_PATH}")
        return
        
    cu.load_test_checkpoint(cfg, model_fp)
    
    # Set model to evaluation mode
    model_fp.eval()
    
    # Print model information
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model_fp, cfg, use_train_input=False)
    
    # Setup quantization configuration
    logger.info("Setting up quantization configuration...")
    
    # Set qconfig for the model
    if backend == 'fbgemm':
        qconfig = get_default_qconfig('fbgemm')
    else:
        qconfig = get_default_qconfig('qnnpack')
    
    # Define qconfig mapping to skip certain layers
    #qconfig_mapping = torch.quantization.QConfigMapping()
    #qconfig_mapping.global_qconfig = qconfig 
    
    # Optional: Skip quantization for specific modules (adapt as needed)
    # qconfig_mapping.set_module_name("model.head", None)  # Skip head quantization if problematic
    
    # Create a modified version of the model with QuantStubs and DeQuantStubs
    class ModelWithStubs(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.quant = torch.quantization.QuantStub()
            self.model = model
            self.dequant = torch.quantization.DeQuantStub()
            
        def forward(self, x):
            if isinstance(x, list):
                # Handle list inputs for SlowFast
                result = []
                for pathway in x:
                    pathway = self.quant(pathway)
                    # We leave the model's internal operations unmodified
                result = self.model(x)
                return result
            else:
                x = self.quant(x)
                x = self.model(x)
                x = self.dequant(x)
                return x
    
    # Create model with stubs and set to CPU
    model_fp = ModelWithStubs(model_fp)
    model_fp.qconfig = qconfig
    
    model_fp.cpu()
    model_fp.eval()  # Set to evaluation mode
    
    # Prepare model for static quantization
    logger.info("Preparing model for static quantization...")
    prepared_model = prepare(model_fp)
    
    # Calibrate the model
    logger.info("Calibrating model...")
    with torch.no_grad():
        batch_count = 0
        for inputs, _, _, _, _ in calib_loader:
            # Move inputs to CPU as quantization only supports CPU
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cpu()
            else:
                inputs = inputs.cpu()
            
            # Forward pass to collect statistics for quantization
            prepared_model(inputs)
            
            batch_count += 1
            if batch_count >= cfg.QUANTIZATION.CALIBRATION_NUM_BATCHES:
                break
    
    # Convert the model to quantized version
    logger.info("Converting model to quantized format...")
    try:
        quantized_model = convert(prepared_model, inplace=False)
        logger.info("Model successfully converted to quantized format")
    except Exception as e:
        logger.error(f"Error during model conversion: {e}")
        logger.info("Attempting fallback approach with dequantized problematic layers...")
        # If full quantization fails, try to dequantize problematic layers
        quantized_model = dequantize_model(prepared_model)
    
    # Save the quantized model
    quantized_model_path = os.path.join(cfg.OUTPUT_DIR, "static_quantized_model.pth")
    torch.save({
        "model_state": quantized_model.state_dict(),
        "cfg": cfg,
    }, quantized_model_path)
    logger.info(f"Quantized model saved to: {quantized_model_path}")
    
    # Create test data loader
    logger.info("Creating test data loader...")
    test_loader = loader.construct_loader(cfg, "test")
    
    # Create meters for testing
    test_meter = ValMeter(len(test_loader), cfg)
    
    # Benchmark and evaluate floating point model
    logger.info("Benchmarking and evaluating floating point model...")
    benchmark_model(model_fp, "Floating Point Model", test_loader, cfg, test_meter)
    
    # Benchmark and evaluate quantized model
    logger.info("Benchmarking and evaluating quantized model...")
    benchmark_model(quantized_model, "Quantized Model", test_loader, cfg, test_meter)
    
    # Print model size comparison
    fp_size = get_model_size(model_fp)
    q_size = get_model_size(quantized_model)
    logger.info(f"Floating point model size: {fp_size:.2f} MB")
    logger.info(f"Quantized model size: {q_size:.2f} MB")
    logger.info(f"Size reduction: {(1 - q_size/fp_size) * 100:.2f}%")
    
    return quantized_model

def get_model_size(model):
    """Calculate model size in MB"""
    torch.save(model.state_dict(), "temp_model.pt")
    size_mb = os.path.getsize("temp_model.pt") / (1024 * 1024)
    os.remove("temp_model.pt")
    return size_mb

def benchmark_model(model, model_name, test_loader, cfg, test_meter):
    """Benchmark and evaluate a model"""
    # Reset meter
    test_meter.reset()
    test_meter.iter_tic()
    
    # Set model to evaluation mode
    model.eval()
    
    # Move model to appropriate device
    device = "cpu" if not cfg.NUM_GPUS else "cuda"
    model = model.to(device)
    
    # Warming up
    logger.info(f"Warming up {model_name}...")
    dummy_inputs = next(iter(test_loader))[0]
    if isinstance(dummy_inputs, (list,)):
        for i in range(len(dummy_inputs)):
            dummy_inputs[i] = dummy_inputs[i].to(device)
    else:
        dummy_inputs = dummy_inputs.to(device)
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_inputs)
    
    # Timing
    logger.info(f"Measuring inference time for {model_name}...")
    inference_times = []
    
    # Test loop
    for cur_iter, (inputs, labels, _, _, meta) in enumerate(test_loader):
        # Transfer the data to the current device
        if device == "cuda":
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
        
        test_meter.data_toc()
        
        # Measure inference time
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            preds = model(inputs)
            
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        inference_times.append(end_time - start_time)
        
        # Compute error metrics
        if not cfg.DATA.MULTI_LABEL:
            # Compute the errors
            num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
            
            # Combine the errors across the GPUs
            top1_err, top5_err = [
                (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
            ]
            
            if cfg.NUM_GPUS > 1:
                top1_err, top5_err = du.all_reduce([top1_err, top5_err])
                
            # Copy the errors from GPU to CPU (sync point)
            top1_err, top5_err = top1_err.item(), top5_err.item()
            
            test_meter.iter_toc()
            # Update and log stats
            test_meter.update_stats(
                top1_err,
                top5_err,
                inputs[0].size(0) if isinstance(inputs, list) else inputs.size(0)
            )
            test_meter.update_predictions(preds, labels)
        
        test_meter.log_iter_stats(0, cur_iter)
        test_meter.iter_tic()
    
    # Log final stats
    test_meter.log_epoch_stats(0)
    
    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)
    logger.info(f"{model_name} - Average inference time: {avg_inference_time*1000:.2f} ms")
    
    # Calculate parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"{model_name} - Total parameters: {total_params/1e6:.2f}M")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-Training Static Quantization for Video Classification"
    )
    
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/configs/meccano/X3D_M.yaml",
        type=str,
    )
    
    parser.add_argument(
        "--backend",
        dest="backend",
        help="Quantization backend: 'fbgemm' (x86) or 'qnnpack' (ARM)",
        default="x86",
        choices=["fbgemm", "qnnpack", "x86"],
        type=str,
    )
    
    parser.add_argument(
        "--calib_batches",
        dest="calib_batches",
        help="Number of batches for calibration",
        default=10,
        type=int,
    )
    
    args = parser.parse_args()
    
    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_file)
    
    # Update config with command line arguments

    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    
    # Set quantization parameters
    cfg.QUANTIZATION.ENABLE = True
    cfg.QUANTIZATION.QAT = False  # Using PTQ, not QAT
    cfg.QUANTIZATION.BACKEND = args.backend
    cfg.QUANTIZATION.CALIBRATION_NUM_BATCHES = args.calib_batches
    
    # Print config
    logger.info(pprint.pformat(cfg))
    
    # Apply post-training static quantization
    apply_static_quantization(cfg)