#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Implementation of one-shot structured global pruning for PyTorch models.
This type of pruning removes entire filters/channels rather than individual weights,
which can lead to actual hardware acceleration.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
import argparse
from functools import partial
import time
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("structured_pruning")

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="One-shot Structured Global Pruning for PyTorch Models")
    parser.add_argument("--model_path", type=str, required=True, 
                       help="Path to the pretrained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./pruned_models", 
                       help="Directory to save pruned models")
    parser.add_argument("--sparsity", type=float, default=0.5,
                       help="Target pruning ratio (0.0-1.0)")
    parser.add_argument("--dim", type=int, default=0, choices=[0, 1],
                       help="Dimension to prune: 0=filter pruning, 1=channel pruning")
    parser.add_argument("--norm", type=str, default="L1", choices=["L1", "L2"],
                       help="Norm to use for ranking importance")
    parser.add_argument("--module_types", type=str, default="Conv2d,Conv3d", 
                       help="Comma-separated list of module types to prune")
    parser.add_argument("--config_file", type=str, default=None,
                       help="Path to model configuration file (if needed)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for pruning")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model before and after pruning")
    parser.add_argument("--save_mask", action="store_true",
                       help="Save pruning masks for future reference")
    return parser.parse_args()

class StructuredGlobalPruning:
    """
    Implements one-shot structured global pruning across a PyTorch model.
    This pruning removes entire filters/channels based on global ranking of their importance.
    """
    def __init__(self, model, sparsity=0.5, dim=0, norm_type="L1", module_types=None):
        """
        Args:
            model: PyTorch model to prune
            sparsity: Target pruning ratio (0.0-1.0)
            dim: Dimension to prune (0=filters, 1=channels)
            norm_type: Norm to use for importance ranking ("L1" or "L2")
            module_types: List of module types to prune (e.g., [nn.Conv2d, nn.Conv3d])
        """
        self.model = model
        self.sparsity = sparsity
        self.dim = dim  # 0 for filter pruning, 1 for channel pruning
        self.norm_type = norm_type
        
        if module_types is None:
            self.module_types = [nn.Conv2d, nn.Conv3d]
        else:
            self.module_types = module_types
            
        # Map modules to keep track of prunable layers
        self.prunable_modules = []
        self._map_modules()
        
    def _map_modules(self):
        """
        Find all prunable modules in the model.
        """
        for name, module in self.model.named_modules():
            if any(isinstance(module, module_type) for module_type in self.module_types):
                self.prunable_modules.append((name, module))
                
        logger.info(f"Found {len(self.prunable_modules)} prunable modules")
        
    def compute_importance_scores(self):
        """
        Compute importance scores for filters/channels in all prunable modules.
        Returns a global ranking of all filters/channels based on their importance.
        """
        all_scores = []
        
        for name, module in self.prunable_modules:
            weight = module.weight.data
            
            if self.dim == 0:  # Filter pruning
                # For each output filter, compute its importance
                if self.norm_type == "L1":
                    scores = weight.abs().sum(dim=[1, 2, 3] if weight.dim() == 4 else [1, 2, 3, 4])
                else:  # L2
                    scores = torch.sqrt((weight ** 2).sum(dim=[1, 2, 3] if weight.dim() == 4 else [1, 2, 3, 4]))
            else:  # Channel pruning (dim == 1)
                # For each input channel, compute its importance across all filters
                if self.norm_type == "L1":
                    scores = weight.abs().sum(dim=[0, 2, 3] if weight.dim() == 4 else [0, 2, 3, 4])
                else:  # L2
                    scores = torch.sqrt((weight ** 2).sum(dim=[0, 2, 3] if weight.dim() == 4 else [0, 2, 3, 4]))
            
            # Store (module_index, filter_index, score)
            for i, score in enumerate(scores):
                all_scores.append((name, i, score.item()))
        
        return all_scores
    
    def prune(self):
        """
        Perform one-shot structured global pruning.
        """
        logger.info(f"Starting structured {'filter' if self.dim == 0 else 'channel'} pruning with sparsity {self.sparsity}")
        
        # Get importance scores
        all_scores = self.compute_importance_scores()
        
        # Sort scores by importance (ascending, so less important first)
        all_scores.sort(key=lambda x: x[2])
        
        # Calculate number of filters/channels to prune
        num_filters_to_prune = int(len(all_scores) * self.sparsity)
        
        logger.info(f"Pruning {num_filters_to_prune} out of {len(all_scores)} {'filters' if self.dim == 0 else 'channels'}")
        
        # Create pruning masks
        masks = {}
        for module_name, module in self.prunable_modules:
            masks[module_name] = torch.ones_like(module.weight.data, dtype=torch.bool)
        
        # Set masks for filters/channels to prune
        for module_name, filter_idx, _ in all_scores[:num_filters_to_prune]:
            if self.dim == 0:  # Filter pruning
                masks[module_name][filter_idx] = 0
            else:  # Channel pruning
                masks[module_name][:, filter_idx] = 0
        
        # Apply masks
        for module_name, mask in masks.items():
            # Find the module by name
            module = dict(self.prunable_modules)[module_name]
            
            # Apply the mask
            module.weight.data *= mask
            
            # Also zero out the corresponding bias if exists
            if hasattr(module, 'bias') and module.bias is not None:
                if self.dim == 0:  # For filter pruning, zero out corresponding bias
                    module.bias.data *= mask[:, 0, 0, 0] if mask.dim() == 4 else mask[:, 0, 0, 0, 0]
        
        # Calculate actual sparsity after pruning
        total_params = 0
        zero_params = 0
        
        for _, module in self.prunable_modules:
            weight = module.weight.data
            total_params += weight.numel()
            zero_params += (weight == 0).sum().item()
        
        actual_sparsity = zero_params / total_params if total_params > 0 else 0
        logger.info(f"Actual weight sparsity after pruning: {actual_sparsity:.4f}")
        
        return masks
    
    def remove_pruned_filters(self):
        """
        Advanced feature: Actually remove pruned filters/channels from the model
        to create a smaller model. This is more complex and requires modifying
        model architecture, not just zeroing weights.
        
        Note: This is a placeholder implementation that would need to be expanded
        for actual use with specific model architectures.
        """
        logger.warning("Filter removal is not yet implemented - the model still has the same structure with zeroed weights")
        # This would require a more sophisticated implementation that analyzes the model graph
        # and rebuilds a new model with removed filters/channels
        pass

def save_model(model, output_dir, sparsity, masks=None):
    """
    Save pruned model and optionally the pruning masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save pruned model
    model_filename = os.path.join(output_dir, f"pruned_model_sparsity_{int(sparsity*100)}.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "sparsity": sparsity,
    }, model_filename)
    logger.info(f"Pruned model saved to {model_filename}")
    
    # Optionally save masks
    if masks is not None:
        masks_filename = os.path.join(output_dir, f"pruning_masks_sparsity_{int(sparsity*100)}.pth")
        torch.save(masks, masks_filename)
        logger.info(f"Pruning masks saved to {masks_filename}")

def evaluate_model(model, data_loader, device):
    """
    Simple evaluation function - placeholder for actual model evaluation.
    """
    logger.info("Model evaluation placeholder - implement your own evaluation logic")
    # Placeholder for model evaluation logic
    # In an actual implementation, you would:
    # 1. Load a validation dataset
    # 2. Run inference on the dataset
    # 3. Calculate and return metrics (accuracy, etc.)
    return {"accuracy": 0.0, "loss": 0.0}  # Return placeholder metrics

def measure_inference_time(model, input_shape, device, num_iterations=100):
    """
    Measure model inference time.
    """
    model.eval()
    model.to(device)
    
    # Create random input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    return avg_time

def load_model(model_path, config_file=None, device="cuda:0"):
    """
    Load PyTorch model from checkpoint.
    For specialized frameworks like SlowFast, you might need custom logic here.
    """
    # This is a placeholder implementation that would need to be adapted based on your needs
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try to directly load the model based on common checkpoint formats
        if "model_state_dict" in checkpoint:
            # Create model instance based on config
            if config_file:
                # This part depends on your model framework
                logger.error("Custom model loading from config not implemented")
                return None
            
            # For simple cases where we just need to load state dict
            model = torch.nn.Module()  # Placeholder, should be your actual model
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "model_state" in checkpoint:
            model = torch.nn.Module()  # Placeholder
            model.load_state_dict(checkpoint["model_state"])
        elif isinstance(checkpoint, dict) and not any(k in checkpoint for k in ["model_state_dict", "model_state"]):
            # Assume checkpoint is directly the state dict
            model = torch.nn.Module()  # Placeholder
            model.load_state_dict(checkpoint)
        else:
            logger.error("Unsupported checkpoint format")
            return None
            
        model.to(device)
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

def main():
    """
    Main function to perform structured global pruning.
    """
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(args.output_dir, "pruning_log.txt")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Starting structured pruning with args: {args}")
    
    # 1. Load model
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    logger.info(f"Using device: {device}")
    
    # Custom model loading function (placeholder)
    # You would need to customize this based on your model format
    model = load_model(args.model_path, args.config_file, device)
    if model is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Convert module type string to actual types
    module_types = []
    for module_type in args.module_types.split(","):
        if module_type.strip() == "Conv2d":
            module_types.append(nn.Conv2d)
        elif module_type.strip() == "Conv3d":
            module_types.append(nn.Conv3d)
        elif module_type.strip() == "Linear":
            module_types.append(nn.Linear)
        # Add more module types as needed
    
    # 2. Evaluate original model if specified
    if args.evaluate:
        logger.info("Evaluating original model")
        # This would be replaced with your actual evaluation logic
        # For example: original_metrics = evaluate_model(model, validation_loader)
        
        # Measure inference time (placeholder)
        original_inference_time = measure_inference_time(
            model, 
            (1, 3, 8, 224, 224),  # Example shape for video model, adjust as needed
            device
        )
        logger.info(f"Original model inference time: {original_inference_time*1000:.2f} ms")
    
    # 3. Create pruner instance
    pruner = StructuredGlobalPruning(
        model=model,
        sparsity=args.sparsity,
        dim=args.dim,
        norm_type=args.norm,
        module_types=module_types
    )
    
    # 4. Perform pruning
    masks = pruner.prune()
    
    # 5. Evaluate pruned model if specified
    if args.evaluate:
        logger.info("Evaluating pruned model")
        # pruned_metrics = evaluate_model(model, validation_loader)
        
        # Measure inference time for pruned model
        pruned_inference_time = measure_inference_time(
            model, 
            (1, 3, 8, 224, 224),  # Example shape, adjust as needed
            device
        )
        logger.info(f"Pruned model inference time: {pruned_inference_time*1000:.2f} ms")
        
        # Calculate speedup
        if 'original_inference_time' in locals():
            speedup = original_inference_time / pruned_inference_time
            logger.info(f"Speedup factor: {speedup:.2f}x")
    
    # 6. Save pruned model
    save_model(
        model=model,
        output_dir=args.output_dir,
        sparsity=args.sparsity,
        masks=masks if args.save_mask else None
    )
    
    logger.info("Pruning completed successfully")

if __name__ == "__main__":
    main()