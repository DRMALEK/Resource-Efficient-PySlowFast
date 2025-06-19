import os
import torch
import torch.nn as nn
import nni.algorithms.compression.pytorch.pruning as pruning
import argparse
from slowfast.utils.parser import load_config, parse_args
from slowfast.models import build_model
from slowfast.config.defaults import get_cfg
from slowfast.utils.checkpoint import load_checkpoint
from slowfast.utils.misc import launch_job
from slowfast.utils.distributed import init_distributed_training
from slowfast.engine import train_net, test_net
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

def parse_custom_args():
    parser = argparse.ArgumentParser(description="X3D Model Pruning Pipeline")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to the config file", 
                      default="/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast/configs/Kinetics/X3D_M.yaml", type=str)
    parser.add_argument("--pretrained", help="Path to pretrained model", type=str, default="")
    parser.add_argument("--pruning_method", help="NNI pruning method", 
                      choices=["level", "l1", "l2", "fpgm", "slim", "taylor"], default="l1")
    parser.add_argument("--sparsity", help="Target sparsity level (0.0-1.0)", type=float, default=0.5)
    parser.add_argument("--mode", help="Pipeline stage", 
                      choices=["pretrain", "prune", "finetune", "full"], default="full")
    parser.add_argument("--pruned_model", help="Path to pruned model for finetuning", type=str, default="")
    parser.add_argument("--output_dir", help="Output directory", type=str, default="./pruning_output")
    parser.add_argument("--eval_after_finetune", help="Evaluate model after finetuning", action="store_true")
    parser.add_argument("--sparsity_step", help="Sparsity increment per iteration", type=float, default=0.1)
    parser.add_argument("--target_sparsity", help="Target final sparsity", type=float, default=0.9)
    
    return parser.parse_args()

def setup_cfg(args):
    """
    Set up the configuration for model training.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.cfg_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Create output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    # Create checkpoints directory if it doesn't exist
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        
    # Set output directory
    cfg.OUTPUT_DIR = args.output_dir
        
    return cfg

def prepare_data_loader(cfg):
    """
    Prepare a minimal data loader for methods that need sample data
    """
    # This is a placeholder. In a real implementation, you would create
    # a proper data loader based on your dataset
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy dataset for Taylor pruning
    # For actual implementation, use your real dataset
    dummy_data = torch.rand(10, 3, cfg.DATA.NUM_FRAMES, 
                           cfg.DATA.TRAIN_CROP_SIZE, 
                           cfg.DATA.TRAIN_CROP_SIZE)
    dummy_labels = torch.randint(0, cfg.MODEL.NUM_CLASSES, (10,))
    dummy_dataset = TensorDataset(dummy_data, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=2)
    
    return dummy_loader

def prune_model(cfg, args, checkpoint_path):
    """
    Prune the model using NNI
    """
    # Build the model
    model = build_model(cfg)
    
    # Load pretrained weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Handle both raw model state_dict and checkpoint dictionary formats
        if "model_state" in checkpoint:
            load_checkpoint(checkpoint_path, model, data_parallel=False)
        else:
            model.load_state_dict(checkpoint["model_state"])
            
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return None
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Configure pruner based on specified method
    config = [{'sparsity': args.sparsity, 'op_types': ['Conv3d']}]
    
    if args.pruning_method == "level":
        pruner = pruning.LevelPruner(model, config)
    elif args.pruning_method == "l1":
        pruner = pruning.L1NormPruner(model, config)
    elif args.pruning_method == "l2":
        pruner = pruning.L2NormPruner(model, config)
    elif args.pruning_method == "fpgm":
        pruner = pruning.FPGMPruner(model, config)
    elif args.pruning_method == "slim":
        # For slim pruner, we need to first do a training phase with sparsity regularization
        logger.warning("SlimPruner requires pre-training with BatchNorm scaling factors regularization")
        pruner = pruning.SlimPruner(model, [{'sparsity': args.sparsity, 'op_types': ['BatchNorm3d']}])
    elif args.pruning_method == "taylor":
        # Taylor pruning needs real data to compute importance scores
        dummy_loader = prepare_data_loader(cfg)
        pruner = pruning.TaylorFOWeightPruner(
            model, config, trainer=None, traced_optimizer=None,
            criterion=torch.nn.CrossEntropyLoss(), training_batches=5,
            num_iterations=5, data_loader=dummy_loader
        )
    
    # Compress the model
    _, masks = pruner.compress()
    
    # Model speedup by removing pruned weights
    from nni.compression.pytorch.speedup import ModelSpeedup
    dummy_input = torch.rand(1, 3, cfg.DATA.NUM_FRAMES, cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE).to(device)
    
    # Speedup the model
    logger.info("Applying model speedup...")
    ModelSpeedup(model, dummy_input, masks, device).speedup_model()
    
    # Save pruned model
    pruned_model_path = os.path.join(args.output_dir, f"pruned_model_{args.pruning_method}_sparsity{int(args.sparsity*100)}.pyth")
    torch.save({
        'model_state': model.state_dict(),
        'cfg': cfg,
        'pruning_method': args.pruning_method,
        'sparsity': args.sparsity
    }, pruned_model_path)
    
    logger.info(f"Pruned model saved to {pruned_model_path}")
    
    # Calculate model stats
    total_params = sum(p.numel() for p in model.parameters())
    non_zero = sum(torch.count_nonzero(p).item() for p in model.parameters())
    actual_sparsity = 1.0 - non_zero / total_params
    
    logger.info(f"Model parameters: {total_params}")
    logger.info(f"Non-zero parameters: {non_zero}")
    logger.info(f"Actual sparsity: {actual_sparsity:.4f}")
    
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
    cfg.SOLVER.MAX_EPOCH = max(5, cfg.SOLVER.MAX_EPOCH // 2)  # Shorter training for finetuning, but at least 5 epochs
    
    # Launch finetuning job
    launch_job(cfg=cfg, init_method=cfg.DIST.INIT_METHOD, func=train_net)
    
    # Return path to finetuned model
    finetuned_model_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_epoch_final.pyth")
    
    if not os.path.exists(finetuned_model_path):
        logger.warning(f"Expected finetuned model not found at {finetuned_model_path}")
        # Try to find the latest checkpoint
        checkpoints_dir = os.path.join(cfg.OUTPUT_DIR, "checkpoints")
        if os.path.exists(checkpoints_dir):
            checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith("checkpoint_epoch")]
            if checkpoints:
                latest = sorted(checkpoints)[-1]
                finetuned_model_path = os.path.join(checkpoints_dir, latest)
                logger.info(f"Using latest checkpoint instead: {finetuned_model_path}")
    
    return finetuned_model_path

def evaluate_model(cfg, model_path):
    """
    Evaluate the model on validation set
    """
    # Update config for evaluation
    cfg.TRAIN.ENABLE = False
    cfg.TEST.ENABLE = True
    cfg.TRAIN.CHECKPOINT_FILE_PATH = model_path
    cfg.NUM_GPUS = 1  # Simplify for testing
    
    # Launch evaluation job
    results = launch_job(cfg=cfg, init_method=cfg.DIST.INIT_METHOD, func=test_net)
    
    logger.info(f"Model evaluation results: {results}")
    return results

def run_pipeline(args):
    """
    Run the complete pruning pipeline based on the specified mode
    """
    cfg = setup_cfg(args)
    
    if args.pretrained:
        model_path = args.pretrained
    else:
        logger.error("No pretrained model provided. Please provide a path to a pretrained model.")
        return None  
    
    if args.mode == "prune" or args.mode == "full":
        logger.info("=== PRUNING MODEL ===")
        pruned_model_path = prune_model(cfg, args, model_path)
        if not pruned_model_path:
            return None
    else:
        pruned_model_path = args.pruned_model
        
    if args.mode == "finetune" or args.mode == "full":
        logger.info("=== FINETUNING PRUNED MODEL ===")
        finetuned_model_path = finetune_model(cfg, pruned_model_path)
        logger.info(f"Finetuned model saved to: {finetuned_model_path}")
        
        if args.eval_after_finetune:
            logger.info("=== EVALUATING FINETUNED MODEL ===")
            evaluate_model(cfg, finetuned_model_path)
        
        return finetuned_model_path
    
    return pruned_model_path

def main():
    args = parse_custom_args()
    
    # Record starting model
    starting_model = args.pretrained
    
    # Save results for each iteration
    results = []
    current_sparsity = args.sparsity
    
    if args.mode == "full":
        # Iterative pruning and finetuning
        while current_sparsity <= args.target_sparsity:
            logger.info(f"=== ITERATIVE PRUNING: SPARSITY {current_sparsity:.2f} ===")
            
            # Set current sparsity
            args.sparsity = current_sparsity
            
            # Run pruning and finetuning for current sparsity
            finetuned_model_path = run_pipeline(args)
            
            if not finetuned_model_path:
                logger.error(f"Failed at sparsity {current_sparsity}. Stopping.")
                break
                
            # Save iteration results
            result = {
                'sparsity': current_sparsity,
                'model_path': finetuned_model_path
            }
            results.append(result)
            
            # Save a copy with descriptive name for easier reference
            descriptive_path = os.path.join(os.path.dirname(finetuned_model_path), 
                                          f"finetuned_{args.pruning_method}_sparsity{int(current_sparsity*100)}.pyth")
            torch.save(torch.load(finetuned_model_path), descriptive_path)
            
            # Use finetuned model as starting point for next iteration
            args.pretrained = finetuned_model_path
            
            # Increment sparsity for next iteration
            current_sparsity += args.sparsity_step
    else:
        # Single run mode
        run_pipeline(args)
    
    # Print summary of iterative results
    if results:
        logger.info("=== ITERATIVE PRUNING SUMMARY ===")
        for result in results:
            logger.info(f"Sparsity {result['sparsity']:.2f}: {result['model_path']}")

        # Save final results
        results_path = os.path.join(args.output_dir, "results.txt")
        with open(results_path, "w") as f:
            for result in results:
                f.write(f"Sparsity {result['sparsity']:.2f}: {result['model_path']}\n")

if __name__ == "__main__":
    main()