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
    
    # Add distributed training arguments
    parser.add_argument("opts", help="additional options for config", default=None, 
                      nargs=argparse.REMAINDER)
    parser.add_argument("--num_shards", help="Number of shards", default=1, type=int)
    parser.add_argument("--shard_id", help="Shard id", default=0, type=int)
    parser.add_argument("--dist_url", help="distributed training url", default="tcp://localhost:9999")
    parser.add_argument("--port", help="Port to use", default=-1)
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
        
    # Set output directory
    cfg.OUTPUT_DIR = args.output_dir
        
    return cfg

def pretrain_model(cfg):
    """
    Pretrain the model or use a pretrained checkpoint
    """
    # Launch training job
    launch_job(cfg=cfg, init_method=cfg.DIST.INIT_METHOD, func=train_net)
    return os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_epoch_final.pyth")

def prune_model(cfg, args, checkpoint_path):
    """
    Prune the model using NNI
    """
    # Build the model
    model = build_model(cfg)
    
    # Load pretrained weights
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, data_parallel=False)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return None
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Configure pruner based on specified method
    if args.pruning_method == "level":
        pruner = pruning.LevelPruner(model, [{'sparsity': args.sparsity, 'op_types': ['Conv3d']}])
    elif args.pruning_method == "l1":
        pruner = pruning.L1NormPruner(model, [{'sparsity': args.sparsity, 'op_types': ['Conv3d']}])
    elif args.pruning_method == "l2":
        pruner = pruning.L2NormPruner(model, [{'sparsity': args.sparsity, 'op_types': ['Conv3d']}])
    elif args.pruning_method == "fpgm":
        pruner = pruning.FPGMPruner(model, [{'sparsity': args.sparsity, 'op_types': ['Conv3d']}])
    elif args.pruning_method == "slim":
        # For slim pruner, we need to first do a training phase with sparsity regularization
        logger.warning("SlimPruner requires pre-training with BatchNorm scaling factors regularization")
        pruner = pruning.SlimPruner(model, [{'sparsity': args.sparsity, 'op_types': ['BatchNorm3d']}])
    elif args.pruning_method == "taylor":
        pruner = pruning.TaylorFOWeightPruner(model, [{'sparsity': args.sparsity, 'op_types': ['Conv3d']}])
    
    # Compress the model
    _, masks = pruner.compress()
    
    # Model speedup by removing pruned weights
    from nni.compression.pytorch.speedup import ModelSpeedup
    dummy_input = torch.rand(1, 3, cfg.DATA.NUM_FRAMES, cfg.DATA.TRAIN_CROP_SIZE, cfg.DATA.TRAIN_CROP_SIZE).to(device)
    
    # Speedup the model
    logger.info("Applying model speedup...")
    ModelSpeedup(model, dummy_input, masks, device).speedup_model()
    
    # Save pruned model
    pruned_model_path = os.path.join(args.output_dir, f"pruned_model_{args.pruning_method}_{int(args.sparsity*100)}.pyth")
    torch.save({
        'model_state': model.state_dict(),
        'cfg': cfg,
        'pruning_method': args.pruning_method,
        'sparsity': args.sparsity
    }, pruned_model_path)
    
    logger.info(f"Pruned model saved to {pruned_model_path}")
    
    # Calculate model stats
    total_params = sum(p.numel() for p in model.parameters())
    non_zero = sum(p.nonzero().size(0) for p in model.parameters())
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
    
    # Adjust learning rate for finetuning
    cfg.SOLVER.BASE_LR = cfg.SOLVER.BASE_LR * 0.1
    cfg.SOLVER.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH // 2  # Shorter training for finetuning
    
    # Launch finetuning job
    launch_job(cfg=cfg, init_method=cfg.DIST.INIT_METHOD, func=train_net)
    
    # Test finetuned model
    launch_job(cfg=cfg, init_method=cfg.DIST.INIT_METHOD, func=test_net)
    
    return os.path.join(cfg.OUTPUT_DIR, "checkpoints", "checkpoint_epoch_final.pyth")

def run_pipeline(args):
    """
    Run the complete pruning pipeline based on the specified mode
    """
    cfg = setup_cfg(args)
    
    if args.pretrained:
        pretrained_model_path = args.pretrained
    else:
        print("No pretrained model provided. please povide a path to a pretrained model.")
        exit()   
    
    if args.mode == "prune" or args.mode == "full":
        logger.info("=== PRUNING MODEL ===")
        pruned_model_path = prune_model(cfg, args, pretrained_model_path)
    else:
        pruned_model_path = args.pruned_model
        
    if args.mode == "finetune" or args.mode == "full":
        logger.info("=== FINETUNING PRUNED MODEL ===")
        finetuned_model_path = finetune_model(cfg, pruned_model_path)
        logger.info(f"Finetuned model saved to: {finetuned_model_path}")

if __name__ == "__main__":
    args = parse_custom_args()
    run_pipeline(args)