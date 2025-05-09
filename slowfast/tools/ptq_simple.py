import torch
import torch.quantization as quantization
import logging
import sys


# Set up logging
logger = logging.getLogger(__name__)

# Add the path to the slowfast module or via export 'export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH'
sys.path.insert(0, '/home/milkyway/Desktop/Student Thesis/Slowfast/slowfast')

from slowfast.models import build_model
import slowfast.utils.checkpoint as cu
from slowfast.datasets import loader
from slowfast.config.defaults import get_cfg


# Load config
cfg = get_cfg()
cfg.merge_from_file("/home/milkyway/Desktop/Student Thesis/results/x3d_M_exp1/X3D_M.yaml")
cfg.NUM_GPUS = 0

# first load the pretrained model
logger.info('Loading the pretrained model...')
model_fp = build_model(cfg)

# load checkpoint
cu.load_test_checkpoint(cfg, model_fp, quantized=False)

# set the model to evaluation mode
model_fp.eval()

# specify the quantization configuration
model_fp.qconfig = quantization.get_default_qconfig("fbgemm")


# fuse the model layers
####

# prepare the model for quantization
logger.info('Preparing the model for quantization...')
model_fp_prepared = quantization.prepare(model_fp, inplace=False)


# Create calibration data loader
calib_loader = loader.construct_loader(cfg, "train")

# Limit calibration batches for efficiency
Calibration_min_batches = 10

# calibrate the model with a representative dataset
logger.info('Calibrating the model...')
with torch.no_grad():
        batch_count = 0
        for inputs, _, _, _, _ in calib_loader:
            # Move inputs to CPU as quantization only supports CPU
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i]
            else:
                inputs = inputs
            
            # Forward pass to collect statistics for quantization
            model_fp_prepared(inputs)
            
            batch_count += 1
            if batch_count >= cfg.QUANTIZATION.CALIBRATION_NUM_BATCHES:
                break

# Convert the model to a quantized 
logger.info('Converting the model to a quantized model...')
quantized_model = torch.quantization.convert(model_fp_prepared, inplace=False)


# Save the quantized model
quantized_model_fp = "/home/milkyway/Desktop/Student Thesis/results/x3d_M_exp1/quantized_model.pth"
logger.info(f'Saving the quantized model to {quantized_model_fp}...')
torch.save(quantized_model.state_dict(), quantized_model_fp)