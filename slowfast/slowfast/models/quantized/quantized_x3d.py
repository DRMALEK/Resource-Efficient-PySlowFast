#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Quantized X3D model implementation with Quantization Aware Training (QAT) support."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.qconfig import QConfig
import torch.quantization as quant
import pytorchvideo.layers.swish

import slowfast.utils.logging as logging
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.utils import round_width

from ..build import MODEL_REGISTRY
from .. import head_helper, resnet_helper, stem_helper

logger = logging.get_logger(__name__)

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

_POOL1 = {
    "x3d": [[1, 1, 1]],
}


@MODEL_REGISTRY.register()
class QuantizedX3D(nn.Module):
    """
    Quantized X3D model builder for Quantization-Aware Training (QAT).

    Based on original X3D:
    Christoph Feichtenhofer, "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    
    This version adds support for Quantization-Aware Training
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(QuantizedX3D, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self.cfg = cfg
        
        # Quantization configuration
        self.quantize_model = cfg.QUANTIZATION.ENABLE
        self.quant_aware_training = cfg.QUANTIZATION.QAT
        
        # QAT stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        exp_stage = 2.0
        self.dim_c1 = cfg.X3D.DIM_C1

        self.dim_res2 = (
            round_width(self.dim_c1, exp_stage, divisor=8)
            if cfg.X3D.SCALE_RES2
            else self.dim_c1
        )
        self.dim_res3 = round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )
        
        # Set QAT configurations if enabled
        if self.quantize_model and self.quant_aware_training:
            self.prepare_qat()

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def prepare_qat(self):
        """
        Prepare model for Quantization Aware Training
        """
        # Set QAT configuration
        if self.cfg.QUANTIZATION.BACKEND == 'fbgemm':
            # Use FBGEMM for x86 CPU
            self.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        else:
            # Use QNNPACK for ARM CPU
            self.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
                
        # Fuse modules before preparing the model for QAT
        self._fuse_modules()
        
        # set model to training mode
        self.train()

        # Prepare the model for QAT
        torch.quantization.prepare_qat(self, inplace=True)
        
        logger.info("Model prepared for Quantization Aware Training (QAT)")
        
    def _fuse_modules(self):
        """
        Fuse Conv+BN+ReLU modules for better quantization based on X3D architecture,
        but skip Swish activations since they're not supported by PyTorch's fusion.
        """
        
        self.eval()  # Set to eval mode for fusing
        
        # Fuse the stem modules
        if hasattr(self.s1, "pathway0_stem"):
            stem = self.s1.pathway0_stem
            # Fuse Conv+BN+ReLU pattern in stem
            if hasattr(stem, "conv") and hasattr(stem, "bn") and hasattr(stem, "relu"):
                torch.quantization.fuse_modules(
                    stem,
                    ["conv", "bn", "relu"],
                    inplace=True
                )
        
        # Fuse ResStage modules for stages 2-5
        for stage_idx in range(2, 6):
            if hasattr(self, f"s{stage_idx}"):
                stage = getattr(self, f"s{stage_idx}")
                
                # Get the number of ResBlocks in this stage
                num_blocks = 0
                while hasattr(stage, f"pathway0_res{num_blocks}"):
                    num_blocks += 1
                
                # Process each ResBlock in the stage
                for i in range(num_blocks):
                    res_block = getattr(stage, f"pathway0_res{i}")
                    
                    # Fuse branch1 (skip connection) if it exists
                    if hasattr(res_block, "branch1"):
                        torch.quantization.fuse_modules(
                            res_block, 
                            ["branch1", "branch1_bn"],
                            inplace=True
                        )
                    
                    # Fuse X3DTransform modules in branch2
                    if hasattr(res_block, "branch2"):
                        transform = res_block.branch2
                        
                        # 1. Fuse a + a_bn + a_relu (first 1x1x1 conv)
                        if hasattr(transform, "a") and hasattr(transform, "a_bn") and hasattr(transform, "a_relu"):
                            torch.quantization.fuse_modules(
                                transform,
                                ["a", "a_bn", "a_relu"],
                                inplace=True
                            )
                        
                        # 2. Handle b pathway - fuse only Conv+BN and skip the activation if it's Swish
                        if hasattr(transform, "b") and hasattr(transform, "b_bn"):
                            # Always fuse just Conv+BN, regardless of SE module presence or activation type
                            torch.quantization.fuse_modules(
                                transform,
                                ["b", "b_bn"],
                                inplace=True
                            )
                        
                        # 3. Fuse c + c_bn (final 1x1x1 conv, no activation follows)
                        if hasattr(transform, "c") and hasattr(transform, "c_bn"):
                            torch.quantization.fuse_modules(
                                transform,
                                ["c", "c_bn"],
                                inplace=True
                            )
        
        # Fuse X3DHead modules
        if hasattr(self, "head"):
            head = self.head
            
            # 1. Fuse conv_5 + conv_5_bn + conv_5_relu
            if hasattr(head, "conv_5") and hasattr(head, "conv_5_bn") and hasattr(head, "conv_5_relu"):
                torch.quantization.fuse_modules(
                    head,
                    ["conv_5", "conv_5_bn", "conv_5_relu"],
                    inplace=True
                )
            
            # 2. Handle lin_5 pathway
            if hasattr(head, "lin_5") and hasattr(head, "lin_5_relu"):
                if hasattr(head, "lin_5_bn"):
                    # If BatchNorm exists for lin_5, fuse it too
                    torch.quantization.fuse_modules(
                        head,
                        ["lin_5", "lin_5_bn", "lin_5_relu"],
                        inplace=True
                    )
                else:
                    # Otherwise just fuse lin_5 + relu
                    torch.quantization.fuse_modules(
                        head,
                        ["lin_5", "lin_5_relu"],
                        inplace=True
                    )
                
        logger.info("Model modules fused for QAT based on X3D architecture (skipping Swish activations)")

    def _construct_network(self, cfg):
        """
        Builds a quantization-friendly single pathway X3D model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        assert cfg.RESNET.DEPTH in {50, 101, 152}

        (d2, d3, d4, d5) = {
            50: (3, 4, 6, 3),
            101: (3, 4, 23, 3),
            152: (3, 8, 36, 3),
        }[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = cfg.X3D.WIDTH_FACTOR
        d_mul = cfg.X3D.DEPTH_FACTOR
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(stage + 2)  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner] if cfg.X3D.CHANNELWISE_3x3x3 else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_pool=cfg.NONLOCAL.POOL[0],
                instantiation=cfg.NONLOCAL.INSTANTIATION,
                trans_func_name=cfg.RESNET.TRANS_FUNC,
                stride_1x1=cfg.RESNET.STRIDE_1X1,
                norm_module=self.norm_module,
                dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            raise NotImplementedError("Detection not supported with quantization")
        else:
            spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            self.head = head_helper.X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                dim_out=cfg.X3D.DIM_C5,
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                bn_lin5_on=cfg.X3D.BN_LIN5,
            )

    def forward(self, x, bboxes=None):
        if self.quantize_model:
            x = x[0] 
            x = self.quant(x)
            
        # Process through the standard X3D model stages        
        x = [x] # Convert to list for single pathway
        x = self.s1(x)
        
        for stage_idx in range(2, 6):
            stage = getattr(self, f"s{stage_idx}")
            x = stage(x)
            
        # Head
        x = self.head(x)

        # Dequantize the output if quantization is enabled
        if self.quantize_model:
            x = self.dequant(x)
            
        return x
        
    def convert_to_quantized_model(self):
        """
        Convert the model to fully quantized for inference
        """
        if not self.quantize_model:
            logger.error("Cannot convert to quantized model: quantization not enabled")
            return

        # set the model to eval mode
        self.eval()

        # move the model to cpu before conversion
        #self.cpu()    
        
        torch.quantization.convert(self, inplace=True)
        
        logger.info("Model converted to quantized format")
