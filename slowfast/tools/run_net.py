#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""

import os
import sys

# Add the path to the slowfast module or via export 'export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH'
sys.path.insert(0, '/home/malek/Master Thesis/Code/slowfast')


from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.utils.parser import load_config, parse_args
from demo_net import demo
from test_net import test
from train_net import train
from visualization import visualize
import slowfast.utils.logging as logging

# Set up proper logging
logger = logging.get_logger(__name__)


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    print("config files: {}".format(args.cfg_files))

    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

        # Perform training.
        if cfg.TRAIN.ENABLE:
            launch_job(cfg=cfg, init_method=args.init_method, func=train)
            logger.info("Training completed.")

        # Perform Pruning (Pruning and benchmarking).
        if cfg.PRUNING.ENABLE:
            pass

        # Perform Fine-tuning.
        if cfg.FINE_TUNING.ENABLE:
            pass

        # Perform multi-clip testing.
        if cfg.TEST.ENABLE:
            if cfg.TEST.NUM_ENSEMBLE_VIEWS == -1:
                num_view_list = [1, 3, 5, 7, 10]
                for num_view in num_view_list:
                    cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view
                    launch_job(cfg=cfg, init_method=args.init_method, func=test)
            else:
                launch_job(cfg=cfg, init_method=args.init_method, func=test)

        # Perform model visualization (optional).
        if cfg.TENSORBOARD.ENABLE and (
            cfg.TENSORBOARD.MODEL_VIS.ENABLE or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
        ):
            launch_job(cfg=cfg, init_method=args.init_method, func=visualize)

        # Run demo (for deployment).
        if cfg.DEMO.ENABLE:
            demo(cfg)


if __name__ == "__main__":
    main()
