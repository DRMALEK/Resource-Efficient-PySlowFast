#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Early stopping utility for training."""

import logging

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping utility to stop training when validation metric stops improving.
    
    Args:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in the monitored metric to qualify as improvement.
        mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the
                   metric has stopped decreasing; in 'max' mode it will stop when the
                   metric has stopped increasing.
    """
    
    def __init__(self, patience=10, min_delta=0.001, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
            
        logger.info(f"EarlyStopping initialized with patience={patience}, "
                   f"min_delta={min_delta}, mode={mode}")
    
    def __call__(self, score, epoch):
        """
        Check if training should stop early.
        
        Args:
            score (float): Current validation metric score.
            epoch (int): Current epoch number.
            
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            logger.info(f"Initial best score: {self.best_score:.4f} at epoch {epoch}")
        elif self._is_better(score):
            improvement = abs(score - self.best_score)
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            logger.info(f"New best score: {self.best_score:.4f} at epoch {epoch} "
                       f"(improvement: {improvement:.4f})")
        else:
            self.counter += 1
            logger.info(f"No improvement for {self.counter}/{self.patience} epochs "
                       f"(current: {score:.4f}, best: {self.best_score:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered! Best score: {self.best_score:.4f} "
                           f"at epoch {self.best_epoch}")
                
        return self.early_stop
    
    def _is_better(self, score):
        """Check if the current score is better than the best score."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:  # mode == "max"
            return score > self.best_score + self.min_delta
    
    def get_best_score(self):
        """Get the best score achieved."""
        return self.best_score
    
    def get_best_epoch(self):
        """Get the epoch with the best score."""
        return self.best_epoch
    
    def reset(self):
        """Reset the early stopping state."""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0
        logger.info("EarlyStopping state reset")
