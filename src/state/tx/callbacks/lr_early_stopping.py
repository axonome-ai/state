"""
Learning Rate Early Stopping Callback

This callback monitors the learning rate during training and stops training
when the learning rate drops below a specified threshold.
"""

import logging
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import Callback


class LearningRateEarlyStopping(Callback):
    """
    Early stopping callback that monitors learning rate and stops training
    when the learning rate drops below a specified threshold.
    
    This is useful for preventing overfitting when using learning rate schedules
    that decay the learning rate over time, as it can indicate that the model
    has converged and further training may not be beneficial.
    """
    
    def __init__(
        self,
        min_lr: float = 1e-7,
        patience: int = 0,
        monitor: str = "learning_rate",
        verbose: bool = True,
        check_frequency: int = 1,
    ):
        """
        Initialize the learning rate early stopping callback.
        
        Args:
            min_lr: Minimum learning rate threshold. Training will stop when
                   the learning rate drops below this value.
            patience: Number of checks to wait before stopping. If 0, stops
                     immediately when threshold is reached.
            monitor: The metric to monitor. Should be "learning_rate" or
                    "train/lr" depending on how it's logged.
            verbose: Whether to print messages when stopping.
            check_frequency: How often to check the learning rate (in steps).
        """
        super().__init__()
        self.min_lr = min_lr
        self.patience = patience
        self.monitor = monitor
        self.verbose = verbose
        self.check_frequency = check_frequency
        
        self.wait_count = 0
        self.stopped_epoch = 0
        
    def on_train_batch_start(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule, 
        batch, 
        batch_idx
    ) -> None:
        """Check learning rate at the start of each training batch."""
        if trainer.global_step % self.check_frequency != 0:
            return
            
        # Get current learning rate from optimizer
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Log the current learning rate for monitoring
            pl_module.log("monitor/learning_rate", current_lr, on_step=True, on_epoch=False)
            
            # Check if learning rate is below threshold
            if current_lr < self.min_lr:
                self.wait_count += 1
                
                if self.verbose:
                    logging.info(
                        f"Learning rate {current_lr:.2e} is below threshold {self.min_lr:.2e}. "
                        f"Patience: {self.wait_count}/{self.patience + 1}"
                    )
                
                if self.wait_count > self.patience:
                    self.stopped_epoch = trainer.current_epoch
                    trainer.should_stop = True
                    
                    if self.verbose:
                        logging.info(
                            f"Early stopping triggered! Learning rate {current_lr:.2e} "
                            f"has been below threshold {self.min_lr:.2e} for "
                            f"{self.wait_count} consecutive checks."
                        )
            else:
                # Reset patience counter if learning rate is above threshold
                self.wait_count = 0
                
    def on_train_epoch_start(
        self, 
        trainer: L.Trainer, 
        pl_module: L.LightningModule
    ) -> None:
        """Check learning rate at the start of each epoch as well."""
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            
            # Log the current learning rate for monitoring
            pl_module.log("monitor/learning_rate", current_lr, on_step=False, on_epoch=True)
            
            if self.verbose and trainer.global_step % 100 == 0:  # Log every 100 steps
                logging.info(f"Current learning rate: {current_lr:.2e}")
    
    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log final learning rate when training ends."""
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            final_lr = optimizer.param_groups[0]["lr"]
            
            if self.verbose:
                if self.stopped_epoch > 0:
                    logging.info(
                        f"Training stopped early at epoch {self.stopped_epoch} "
                        f"with final learning rate: {final_lr:.2e}"
                    )
                else:
                    logging.info(f"Training completed with final learning rate: {final_lr:.2e}")
