import time
from typing import Optional

from lightning.pytorch.callbacks import Callback


class EpochTimingCallback(Callback):
    """
    Callback that tracks and logs epoch timing metrics to wandb.
    
    Key metrics for wandb plotting:
    - epoch_duration_minutes: Y-axis for time series plot (epoch duration in minutes)
    - epoch_duration_seconds: Y-axis alternative (epoch duration in seconds)
    - X-axis: epoch number (automatically provided by wandb)
    
    This creates a time series plot showing epoch duration over time.
    """

    def __init__(self):
        """Initialize the epoch timing callback."""
        super().__init__()
        
        # Timing tracking
        self.epoch_start_time: Optional[float] = None
        self.epoch_durations: list[float] = []
        self.cumulative_time: float = 0.0
        self.training_start_time: Optional[float] = None

    def on_train_start(self, trainer, pl_module):
        """Initialize timing tracking when training starts."""
        self.training_start_time = time.time()
        self.epoch_durations = []
        self.cumulative_time = 0.0

    def on_train_epoch_start(self, trainer, pl_module):
        """Record the start time of each epoch."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Calculate and log epoch timing metrics."""
        if self.epoch_start_time is None:
            return

        # Calculate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - self.epoch_start_time
        
        # Update tracking
        self.epoch_durations.append(epoch_duration)
        self.cumulative_time += epoch_duration

        # Log epoch duration in minutes
        pl_module.log("timing/epoch_duration_minutes", epoch_duration / 60.0, on_epoch=True)

    def on_train_end(self, trainer, pl_module):
        """Training ended - no additional logging needed."""
        pass
    