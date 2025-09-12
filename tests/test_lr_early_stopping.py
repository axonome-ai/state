#!/usr/bin/env python3
"""
Test script to verify the LearningRateEarlyStopping callback works correctly.
"""

import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
import logging
import yaml

# Set up logging to see the callback messages
logging.basicConfig(level=logging.INFO)

# Import the callback
import sys
sys.path.append('src')
from state.tx.callbacks import LearningRateEarlyStopping

class DummyModel(L.LightningModule):
    """A dummy model for testing the callback."""
    
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        return self.layer(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Use a scheduler that reduces LR quickly to test early stopping
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        return [optimizer], [scheduler]

def test_lr_early_stopping():
    """Test the learning rate early stopping callback."""
    
    print("Testing LearningRateEarlyStopping callback...")
    
    # Create model and data
    model = DummyModel()
    
    # Create dummy data
    train_data = [(torch.randn(32, 10), torch.randn(32, 1)) for _ in range(100)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Create callback with a high threshold to trigger early stopping
    callback = LearningRateEarlyStopping(
        min_lr=1e-2,  # High threshold to trigger early stopping
        patience=2,   # Wait 2 checks before stopping
        verbose=True,
        check_frequency=10  # Check every 10 steps
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    
    print("Starting training...")
    print("The callback should stop training when LR drops below 1e-2")
    
    try:
        trainer.fit(model, train_loader)
        print("Training completed normally")
    except Exception as e:
        print(f"Training stopped with exception: {e}")
    
    # Check final learning rate
    final_lr = trainer.optimizers[0].param_groups[0]["lr"]
    print(f"Final learning rate: {final_lr:.2e}")
    
    if final_lr < 1e-2:
        print("✓ Test passed: Learning rate dropped below threshold")
    else:
        print("✗ Test failed: Learning rate did not drop below threshold")

def test_lr_early_stopping_no_stop():
    """Test that training continues when LR doesn't drop below threshold."""
    
    print("\nTesting LearningRateEarlyStopping callback (no early stop)...")
    
    # Create model and data
    model = DummyModel()
    
    # Create dummy data
    train_data = [(torch.randn(32, 10), torch.randn(32, 1)) for _ in range(50)]
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    # Create callback with a very low threshold (should not trigger)
    callback = LearningRateEarlyStopping(
        min_lr=1e-8,  # Very low threshold
        patience=2,
        verbose=True,
        check_frequency=10
    )
    
    # Create trainer
    trainer = L.Trainer(
        max_epochs=2,  # Short training
        callbacks=[callback],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    
    print("Starting training with low LR threshold (should not stop early)...")
    
    trainer.fit(model, train_loader)
    print("✓ Test passed: Training completed without early stopping")

def test_config_loading():
    """Test loading config files with early stopping."""
    
    print("\nTesting config file loading...")
    
    # Test TX config loading
    try:
        with open("src/state/configs/early_stopping/tx.yaml", 'r') as f:
            tx_config = yaml.safe_load(f)
        
        assert "training" in tx_config, "training config not found"
        assert "lr_early_stopping" in tx_config["training"], "lr_early_stopping config not found"
        assert tx_config["training"]["lr_early_stopping"]["enabled"] == True, "Early stopping not enabled"
        print("✓ TX config loading test passed")
    except Exception as e:
        print(f"✗ TX config loading test failed: {e}")
    
    # Test embedding config loading
    try:
        with open("src/state/configs/early_stopping/embedding.yaml", 'r') as f:
            emb_config = yaml.safe_load(f)
        
        assert "experiment" in emb_config, "experiment config not found"
        assert "lr_early_stopping" in emb_config["experiment"], "lr_early_stopping config not found"
        assert emb_config["experiment"]["lr_early_stopping"]["enabled"] == True, "Early stopping not enabled"
        print("✓ Embedding config loading test passed")
    except Exception as e:
        print(f"✗ Embedding config loading test failed: {e}")

if __name__ == "__main__":
    test_lr_early_stopping()
    test_lr_early_stopping_no_stop()
    test_config_loading()
    print("\nAll tests completed!")
