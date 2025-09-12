"""
Pytest configuration for the axonome-state project.

This file configures pytest to suppress known harmless warnings
so they don't interfere with real warning detection.
"""

import warnings
import pytest

# Suppress known harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, module="requests")
warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
warnings.filterwarnings("ignore", message=".*loss_fn.*already saved during checkpointing.*")
warnings.filterwarnings("ignore", message=".*Tensor Cores.*")

@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress known harmless warnings during tests."""
    warnings.filterwarnings("ignore", category=UserWarning, module="requests")
    warnings.filterwarnings("ignore", message=".*RequestsDependencyWarning.*")
    warnings.filterwarnings("ignore", message=".*loss_fn.*already saved during checkpointing.*")
    warnings.filterwarnings("ignore", message=".*Tensor Cores.*")
