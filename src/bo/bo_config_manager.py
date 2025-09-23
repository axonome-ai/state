#!/usr/bin/env python3
"""
BO Config Manager for handling config-based Bayesian Optimization runs.

This module provides:
- Config-based folder management
- Run continuation capabilities
- State persistence and recovery
- Config validation and change detection
"""

import os
import json
import hashlib
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BOConfigManager:
    """Manages BO runs based on configuration files with continuation support."""
    
    def __init__(self, base_dir: str = "bo_runs"):
        """
        Initialize the BO Config Manager.
        
        Args:
            base_dir: Base directory for all BO runs
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Registry file for tracking active configs
        self.registry_file = self.base_dir / "active_configs.txt"
        self._load_registry()
    
    def _load_registry(self):
        """Load the active configs registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.active_configs = set(line.strip() for line in f if line.strip())
        else:
            self.active_configs = set()
    
    def _save_registry(self):
        """Save the active configs registry."""
        with open(self.registry_file, 'w') as f:
            for config_name in sorted(self.active_configs):
                f.write(f"{config_name}\n")
    
    def _get_config_hash(self, config_path: str) -> str:
        """Calculate hash of config file for change detection."""
        with open(config_path, 'rb') as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()[:16]  # Use first 16 chars
    
    def _get_config_name(self, config_path: str) -> str:
        """Extract config name from file path."""
        return Path(config_path).stem
    
    def _get_run_dir(self, config_name: str) -> Path:
        """Get the run directory for a config name."""
        return self.base_dir / config_name
    
    def _validate_config_continuation(self, config_path: str, run_dir: Path) -> bool:
        """
        Validate if we can continue an existing run.
        
        Returns:
            True if we can continue, False if we need a new run
        """
        config_hash_file = run_dir / "config_hash.txt"
        
        if not config_hash_file.exists():
            logger.warning(f"No config hash found in {run_dir}, creating new run")
            return False
        
        # Read stored hash
        with open(config_hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        # Calculate current hash
        current_hash = self._get_config_hash(config_path)
        
        if stored_hash != current_hash:
            raise ValueError(
                f"Config has changed! Stored hash: {stored_hash}, Current hash: {current_hash}\n"
                f"To continue with changes, use --continue flag\n"
                f"To start fresh, use --new flag\n"
                f"Existing run directory: {run_dir}"
            )
        
        logger.info(f"Config hash matches, can continue existing run")
        return True
    
    def _create_new_run_dir(self, config_name: str, config_path: str) -> Path:
        """Create a new run directory for the config."""
        base_run_dir = self._get_run_dir(config_name)
        
        # If directory exists, create with increment
        counter = 1
        run_dir = base_run_dir
        while run_dir.exists():
            run_dir = Path(f"{base_run_dir}_{counter}")
            counter += 1
        
        # Create directory structure
        run_dir.mkdir(parents=True)
        (run_dir / "runs").mkdir()
        (run_dir / "checkpoints").mkdir()
        
        # Store config and hash
        config_hash = self._get_config_hash(config_path)
        with open(run_dir / "config_hash.txt", 'w') as f:
            f.write(config_hash)
        
        # Copy config file
        shutil.copy2(config_path, run_dir / "config.yaml")
        
        # Add to registry
        self.active_configs.add(run_dir.name)
        self._save_registry()
        
        logger.info(f"Created new run directory: {run_dir}")
        return run_dir
    
    def get_or_create_run_dir(self, config_path: str) -> Tuple[Path, bool]:
        """
        Get existing run directory or create new one.
        
        Args:
            config_path: Path to the BO config file
            
        Returns:
            Tuple of (run_directory, is_continuation)
        """
        config_name = self._get_config_name(config_path)
        base_run_dir = self._get_run_dir(config_name)
        
        # Check if we can continue existing run
        if base_run_dir.exists():
            try:
                if self._validate_config_continuation(config_path, base_run_dir):
                    logger.info(f"Continuing existing run: {base_run_dir}")
                    return base_run_dir, True
            except ValueError as e:
                # Config changed - raise error instead of creating new run
                raise e
        else:
            logger.info(f"No existing run found, creating new run for {config_name}")
            return self._create_new_run_dir(config_name, config_path), False
    
    def load_bo_state(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        """Load existing BO state from run directory."""
        state_file = run_dir / "bo_state.json"
        
        if not state_file.exists():
            logger.info("No existing BO state found")
            return None
        
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            logger.info(f"Loaded BO state from {state_file}")
            return state
        except Exception as e:
            logger.error(f"Failed to load BO state: {e}")
            return None
    
    def save_bo_state(self, run_dir: Path, state: Dict[str, Any]):
        """Save BO state to run directory."""
        state_file = run_dir / "bo_state.json"
        
        try:
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.info(f"Saved BO state to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save BO state: {e}")
    
    def get_next_run_id(self, run_dir: Path, config_name: str = None) -> str:
        """Get the next run ID for training runs."""
        from ..state.utils.naming import generate_run_name
        
        runs_dir = run_dir / "runs"
        runs_dir.mkdir(exist_ok=True)
        
        # Use config name as prefix if provided, otherwise use bo_run
        prefix = config_name if config_name else "bo_run"
        return generate_run_name(prefix)
    
    def get_run_output_dir(self, run_dir: Path, run_id: str) -> Path:
        """Get the output directory for a specific training run."""
        # Use the run_id directly without adding another timestamp
        return run_dir / "runs" / run_id
    
    def list_active_runs(self) -> Dict[str, Dict[str, Any]]:
        """List all active BO runs with their status."""
        runs_info = {}
        
        for config_name in self.active_configs:
            run_dir = self.base_dir / config_name
            if not run_dir.exists():
                continue
            
            # Load basic info
            info = {
                "config_name": config_name,
                "run_dir": str(run_dir),
                "created": datetime.fromtimestamp(run_dir.stat().st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(),
            }
            
            # Load BO state if available
            state = self.load_bo_state(run_dir)
            if state:
                info.update({
                    "n_iterations": len(state.get("results", [])),
                    "best_score": state.get("best_score"),
                    "status": "active" if state.get("n_calls", 0) > len(state.get("results", [])) else "completed"
                })
            else:
                info["status"] = "new"
            
            runs_info[config_name] = info
        
        return runs_info
    
    def cleanup_failed_runs(self, run_dir: Path, max_age_hours: int = None):
        """Clean up failed or incomplete runs (no age limit)."""
        # Clean up V2 structure (bo_runs/config_name/runs/)
        runs_dir = run_dir / "runs"
        if runs_dir.exists():
            for run_path in runs_dir.iterdir():
                if run_path.is_dir():
                    # Check if run is incomplete (no results file)
                    results_file = run_path / "agg_results.csv"
                    if not results_file.exists():
                        logger.info(f"Cleaning up incomplete run: {run_path}")
                        shutil.rmtree(run_path)
        
        # Clean up V1 double-nested structure (bo_runs/bo_runs/config_name/runs/)
        double_nested_dir = run_dir.parent / "bo_runs" / run_dir.name / "runs"
        if double_nested_dir.exists():
            logger.info(f"Found legacy double-nested structure: {double_nested_dir}")
            for run_path in double_nested_dir.iterdir():
                if run_path.is_dir():
                    # Check if run is incomplete (no results file)
                    results_file = run_path / "agg_results.csv"
                    if not results_file.exists():
                        logger.info(f"Cleaning up incomplete legacy run: {run_path}")
                        shutil.rmtree(run_path)
            
            # If the double-nested directory is now empty, remove it
            if not any(double_nested_dir.iterdir()):
                logger.info(f"Removing empty legacy directory: {double_nested_dir}")
                shutil.rmtree(double_nested_dir)
                # Also remove parent directories if empty
                parent_dir = double_nested_dir.parent
                if parent_dir.exists() and not any(parent_dir.iterdir()):
                    logger.info(f"Removing empty legacy parent directory: {parent_dir}")
                    shutil.rmtree(parent_dir)


def main():
    """Test the BO Config Manager."""
    manager = BOConfigManager()
    
    print("Active BO runs:")
    runs = manager.list_active_runs()
    for name, info in runs.items():
        n_iterations = info.get('n_iterations', 0)
        print(f"  {name}: {info['status']} ({n_iterations} iterations)")
    
    # Test with a config
    config_path = "src/state/configs/bo/bo_config_quick.yaml"
    if os.path.exists(config_path):
        run_dir, is_continuation = manager.get_or_create_run_dir(config_path)
        print(f"\nConfig: {config_path}")
        print(f"Run dir: {run_dir}")
        print(f"Is continuation: {is_continuation}")


if __name__ == "__main__":
    main()

