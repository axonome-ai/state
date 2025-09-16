"""
Perturbation filtering for ESM2 embeddings.

This module provides monkey patching functionality to filter out perturbations
that don't have ESM2 embeddings when esm_perts_only=True.
"""

import logging
import numpy as np
import torch
from typing import Set

logger = logging.getLogger(__name__)

# Global variables to store original methods and available perturbations
_original_methods = {}
_available_perts: Set[str] = set()


def patch_perturbation_data_module_for_filtering(perturbation_features_file: str):
    """
    Apply monkey patches to PerturbationDataModule to filter perturbations without ESM2 embeddings.
    
    Args:
        perturbation_features_file: Path to the ESM2 perturbation features file
    """
    global _available_perts, _original_methods
    
    # Load available perturbations from ESM2 features file
    try:
        esm2_features = torch.load(perturbation_features_file, map_location='cpu')
        _available_perts = set(esm2_features.keys())
        logger.info(f"🔧 Loaded {len(_available_perts)} ESM2 perturbations from {perturbation_features_file}")
    except Exception as e:
        logger.error(f"🔧 Failed to load ESM2 perturbations from {perturbation_features_file}: {e}")
        _available_perts = set()
        return
    
    # Import PerturbationDataModule
    from cell_load.data_modules import PerturbationDataModule
    
    # Store original methods
    _original_methods['_split_fewshot_celltype'] = PerturbationDataModule._split_fewshot_celltype
    _original_methods['_process_celltype'] = PerturbationDataModule._process_celltype
    
    # Apply patches
    PerturbationDataModule._split_fewshot_celltype = _filtered_split_fewshot_celltype
    PerturbationDataModule._process_celltype = _filtered_process_celltype
    
    logger.info("🔧 Applied perturbation filtering patches to PerturbationDataModule")


def _filtered_split_fewshot_celltype(self, ds, pert_indices, ctrl_indices, cache, pert_config):
    """
    Filtered version of _split_fewshot_celltype that excludes perturbations without ESM2 embeddings.
    """
    # Check if filtering should be applied
    esm_perts_only = getattr(self, 'esm_perts_only', False)
    
    if not esm_perts_only or not _available_perts:
        logger.debug("🔧 Skipping perturbation filtering (esm_perts_only=False or no ESM2 perts loaded)")
        return _original_methods['_split_fewshot_celltype'](self, ds, pert_indices, ctrl_indices, cache, pert_config)
    
    logger.info(f"🔧 Filtering perturbations for fewshot split - ESM2 perts: {len(_available_perts)}")
    
    # Filter the pert_config to only include perturbations with ESM2 embeddings
    original_val_perts = pert_config.get("val", [])
    original_test_perts = pert_config.get("test", [])
    
    # Filter perturbations, but keep non-targeting (control) perturbations
    val_perts_filtered = [p for p in original_val_perts if p in _available_perts or p == "non-targeting"]
    test_perts_filtered = [p for p in original_test_perts if p in _available_perts or p == "non-targeting"]
    
    # Get all available perturbation names and filter them for training
    all_pert_names = set(cache.pert_categories)
    train_pert_names = all_pert_names - set(original_val_perts) - set(original_test_perts)
    train_perts_filtered = [p for p in train_pert_names if p in _available_perts or p == "non-targeting"]
    
    # Log filtering results
    logger.info(f"🔧 Fewshot filtering results:")
    logger.info(f"  Val: {len(original_val_perts)} -> {len(val_perts_filtered)} perts")
    logger.info(f"  Test: {len(original_test_perts)} -> {len(test_perts_filtered)} perts")
    logger.info(f"  Train: {len(train_pert_names)} -> {len(train_perts_filtered)} perts")
    
    # Log filtered out perturbations
    val_filtered_out = [p for p in original_val_perts if p not in val_perts_filtered]
    test_filtered_out = [p for p in original_test_perts if p not in test_perts_filtered]
    train_filtered_out = [p for p in train_pert_names if p not in train_perts_filtered]
    
    logger.info(f"🔧 Sample filtered out perturbations:")
    logger.info(f"  Val filtered out (first 10): {val_filtered_out[:10]}")
    logger.info(f"  Test filtered out (first 10): {test_filtered_out[:10]}")
    logger.info(f"  Train filtered out (first 10): {train_filtered_out[:10]}")
    
    # Check if TAZ was filtered out
    if "TAZ" in val_filtered_out:
        logger.info("✅ TAZ successfully filtered from validation perturbations")
    if "TAZ" in test_filtered_out:
        logger.info("✅ TAZ successfully filtered from test perturbations")
    if "TAZ" in train_filtered_out:
        logger.info("✅ TAZ successfully filtered from training perturbations")
    
    # Now implement the corrected logic with explicit train_pert_codes
    counts = {"train": 0, "val": 0, "test": 0}
    
    # Get perturbation codes for this cell type
    pert_codes = cache.pert_codes[pert_indices]
    
    # Create sets of perturbation codes for each split
    val_pert_names = set(val_perts_filtered)
    test_pert_names = set(test_perts_filtered)
    train_pert_names = set(train_perts_filtered)
    
    val_pert_codes = set()
    test_pert_codes = set()
    train_pert_codes = set()
    
    for i, pert_name in enumerate(cache.pert_categories):
        if pert_name in val_pert_names:
            val_pert_codes.add(i)
        if pert_name in test_pert_names:
            test_pert_codes.add(i)
        if pert_name in train_pert_names:
            train_pert_codes.add(i)
    
    # Split perturbation indices by their codes using explicit train_pert_codes
    val_mask = np.isin(pert_codes, list(val_pert_codes))
    test_mask = np.isin(pert_codes, list(test_pert_codes))
    train_mask = np.isin(pert_codes, list(train_pert_codes))  # This is the corrected line!
    
    val_pert_indices = pert_indices[val_mask]
    test_pert_indices = pert_indices[test_mask]
    train_pert_indices = pert_indices[train_mask]
    
    # Split controls proportionally
    rng = np.random.default_rng(self.random_seed)
    ctrl_indices_shuffled = rng.permutation(ctrl_indices)
    
    n_val = len(val_pert_indices)
    n_test = len(test_pert_indices)
    n_train = len(train_pert_indices)
    total_pert = n_val + n_test + n_train
    
    if total_pert > 0:
        n_ctrl_val = int(len(ctrl_indices) * n_val / total_pert)
        n_ctrl_test = int(len(ctrl_indices) * n_test / total_pert)
        
        val_ctrl_indices = ctrl_indices_shuffled[:n_ctrl_val]
        test_ctrl_indices = ctrl_indices_shuffled[
            n_ctrl_val : n_ctrl_val + n_ctrl_test
        ]
        train_ctrl_indices = ctrl_indices_shuffled[n_ctrl_val + n_ctrl_test :]
        
        # Create subsets
        if len(val_pert_indices) > 0:
            subset = ds.to_subset_dataset("val", val_pert_indices, val_ctrl_indices)
            self.val_datasets.append(subset)
            counts["val"] = len(subset)
        
        if len(test_pert_indices) > 0:
            subset = ds.to_subset_dataset("test", test_pert_indices, test_ctrl_indices)
            self.test_datasets.append(subset)
            counts["test"] = len(subset)
        
        if len(train_pert_indices) > 0:
            subset = ds.to_subset_dataset("train", train_pert_indices, train_ctrl_indices)
            self.train_datasets.append(subset)
            counts["train"] = len(subset)
    
    return counts


def _filtered_process_celltype(self, ds, celltype, ct_indices, ctrl_indices, pert_indices, cache, dataset_name, zeroshot_celltypes, fewshot_celltypes, is_training_dataset):
    """
    Filtered version of _process_celltype that excludes perturbations without ESM2 embeddings.
    """
    # Check if filtering should be applied
    esm_perts_only = getattr(self, 'esm_perts_only', False)
    
    if not esm_perts_only or not _available_perts:
        logger.debug("🔧 Skipping perturbation filtering (esm_perts_only=False or no ESM2 perts loaded)")
        return _original_methods['_process_celltype'](self, ds, celltype, ct_indices, ctrl_indices, pert_indices, cache, dataset_name, zeroshot_celltypes, fewshot_celltypes, is_training_dataset)
    
    # Filter perturbation indices for training datasets
    if is_training_dataset:
        logger.info(f"🔧 Filtering training perturbations for {dataset_name}.{celltype}")
        
        # Get perturbation codes for this cell type
        pert_codes = cache.pert_codes[pert_indices]
        
        # Filter perturbation codes to only include those with ESM2 embeddings
        valid_pert_codes = []
        for i, pert_name in enumerate(cache.pert_categories):
            if pert_name in _available_perts or pert_name == "non-targeting":
                valid_pert_codes.append(i)
        
        valid_pert_codes = set(valid_pert_codes)
        
        # Filter perturbation indices
        valid_mask = np.isin(pert_codes, list(valid_pert_codes))
        filtered_pert_indices = pert_indices[valid_mask]
        
        logger.info(f"🔧 Training filtering results:")
        logger.info(f"  Original perts: {len(pert_indices)} -> {len(filtered_pert_indices)} perts")
        
        # Check what perturbations were filtered out
        original_pert_names = [cache.pert_categories[code] for code in pert_codes]
        filtered_pert_names = [cache.pert_categories[code] for code in pert_codes[valid_mask]]
        filtered_out_perts = set(original_pert_names) - set(filtered_pert_names)
        
        if filtered_out_perts:
            logger.info(f"🔧 Filtered out perturbations: {sorted(filtered_out_perts)}")
        
        # Use filtered perturbation indices
        pert_indices = filtered_pert_indices
    
    # Call the original method with potentially filtered pert_indices
    return _original_methods['_process_celltype'](self, ds, celltype, ct_indices, ctrl_indices, pert_indices, cache, dataset_name, zeroshot_celltypes, fewshot_celltypes, is_training_dataset)






def restore_perturbation_data_module():
    """Restore original PerturbationDataModule methods."""
    global _original_methods
    
    if not _original_methods:
        logger.warning("🔧 No original methods to restore")
        return
    
    from cell_load.data_modules import PerturbationDataModule
    
    # Restore original methods
    PerturbationDataModule._split_fewshot_celltype = _original_methods['_split_fewshot_celltype']
    PerturbationDataModule._process_celltype = _original_methods['_process_celltype']
    
    # Clear stored methods
    _original_methods = {}
    
    logger.info("🔧 Restored original PerturbationDataModule methods")