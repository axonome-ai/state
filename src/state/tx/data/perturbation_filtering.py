"""
Perturbation filtering utilities for data modules.
"""

import torch
import h5py
import logging
from pathlib import Path
from cell_load.data_modules.perturbation_dataloader import safe_decode_array, generate_onehot_map

logger = logging.getLogger(__name__)


def create_filtered_setup_global_maps(data_module):
    """
    Create a filtered version of _setup_global_maps that excludes perturbations
    missing from ESM2 features but keeps the control perturbation.
    
    Args:
        data_module: The PerturbationDataModule instance to create the method for
        
    Returns:
        A function that can replace the original _setup_global_maps method
    """
    
    def filtered_setup_global_maps():
        """Filtered version that excludes missing perturbations but keeps non-targeting."""
        all_perts = set()
        all_batches = set()
        all_celltypes = set()

        for dataset_name in data_module.config.get_all_datasets():
            dataset_path = Path(data_module.config.datasets[dataset_name])
            files = data_module._find_dataset_files(dataset_path)

            for _fname, fpath in files.items():
                with h5py.File(fpath, "r") as f:
                    pert_arr = f[f"obs/{data_module.pert_col}/categories"][:]
                    perts = set(safe_decode_array(pert_arr))
                    all_perts.update(perts)

                    try:
                        batch_arr = f[f"obs/{data_module.batch_col}/categories"][:]
                    except KeyError:
                        batch_arr = f[f"obs/{data_module.batch_col}"][:]
                    batches = set(safe_decode_array(batch_arr))
                    all_batches.update(batches)

                    try:
                        celltype_arr = f[f"obs/{data_module.cell_type_key}/categories"][:]
                    except KeyError:
                        celltype_arr = f[f"obs/{data_module.cell_type_key}"][:]
                    celltypes = set(safe_decode_array(celltype_arr))
                    all_celltypes.update(celltypes)

        # Create one-hot maps
        if data_module.perturbation_features_file:
            # Load the custom featurizations from a torch file
            featurization_dict = torch.load(data_module.perturbation_features_file)
            
            # Filter out perturbations that are missing from ESM2 features
            available_perts = set(featurization_dict.keys())
            missing_perts = all_perts - available_perts
            
            # Never skip the control perturbation (non-targeting)
            if data_module.control_pert in missing_perts:
                missing_perts.remove(data_module.control_pert)
                logger.info(f"Keeping control perturbation '{data_module.control_pert}' even though it's missing from ESM2 features")
            
            if len(missing_perts) > 0:
                logger.info(f"Filtering out {len(missing_perts)} perturbations missing from ESM2 features:")
                for pert in sorted(missing_perts):
                    logger.info(f"  - SKIPPING: {pert}")
                
                # Only keep perturbations that have ESM2 features OR are the control perturbation
                filtered_perts = (all_perts & available_perts) | {data_module.control_pert}
                filtered_featurization_dict = {k: v for k, v in featurization_dict.items() if k in filtered_perts}
                
                # Add control perturbation with zero vector if it was missing
                if data_module.control_pert not in filtered_featurization_dict:
                    feature_dim = next(iter(featurization_dict.values())).shape[-1]
                    filtered_featurization_dict[data_module.control_pert] = torch.zeros(feature_dim)
                    logger.info(f"Added zero vector for control perturbation '{data_module.control_pert}'")
                
                data_module.pert_onehot_map = filtered_featurization_dict
                
                # Update all_perts to only include filtered perturbations
                all_perts = filtered_perts
                
                # Log summary of kept perturbations
                kept_perts = sorted(filtered_perts)
                logger.info(f"Kept {len(kept_perts)} perturbations after filtering:")
                for pert in kept_perts:
                    if pert == data_module.control_pert:
                        logger.info(f"  - KEPT (control): {pert}")
                    else:
                        logger.info(f"  - KEPT: {pert}")
            else:
                data_module.pert_onehot_map = featurization_dict
                logger.info("All perturbations have ESM2 features available.")
        else:
            # Fall back to default: generate one-hot mapping
            data_module.pert_onehot_map = generate_onehot_map(all_perts)

        data_module.batch_onehot_map = generate_onehot_map(all_batches)
        data_module.cell_type_onehot_map = generate_onehot_map(all_celltypes)
    
    return filtered_setup_global_maps


def apply_perturbation_filtering(data_module, esm_perts_only=False):
    """
    Apply perturbation filtering to a data module if esm_perts_only is enabled.
    
    Args:
        data_module: The PerturbationDataModule instance to modify
        esm_perts_only: Whether to filter out perturbations missing from ESM2 features
    """
    if esm_perts_only and data_module.perturbation_features_file:
        logger.info("Applying perturbation filtering to exclude missing ESM2 features")
        filtered_method = create_filtered_setup_global_maps(data_module)
        data_module._setup_global_maps = filtered_method
    else:
        logger.info("No perturbation filtering applied")
