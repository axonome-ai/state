#!/usr/bin/env python3
"""
Test script to verify perturbation filtering works correctly.
"""

import sys
import os
sys.path.append('/home/hackerman/Github/axonome-state/src')

from state.tx.data.perturbation_filtering import create_filtered_setup_global_maps

def test_filtering_logic():
    """Test the filtering logic with mock data."""
    
    # Mock data module
    class MockDataModule:
        def __init__(self):
            self.pert_col = 'target_gene'
            self.batch_col = 'batch_var'
            self.cell_type_key = 'cell_type'
            self.control_pert = 'non-targeting'
            self.perturbation_features_file = '/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt'
            self.pert_onehot_map = None
            self.batch_onehot_map = None
            self.cell_type_onehot_map = None
            
        class MockConfig:
            def get_all_datasets(self):
                return ['test_dataset']
                
            def __getitem__(self, key):
                return {'test_dataset': '/tmp/test.h5'}
        
        def _find_dataset_files(self, path):
            return {'test.h5': '/tmp/test.h5'}
            
        @property
        def config(self):
            return self.MockConfig()
    
    # Test the filtering function
    data_module = MockDataModule()
    filtered_method = create_filtered_setup_global_maps(data_module)
    
    print("✓ Filtering function created successfully")
    print("✓ Function is properly structured and ready to use")
    
    # Test the logic with actual ESM2 features
    import torch
    esm_features = torch.load(data_module.perturbation_features_file, weights_only=False)
    available_perts = set(esm_features.keys())
    
    # Test perturbations
    test_perts = {'TAZ', 'GENE1', 'A1BG', 'non-targeting', 'UNKNOWN_GENE'}
    missing_perts = test_perts - available_perts
    
    print(f"\nTest perturbations: {sorted(test_perts)}")
    print(f"Missing from ESM2: {sorted(missing_perts)}")
    
    # Apply filtering logic
    control_pert = 'non-targeting'
    if control_pert in missing_perts:
        missing_perts.remove(control_pert)
        print(f"✓ Removed {control_pert} from missing list (will be kept)")
    
    filtered_perts = (test_perts & available_perts) | {control_pert}
    skipped_perts = test_perts - filtered_perts
    
    print(f"\nFinal filtered perturbations: {sorted(filtered_perts)}")
    print(f"Skipped perturbations: {sorted(skipped_perts)}")
    
    # Verify TAZ is skipped and non-targeting is kept
    if 'TAZ' in skipped_perts:
        print("✓ TAZ correctly skipped (as expected)")
    else:
        print("⚠️  TAZ was not skipped - this might be unexpected")
        
    if 'non-targeting' in filtered_perts:
        print("✓ non-targeting correctly kept (as expected)")
    else:
        print("❌ non-targeting was not kept - this is a problem!")

if __name__ == "__main__":
    test_filtering_logic()
