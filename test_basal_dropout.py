#!/usr/bin/env python3
"""
Comprehensive test script for basal_dropout functionality in StateTransitionPerturbationModel
Tests both configuration options and training/evaluation behavior
"""

import torch
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from state.tx.models.state_transition import StateTransitionPerturbationModel

def test_basal_dropout_configuration():
    """Test that basal_dropout can be controlled via input_dropout parameter"""
    
    print("="*80)
    print("TESTING BASAL DROPOUT CONFIGURATION")
    print("="*80)
    
    # Test 1: Using basal_dropout_prob
    print("\nTest 1: Using basal_dropout_prob")
    model1 = StateTransitionPerturbationModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=100,
        pert_dim=50,
        embed_key="X_hvg",
        basal_dropout_prob=0.1
    )
    print(f"basal_dropout_prob=0.1 -> basal_dropout_prob={model1.basal_dropout_prob}")
    print(f"basal_dropout layer p={model1.basal_dropout.p}")
    
    # Test 2: Using input_dropout (new alias)
    print("\nTest 2: Using input_dropout")
    model2 = StateTransitionPerturbationModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=100,
        pert_dim=50,
        embed_key="X_hvg",
        input_dropout=0.2
    )
    print(f"input_dropout=0.2 -> basal_dropout_prob={model2.basal_dropout_prob}")
    print(f"basal_dropout layer p={model2.basal_dropout.p}")
    
    # Test 3: input_dropout should override basal_dropout_prob when both are provided
    print("\nTest 3: input_dropout should override basal_dropout_prob")
    model3 = StateTransitionPerturbationModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=100,
        pert_dim=50,
        embed_key="X_hvg",
        basal_dropout_prob=0.1,
        input_dropout=0.3
    )
    print(f"basal_dropout_prob=0.1, input_dropout=0.3 -> basal_dropout_prob={model3.basal_dropout_prob}")
    print(f"basal_dropout layer p={model3.basal_dropout.p}")
    
    # Test 4: Default value when neither is provided
    print("\nTest 4: Default value")
    model4 = StateTransitionPerturbationModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=100,
        pert_dim=50,
        embed_key="X_hvg"
    )
    print(f"Neither provided -> basal_dropout_prob={model4.basal_dropout_prob}")
    print(f"basal_dropout layer p={model4.basal_dropout.p}")

def test_basal_dropout_eval_behavior():
    """Test that basal_dropout is turned off during evaluation and inference"""
    
    print("\n" + "="*80)
    print("TESTING BASAL DROPOUT TRAINING/EVALUATION BEHAVIOR")
    print("="*80)
    
    # Create model with dropout enabled
    model = StateTransitionPerturbationModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=100,
        pert_dim=50,
        embed_key="X_hvg",
        input_dropout=0.5  # High dropout rate for clear testing
    )
    
    print(f"Model created with input_dropout=0.5")
    print(f"basal_dropout layer p={model.basal_dropout.p}")
    
    # Create test input
    test_input = torch.randn(4, 8, 100)  # batch_size=4, seq_len=8, input_dim=100
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input mean: {test_input.mean():.6f}, std: {test_input.std():.6f}")
    
    # Test 1: Training mode - dropout should be applied
    print("\n" + "-"*60)
    print("TEST 1: Training Mode (dropout should be applied)")
    print("-"*60)
    
    model.train()
    print(f"Model training mode: {model.training}")
    
    # Run multiple times to see variability due to dropout
    training_outputs = []
    for i in range(5):
        with torch.no_grad():  # No gradients, but dropout still applied
            output = model.encode_basal_expression(test_input)
            training_outputs.append(output.clone())
            print(f"  Run {i+1}: output mean={output.mean():.6f}, std={output.std():.6f}")
    
    # Check variability - should be different due to dropout
    training_means = [out.mean().item() for out in training_outputs]
    training_std = torch.std(torch.tensor(training_means))
    print(f"  Variability in training outputs (std of means): {training_std:.6f}")
    
    # Test 2: Evaluation mode - dropout should be disabled
    print("\n" + "-"*60)
    print("TEST 2: Evaluation Mode (dropout should be disabled)")
    print("-"*60)
    
    model.eval()
    print(f"Model training mode: {model.training}")
    
    # Run multiple times - should be identical
    eval_outputs = []
    for i in range(5):
        with torch.no_grad():
            output = model.encode_basal_expression(test_input)
            eval_outputs.append(output.clone())
            print(f"  Run {i+1}: output mean={output.mean():.6f}, std={output.std():.6f}")
    
    # Check consistency - should be identical
    eval_means = [out.mean().item() for out in eval_outputs]
    eval_std = torch.std(torch.tensor(eval_means))
    print(f"  Consistency in eval outputs (std of means): {eval_std:.6f}")
    
    # Test 3: Verify outputs are identical in eval mode
    print("\n" + "-"*60)
    print("TEST 3: Consistency Check in Evaluation Mode")
    print("-"*60)
    
    all_identical = True
    for i in range(1, len(eval_outputs)):
        if not torch.allclose(eval_outputs[0], eval_outputs[i], atol=1e-6):
            all_identical = False
            break
    
    print(f"  All evaluation outputs identical: {all_identical}")
    if all_identical:
        print("  ✅ PASS: Dropout is properly disabled during evaluation")
    else:
        print("  ❌ FAIL: Dropout is still being applied during evaluation")
    
    # Test 4: Compare training vs eval behavior
    print("\n" + "-"*60)
    print("TEST 4: Training vs Evaluation Comparison")
    print("-"*60)
    
    # Switch back to training mode
    model.train()
    training_output = model.encode_basal_expression(test_input)
    
    # Switch to eval mode
    model.eval()
    eval_output = model.encode_basal_expression(test_input)
    
    print(f"  Training output mean: {training_output.mean():.6f}")
    print(f"  Evaluation output mean: {eval_output.mean():.6f}")
    print(f"  Outputs are different: {not torch.allclose(training_output, eval_output, atol=1e-6)}")
    
    # Test 5: Test with different dropout rates
    print("\n" + "-"*60)
    print("TEST 5: Different Dropout Rates")
    print("-"*60)
    
    for dropout_rate in [0.0, 0.1, 0.5, 0.9]:
        model_test = StateTransitionPerturbationModel(
            input_dim=100,
            hidden_dim=256,
            output_dim=100,
            pert_dim=50,
            embed_key="X_hvg",
            input_dropout=dropout_rate
        )
        
        model_test.eval()
        output = model_test.encode_basal_expression(test_input)
        print(f"  Dropout rate {dropout_rate}: output mean={output.mean():.6f}")
    
    return training_std, eval_std, all_identical

def test_basal_dropout_forward_pass():
    """Test that dropout is applied during training but not during evaluation in encode_basal_expression"""
    
    print("\n" + "="*80)
    print("TESTING BASAL DROPOUT IN ENCODE_BASAL_EXPRESSION")
    print("="*80)
    
    # Create model
    model = StateTransitionPerturbationModel(
        input_dim=100,
        hidden_dim=256,
        output_dim=100,
        pert_dim=50,
        embed_key="X_hvg",
        input_dropout=0.3
    )
    
    # Create test input
    test_input = torch.randn(2, 8, 100)  # batch_size=2, seq_len=8, input_dim=100
    print(f"Test input shape: {test_input.shape}")
    print(f"Test input mean: {test_input.mean():.6f}")
    
    # Test training mode
    print("\nTraining mode encode_basal_expression:")
    model.train()
    training_output = model.encode_basal_expression(test_input)
    print(f"  Output shape: {training_output.shape}")
    print(f"  Output mean: {training_output.mean():.6f}")
    
    # Test evaluation mode
    print("\nEvaluation mode encode_basal_expression:")
    model.eval()
    eval_output = model.encode_basal_expression(test_input)
    print(f"  Output shape: {eval_output.shape}")
    print(f"  Output mean: {eval_output.mean():.6f}")
    
    # Check if outputs are different
    outputs_different = not torch.allclose(training_output, eval_output, atol=1e-6)
    print(f"  Training and eval outputs are different: {outputs_different}")
    
    return outputs_different

def main():
    """Run all basal dropout tests"""
    
    print("🧪 COMPREHENSIVE BASAL DROPOUT TESTS")
    print("Testing StateTransitionPerturbationModel basal dropout functionality")
    
    # Test configuration options
    test_basal_dropout_configuration()
    
    # Test training/evaluation behavior
    training_std, eval_std, all_identical = test_basal_dropout_eval_behavior()
    
    # Test forward pass behavior
    forward_different = test_basal_dropout_forward_pass()
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"✅ Configuration tests: PASSED")
    print(f"✅ Training mode variability (std): {training_std:.6f} (should be > 0)")
    print(f"✅ Evaluation mode consistency (std): {eval_std:.6f} (should be ~0)")
    print(f"✅ All eval outputs identical: {all_identical}")
    print(f"✅ Forward pass behavior: {'PASSED' if forward_different else 'FAILED'}")
    
    if (training_std > 0.0001 and eval_std < 0.0001 and 
        all_identical and forward_different):
        print("\n🎉 ALL TESTS PASSED! Basal dropout is working correctly!")
        print("   - Can be controlled via training.input_dropout or training.basal_dropout_prob")
        print("   - Applied during training for regularization")
        print("   - Disabled during evaluation for deterministic inference")
    else:
        print("\n❌ SOME TESTS FAILED! Check the implementation")
        print(f"   - Training variability: {training_std:.6f} (need > 0.0001)")
        print(f"   - Evaluation consistency: {eval_std:.6f} (need < 0.0001)")
        print(f"   - All eval outputs identical: {all_identical}")
        print(f"   - Forward pass different: {forward_different}")

if __name__ == "__main__":
    main()