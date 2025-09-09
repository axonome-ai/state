# Loss Function Testing

This directory contains several test scripts to verify that all loss functions work correctly with the StateTransitionPerturbationModel.

## Test Scripts

### 1. `test_loss_unit.py` - Fast Unit Tests (Recommended)
**Fastest option - tests without full training pipeline**

```bash
python test_loss_unit.py
```

This script:
- Tests loss function instantiation with dummy data
- Tests model creation with different loss functions
- Tests forward pass and training step
- Runs in seconds, no data required

### 2. `test_training_short.py` - Short Training Tests
**Uses your exact command but with minimal training**

```bash
# Test all losses
python test_training_short.py

# Test specific loss
python test_training_short.py energy
```

This script:
- Uses your exact training command
- Reduces `max_steps` from 60000 to 20
- Disables validations for speed
- **Disables wandb completely** (no logging to wandb)
- Takes 1-2 minutes per loss function

### 3. `quick_loss_test.py` - Quick Training Tests
**Similar to above but with even shorter training**

```bash
# Test all losses
python quick_loss_test.py

# Test specific loss
python quick_loss_test.py wasserstein
```

### 4. `test_loss_functions.py` - Comprehensive Tests
**Full test suite with detailed reporting**

```bash
python test_loss_functions.py
```

This script:
- Tests all loss functions with proper parameters
- Provides detailed success/failure reporting
- **Disables wandb completely** (no logging to wandb)
- Cleans up test outputs
- Takes 5-10 minutes total

## Available Loss Functions

The following loss functions are tested:

1. `energy` - Energy distance (default)
2. `mse` - Mean Squared Error
3. `se` - Combined Sinkhorn + Energy
4. `sinkhorn` - Sinkhorn loss
5. `cross_entropy` - Binary Cross Entropy with Logits
6. `wasserstein` - Wasserstein distance
7. `kl_divergence` - KL Divergence
8. `mmd` - Maximum Mean Discrepancy
9. `tabular` - Tabular loss

## Usage Examples

### Test a specific loss function:
```bash
python test_training_short.py kl_divergence
```

### Run unit tests only:
```bash
python test_loss_unit.py
```

### Run comprehensive tests:
```bash
python test_loss_functions.py
```

## Expected Output

Successful tests will show:
```
✅ energy: Training completed successfully
✅ mse: Training completed successfully
✅ wasserstein: Training completed successfully
...
Summary: 9/9 loss functions worked
```

Failed tests will show:
```
❌ some_loss: Training failed with return code 1
Error: [error message]
```

## Wandb Disabling

All test scripts automatically disable wandb to prevent:
- Unwanted logging during tests
- Network calls during testing
- Test data pollution in your wandb projects

The scripts use multiple methods to disable wandb:
- Environment variables: `WANDB_MODE=disabled` and `WANDB_DISABLED=true`
- Configuration parameters: `use_wandb=false` and `wandb.enable=false`

## Troubleshooting

1. **Data file not found**: Make sure the data file exists at the specified path
2. **CUDA errors**: Try setting `training.devices=1` and `training.strategy=auto`
3. **Memory issues**: Reduce batch size or use CPU-only mode
4. **Timeout errors**: Increase the timeout in the test scripts
5. **Wandb still logging**: Check that environment variables are set correctly

## Cleanup

Test scripts automatically clean up temporary outputs, but you can manually remove:
- `test_outputs/` directory
- `quick_test/` directory  
- `test_short_training/` directory
- `wandb_logs/` directory
