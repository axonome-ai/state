# Learning Rate Policies for State Transition Models

This module provides flexible and configurable learning rate scheduling strategies for state transition models. Learning rate policies can significantly impact training stability, convergence speed, and final model performance.

## Overview

The learning rate policy system consists of:

- **Policy Classes**: Abstract base class and concrete implementations for different scheduling strategies
- **Factory Pattern**: Easy creation of policies by name
- **Configuration Support**: YAML-based configuration files
- **Visualization Tools**: Plotting and comparison utilities
- **Integration**: Seamless integration with PyTorch Lightning models

## Available Policies

### 1. Cosine Annealing (`cosine_annealing`)
Implements cosine annealing with optional warmup. Often effective for transformer-based models.

**Parameters:**
- `max_lr`: Maximum learning rate (default: 1e-4)
- `min_lr`: Minimum learning rate (default: 1e-6)
- `warmup_steps`: Number of warmup steps (default: 1000)
- `warmup_ratio`: Warmup start factor (default: 0.1)
- `T_max`: Maximum number of steps (default: None, uses total steps)
- `eta_min_ratio`: Minimum LR as ratio of max LR (default: 0.1)

### 2. One Cycle (`one_cycle`)
Implements the 1cycle learning rate schedule for super-convergence.

**Parameters:**
- `max_lr`: Maximum learning rate (default: 1e-4)
- `total_steps`: Total training steps (default: None, uses total steps)
- `pct_start`: Percentage of steps for increasing phase (default: 0.3)
- `anneal_strategy`: Annealing strategy ("cos" or "linear", default: "cos")
- `div_factor`: Initial LR = max_lr / div_factor (default: 25.0)
- `final_div_factor`: Final LR = max_lr / final_div_factor (default: 10000.0)

### 3. Warmup Cosine Restarts (`warmup_cosine_restarts`)
Cosine annealing with warm restarts to escape local minima.

**Parameters:**
- `max_lr`: Maximum learning rate (default: 1e-4)
- `min_lr`: Minimum learning rate (default: 1e-6)
- `warmup_steps`: Number of warmup steps (default: 1000)
- `T_0`: Number of steps for first restart (default: 10000)
- `T_mult`: Multiplication factor for restart period (default: 1)
- `eta_min_ratio`: Minimum LR as ratio of max LR (default: 0.1)

### 4. Reduce on Plateau (`reduce_on_plateau`)
Reduces learning rate when validation loss plateaus.

**Parameters:**
- `max_lr`: Maximum learning rate (default: 1e-4)
- `min_lr`: Minimum learning rate (default: 1e-6)
- `warmup_steps`: Number of warmup steps (default: 1000)
- `mode`: Metric to monitor ("min" or "max", default: "min")
- `factor`: Factor by which LR is reduced (default: 0.5)
- `patience`: Steps to wait before reducing (default: 10)
- `threshold`: Threshold for measuring improvement (default: 1e-4)
- `cooldown`: Steps to wait after reduction (default: 0)

### 5. Polynomial Decay (`polynomial_decay`)
Polynomial decay learning rate schedule.

**Parameters:**
- `max_lr`: Maximum learning rate (default: 1e-4)
- `min_lr`: Minimum learning rate (default: 1e-6)
- `warmup_steps`: Number of warmup steps (default: 1000)
- `power`: Power of polynomial (default: 1.0)

### 6. Custom Lambda (`custom_lambda`)
Custom lambda-based learning rate function.

**Parameters:**
- `max_lr`: Maximum learning rate (default: 1e-4)
- `lr_lambda`: Custom lambda function for LR calculation

## Usage

### Basic Usage

```python
from state.tx.models.state_transition import StateTransitionPerturbationModel

# Create model with learning rate policy
model = StateTransitionPerturbationModel(
    input_dim=5120,
    hidden_dim=512,
    output_dim=512,
    pert_dim=512,
    lr_policy_config={
        "name": "cosine_annealing",
        "max_lr": 1e-4,
        "min_lr": 1e-6,
        "warmup_steps": 1000,
        "eta_min_ratio": 0.1
    }
)
```

### Using Configuration Files

```python
import yaml
from state.tx.models.lr_policy_utils import create_lr_policy_from_yaml

# Load from YAML file
policy = create_lr_policy_from_yaml("configs/lr_policies/cosine_annealing.yaml")

# Use with model
model = StateTransitionPerturbationModel(
    input_dim=5120,
    hidden_dim=512,
    output_dim=512,
    pert_dim=512,
    lr_policy_config=policy
)
```

### Creating Custom Policies

```python
from state.tx.models.lr_policies import LearningRatePolicyFactory

# Create policy programmatically
policy = LearningRatePolicyFactory.create_policy(
    "cosine_annealing",
    max_lr=2e-4,
    min_lr=1e-7,
    warmup_steps=2000
)
```

## Visualization

### Plot Single Policy

```python
from state.tx.models.lr_policy_utils import visualize_lr_schedule

policy = LearningRatePolicyFactory.create_policy("cosine_annealing")
visualize_lr_schedule(
    policy,
    total_steps=10000,
    save_path="cosine_schedule.png"
)
```

### Compare Multiple Policies

```python
from state.tx.models.lr_policy_utils import compare_lr_policies

policy_configs = [
    {"name": "cosine_annealing", "max_lr": 1e-4},
    {"name": "one_cycle", "max_lr": 1e-4},
    {"name": "warmup_cosine_restarts", "max_lr": 1e-4}
]

compare_lr_policies(
    policy_configs,
    total_steps=10000,
    save_path="policy_comparison.png"
)
```

## Recommendations

### For State Transition Models

**Small Datasets (< 10K samples):**
- Use `cosine_annealing` with shorter warmup
- `max_lr`: 1e-4, `warmup_steps`: 500

**Medium Datasets (10K-100K samples):**
- Use `cosine_annealing` with standard settings
- `max_lr`: 1e-4, `warmup_steps`: 1000

**Large Datasets (> 100K samples):**
- Use `warmup_cosine_restarts` for better convergence
- `max_lr`: 1e-4, `T_0`: total_steps // 4

### For Transformer Models

**Fast Training:**
- Use `one_cycle` for super-convergence
- `max_lr`: 1e-4, `pct_start`: 0.3

**Stable Training:**
- Use `cosine_annealing` with warmup
- `max_lr`: 1e-4, `warmup_steps`: 1000

**Fine-tuning:**
- Use `reduce_on_plateau`
- `max_lr`: 1e-5, `patience`: 5

## Monitoring

### Learning Rate Callback

```python
from state.tx.models.lr_policies import LearningRateMonitor

# Add to trainer callbacks
callback = LearningRateMonitor(logging_interval="step")
trainer = Trainer(callbacks=[callback])
```

### Logging

The learning rate is automatically logged as `learning_rate` in the training logs when using the `LearningRateMonitor` callback.

## Configuration Files

Configuration files are located in `src/state/configs/lr_policies/`:

- `cosine_annealing.yaml` - Cosine annealing with warmup
- `one_cycle.yaml` - One cycle learning rate
- `warmup_cosine_restarts.yaml` - Cosine with restarts
- `reduce_on_plateau.yaml` - Reduce on plateau
- `polynomial_decay.yaml` - Polynomial decay
- `example_state_transition.yaml` - Complete example

## Examples

See `examples/lr_policy_example.py` for a comprehensive demonstration of all features.

## Integration with PyTorch Lightning

The learning rate policies are fully integrated with PyTorch Lightning's `configure_optimizers()` method. The policies automatically handle:

- Optimizer creation (AdamW with configurable weight decay)
- Scheduler creation and configuration
- Step/epoch-based scheduling
- Monitoring and logging

## Best Practices

1. **Start with cosine annealing** for most use cases
2. **Use warmup** for transformer-based models (10-20% of total steps)
3. **Monitor validation loss** to adjust policy parameters
4. **Use reduce on plateau** for fine-tuning scenarios
5. **Visualize schedules** before training to understand the LR curve
6. **Experiment with different policies** for your specific dataset and model

## Troubleshooting

### Common Issues

1. **Learning rate too high**: Reduce `max_lr` or increase warmup steps
2. **Learning rate too low**: Increase `max_lr` or reduce `min_lr`
3. **Training instability**: Add warmup or reduce learning rate
4. **Slow convergence**: Try `one_cycle` or increase learning rate
5. **Overfitting**: Use `reduce_on_plateau` or reduce learning rate

### Debugging

```python
# Print available policies
from state.tx.models.lr_policy_utils import print_available_policies
print_available_policies()

# Get policy information
from state.tx.models.lr_policy_utils import print_policy_info
print_policy_info("cosine_annealing")
```
