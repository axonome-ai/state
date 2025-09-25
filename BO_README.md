# Bayesian Optimization (BO) Setup and Usage

This README provides instructions for setting up and running Bayesian Optimization on the `bo-improvements` branch.

## Prerequisites

- Python 3.10-3.12
- Git access to the repository

## Installation

1. **Clone the repository and checkout the BO improvements branch:**
   ```bash
   git clone https://github.com/axonome-ai/state.git
   cd state
   git checkout bo-improvements
   ```

2. **Install the package in editable mode:**
   ```bash
   pip install -e .
   ```

## Data Preparation

1. **Download and extract the dataset:**
   - Download the required dataset zip file
   - Extract it to any location on your system
   - Note the path to the extracted dataset directory

2. **Update the dataset path in the config:**
   - Open `src/state/configs/bo/bo_config_example.yaml`
   - Update the `dataset_dir_path` to point to your extracted dataset location:
     ```yaml
     dataset_dir_path: "/path/to/your/extracted/dataset"
     ```

## Running Bayesian Optimization

**Run the BO example:**
```bash
python -m src.bo.run_bo_v2 src/state/configs/bo/bo_config_example.yaml --new
```

### Command Options

- `--new`: Force start a new run (overwrites existing)
- `--continue`: Force continue with config changes

### Example Commands

```bash
# Start a new BO run
python -m src.bo.run_bo_v2 src/state/configs/bo/bo_config_example.yaml --new

# Continue an existing run
python -m src.bo.run_bo_v2 src/state/configs/bo/bo_config_example.yaml --continue

# Run with quick config (shorter training)
python -m src.bo.run_bo_v2 src/state/configs/bo/bo_config_quick.yaml --new
```

## Output

- **Results directory:** `bo_runs/bo_config_example/`
- **Logs:** `bo_runs/bo_config_example/bo_log.txt`
- **Results:** `bo_runs/bo_config_example/bo_results.json`
- **Wandb tracking:** Available at the provided wandb URL

## Troubleshooting

- **Memory issues:** The BO runs can be memory-intensive. Monitor system resources.
- **Path errors:** Ensure you're running from the project root directory.
- **Dataset not found:** Verify the `dataset_dir_path` in the config file is correct.

## Configuration Files

- `src/state/configs/bo/bo_config_example.yaml` - Main BO configuration
- `src/state/configs/bo/bo_config_quick.yaml` - Quick test configuration
- `src/state/configs/bo/bo_config.yaml` - Full configuration

## Dependencies

All required dependencies are automatically installed with `pip install -e .`, including:
- scikit-optimize
- tabulate
- matplotlib
- cell-load
- cell-eval
- And all other project dependencies
