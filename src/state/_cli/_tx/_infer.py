import argparse
import glob
from pathlib import Path
from typing import Dict

import h5py
from cell_load.config import ExperimentConfig
from cell_load.utils.data_utils import generate_onehot_map, safe_decode_array
from hydra import compose, initialize_config_dir

from state._cli._tx.dataloader import PerturbationDataModuleFromExperimentConfig


def add_arguments_infer(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to model checkpoint (.ckpt). If not provided, will use model_dir/checkpoints/final.ckpt",
    )
    parser.add_argument("--adata", type=str, required=True, help="Path to input AnnData file (.h5ad)")
    parser.add_argument("--embed_key", type=str, default=None, help="Key in adata.obsm for input features")
    parser.add_argument(
        "--pert_col", type=str, default="drugname_drugconc", help="Column in adata.obs for perturbation labels"
    )
    parser.add_argument(
        "--ctrl_pert", type=str, default="non-targeting", help="Control gene used for control_pert option"
    )
    parser.add_argument("--output", type=str, default=None, help="Path to output AnnData file (.h5ad)")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model_dir containing the config.yaml file and the pert_onehot_map.pt file that was saved during training.",
    )
    parser.add_argument(
        "--celltype_col", type=str, default=None, help="Column in adata.obs for cell type labels (optional)"
    )
    parser.add_argument(
        "--celltypes", type=str, default=None, help="Comma-separated list of cell types to include (optional)"
    )
    parser.add_argument(
        "--preprocess_with_cell_eval", type=bool, default=False, help="Should we preprocess using cell-load?"
    )
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference (default: 1000)")
    parser.add_argument("--ctrl_pert_option", choices=["replace", None])


def _expand_braces(pattern: str) -> list[str]:
    """Expand brace patterns like {a,b,c} into multiple patterns."""

    def expand_single_brace(text: str) -> list[str]:
        # Find the first brace group
        import re
        match = re.search(r"\{([^}]+)\}", text)
        if not match:
            return [text]

        # Extract the options and expand them
        before = text[: match.start()]
        after = text[match.end() :]
        options = match.group(1).split(",")

        results = []
        for option in options:
            new_text = before + option.strip() + after
            # Recursively expand any remaining braces
            results.extend(expand_single_brace(new_text))

        return results

    return expand_single_brace(pattern)
def _find_dataset_files(dataset_path: Path) -> dict[str, Path]:
    files: Dict[str, Path] = {}
    path_str = str(dataset_path)

    # Check if path contains glob patterns
    if any(char in path_str for char in "*?[]{}"):
        # Handle brace expansion manually since Python glob doesn't support it
        expanded_patterns = _expand_braces(path_str)

        for pattern in expanded_patterns:
            if pattern.startswith("/"):
                # Absolute path - use glob.glob()
                if pattern.endswith((".h5", ".h5ad")):
                    # Pattern already specifies file extension
                    for fpath_str in sorted(glob.glob(pattern)):
                        fpath = Path(fpath_str)
                        files[fpath.stem] = fpath
                else:
                    # Pattern doesn't specify extension, add file patterns
                    for ext in ("*.h5", "*.h5ad"):
                        full_pattern = f"{pattern.rstrip('/')}/{ext}"
                        for fpath_str in sorted(glob.glob(full_pattern)):
                            fpath = Path(fpath_str)
                            files[fpath.stem] = fpath
            else:
                # Relative path - use Path.glob()
                if pattern.endswith((".h5", ".h5ad")):
                    for fpath in sorted(Path().glob(pattern)):
                        files[fpath.stem] = fpath
                else:
                    for ext in ("*.h5", "*.h5ad"):
                        full_pattern = f"{pattern.rstrip('/')}/{ext}"
                        for fpath in sorted(Path().glob(full_pattern)):
                            files[fpath.stem] = fpath
    else:
        # No glob patterns - treat as regular path
        if dataset_path.is_file():
            # Single file
            files[dataset_path.stem] = dataset_path
        else:
            # Directory - search for files
            for ext in ("*.h5", "*.h5ad"):
                for fpath in sorted(dataset_path.glob(ext)):
                    files[fpath.stem] = fpath

    return files

def _setup_global_maps(dataset_paths, pert_col, batch_col, cell_type_key):
    """
    Set up global one-hot maps for perturbations and batches.
    For perturbations, we scan through all files in all train_specs and test_specs.
    """
    all_perts = set()
    all_batches = set()
    all_celltypes = set()

    for dataset_path in dataset_paths:
        files = _find_dataset_files(dataset_path)

        for _fname, fpath in files.items():
            with h5py.File(fpath, "r") as f:
                pert_arr = f[f"obs/{pert_col}/categories"][:]
                perts = set(safe_decode_array(pert_arr))
                all_perts.update(perts)

                try:
                    batch_arr = f[f"obs/{batch_col}/categories"][:]
                except KeyError:
                    batch_arr = f[f"obs/{batch_col}"][:]
                batches = set(safe_decode_array(batch_arr))
                all_batches.update(batches)

                try:
                    celltype_arr = f[f"obs/{cell_type_key}/categories"][:]
                except KeyError:
                    celltype_arr = f[f"obs/{cell_type_key}"][:]
                celltypes = set(safe_decode_array(celltype_arr))
                all_celltypes.update(celltypes)

    batch_onehot_map = generate_onehot_map(all_batches)
    cell_type_onehot_map = generate_onehot_map(all_celltypes)
    return batch_onehot_map, cell_type_onehot_map


def run_tx_infer(args):
    import logging
    import os
    import pickle

    import numpy as np
    import scanpy as sc
    import torch
    import yaml
    from tqdm import tqdm

    from ...tx.models.state_transition import StateTransitionPerturbationModel

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    preprocess_with_cell_eval = args.preprocess_with_cell_eval

    def load_config(cfg_path: str) -> dict:
        """Load config from the YAML file that was dumped during training."""
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Could not find config file: {cfg_path}")
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    # Load the config
    config_path = os.path.join(args.model_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # Determine checkpoint path
    checkpoint_dir = Path(args.model_dir).resolve() / 'checkpoints'
    final_checkpoint_path = checkpoint_dir / 'final.ckpt'
    if args.checkpoint is not None:
        if args.checkpoint == 'best':
            checkpoint_path = str(next((p for p in Path(checkpoint_dir).iterdir() if 'val_loss' in p.stem)))
        elif args.checkpoint == 'final':
            checkpoint_path = final_checkpoint_path
        else:
            checkpoint_path = args.checkpoint
    else:
        checkpoint_path = str(final_checkpoint_path)
        logger.info(f"No checkpoint provided, reverting to default: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        if not Path(checkpoint_path).exists():
            logger.info(f'Failed to find {checkpoint_path}, looking in model dir.')
            checkpoint_path = checkpoint_dir / checkpoint_path
            if not Path(checkpoint_path).exists():
                raise FileNotFoundError(checkpoint_path)

    # Get perturbation dimensions and mapping from data module
    var_dims_path = os.path.join(args.model_dir, "var_dims.pkl")
    with open(var_dims_path, "rb") as f:
        var_dims = pickle.load(f)
    pert_dim = var_dims["pert_dim"]

    # Load model
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = StateTransitionPerturbationModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    cell_sentence_len = model.cell_sentence_len
    device = next(model.parameters()).device

    # Load AnnData
    logger.info(f"Loading AnnData from: {args.adata}")
    adata = sc.read_h5ad(args.adata)

    # Optionally filter by cell type
    if args.celltype_col is not None and args.celltypes is not None:
        celltypes = [ct.strip() for ct in args.celltypes.split(",")]
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        initial_n = adata.n_obs
        adata = adata[adata.obs[args.celltype_col].isin(celltypes)].copy()
        logger.info(f"Filtered AnnData to {adata.n_obs} cells of types {celltypes} (from {initial_n} cells)")
    elif args.celltype_col is not None:
        if args.celltype_col not in adata.obs:
            raise ValueError(f"Column '{args.celltype_col}' not found in adata.obs.")
        logger.info(f"No cell type filtering applied, but cell type column '{args.celltype_col}' is available.")

    # Get input features
    if args.embed_key in adata.obsm:
        X = adata.obsm[args.embed_key]
        logger.info(f"Using adata.obsm['{args.embed_key}'] as input features: shape {X.shape}")
    else:
        try:
            X = adata.X.toarray()
        except:
            X = adata.X
        logger.info(f"Using adata.X as input features: shape {X.shape}")

    # Prepare perturbation tensor using the data module's mapping
    pert_names = adata.obs[args.pert_col].values
    pert_tensor = torch.zeros((len(pert_names), pert_dim), device="cpu")  # Keep on CPU initially
    logger.info(f"Perturbation tensor shape: {pert_tensor.shape}")

    # Load perturbation mapping from torch file
    pert_onehot_map_path = os.path.join(args.model_dir, "pert_onehot_map.pt")
    pert_onehot_map = torch.load(pert_onehot_map_path, weights_only=False)

    logger.info(f"Data module has {len(pert_onehot_map)} perturbations in mapping")
    logger.info(f"First 10 perturbations in data module: {list(pert_onehot_map.keys())[:10]}")

    unique_pert_names = sorted(set(pert_names))
    logger.info(f"AnnData has {len(unique_pert_names)} unique perturbations")
    logger.info(f"First 10 perturbations in AnnData: {unique_pert_names[:10]}")

    # Check overlap
    overlap = set(unique_pert_names) & set(pert_onehot_map.keys())
    logger.info(f"Overlap between AnnData and data module: {len(overlap)} perturbations")
    if len(overlap) < len(unique_pert_names):
        missing = set(unique_pert_names) - set(pert_onehot_map.keys())
        logger.warning(f"Missing perturbations: {list(missing)[:10]}")

    # Check if there's a control perturbation that might match
    control_pert = args.ctrl_pert
    if args.pert_col == "drugname_drugconc":  # quick hack for tahoe
        control_pert = "[('DMSO_TF', 0.0, 'uM')]"
    logger.info(f"Control perturbation in data module: '{control_pert}'")

    matched_count = 0
    for idx, name in enumerate(pert_names):
        if name in pert_onehot_map:
            pert_tensor[idx] = pert_onehot_map[name]
            matched_count += 1
        else:
            # For now, use control perturbation as fallback
            if control_pert in pert_onehot_map:
                pert_tensor[idx] = pert_onehot_map[control_pert]
            else:
                # Use first available perturbation as fallback
                first_pert = list(pert_onehot_map.keys())[0]
                pert_tensor[idx] = pert_onehot_map[first_pert]

    logger.info(f"Matched {matched_count} out of {len(pert_names)} perturbations")



    # Process in batches with progress bar
    # Use cell_sentence_len as batch size since model expects this
    n_samples = len(pert_names)
    batch_size = cell_sentence_len  # Model requires this exact batch size
    n_batches = (n_samples + batch_size - 1) // batch_size  # Ceiling division

    cfg_dir = str(Path(__file__).resolve().parents[2] / "configs")

    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        default_cfg = compose(config_name="config")

    kwargs = {**default_cfg['data']['kwargs']}

    kwargs['num_workers'] = 1
    kwargs['batch_col'] = "batch_var"
    kwargs['pert_col'] = "target_gene"
    kwargs['cell_type_key'] = "cell_type"
    kwargs['control_pert'] = "non-targeting"
    kwargs['perturbation_features_file'] = "/home/hackerman/Github/state/competition_support_set/ESM2_pert_features.pt"
    kwargs["batch_size"] = batch_size
    kwargs["cell_sentence_len"] = cell_sentence_len
    kwargs['pert_col'] = args.pert_col
    kwargs['embed_key'] = args.embed_key
    kwargs['control_pert'] = control_pert
    kwargs['batch_size'] = batch_size

    exp_config = ExperimentConfig(datasets={'replogle_h1': str(Path(args.adata).parent)}, training={},
                                  zeroshot={f'replogle_h1.{Path(args.adata).stem.split("_")[0]}': 'test'},
                                  fewshot={})

    data_module = PerturbationDataModuleFromExperimentConfig(exp_config,
                                                            **kwargs,
                                                            )
    data_module.setup(stage="fit")
    dl = data_module.test_dataloader()
    print("num_workers:", dl.num_workers)
    print("batch size:", dl.batch_size)


    logger.info(
        f"Running inference on {n_samples} samples in {n_batches} batches of size {batch_size} (model's cell_sentence_len)..."
    )

    all_preds = []
    # all_gt = []

    with torch.no_grad():
        device = torch.device("cuda", 0)

        progress_bar = tqdm(total=n_samples, desc="Processing samples", unit="samples")
        for batch_idx, batch in enumerate(dl):

            current_batch_size = batch['pert_emb'].shape[0]
            batch_gpu = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}

            if current_batch_size < cell_sentence_len:
                # Pad with zeros for embeddings
                X_batch = batch_gpu['ctrl_cell_emb']
                pert_batch = batch_gpu['pert_emb']
                padding_size = cell_sentence_len - current_batch_size
                X_pad = torch.zeros((padding_size, X_batch.shape[1]), device=device)
                X_batch = torch.cat([X_batch, X_pad], dim=0)

                # Pad perturbation tensor with control perturbation
                pert_pad = torch.zeros((padding_size, pert_batch.shape[1]), device=device)
                if control_pert in pert_onehot_map:
                    pert_pad[:] = pert_onehot_map[control_pert].to(device)
                else:
                    pert_pad[:, 0] = 1  # Default to first perturbation
                pert_batch = torch.cat([pert_batch, pert_pad], dim=0)
                batch_gpu['ctrl_cell_emb'] = X_batch
                batch_gpu['pert_emb'] = pert_batch
                batch_gpu["batch"]: torch.zeros((1, cell_sentence_len), device=device)


            batch_preds = model.predict_step(batch_gpu, batch_idx=batch_idx, padded=False)

            # Extract predictions from the dictionary returned by predict_step
            # Use gene decoder output if available, otherwise use latent predictions
            if "pert_cell_counts_preds" in batch_preds and batch_preds["pert_cell_counts_preds"] is not None:
                # Use gene space predictions (from decoder)
                pred_tensor = batch_preds["pert_cell_counts_preds"]
            else:
                # Use latent space predictions
                pred_tensor = batch_preds["preds"]

            # Only keep predictions for the actual samples (not padding)
            actual_preds = pred_tensor[:current_batch_size]
            if args.ctrl_pert_option == "replace":
                mask = torch.tensor([n == control_pert for n in batch["pert_name"][:current_batch_size]], dtype=torch.bool).to(device)
                actual_preds = torch.where(mask.unsqueeze(1),
                                           batch_gpu['ctrl_cell_emb'][:current_batch_size],
                                           actual_preds[:current_batch_size])

            all_preds.append(actual_preds.cpu().numpy())
            # all_gt.append(batch["ctrl_cell_emb"].cpu().numpy())

            # Update progress bar
            progress_bar.update(current_batch_size)

        progress_bar.close()

    # Concatenate all predictions
    preds_np = np.concatenate(all_preds, axis=0)

    # Save predictions to AnnData
    adata.X = preds_np
    # adata.obsm['GT'] = np.concatenate(all_gt, axis=0)
    output_path = args.output or args.adata.replace(".h5ad", "_with_preds.h5ad")
    adata.write_h5ad(output_path)
    logger.info(f"Saved predictions to {output_path} (in adata.X)")


def main():
    parser = argparse.ArgumentParser(description="Run inference on AnnData with a trained model checkpoint.")
    add_arguments_infer(parser)
    args = parser.parse_args()
    run_tx_infer(args)


if __name__ == "__main__":
    main()
