from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from typing import Sequence


def build_anndata(
    x_matrix: np.ndarray,          # 2-D (N, D)  or 3-D (N, L, D)
    embeddings: np.ndarray,        # 2-D or 3-D
    perturbations: Sequence[str],
    cell_types:    Sequence[str],
    pert_col_name: str,               # ← pass cfg[...] here
    celltype_col_name: str = "cell_type",
    batches:       Sequence[str] | None = None,
    batch_col_name: str = "batch",
    var_names:     Sequence[str] | None = None,
) -> sc.AnnData:
    """
    • If an input is 3-D, reshape (N, L, D) → (N*L, D).
    • After flattening, the two matrices **must** have the same
      number of rows; otherwise we raise.
    """

    # --- flatten if needed -------------------------------------------
    if x_matrix.ndim == 3:
        x_matrix = x_matrix.reshape(-1, x_matrix.shape[-1])
    if embeddings.ndim == 3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])

    # ---- 2. sanity-check row counts ---------------------------------
    if len(x_matrix) != len(embeddings):
        raise ValueError(
            f"Row mismatch after flattening: "
            f"x_matrix has {len(x_matrix)}, embeddings has {len(embeddings)}."
        )
    n_obs = len(x_matrix)

    # ---- 3. metadata length check -----------------------------------
    if not (len(perturbations) == len(cell_types) == n_obs):
        raise ValueError(
            "Metadata lengths differ from data rows. "
            f"n_obs = {n_obs}, "
            f"len(perturbations) = {len(perturbations)}, "
            f"len(cell_types) = {len(cell_types)}."
        )
    if batches is not None and len(batches) != n_obs:
        raise ValueError(
            f"len(batches) = {len(batches)} differs from n_obs = {n_obs}."
        )

    # --- obs dataframe ------------------------------------------------
    obs = {
        pert_col_name:     perturbations,
        celltype_col_name: cell_types,
    }
    if batches is not None:
        obs[batch_col_name] = batches

    adata = sc.AnnData(X=x_matrix, obs=pd.DataFrame(obs))
    adata.obsm["X_emb"] = embeddings
    if var_names is not None:
        adata.var_names = var_names
    return adata
