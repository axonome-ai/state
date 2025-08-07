"""
Stateless utilities for validation metrics.

Keeping these pure makes them easy to test and reuse.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from pdex import parallel_differential_expression

from state.emb.utils import compute_pearson_delta, compute_perturbation_ranking_score, compute_gene_overlap_cross_pert

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




def _global_mean_offsets(
    adata: sc.AnnData,
    col_id: str,
    ctrl_label: str,
) -> dict[str, np.ndarray]:
    """
    Average perturbation effect across the whole validation set
    (stratified by cell type so we capture cell-type-specific baselines).
    """
    dfs: list[pd.DataFrame] = []

    for ct in adata.obs["cell_type"].unique():
        subset = adata[adata.obs["cell_type"] == ct]
        ctrl   = subset[subset.obs[col_id] == ctrl_label]
        pert   = subset[subset.obs[col_id] != ctrl_label]

        mean_ctrl = ctrl.obsm["X_emb"].mean(0)
        offsets   = pert.obsm["X_emb"] - mean_ctrl

        df = pd.DataFrame(
            offsets,
            index   = pert.obs_names,
            columns = [f"emb_{i}" for i in range(offsets.shape[1])],
        )
        df[col_id] = pert.obs[col_id].values
        dfs.append(df.groupby(col_id).mean())

    merged = pd.concat(dfs).groupby(level=0).mean()
    mapping = {k: v.values for k, v in merged.iterrows()}
    mapping[ctrl_label] = np.zeros_like(next(iter(mapping.values())))
    return mapping


# --------------------------------------------------------------------- #
#                            Perturbation metric                        #
# --------------------------------------------------------------------- #
def perturbation_metrics(
    adata: sc.AnnData,
    col_id: str,
    ctrl_label: str,
) -> Tuple[float, float]:
    """
    Pearson-delta and ranking score aggregated over *all* cells.
    The same AnnData is used both to derive mean perturbation effects
    and to score predictions – no separate train/test split.
    """
    # 1.  Mean perturbation offsets across *all* cell types
    pert_offsets = _global_mean_offsets(adata, col_id, ctrl_label)

    # 2.  Build predictions for every cell
    pred_x = np.zeros_like(adata.obsm["X_emb"])
    ctrl_mask = adata.obs[col_id] == ctrl_label

    for i, idx in enumerate(adata.obs.index):
        pert = adata.obs.at[idx, col_id]
        if pert not in pert_offsets:            # unseen perturbation
            continue

        if pert == ctrl_label:
            basal = adata.obsm["X_emb"][i]      # the cell itself
        else:
            # random control cell of the *same* cell type
            same_ct = adata.obs["cell_type"] == adata.obs.at[idx, "cell_type"]
            ctrl_pool = np.where(ctrl_mask & same_ct)[0]
            basal = adata.obsm["X_emb"][np.random.choice(ctrl_pool)]

        pred_x[i] = basal + pert_offsets[pert]

    # 3.  Metrics
    pred = sc.AnnData(X=pred_x, obs=adata.obs.copy())
    ctrl = pred[pred.obs[col_id] == ctrl_label]

    corr = compute_pearson_delta(pred.X, adata.X, ctrl.X, ctrl.X)
    rank = compute_perturbation_ranking_score(pred, adata, pert_col=col_id)
    return float(corr), float(rank)


def _cell_type_means(
    adata: sc.AnnData,
    col_id: str,
    ctrl_label: str,
) -> dict[str, np.ndarray]:
    """
    Average perturbation effect per cell-type; returns mapping
    perturbation → mean offset vector.
    """
    dfs: list[pd.DataFrame] = []

    for ct in adata.obs["cell_type"].unique():
        subset = adata[adata.obs["cell_type"] == ct]
        ctrl = subset[subset.obs[col_id] == ctrl_label]
        pert = subset[subset.obs[col_id] != ctrl_label]

        mean_ctrl = ctrl.obsm["X_emb"].mean(0)
        offsets = pert.obsm["X_emb"] - mean_ctrl

        df = pd.DataFrame(
            offsets,
            index=pert.obs_names,
            columns=[f"emb_{i}" for i in range(offsets.shape[1])],
        )
        df[col_id] = pert.obs[col_id].values
        dfs.append(df.groupby(col_id).mean())

    merged = pd.concat(dfs).groupby(level=0).mean()
    mapping = {k: v.values for k, v in merged.iterrows()}
    mapping[ctrl_label] = np.zeros_like(next(iter(mapping.values())))
    return mapping


def _evaluate_on_holdout(
    test: sc.AnnData,
    mean_perturbations: dict[str, np.ndarray],
    col_id: str,
    ctrl_label: str,
) -> Tuple[float, float]:
    """
    Compute correlation and ranking score for a single hold-out cell type.
    """
    pred_x = np.zeros_like(test.obsm["X_emb"])
    ctrl_idxs = test[test.obs[col_id] == ctrl_label].obs.index

    for i, idx in enumerate(test.obs.index):
        pert = test.obs.at[idx, col_id]
        if pert not in mean_perturbations:
            continue
        ctrl_idx = idx if pert == ctrl_label else np.random.choice(ctrl_idxs)
        pred_x[i] = (
            test[ctrl_idx].obsm["X_emb"] + mean_perturbations[pert]
        )

    pred = sc.AnnData(X=pred_x, obs=test.obs.copy())
    pred = pred[pred.obs[col_id].isin(mean_perturbations)]
    real = test[test.obs[col_id].isin(mean_perturbations)]
    ctrl = pred[pred.obs[col_id] == ctrl_label]

    corr = compute_pearson_delta(pred.X, real.X, ctrl.X, ctrl.X)
    rank = compute_perturbation_ranking_score(pred, real)
    return corr, rank


# --------------------------------------------------------------------- #
#                       Differential-expression metric                  #
# --------------------------------------------------------------------- #
def de_metric(
    adata_pred: sc.AnnData,
    adata_real: sc.AnnData,
    col_id: str,
    ctrl_label: str,
    k: int,
    method: str,
) -> float:
    """Mean gene-overlap score across perturbations."""
    true_top = _top_k_genes(adata_real, col_id, ctrl_label, k, method)
    pred_top = _top_k_genes(adata_pred, col_id, ctrl_label, k, method)
    overlap = compute_gene_overlap_cross_pert(pred_top, true_top, k=k)
    return float(np.mean(list(overlap.values())))


def _top_k_genes(
    adata: sc.AnnData,
    col_id: str,
    ctrl_label: str,
    k: int,
    method: str,
) -> pd.DataFrame:
    """
    Returns DataFrame shaped (perturbations × k) with top-k genes per perturb.
    """
    de = parallel_differential_expression(
        adata,
        reference=ctrl_label,
        groupby_key=col_id,
        metric=method,
        is_log1p=True,
        exp_post_agg=True,
        num_workers=1,
    )
    de = (
        de.groupby("target")
          .apply(lambda df: df.nlargest(k, "abs_stat"))
          .reset_index(drop=True)
          .pivot(columns="target", values="feature")
    )
    return de
