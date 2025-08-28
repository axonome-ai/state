import logging
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

from cell_load.config import ExperimentConfig
from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule
from cell_load.dataset._metadata import MetadataConcatDataset
from cell_load.dataset._perturbation import PerturbationDataset
# from cell_load.mapping_strategies.random import RandomMappingStrategy
from cell_load.mapping_strategies.batch import BatchMappingStrategy
from cell_load.mapping_strategies.first_one import FirstOneMappingStrategy
from cell_load.utils.data_utils import GlobalH5MetadataCache
from tqdm import tqdm

from state._cli._tx.linear_scheduler import RandomMappingStrategy

logger = logging.getLogger(__name__)


from typing import Literal, Set, Iterator, Iterable
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


from typing import List

class ListSampler(Sampler[int]):
    """
    Yields exactly the indices you provide, in that order.
    """
    def __init__(self, indices: Iterable[int]):
        self.indices: List[int] = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

def to_subset_dataset(
    self,
    split: str,
    perturbed_indices: np.ndarray,
    control_indices: np.ndarray,
) -> Subset:
    """
    Creates a Subset of this dataset that includes only the specified perturbed_indices.
    If `self.should_yield_control_cells` flag is True, the Subset will also yield control cells.

    Args:
        split: Name of the split to create, one of 'train', 'val', 'test', or 'train_eval'
        perturbed_indices: Indices of perturbed cells to include
        control_indices: Indices of control cells to include
    """

    # sort them for stable ordering
    perturbed_indices = np.sort(perturbed_indices)
    control_indices = np.sort(control_indices)

    # Register them in the dataset
    self._register_split_indices(split, perturbed_indices, control_indices)

    # Return a Subset containing perturbed cells and optionally control cells
    if self.should_yield_control_cells:
        all_indices = np.sort(np.concatenate([perturbed_indices, control_indices]))
        return Subset(self, all_indices)
    else:
        return Subset(self, perturbed_indices)


class PerturbationDataModuleFromExperimentConfig(PerturbationDataModule):
    """
    Accepts a pre-parsed ExperimentConfig instead of a TOML path.
    We bypass PerturbationDataModule.__init__ (which expects a path),
    but still initialize LightningDataModule via an MRO-safe super() call.
    """

    def __init__(
        self,
        config: "ExperimentConfig",
        *,
        batch_size: int = 128,
        num_workers: int = 8,
        random_seed: int = 42,  # to be superseded by seed_everything
        pert_col: str = "gene",
        batch_col: str = "gem_group",
        cell_type_key: str = "cell_type",
        control_pert: str = "non-targeting",
        embed_key: Literal["X_hvg", "X_state"] | None = None,
        output_space: Literal["gene", "all"] = "gene",
        basal_mapping_strategy: Literal["batch", "random", "first_one"] = "random",
        n_basal_samples: int = 1,
        should_yield_control_cells: bool = True,
        cell_sentence_len: int = 512,
        **kwargs,
    ):
        # IMPORTANT: MRO-safe init of LightningDataModule without touching the parent ctor
        super(PerturbationDataModule, self).__init__()  # calls LightningDataModule.__init__

        # Use provided config directly
        self.config = config
        self.config.validate()

        # Optional breadcrumb if your ExperimentConfig carries a source path
        self.toml_config_path = getattr(config, "source_path", None)

        # Experiment-level params
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)

        # H5 field names
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.embed_key = embed_key
        self.output_space = output_space

        # Sampling/mapping
        self.n_basal_samples = n_basal_samples
        self.cell_sentence_len = cell_sentence_len
        self.should_yield_control_cells = should_yield_control_cells

        # Optional behaviors (mirror base kwargs)
        self.map_controls = kwargs.get("map_controls", True)
        self.perturbation_features_file = kwargs.get("perturbation_features_file")
        self.int_counts = kwargs.get("int_counts", False)
        self.normalize_counts = kwargs.get("normalize_counts", False)
        self.store_raw_basal = kwargs.get("store_raw_basal", False)
        self.barcode = kwargs.get("barcode", False)

        logger.info(
            f"Initializing DataModuleFromExperimentConfig: batch_size={batch_size}, "
            f"workers={num_workers}, random_seed={random_seed}"
        )

        # Mapping strategy
        self.basal_mapping_strategy = basal_mapping_strategy
        mapping = {
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
            "first_one": FirstOneMappingStrategy,
        }
        if basal_mapping_strategy not in mapping:
            raise ValueError(
                f"Unsupported basal_mapping_strategy={basal_mapping_strategy!r}. "
                f"Choose from {list(mapping.keys())}."
            )
        self.mapping_strategy_cls = mapping[basal_mapping_strategy]

        # Determine if raw expression is needed (mirror base logic)
        self.store_raw_expression = bool(
            self.embed_key
            and (
                (self.embed_key != "X_hvg" and self.output_space == "gene")
                or self.output_space == "all"
            )
        )

        # Prepare dataset lists and maps
        self.train_datasets: list[Dataset] = []
        self.val_datasets: list[Dataset] = []
        self.test_datasets: list[Dataset] = []

        self.all_perts: Set[str] = set()
        self.pert_onehot_map: dict[str, torch.Tensor] | None = None
        self.batch_onehot_map: dict[str, torch.Tensor] | None = None
        self.cell_type_onehot_map: dict[str, torch.Tensor] | None = None

        # Initialize global maps
        self._setup_global_maps()

    def test_dataloader(self):
        if len(self.test_datasets) == 0:
            return None
        return self._create_dataloader(self.test_datasets, batch_size=self.batch_size)

    def _create_dataloader(
        self,
        datasets: list[Dataset],
        test: bool = False,
        batch_size: int | None = None,
        shuffle=False
    ):
        """Create a DataLoader with appropriate configuration."""
        use_int_counts = "int_counts" in self.__dict__ and self.int_counts
        collate_fn = partial(PerturbationDataset.collate_fn, int_counts=use_int_counts)

        ds = MetadataConcatDataset(datasets)
        # pert_name_cell_type_to_indices = defaultdict(list)
        # for i, d in enumerate(ds):
        #     pert_name_cell_type_to_indices[d['pert_name'], d['cell_type']].append(i)
        # indices = sum(pert_name_cell_type_to_indices.values(), [])
        # sampler = ListSampler(indices)
        batch_size = batch_size or (1 if test else self.batch_size)

        # use_batch = self.basal_mapping_strategy == "batch"
        # sampler = PerturbationBatchSampler(
        #     dataset=ds,
        #     batch_size=batch_size,
        #     drop_last=False,
        #     cell_sentence_len=self.cell_sentence_len,
        #     test=test,
        #     use_batch=use_batch,
        # )
        sampler = None
        batch_sampler = None
        out = DataLoader(
            ds,
            sampler=sampler,
            batch_size=batch_size if batch_sampler is None else 1,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=4 if not test else None,
        )
        print()
        return out

    def _process_celltype(
        self,
        ds: PerturbationDataset,
        celltype: str,
        ct_indices: np.ndarray,
        ctrl_indices: np.ndarray,
        pert_indices: np.ndarray,
        cache,
        dataset_name: str,
        zeroshot_celltypes: dict[str, str],
        fewshot_celltypes: dict[str, dict[str, list[str]]],
        is_training_dataset: bool,
    ) -> dict[str, int]:
        """Process a single cell type and return counts for each split."""
        counts = {"train": 0, "val": 0, "test": 0}

        if celltype in zeroshot_celltypes:
            # Zeroshot: all cells go to specified split
            split = zeroshot_celltypes[celltype]
            subset = to_subset_dataset(ds, split, pert_indices, ctrl_indices)
            # subset = ds.to_subset_dataset(split, pert_indices, ctrl_indices)

            if split == "train":
                self.train_datasets.append(subset)
            elif split == "val":
                self.val_datasets.append(subset)
            elif split == "test":
                self.test_datasets.append(subset)

            counts[split] = len(subset)

        elif celltype in fewshot_celltypes:
            # Fewshot: split perturbations according to config
            pert_config = fewshot_celltypes[celltype]
            split_counts = self._split_fewshot_celltype(
                ds, pert_indices, ctrl_indices, cache, pert_config
            )
            for split, count in split_counts.items():
                counts[split] += count

        elif is_training_dataset:
            # Regular training cell type
            subset = to_subset_dataset(ds, "train", pert_indices, ctrl_indices)
            # subset = ds.to_subset_dataset("train", pert_indices, ctrl_indices)
            self.train_datasets.append(subset)
            counts["train"] = len(subset)

        return counts
    def _create_base_dataset(
        self, dataset_name: str, fpath: Path
    ) -> PerturbationDataset:
        """Create a base PerturbationDataset instance."""
        mapping_kwargs = {"map_controls": self.map_controls}

        return PerturbationDataset(
            name=dataset_name,
            h5_path=fpath,
            mapping_strategy=self.mapping_strategy_cls(
                random_state=self.random_seed,
                n_basal_samples=self.n_basal_samples,
                **mapping_kwargs,
            ),
            embed_key=self.embed_key,
            pert_onehot_map=self.pert_onehot_map,
            batch_onehot_map=self.batch_onehot_map,
            cell_type_onehot_map=self.cell_type_onehot_map,
            pert_col=self.pert_col,
            cell_type_key=self.cell_type_key,
            batch_col=self.batch_col,
            control_pert=self.control_pert,
            random_state=self.random_seed,
            should_yield_control_cells=self.should_yield_control_cells,
            store_raw_expression=self.store_raw_expression,
            output_space=self.output_space,
            store_raw_basal=self.store_raw_basal,
            barcode=self.barcode,
        )

    def _setup_datasets(self):
        """
        Set up training datasets with proper handling of zeroshot/fewshot splits w/ TOML.
        Uses H5MetadataCache for faster metadata access.
        """

        for dataset_name in self.config.get_all_datasets():
            dataset_path = Path(self.config.datasets[dataset_name])
            files = self._find_dataset_files(dataset_path)

            # Get configuration for this dataset
            zeroshot_celltypes = self.config.get_zeroshot_celltypes(dataset_name)
            fewshot_celltypes = self.config.get_fewshot_celltypes(dataset_name)
            is_training_dataset = self.config.training.get(dataset_name) == "train"

            logger.info(f"Processing dataset {dataset_name}:")
            logger.info(f"  - Training dataset: {is_training_dataset}")
            logger.info(f"  - Zeroshot cell types: {list(zeroshot_celltypes.keys())}")
            logger.info(f"  - Fewshot cell types: {list(fewshot_celltypes.keys())}")

            # Process each file in the dataset
            for fname, fpath in tqdm(
                list(files.items()), desc=f"Processing {dataset_name}"
            ):
                # Create metadata cache
                cache = GlobalH5MetadataCache().get_cache(
                    str(fpath),
                    self.pert_col,
                    self.cell_type_key,
                    self.control_pert,
                    self.batch_col,
                )

                # Create base dataset
                ds = self._create_base_dataset(dataset_name, fpath)
                train_sum = val_sum = test_sum = 0

                # Process each cell type in this file
                for ct_idx, ct in enumerate(cache.cell_type_categories):
                    ct_mask = cache.cell_type_codes == ct_idx
                    n_cells = np.sum(ct_mask)

                    if n_cells == 0:
                        continue

                    ct_indices = np.where(ct_mask)[0]

                    # Split into control and perturbed indices
                    ctrl_mask = cache.pert_codes[ct_indices] == cache.control_pert_code
                    ctrl_indices = ct_indices[ctrl_mask]
                    pert_indices = ct_indices[~ctrl_mask]

                    # Determine how to handle this cell type
                    counts = self._process_celltype(
                        ds,
                        ct,
                        ct_indices,
                        ctrl_indices,
                        pert_indices,
                        cache,
                        dataset_name,
                        zeroshot_celltypes,
                        fewshot_celltypes,
                        is_training_dataset,
                    )

                    train_sum += counts["train"]
                    val_sum += counts["val"]
                    test_sum += counts["test"]

                tqdm.write(
                    f"Processed {fname}: {train_sum} train, {val_sum} val, {test_sum} test"
                )

            logger.info("\n")
        logger.info("\n")


    def _split_fewshot_celltype(
        self,
        ds: PerturbationDataset,
        pert_indices: np.ndarray,
        ctrl_indices: np.ndarray,
        cache,
        pert_config: dict[str, list[str]],
    ) -> dict[str, int]:
        """Split a fewshot cell type according to perturbation assignments."""
        counts = {"train": 0, "val": 0, "test": 0}

        # Get perturbation codes for this cell type
        pert_codes = cache.pert_codes[pert_indices]

        # Create sets of perturbation codes for each split
        val_pert_names = set(pert_config.get("val", []))
        test_pert_names = set(pert_config.get("test", []))

        val_pert_codes = set()
        test_pert_codes = set()

        for i, pert_name in enumerate(cache.pert_categories):
            if pert_name in val_pert_names:
                val_pert_codes.add(i)
            if pert_name in test_pert_names:
                test_pert_codes.add(i)

        # Split perturbation indices by their codes
        val_mask = np.isin(pert_codes, list(val_pert_codes))
        test_mask = np.isin(pert_codes, list(test_pert_codes))
        train_mask = ~(val_mask | test_mask)

        val_pert_indices = pert_indices[val_mask]
        test_pert_indices = pert_indices[test_mask]
        train_pert_indices = pert_indices[train_mask]

        # Split controls proportionally
        # rng = np.random.default_rng(self.random_seed)
        # ctrl_indices_shuffled = rng.permutation(ctrl_indices)
        ctrl_indices_shuffled = ctrl_indices

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
                subset = to_subset_dataset(ds, "val", val_pert_indices, val_ctrl_indices)
                # subset = ds.to_subset_dataset("val", val_pert_indices, val_ctrl_indices)
                self.val_datasets.append(subset)
                counts["val"] = len(subset)

            if len(test_pert_indices) > 0:
                subset = to_subset_dataset(ds, "test", test_pert_indices, test_ctrl_indices)
                # subset = ds.to_subset_dataset(
                #     "test", test_pert_indices, test_ctrl_indices
                # )
                self.test_datasets.append(subset)
                counts["test"] = len(subset)

            if len(train_pert_indices) > 0:
                subset = to_subset_dataset(ds, "train", train_pert_indices, train_ctrl_indices)
                # subset = ds.to_subset_dataset(
                #     "train", train_pert_indices, train_ctrl_indices
                # )
                self.train_datasets.append(subset)
                counts["train"] = len(subset)

        return counts