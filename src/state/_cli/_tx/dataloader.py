import logging
import random
from functools import partial

from cell_load.config import ExperimentConfig
from cell_load.data_modules.perturbation_dataloader import PerturbationDataModule
from cell_load.data_modules.samplers import PerturbationBatchSampler
from cell_load.dataset import PerturbationDataset, MetadataConcatDataset
from cell_load.mapping_strategies import FirstOneMappingStrategy, RandomMappingStrategy, BatchMappingStrategy


logger = logging.getLogger(__name__)


from typing import Literal, Set
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


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
        return self._create_dataloader(self.test_datasets, test=True, batch_size=1)

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
        use_batch = self.basal_mapping_strategy == "batch"

        batch_size = batch_size or (1 if test else self.batch_size)

        sampler = PerturbationBatchSampler(
            dataset=ds,
            batch_size=batch_size,
            drop_last=False,
            cell_sentence_len=self.cell_sentence_len,
            test=test,
            use_batch=use_batch,
        )
        out = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=4 if not test else None,
        )
        print()
        return out
