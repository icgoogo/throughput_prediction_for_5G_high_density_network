from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DatasetConfig:
    """Kaggle/local dataset paths."""

    input_root: Path = Path("/kaggle/input/datasets/hammadsyarif/l5ghdd-ds")
    working_dir: Path = Path("/kaggle/working")
    arena_name: str = "Acc Arena"
    master_name: str = "master_quantile_full_df.parquet"
    engineered_name: str = "arena_quantile_full_df.parquet"
    max_user_id: int = 11_999
    chunk_size: int = 500

    @property
    def acc_arena_dir(self) -> Path:
        return self.input_root / self.arena_name

    @property
    def cached_master_path(self) -> Path:
        return self.working_dir / self.master_name

    @property
    def cached_feature_path(self) -> Path:
        return self.working_dir / self.engineered_name

    @property
    def input_master_path(self) -> Path:
        return self.input_root / self.master_name

    @property
    def input_feature_path(self) -> Path:
        return self.input_root / self.engineered_name

    @property
    def model_dir(self) -> Path:
        return self.working_dir / "saved_models"


@dataclass
class RFConfig:
    """Random Forest settings.

    The `full_safe_*` settings are for all-row training without creating an
    unnecessarily huge tree ensemble. They do not reduce row count.
    """

    random_state: int = 42
    active_threshold: float = 0.5
    log_rf_trees: int = 1
    sample_n_s1: Optional[int] = 800_000
    sample_n_s2: Optional[int] = 500_000
    use_gpu_rf: bool = False
    batch_size: int = 500_000

    # Original notebook-like defaults for feature search.
    n_estimators: int = 100
    max_depth: int = 20
    n_jobs: int = -1

    # Safer full-data defaults.
    full_safe_n_estimators: int = 30
    full_safe_max_depth: int = 12
    full_safe_max_leaf_nodes: int = 2048
    full_safe_min_samples_leaf: int = 200
    full_safe_n_jobs: int = 1
    full_safe_bootstrap: bool = False


@dataclass
class SplitConfig:
    test_id_start: int = 10_000
    val_frac: float = 0.20
    random_seed: int = 42
    is_undersampling: bool = False


@dataclass
class SequenceConfig:
    seed: int = 42
    seq_len: int = 16
    stride: int = 4
    batch_size: int = 256
    lstm_hidden: int = 32
    mlp_hidden: int = 16
    fusion_hidden: int = 32
    num_layers: int = 1
    dropout: float = 0.20
    epochs: int = 20
    patience: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class PipelineConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    rf: RFConfig = field(default_factory=RFConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    target: str = "Throughput"
    run_sequence_model: bool = False
    fast_mode: bool = True
