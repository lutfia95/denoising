from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml

from src.features.fdr_weights import FDRWeightConfig
from src.features.peak_features import PeakFeatureConfig
from src.features.spectrum_features import SpectrumFeatureConfig
from src.model.mlp import MLPConfig
from src.model.transformer import PeakTransformerConfig
from src.splitting.splitter import SplitConfig


@dataclass(slots=True)
class AppConfig:
    peak_features: PeakFeatureConfig
    spectrum_features: SpectrumFeatureConfig
    fdr: FDRWeightConfig
    split: SplitConfig


@dataclass(slots=True)
class DataConfig:
    train_path: str
    val_path: str
    test_path: str
    target_column: str
    use_training_weights: bool
    weight_column: str


@dataclass(slots=True)
class FeatureSelectionConfig:
    peak_feature_columns: list[str]
    spectrum_feature_columns: list[str]
    use_raw_peak_mz: bool
    raw_peak_mz_column: str
    use_raw_peak_intensity: bool
    raw_peak_intensity_column: str
    sort_raw_peak_inputs_by_mz: bool
    broadcast_spectrum_features_to_peaks: bool
    normalize_peak_features: bool
    normalize_spectrum_features: bool
    normalization_fit_split: str


@dataclass(slots=True)
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float


@dataclass(slots=True)
class SchedulerConfig:
    enabled: bool


@dataclass(slots=True)
class TrainingLoopConfig:
    seed: int
    batch_size: int
    num_workers: int
    max_epochs: int
    cpu_num_threads: int | None
    cpu_num_interop_threads: int | None
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    dataloader_pin_memory: bool | None
    enable_amp: bool
    compile_model: bool
    cache_dataset_in_memory: bool
    save_latest_checkpoint: bool
    resume_from_checkpoint: str | None
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    gradient_clip_norm: float


@dataclass(slots=True)
class LossConfig:
    name: str
    use_pos_weight: bool
    pos_weight: float | None
    reduction: str


@dataclass(slots=True)
class EarlyStoppingConfig:
    enabled: bool
    monitor: str
    mode: str
    patience: int
    min_delta: float


@dataclass(slots=True)
class EvaluationConfig:
    primary_metric: str
    threshold_for_binary_metrics: float
    report_metrics: list[str]
    retained_peak_fractions: list[float]


@dataclass(slots=True)
class TrainingConfig:
    data: DataConfig
    features: FeatureSelectionConfig
    model: MLPConfig | PeakTransformerConfig
    training: TrainingLoopConfig
    loss: LossConfig
    early_stopping: EarlyStoppingConfig
    evaluation: EvaluationConfig
    output: OutputConfig

@dataclass(slots=True)
class OutputConfig:
    output_dir: str
    save_best_model: bool
    save_epoch_history: bool
    save_metrics_summary: bool
    save_plots: bool
    save_confusion_matrices: bool
    device_verbose: bool
    enable_file_logging: bool
    log_file_name: str


@dataclass(slots=True)
class AutoGluonConfig:
    label: str
    problem_type: str
    eval_metric: str
    presets: str
    time_limit: int | None
    verbosity: int
    positive_class: int | None
    sample_weight_column: str | None
    weight_evaluation: bool
    save_leaderboard: bool
    save_feature_importance: bool
    feature_importance_subsample_size: int | None
    feature_importance_num_shuffle_sets: int | None
    feature_importance_time_limit: int | None
    dynamic_stacking: bool
    num_stack_levels: int
    use_bag_holdout: bool
    fit_weighted_ensemble: bool
    save_bag_folds: bool


@dataclass(slots=True)
class AutoGluonOutputConfig:
    output_dir: str
    predictor_subdir: str | None
    unique_predictor_subdir: bool
    save_metrics_summary: bool
    save_confusion_matrices: bool
    save_predictions: bool
    save_flattened_tables: bool
    save_fit_summary: bool
    enable_file_logging: bool
    log_file_name: str


@dataclass(slots=True)
class AutoGluonTrainingConfig:
    data: DataConfig
    features: FeatureSelectionConfig
    evaluation: EvaluationConfig
    autogluon: AutoGluonConfig
    output: AutoGluonOutputConfig

def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    feature_cfg = raw.get("feature_engineering", {})
    fdr_cfg = raw.get("fdr", {})
    split_cfg = raw.get("split", {})

    peak_features = PeakFeatureConfig(
        use_log_intensity=bool(feature_cfg.get("use_log_intensity", True)),
        use_relative_intensity=bool(feature_cfg.get("use_relative_intensity", True)),
        use_mz_over_precursor=bool(feature_cfg.get("use_mz_over_precursor", True)),
        use_delta_to_precursor=bool(feature_cfg.get("use_delta_to_precursor", True)),
        use_delta_neighbors=bool(feature_cfg.get("use_delta_neighbors", True)),
        sort_by_mz=bool(feature_cfg.get("sort_by_mz", True)),
        eps=float(feature_cfg.get("eps", 1.0e-8)),
    )

    spectrum_features = SpectrumFeatureConfig(
        use_tic=bool(feature_cfg.get("use_tic", True)),
        use_num_peaks=bool(feature_cfg.get("use_num_peaks", True)),
    )

    fdr = FDRWeightConfig(
        enabled=bool(fdr_cfg.get("enabled", True)),
        clip_min=float(fdr_cfg.get("clip_min", 0.0)),
        clip_max=float(fdr_cfg.get("clip_max", 0.01)),
        weight_min=float(fdr_cfg.get("weight_min", 0.2)),
        mode=str(fdr_cfg.get("mode", "linear")),
    )

    split = SplitConfig(
        train_fraction=float(split_cfg.get("train_fraction", 0.70)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        test_fraction=float(split_cfg.get("test_fraction", 0.15)),
        random_seed=int(split_cfg.get("random_seed", 42)),
        split_method=str(split_cfg.get("split_method", "PeakListFileName")),
        length_weight=bool(split_cfg.get("length_weight", False)),
        length_weight_eps=float(split_cfg.get("length_weight_eps", 1.0)),
        length_weight_min=float(split_cfg.get("length_weight_min", 0.5)),
        length_weight_max=float(split_cfg.get("length_weight_max", 2.0)),
    )

    return AppConfig(
        peak_features=peak_features,
        spectrum_features=spectrum_features,
        fdr=fdr,
        split=split,
    )


def load_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_cfg = raw.get("data", {})
    output_cfg = raw.get("output", {})
    features_cfg = raw.get("features", {})
    model_cfg = raw.get("model", {})
    mlp_cfg = model_cfg.get("mlp", {})
    training_cfg = raw.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {})
    scheduler_cfg = training_cfg.get("scheduler", {})
    loss_cfg = raw.get("loss", {})
    early_cfg = raw.get("early_stopping", {})
    eval_cfg = raw.get("evaluation", {})

    features = FeatureSelectionConfig(
        peak_feature_columns=list(features_cfg.get("peak_feature_columns", [])),
        spectrum_feature_columns=list(features_cfg.get("spectrum_feature_columns", [])),
        use_raw_peak_mz=bool(features_cfg.get("use_raw_peak_mz", True)),
        raw_peak_mz_column=str(features_cfg.get("raw_peak_mz_column", "mz_arr")),
        use_raw_peak_intensity=bool(features_cfg.get("use_raw_peak_intensity", True)),
        raw_peak_intensity_column=str(
            features_cfg.get("raw_peak_intensity_column", "int_arr")
        ),
        sort_raw_peak_inputs_by_mz=bool(
            features_cfg.get("sort_raw_peak_inputs_by_mz", True)
        ),
        broadcast_spectrum_features_to_peaks=bool(
            features_cfg.get("broadcast_spectrum_features_to_peaks", True)
        ),
        normalize_peak_features=bool(features_cfg.get("normalize_peak_features", True)),
        normalize_spectrum_features=bool(
            features_cfg.get("normalize_spectrum_features", True)
        ),
        normalization_fit_split=str(
            features_cfg.get("normalization_fit_split", "train")
        ),
    )
    output = OutputConfig(
        output_dir=str(output_cfg.get("output_dir", "outputs/mlp_baseline")),
        save_best_model=bool(output_cfg.get("save_best_model", True)),
        save_epoch_history=bool(output_cfg.get("save_epoch_history", True)),
        save_metrics_summary=bool(output_cfg.get("save_metrics_summary", True)),
        save_plots=bool(output_cfg.get("save_plots", True)),
        save_confusion_matrices=bool(output_cfg.get("save_confusion_matrices", True)),
        device_verbose=bool(output_cfg.get("device_verbose", True)),
        enable_file_logging=bool(output_cfg.get("enable_file_logging", True)),
        log_file_name=str(output_cfg.get("log_file_name", "training.log")),
    )

    model = MLPConfig(
        peak_input_dim=(
            len(features.peak_feature_columns)
            + int(features.use_raw_peak_mz)
            + int(features.use_raw_peak_intensity)
        ),
        spectrum_input_dim=len(features.spectrum_feature_columns),
        hidden_dims=[int(x) for x in mlp_cfg.get("hidden_dims", [128, 64])],
        dropout=float(mlp_cfg.get("dropout", 0.1)),
        activation=str(mlp_cfg.get("activation", "gelu")),
        use_layer_norm=bool(mlp_cfg.get("use_layer_norm", True)),
        output_dim=int(mlp_cfg.get("output_dim", 1)),
        broadcast_spectrum_features_to_peaks=bool(
            features.broadcast_spectrum_features_to_peaks
        ),
    )

    training = TrainingLoopConfig(
        seed=int(training_cfg.get("seed", 42)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        max_epochs=int(training_cfg.get("max_epochs", 50)),
        cpu_num_threads=(
            None
            if training_cfg.get("cpu_num_threads", None) is None
            else int(training_cfg.get("cpu_num_threads"))
        ),
        cpu_num_interop_threads=(
            None
            if training_cfg.get("cpu_num_interop_threads", None) is None
            else int(training_cfg.get("cpu_num_interop_threads"))
        ),
        dataloader_prefetch_factor=int(training_cfg.get("dataloader_prefetch_factor", 4)),
        dataloader_persistent_workers=bool(
            training_cfg.get("dataloader_persistent_workers", True)
        ),
        dataloader_pin_memory=(
            None
            if training_cfg.get("dataloader_pin_memory", None) is None
            else bool(training_cfg.get("dataloader_pin_memory"))
        ),
        enable_amp=bool(training_cfg.get("enable_amp", True)),
        compile_model=bool(training_cfg.get("compile_model", True)),
        cache_dataset_in_memory=bool(
            training_cfg.get("cache_dataset_in_memory", True)
        ),
        save_latest_checkpoint=bool(
            training_cfg.get("save_latest_checkpoint", True)
        ),
        resume_from_checkpoint=(
            None
            if training_cfg.get("resume_from_checkpoint", None) is None
            else str(training_cfg.get("resume_from_checkpoint"))
        ),
        optimizer=OptimizerConfig(
            name=str(optimizer_cfg.get("name", "adamw")),
            learning_rate=float(optimizer_cfg.get("learning_rate", 1.0e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-4)),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler_cfg.get("enabled", False)),
        ),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
    )

    loss = LossConfig(
        name=str(loss_cfg.get("name", "bce_with_logits")),
        use_pos_weight=bool(loss_cfg.get("use_pos_weight", True)),
        pos_weight=None
        if loss_cfg.get("pos_weight", None) is None
        else float(loss_cfg.get("pos_weight")),
        reduction=str(loss_cfg.get("reduction", "none")),
    )

    early_stopping = EarlyStoppingConfig(
        enabled=bool(early_cfg.get("enabled", True)),
        monitor=str(early_cfg.get("monitor", "val_pr_auc")),
        mode=str(early_cfg.get("mode", "max")),
        patience=int(early_cfg.get("patience", 7)),
        min_delta=float(early_cfg.get("min_delta", 1.0e-4)),
    )

    evaluation = EvaluationConfig(
        primary_metric=str(eval_cfg.get("primary_metric", "pr_auc")),
        threshold_for_binary_metrics=float(
            eval_cfg.get("threshold_for_binary_metrics", 0.5)
        ),
        report_metrics=list(
            eval_cfg.get(
                "report_metrics",
                ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall"],
            )
        ),
        retained_peak_fractions=[
            float(x) for x in eval_cfg.get("retained_peak_fractions", [0.3, 0.5, 0.7])
        ],
    )

    data = DataConfig(
        train_path=str(data_cfg.get("train_path", "data/splits/train.parquet")),
        val_path=str(data_cfg.get("val_path", "data/splits/val.parquet")),
        test_path=str(data_cfg.get("test_path", "data/splits/test.parquet")),
        target_column=str(data_cfg.get("target_column", "annotation_mask")),
        use_training_weights=bool(data_cfg.get("use_training_weights", False)),
        weight_column=str(data_cfg.get("weight_column", "weight")),
    )

    return TrainingConfig(
        data=data,
        features=features,
        model=model,
        training=training,
        loss=loss,
        early_stopping=early_stopping,
        evaluation=evaluation,
        output=output,
    )


def load_transformer_training_config(config_path: str | Path) -> TrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_cfg = raw.get("data", {})
    output_cfg = raw.get("output", {})
    features_cfg = raw.get("features", {})
    model_cfg = raw.get("model", {})
    transformer_cfg = model_cfg.get("transformer", {})
    training_cfg = raw.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {})
    scheduler_cfg = training_cfg.get("scheduler", {})
    loss_cfg = raw.get("loss", {})
    early_cfg = raw.get("early_stopping", {})
    eval_cfg = raw.get("evaluation", {})

    features = FeatureSelectionConfig(
        peak_feature_columns=list(features_cfg.get("peak_feature_columns", [])),
        spectrum_feature_columns=list(features_cfg.get("spectrum_feature_columns", [])),
        use_raw_peak_mz=bool(features_cfg.get("use_raw_peak_mz", False)),
        raw_peak_mz_column=str(features_cfg.get("raw_peak_mz_column", "mz_arr")),
        use_raw_peak_intensity=bool(features_cfg.get("use_raw_peak_intensity", False)),
        raw_peak_intensity_column=str(
            features_cfg.get("raw_peak_intensity_column", "int_arr")
        ),
        sort_raw_peak_inputs_by_mz=bool(
            features_cfg.get("sort_raw_peak_inputs_by_mz", True)
        ),
        broadcast_spectrum_features_to_peaks=bool(
            features_cfg.get("broadcast_spectrum_features_to_peaks", True)
        ),
        normalize_peak_features=bool(features_cfg.get("normalize_peak_features", True)),
        normalize_spectrum_features=bool(
            features_cfg.get("normalize_spectrum_features", True)
        ),
        normalization_fit_split=str(
            features_cfg.get("normalization_fit_split", "train")
        ),
    )
    output = OutputConfig(
        output_dir=str(output_cfg.get("output_dir", "outputs/transformer_baseline")),
        save_best_model=bool(output_cfg.get("save_best_model", True)),
        save_epoch_history=bool(output_cfg.get("save_epoch_history", True)),
        save_metrics_summary=bool(output_cfg.get("save_metrics_summary", True)),
        save_plots=bool(output_cfg.get("save_plots", True)),
        save_confusion_matrices=bool(output_cfg.get("save_confusion_matrices", True)),
        device_verbose=bool(output_cfg.get("device_verbose", True)),
        enable_file_logging=bool(output_cfg.get("enable_file_logging", True)),
        log_file_name=str(output_cfg.get("log_file_name", "training.log")),
    )

    model = PeakTransformerConfig(
        peak_input_dim=(
            len(features.peak_feature_columns)
            + int(features.use_raw_peak_mz)
            + int(features.use_raw_peak_intensity)
        ),
        spectrum_input_dim=len(features.spectrum_feature_columns),
        d_model=int(transformer_cfg.get("d_model", 96)),
        num_heads=int(transformer_cfg.get("num_heads", 4)),
        num_layers=int(transformer_cfg.get("num_layers", 3)),
        ff_multiplier=float(transformer_cfg.get("ff_multiplier", 4.0)),
        dropout=float(transformer_cfg.get("dropout", 0.1)),
        activation=str(transformer_cfg.get("activation", "gelu")),
        use_layer_norm=bool(transformer_cfg.get("use_layer_norm", True)),
        output_dim=int(transformer_cfg.get("output_dim", 1)),
        use_spectrum_context_gating=bool(
            transformer_cfg.get("use_spectrum_context_gating", True)
        ),
        use_peak_positional_projection=bool(
            transformer_cfg.get("use_peak_positional_projection", True)
        ),
    )

    training = TrainingLoopConfig(
        seed=int(training_cfg.get("seed", 42)),
        batch_size=int(training_cfg.get("batch_size", 16)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        max_epochs=int(training_cfg.get("max_epochs", 50)),
        cpu_num_threads=(
            None
            if training_cfg.get("cpu_num_threads", None) is None
            else int(training_cfg.get("cpu_num_threads"))
        ),
        cpu_num_interop_threads=(
            None
            if training_cfg.get("cpu_num_interop_threads", None) is None
            else int(training_cfg.get("cpu_num_interop_threads"))
        ),
        dataloader_prefetch_factor=int(training_cfg.get("dataloader_prefetch_factor", 4)),
        dataloader_persistent_workers=bool(
            training_cfg.get("dataloader_persistent_workers", True)
        ),
        dataloader_pin_memory=(
            None
            if training_cfg.get("dataloader_pin_memory", None) is None
            else bool(training_cfg.get("dataloader_pin_memory"))
        ),
        enable_amp=bool(training_cfg.get("enable_amp", True)),
        compile_model=bool(training_cfg.get("compile_model", True)),
        cache_dataset_in_memory=bool(
            training_cfg.get("cache_dataset_in_memory", True)
        ),
        save_latest_checkpoint=bool(
            training_cfg.get("save_latest_checkpoint", True)
        ),
        resume_from_checkpoint=(
            None
            if training_cfg.get("resume_from_checkpoint", None) is None
            else str(training_cfg.get("resume_from_checkpoint"))
        ),
        optimizer=OptimizerConfig(
            name=str(optimizer_cfg.get("name", "adamw")),
            learning_rate=float(optimizer_cfg.get("learning_rate", 3.0e-4)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-4)),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler_cfg.get("enabled", False)),
        ),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
    )

    loss = LossConfig(
        name=str(loss_cfg.get("name", "bce_with_logits")),
        use_pos_weight=bool(loss_cfg.get("use_pos_weight", True)),
        pos_weight=None
        if loss_cfg.get("pos_weight", None) is None
        else float(loss_cfg.get("pos_weight")),
        reduction=str(loss_cfg.get("reduction", "none")),
    )

    early_stopping = EarlyStoppingConfig(
        enabled=bool(early_cfg.get("enabled", True)),
        monitor=str(early_cfg.get("monitor", "val_pr_auc")),
        mode=str(early_cfg.get("mode", "max")),
        patience=int(early_cfg.get("patience", 7)),
        min_delta=float(early_cfg.get("min_delta", 1.0e-4)),
    )

    evaluation = EvaluationConfig(
        primary_metric=str(eval_cfg.get("primary_metric", "pr_auc")),
        threshold_for_binary_metrics=float(
            eval_cfg.get("threshold_for_binary_metrics", 0.5)
        ),
        report_metrics=list(
            eval_cfg.get(
                "report_metrics",
                ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall"],
            )
        ),
        retained_peak_fractions=[
            float(x) for x in eval_cfg.get("retained_peak_fractions", [0.3, 0.5, 0.7])
        ],
    )

    data = DataConfig(
        train_path=str(data_cfg.get("train_path", "data/splits/train.parquet")),
        val_path=str(data_cfg.get("val_path", "data/splits/val.parquet")),
        test_path=str(data_cfg.get("test_path", "data/splits/test.parquet")),
        target_column=str(data_cfg.get("target_column", "annotation_mask")),
        use_training_weights=bool(data_cfg.get("use_training_weights", False)),
        weight_column=str(data_cfg.get("weight_column", "weight")),
    )

    return TrainingConfig(
        data=data,
        features=features,
        model=model,
        training=training,
        loss=loss,
        early_stopping=early_stopping,
        evaluation=evaluation,
        output=output,
    )


def load_transformer_imp_training_config(config_path: str | Path) -> TrainingConfig:
    from src.model.transformer_imp import PeakTransformerImpConfig

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_cfg = raw.get("data", {})
    output_cfg = raw.get("output", {})
    features_cfg = raw.get("features", {})
    model_cfg = raw.get("model", {})
    transformer_imp_cfg = model_cfg.get("transformer_imp", {})
    training_cfg = raw.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {})
    scheduler_cfg = training_cfg.get("scheduler", {})
    loss_cfg = raw.get("loss", {})
    early_cfg = raw.get("early_stopping", {})
    eval_cfg = raw.get("evaluation", {})

    features = FeatureSelectionConfig(
        peak_feature_columns=list(features_cfg.get("peak_feature_columns", [])),
        spectrum_feature_columns=list(features_cfg.get("spectrum_feature_columns", [])),
        use_raw_peak_mz=bool(features_cfg.get("use_raw_peak_mz", False)),
        raw_peak_mz_column=str(features_cfg.get("raw_peak_mz_column", "mz_arr")),
        use_raw_peak_intensity=bool(features_cfg.get("use_raw_peak_intensity", False)),
        raw_peak_intensity_column=str(
            features_cfg.get("raw_peak_intensity_column", "int_arr")
        ),
        sort_raw_peak_inputs_by_mz=bool(
            features_cfg.get("sort_raw_peak_inputs_by_mz", True)
        ),
        broadcast_spectrum_features_to_peaks=bool(
            features_cfg.get("broadcast_spectrum_features_to_peaks", False)
        ),
        normalize_peak_features=bool(features_cfg.get("normalize_peak_features", True)),
        normalize_spectrum_features=bool(
            features_cfg.get("normalize_spectrum_features", True)
        ),
        normalization_fit_split=str(
            features_cfg.get("normalization_fit_split", "train")
        ),
    )
    output = OutputConfig(
        output_dir=str(output_cfg.get("output_dir", "outputs/transformer_imp_baseline")),
        save_best_model=bool(output_cfg.get("save_best_model", True)),
        save_epoch_history=bool(output_cfg.get("save_epoch_history", True)),
        save_metrics_summary=bool(output_cfg.get("save_metrics_summary", True)),
        save_plots=bool(output_cfg.get("save_plots", True)),
        save_confusion_matrices=bool(output_cfg.get("save_confusion_matrices", True)),
        device_verbose=bool(output_cfg.get("device_verbose", True)),
        enable_file_logging=bool(output_cfg.get("enable_file_logging", True)),
        log_file_name=str(output_cfg.get("log_file_name", "training.log")),
    )

    model = PeakTransformerImpConfig(
        peak_input_dim=(
            len(features.peak_feature_columns)
            + int(features.use_raw_peak_mz)
            + int(features.use_raw_peak_intensity)
        ),
        spectrum_input_dim=len(features.spectrum_feature_columns),
        d_model=int(transformer_imp_cfg.get("d_model", 160)),
        num_heads=int(transformer_imp_cfg.get("num_heads", 8)),
        num_layers=int(transformer_imp_cfg.get("num_layers", 5)),
        ff_multiplier=float(transformer_imp_cfg.get("ff_multiplier", 4.0)),
        dropout=float(transformer_imp_cfg.get("dropout", 0.1)),
        activation=str(transformer_imp_cfg.get("activation", "gelu")),
        use_layer_norm=bool(transformer_imp_cfg.get("use_layer_norm", True)),
        output_dim=int(transformer_imp_cfg.get("output_dim", 1)),
        max_position_embeddings=int(
            transformer_imp_cfg.get("max_position_embeddings", 4096)
        ),
        local_attention_window=int(
            transformer_imp_cfg.get("local_attention_window", 32)
        ),
        use_global_spectrum_token=bool(
            transformer_imp_cfg.get("use_global_spectrum_token", True)
        ),
        use_learned_peak_rank_embedding=bool(
            transformer_imp_cfg.get("use_learned_peak_rank_embedding", True)
        ),
        use_spectrum_scale_shift=bool(
            transformer_imp_cfg.get("use_spectrum_scale_shift", True)
        ),
        raw_mz_feature_index=int(transformer_imp_cfg.get("raw_mz_feature_index", -1)),
        use_mz_relative_bias=bool(
            transformer_imp_cfg.get("use_mz_relative_bias", True)
        ),
        mz_relative_bias_scale=float(
            transformer_imp_cfg.get("mz_relative_bias_scale", 0.25)
        ),
    )

    training = TrainingLoopConfig(
        seed=int(training_cfg.get("seed", 42)),
        batch_size=int(training_cfg.get("batch_size", 8)),
        num_workers=int(training_cfg.get("num_workers", 0)),
        max_epochs=int(training_cfg.get("max_epochs", 60)),
        cpu_num_threads=(
            None
            if training_cfg.get("cpu_num_threads", None) is None
            else int(training_cfg.get("cpu_num_threads"))
        ),
        cpu_num_interop_threads=(
            None
            if training_cfg.get("cpu_num_interop_threads", None) is None
            else int(training_cfg.get("cpu_num_interop_threads"))
        ),
        dataloader_prefetch_factor=int(training_cfg.get("dataloader_prefetch_factor", 4)),
        dataloader_persistent_workers=bool(
            training_cfg.get("dataloader_persistent_workers", True)
        ),
        dataloader_pin_memory=(
            None
            if training_cfg.get("dataloader_pin_memory", None) is None
            else bool(training_cfg.get("dataloader_pin_memory"))
        ),
        enable_amp=bool(training_cfg.get("enable_amp", True)),
        compile_model=bool(training_cfg.get("compile_model", True)),
        cache_dataset_in_memory=bool(
            training_cfg.get("cache_dataset_in_memory", True)
        ),
        save_latest_checkpoint=bool(
            training_cfg.get("save_latest_checkpoint", True)
        ),
        resume_from_checkpoint=(
            None
            if training_cfg.get("resume_from_checkpoint", None) is None
            else str(training_cfg.get("resume_from_checkpoint"))
        ),
        optimizer=OptimizerConfig(
            name=str(optimizer_cfg.get("name", "adamw")),
            learning_rate=float(optimizer_cfg.get("learning_rate", 3.0e-4)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-4)),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(scheduler_cfg.get("enabled", True)),
        ),
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 1.0)),
    )

    loss = LossConfig(
        name=str(loss_cfg.get("name", "bce_with_logits")),
        use_pos_weight=bool(loss_cfg.get("use_pos_weight", True)),
        pos_weight=None
        if loss_cfg.get("pos_weight", None) is None
        else float(loss_cfg.get("pos_weight")),
        reduction=str(loss_cfg.get("reduction", "none")),
    )

    early_stopping = EarlyStoppingConfig(
        enabled=bool(early_cfg.get("enabled", True)),
        monitor=str(early_cfg.get("monitor", "val_pr_auc")),
        mode=str(early_cfg.get("mode", "max")),
        patience=int(early_cfg.get("patience", 10)),
        min_delta=float(early_cfg.get("min_delta", 1.0e-4)),
    )

    evaluation = EvaluationConfig(
        primary_metric=str(eval_cfg.get("primary_metric", "pr_auc")),
        threshold_for_binary_metrics=float(
            eval_cfg.get("threshold_for_binary_metrics", 0.4)
        ),
        report_metrics=list(
            eval_cfg.get(
                "report_metrics",
                ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall"],
            )
        ),
        retained_peak_fractions=[
            float(x) for x in eval_cfg.get("retained_peak_fractions", [0.3, 0.5, 0.7])
        ],
    )

    data = DataConfig(
        train_path=str(data_cfg.get("train_path", "data/splits/train.parquet")),
        val_path=str(data_cfg.get("val_path", "data/splits/val.parquet")),
        test_path=str(data_cfg.get("test_path", "data/splits/test.parquet")),
        target_column=str(data_cfg.get("target_column", "annotation_mask")),
        use_training_weights=bool(data_cfg.get("use_training_weights", False)),
        weight_column=str(data_cfg.get("weight_column", "weight")),
    )

    return TrainingConfig(
        data=data,
        features=features,
        model=model,
        training=training,
        loss=loss,
        early_stopping=early_stopping,
        evaluation=evaluation,
        output=output,
    )


def load_autogluon_training_config(
    config_path: str | Path,
) -> AutoGluonTrainingConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    data_cfg = raw.get("data", {})
    output_cfg = raw.get("output", {})
    features_cfg = raw.get("features", {})
    autogluon_cfg = raw.get("autogluon", {})
    eval_cfg = raw.get("evaluation", {})

    data = DataConfig(
        train_path=str(data_cfg.get("train_path", "data/splits/train.parquet")),
        val_path=str(data_cfg.get("val_path", "data/splits/val.parquet")),
        test_path=str(data_cfg.get("test_path", "data/splits/test.parquet")),
        target_column=str(data_cfg.get("target_column", "annotation_mask")),
        use_training_weights=bool(data_cfg.get("use_training_weights", False)),
        weight_column=str(data_cfg.get("weight_column", "weight")),
    )

    features = FeatureSelectionConfig(
        peak_feature_columns=list(features_cfg.get("peak_feature_columns", [])),
        spectrum_feature_columns=list(features_cfg.get("spectrum_feature_columns", [])),
        use_raw_peak_mz=bool(features_cfg.get("use_raw_peak_mz", False)),
        raw_peak_mz_column=str(features_cfg.get("raw_peak_mz_column", "mz_arr")),
        use_raw_peak_intensity=bool(features_cfg.get("use_raw_peak_intensity", False)),
        raw_peak_intensity_column=str(
            features_cfg.get("raw_peak_intensity_column", "int_arr")
        ),
        sort_raw_peak_inputs_by_mz=bool(
            features_cfg.get("sort_raw_peak_inputs_by_mz", True)
        ),
        broadcast_spectrum_features_to_peaks=bool(
            features_cfg.get("broadcast_spectrum_features_to_peaks", False)
        ),
        normalize_peak_features=bool(features_cfg.get("normalize_peak_features", True)),
        normalize_spectrum_features=bool(
            features_cfg.get("normalize_spectrum_features", True)
        ),
        normalization_fit_split=str(
            features_cfg.get("normalization_fit_split", "train")
        ),
    )

    evaluation = EvaluationConfig(
        primary_metric=str(eval_cfg.get("primary_metric", "pr_auc")),
        threshold_for_binary_metrics=float(
            eval_cfg.get("threshold_for_binary_metrics", 0.5)
        ),
        report_metrics=list(
            eval_cfg.get(
                "report_metrics",
                ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall"],
            )
        ),
        retained_peak_fractions=[
            float(x) for x in eval_cfg.get("retained_peak_fractions", [0.3, 0.5, 0.7])
        ],
    )

    autogluon = AutoGluonConfig(
        label=str(autogluon_cfg.get("label", data.target_column)),
        problem_type=str(autogluon_cfg.get("problem_type", "binary")),
        eval_metric=str(autogluon_cfg.get("eval_metric", "average_precision")),
        presets=str(autogluon_cfg.get("presets", "high_quality")),
        time_limit=(
            None
            if autogluon_cfg.get("time_limit", None) is None
            else int(autogluon_cfg.get("time_limit"))
        ),
        verbosity=int(autogluon_cfg.get("verbosity", 2)),
        positive_class=(
            None
            if autogluon_cfg.get("positive_class", None) is None
            else int(autogluon_cfg.get("positive_class"))
        ),
        sample_weight_column=(
            None
            if autogluon_cfg.get("sample_weight_column", None) is None
            else str(autogluon_cfg.get("sample_weight_column"))
        ),
        weight_evaluation=bool(autogluon_cfg.get("weight_evaluation", False)),
        save_leaderboard=bool(autogluon_cfg.get("save_leaderboard", True)),
        save_feature_importance=bool(
            autogluon_cfg.get("save_feature_importance", True)
        ),
        feature_importance_subsample_size=(
            None
            if autogluon_cfg.get("feature_importance_subsample_size", None) is None
            else int(autogluon_cfg.get("feature_importance_subsample_size"))
        ),
        feature_importance_num_shuffle_sets=(
            None
            if autogluon_cfg.get("feature_importance_num_shuffle_sets", None) is None
            else int(autogluon_cfg.get("feature_importance_num_shuffle_sets"))
        ),
        feature_importance_time_limit=(
            None
            if autogluon_cfg.get("feature_importance_time_limit", None) is None
            else int(autogluon_cfg.get("feature_importance_time_limit"))
        ),
        dynamic_stacking=bool(autogluon_cfg.get("dynamic_stacking", False)),
        num_stack_levels=int(autogluon_cfg.get("num_stack_levels", 1)),
        use_bag_holdout=bool(autogluon_cfg.get("use_bag_holdout", True)),
        fit_weighted_ensemble=bool(autogluon_cfg.get("fit_weighted_ensemble", True)),
        save_bag_folds=bool(autogluon_cfg.get("save_bag_folds", True)),
    )

    output = AutoGluonOutputConfig(
        output_dir=str(output_cfg.get("output_dir", "outputs/autogluon_baseline")),
        predictor_subdir=(
            None
            if output_cfg.get("predictor_subdir", None) is None
            else str(output_cfg.get("predictor_subdir"))
        ),
        unique_predictor_subdir=bool(
            output_cfg.get("unique_predictor_subdir", True)
        ),
        save_metrics_summary=bool(output_cfg.get("save_metrics_summary", True)),
        save_confusion_matrices=bool(output_cfg.get("save_confusion_matrices", True)),
        save_predictions=bool(output_cfg.get("save_predictions", True)),
        save_flattened_tables=bool(output_cfg.get("save_flattened_tables", False)),
        save_fit_summary=bool(output_cfg.get("save_fit_summary", True)),
        enable_file_logging=bool(output_cfg.get("enable_file_logging", True)),
        log_file_name=str(output_cfg.get("log_file_name", "training.log")),
    )

    return AutoGluonTrainingConfig(
        data=data,
        features=features,
        evaluation=evaluation,
        autogluon=autogluon,
        output=output,
    )
