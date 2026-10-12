"""WheeledLab FKD model and training definitions with OptCar-local imports."""
from .config import (
    FKD_model_config,
    TransformerModelConfig,
    TransformerModelConfig_CTX,
    dtype_to_trainer_precision,
    resolve_dtype,
)
from .learning_config import SupervisedLearningConfig

__all__ = [
    "FKD_model_config", "TransformerModelConfig", "TransformerModelConfig_CTX",
    "SupervisedLearningConfig", "TransformerModel_decoderonly", "TransformerModel_decoderonly_CTX",
    "dtype_to_trainer_precision", "resolve_dtype",
]


def __getattr__(name):
    if name in ("TransformerModel_decoderonly", "TransformerModel_decoderonly_CTX"):
        from . import transformer
        return getattr(transformer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
