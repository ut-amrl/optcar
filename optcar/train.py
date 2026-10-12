"""WheeledLab-compatible FKD training entry point.

Model, learning, and dataset settings live in separate Python config factories.
The previous OptCar-specific training recipe has been replaced by the copied
WheeledLab backend. See configs/corl26_model.py and configs/corl26_learning.py.
"""
from optcar.fkd_train import (
    add_training_arguments,
    get_dataloader,
    load_config,
    main,
    run_training,
    split_dataset,
)

train = run_training

__all__ = [
    "main", "train", "run_training", "add_training_arguments",
    "load_config", "split_dataset", "get_dataloader",
]

if __name__ == "__main__":
    main()
