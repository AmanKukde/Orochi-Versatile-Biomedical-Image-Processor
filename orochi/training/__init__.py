"""Training utilities and trainers for Orochi models.

Provides:
- Base Trainer class
- Training callbacks (early stopping, checkpointing, logging)
- Evaluation metrics
- Training utilities

Example:
    >>> from orochi.training import Trainer
    >>> from orochi.training.callbacks import EarlyStopping, ModelCheckpoint
    >>>
    >>> trainer = Trainer(
    ...     model=model,
    ...     config=config,
    ...     callbacks=[EarlyStopping(), ModelCheckpoint()]
    ... )
    >>> trainer.fit(train_loader, val_loader)
"""

__all__ = []
