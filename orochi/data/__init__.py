"""Data loading and preprocessing for biomedical images.

Provides:
- Dataset classes for various tasks
- Data transforms and augmentations
- DataLoader utilities
- Preprocessing functions

Example:
    >>> from orochi.data import BiomedicalDataset
    >>> from orochi.data.transforms import RandomFlip, RandomRotate
    >>>
    >>> dataset = BiomedicalDataset(
    ...     data_path="./data/biosr",
    ...     transform=[RandomFlip(), RandomRotate()]
    ... )
"""

__all__ = []
