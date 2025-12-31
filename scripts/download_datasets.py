#!/usr/bin/env python3
"""
Download biomedical imaging datasets from Hugging Face.

This script automatically downloads the required datasets to ./data if they don't exist:
- hiPSC 2D: eternalaudrey/hipsc_2d
- hiPSC 3D: eternalaudrey/hipsc_3d
- HiP-CT 2D: eternalaudrey/hipct_2d
- IDR 2D: eternalaudrey/idr_2d
- IDR Raw: eternalaudrey/idr-01, idr-02, idr-03, idr-04

Usage:
    python scripts/download_datasets.py [--dataset DATASET_NAME]

    If --dataset is not specified, all datasets will be downloaded.
"""

import os
import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub is not installed.")
    print("Please install it with: pip install huggingface_hub")
    exit(1)


# Dataset configurations
DATASETS = {
    "hipsc_2d": {
        "repo_id": "eternalaudrey/hipsc_2d",
        "local_dir": "hipsc_2d",
        "description": "hiPSC 2D dataset"
    },
    "hipsc_3d": {
        "repo_id": "eternalaudrey/hipsc_3d",
        "local_dir": "hipsc_3d",
        "description": "hiPSC 3D dataset"
    },
    "hipct_2d": {
        "repo_id": "eternalaudrey/hipct_2d",
        "local_dir": "hipct_2d",
        "description": "HiP-CT 2D dataset"
    },
    "idr_2d": {
        "repo_id": "eternalaudrey/idr_2d",
        "local_dir": "idr_2d",
        "description": "IDR 2D dataset"
    },
    "idr_01": {
        "repo_id": "eternalaudrey/idr-01",
        "local_dir": "idr_raw/idr-01",
        "description": "IDR Raw dataset 01"
    },
    "idr_02": {
        "repo_id": "eternalaudrey/idr-02",
        "local_dir": "idr_raw/idr-02",
        "description": "IDR Raw dataset 02"
    },
    "idr_03": {
        "repo_id": "eternalaudrey/idr-03",
        "local_dir": "idr_raw/idr-03",
        "description": "IDR Raw dataset 03"
    },
    "idr_04": {
        "repo_id": "eternalaudrey/idr-04",
        "local_dir": "idr_raw/idr-04",
        "description": "IDR Raw dataset 04"
    },
}


def download_dataset(dataset_name, data_root="./data", force=False):
    """Download a single dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset (key from DATASETS dict)
        data_root: Root directory for data storage (default: ./data)
        force: If True, re-download even if dataset exists

    Returns:
        Path to the downloaded dataset
    """
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}")

    dataset_info = DATASETS[dataset_name]
    repo_id = dataset_info["repo_id"]
    local_dir = Path(data_root) / dataset_info["local_dir"]
    description = dataset_info["description"]

    # Check if dataset already exists
    if local_dir.exists() and not force:
        print(f"✓ {description} already exists at {local_dir}")
        return local_dir

    print(f"📥 Downloading {description} from {repo_id}...")
    print(f"   Target directory: {local_dir}")

    try:
        # Create parent directory if needed
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        # Download the dataset
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False
        )

        print(f"✓ Successfully downloaded {description}")
        return local_dir

    except Exception as e:
        print(f"✗ Error downloading {description}: {e}")
        raise


def download_all_datasets(data_root="./data", force=False):
    """Download all configured datasets.

    Args:
        data_root: Root directory for data storage (default: ./data)
        force: If True, re-download even if datasets exist
    """
    print(f"Starting dataset downloads to {data_root}")
    print(f"Total datasets: {len(DATASETS)}\n")

    # Create data root directory
    Path(data_root).mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_datasets = []

    for dataset_name in DATASETS.keys():
        try:
            download_dataset(dataset_name, data_root, force)
            success_count += 1
        except Exception as e:
            failed_datasets.append((dataset_name, str(e)))
        print()  # Add blank line between datasets

    # Summary
    print("=" * 60)
    print(f"Download Summary:")
    print(f"  Successful: {success_count}/{len(DATASETS)}")

    if failed_datasets:
        print(f"  Failed: {len(failed_datasets)}")
        print("\nFailed datasets:")
        for name, error in failed_datasets:
            print(f"  - {name}: {error}")
    else:
        print("  All datasets downloaded successfully!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download biomedical imaging datasets from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available datasets:
  hipsc_2d   - hiPSC 2D dataset
  hipsc_3d   - hiPSC 3D dataset
  hipct_2d   - HiP-CT 2D dataset
  idr_2d     - IDR 2D dataset
  idr_01     - IDR Raw dataset 01
  idr_02     - IDR Raw dataset 02
  idr_03     - IDR Raw dataset 03
  idr_04     - IDR Raw dataset 04

Examples:
  # Download all datasets
  python scripts/download_datasets.py

  # Download specific dataset
  python scripts/download_datasets.py --dataset hipsc_2d

  # Force re-download all datasets
  python scripts/download_datasets.py --force
        """
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        help="Download specific dataset (if not specified, downloads all)"
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
        help="Root directory for data storage (default: ./data)"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if dataset exists"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )

    args = parser.parse_args()

    # List datasets and exit
    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:12} - {info['description']} ({info['repo_id']})")
        return

    # Download specific dataset or all
    if args.dataset:
        download_dataset(args.dataset, args.data_root, args.force)
    else:
        download_all_datasets(args.data_root, args.force)


if __name__ == "__main__":
    main()
