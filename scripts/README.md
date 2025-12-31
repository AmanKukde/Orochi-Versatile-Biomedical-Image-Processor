# Scripts

Utility scripts for the Orochi Versatile Biomedical Image Processor.

## Dataset Download

### download_datasets.py

Downloads biomedical imaging datasets from Hugging Face Hub.

**Datasets:**
- `hipsc_2d` - hiPSC 2D dataset (eternalaudrey/hipsc_2d)
- `hipsc_3d` - hiPSC 3D dataset (eternalaudrey/hipsc_3d)
- `hipct_2d` - HiP-CT 2D dataset (eternalaudrey/hipct_2d)
- `idr_2d` - IDR 2D dataset (eternalaudrey/idr_2d)
- `idr_01-04` - IDR Raw datasets (eternalaudrey/idr-01 through idr-04)

**Prerequisites:**
```bash
pip install huggingface_hub
```

**Usage:**

Download all datasets to `./data`:
```bash
python scripts/download_datasets.py
```

Download specific dataset:
```bash
python scripts/download_datasets.py --dataset hipsc_2d
```

List available datasets:
```bash
python scripts/download_datasets.py --list
```

Force re-download:
```bash
python scripts/download_datasets.py --force
```

Custom data directory:
```bash
python scripts/download_datasets.py --data-root /path/to/data
```

**Note:** The `./data` directory is excluded from git via `.gitignore` to prevent committing large dataset files.
