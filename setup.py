"""Setup configuration for Orochi package.

Installation:
    pip install -e .                    # Development mode
    pip install -e .[dev]              # With dev dependencies
    pip install -e .[all]              # With all dependencies
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Orochi: Versatile Biomedical Image Processor"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    req_file = Path(__file__).parent / filename
    if req_file.exists():
        with open(req_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Core requirements
install_requires = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-image>=0.20.0",
    "pillow>=9.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "einops>=0.6.0",
    "monai>=1.0.0",
    "nibabel>=5.0.0",
    "simpleitk>=2.2.0",
]

# Development dependencies
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

# Documentation dependencies
docs_requires = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "sphinx-autodoc-typehints>=1.23.0",
]

# Optional dependencies
optional_requires = [
    "wandb>=0.15.0",  # Experiment tracking
    "tensorboard>=2.13.0",  # TensorBoard logging
    "mamba-ssm==2.2.2",  # Mamba state-space models (optional, GPU required)
    "causal-conv1d==1.4.0",  # Required for mamba-ssm
]

setup(
    name="orochi",
    version="1.0.0",
    author="Orochi Team",
    author_email="",
    description="Versatile Biomedical Image Processor based on Mamba architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jnjnnjzch/foundation-mamba-biomed",
    project_urls={
        "Bug Tracker": "https://github.com/jnjnnjzch/foundation-mamba-biomed/issues",
        "Documentation": "https://github.com/jnjnnjzch/foundation-mamba-biomed",
        "Source Code": "https://github.com/jnjnnjzch/foundation-mamba-biomed",
    },
    packages=find_packages(exclude=["tests", "tests.*", "legacy", "legacy.*", "temp", "temp.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": docs_requires,
        "optional": optional_requires,
        "all": dev_requires + docs_requires + optional_requires,
    },
    entry_points={
        "console_scripts": [
            # Add CLI tools here if needed
            # "orochi-train=orochi.cli.train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "deep learning",
        "medical imaging",
        "biomedical image processing",
        "mamba",
        "state space models",
        "super resolution",
        "image registration",
        "image segmentation",
        "pytorch",
    ],
)
