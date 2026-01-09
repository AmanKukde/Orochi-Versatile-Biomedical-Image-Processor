# Contributing to Orochi

Thank you for your interest in contributing to the Orochi Biomedical Image Processor! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

## Code of Conduct

This project follows a code of professional conduct. Be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.10+
- CUDA 12.1+ (for GPU support)
- Git

### Forking the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:
```bash
git clone https://github.com/YOUR_USERNAME/Orochi-Versatile-Biomedical-Image-Processor.git
cd Orochi-Versatile-Biomedical-Image-Processor
```

## Development Setup

### 1. Create Environment

Using conda (recommended):
```bash
conda env create -f temp/environment.yaml
conda activate mamba_biomed
```

Or using pip:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Install Pre-commit Hooks

```bash
pre-commit install
```

This will automatically run code formatting and linting before each commit.

### 3. Verify Installation

```bash
python -c "import orochi; print(orochi.__version__)"
pytest tests/ -v
```

## Code Style

We follow PEP 8 with some modifications:

### Formatting Tools

- **Black**: Code formatter (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Running Formatters

```bash
# Format code
black orochi tests

# Sort imports
isort orochi tests

# Lint
flake8 orochi tests

# Type check
mypy orochi --ignore-missing-imports
```

### Style Guidelines

1. **Line Length**: Maximum 100 characters
2. **Docstrings**: Use Google style docstrings
3. **Type Hints**: Add type hints to all new functions
4. **Comments**: Add inline comments for complex logic

Example function with proper style:

```python
def process_image(
    img: torch.Tensor,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Process an image with optional normalization and resizing.

    Args:
        img: Input image tensor [C, H, W]
        normalize: Whether to normalize to [0, 1]
        target_size: Optional target size (H, W)

    Returns:
        Processed image tensor

    Example:
        >>> img = torch.randn(1, 256, 256)
        >>> processed = process_image(img, normalize=True, target_size=(128, 128))
        >>> print(processed.shape)  # torch.Size([1, 128, 128])
    """
    # Normalize if requested
    if normalize:
        img = (img - img.min()) / (img.max() - img.min())

    # Resize if target size specified
    if target_size is not None:
        img = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear').squeeze(0)

    return img
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=orochi --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v

# Run specific test
pytest tests/unit/test_models.py::TestMambaEncoder::test_encoder_2d_forward -v
```

### Writing Tests

1. Create test files in `tests/unit/` or `tests/integration/`
2. Name test files `test_*.py`
3. Name test classes `Test*`
4. Name test functions `test_*`
5. Use fixtures from `tests/conftest.py`

Example test:

```python
def test_model_forward(sample_2d_image, sample_config_2d):
    """Test model forward pass."""
    model = MambaEncoderHeria(sample_config_2d)
    model.eval()

    with torch.no_grad():
        features = model(sample_2d_image)

    assert isinstance(features, list)
    assert len(features) > 0
```

### Test Coverage

- Aim for >80% code coverage
- All new features must include tests
- Bug fixes should include regression tests

## Submitting Changes

### Workflow

1. Create a new branch for your changes:
```bash
git checkout -b feature/my-new-feature
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: add new feature"
```

3. Push to your fork:
```bash
git push origin feature/my-new-feature
```

4. Open a Pull Request on GitHub

### Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add 3D segmentation decoder
fix: resolve NaN in SSIM loss computation
docs: update installation instructions
test: add unit tests for gradient losses
```

### Pull Request Guidelines

1. **Title**: Use conventional commit format
2. **Description**: Explain what and why, not how
3. **Tests**: Include tests for new features
4. **Documentation**: Update docs if needed
5. **Changes**: Keep PRs focused and reasonably sized

### Code Review Process

1. Automated checks must pass (linting, tests, etc.)
2. At least one maintainer review required
3. Address review feedback
4. Maintainer will merge when approved

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Short description of what the function does.

    Longer description providing more details about the function's
    behavior, algorithms used, or important notes.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative
        TypeError: When param2 is not a string

    Example:
        >>> result = my_function(42, "hello")
        >>> print(result)  # True

    Note:
        Important implementation details or warnings.
    """
    pass
```

### Updating Documentation

1. Update docstrings for code changes
2. Update relevant markdown files in `docs/`
3. Add examples for new features
4. Update CHANGELOG.md

## Project Structure

```
Orochi-Versatile-Biomedical-Image-Processor/
├── orochi/                 # Main package
│   ├── models/            # Model architectures
│   ├── losses/            # Loss functions
│   ├── data/              # Datasets and data utilities
│   ├── metrics/           # Evaluation metrics
│   ├── training/          # Training utilities
│   ├── inference/         # Inference utilities
│   ├── utils/             # General utilities
│   └── cli/               # Command-line interface
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   └── integration/      # Integration tests
├── configs/              # Configuration files
├── scripts/              # Utility scripts
├── docs/                 # Documentation
└── experiments/          # Experiment-specific code
```

## Questions?

If you have questions:

1. Check existing issues and discussions
2. Review the documentation
3. Open a new issue with the "question" label

Thank you for contributing to Orochi!
