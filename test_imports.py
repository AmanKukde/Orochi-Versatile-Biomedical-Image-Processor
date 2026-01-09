#!/usr/bin/env python3
"""
Comprehensive import test for Orochi package.

This script tests all imports to ensure the package structure is correct.
"""

import sys
from typing import List, Tuple

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def test_import(module_path: str, items: List[str] = None) -> Tuple[bool, str]:
    """
    Test importing a module or specific items from a module.

    Args:
        module_path: Module path to import
        items: Optional list of items to import from the module

    Returns:
        Tuple of (success, error_message)
    """
    try:
        if items:
            # Test importing specific items
            for item in items:
                exec(f"from {module_path} import {item}")
        else:
            # Test importing entire module
            exec(f"import {module_path}")
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    """Run comprehensive import tests."""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Orochi Package Import Test{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

    # Track results
    passed = 0
    failed = 0
    errors = []

    # Test cases: (description, module_path, items_to_import)
    test_cases = [
        # Core package
        ("Core package", "orochi", None),
        ("Package version", "orochi", ["__version__"]),
        ("MambaModel alias", "orochi", ["MambaModel"]),
        ("MambaEncoderHeria", "orochi", ["MambaEncoderHeria"]),
        ("get_dataset factory", "orochi", ["get_dataset"]),
        ("get_loss_function factory", "orochi", ["get_loss_function"]),
        ("get_metric factory", "orochi", ["get_metric"]),

        # Models
        ("Models module", "orochi.models", None),
        ("MambaEncoderHeria from models", "orochi.models", ["MambaEncoderHeria"]),
        ("PatchEmbed", "orochi.models", ["PatchEmbed"]),
        ("PatchMerging", "orochi.models", ["PatchMerging"]),
        ("reg_decoder", "orochi.models", ["reg_decoder"]),
        ("SR_decoder", "orochi.models", ["SR_decoder"]),
        ("fus_decoder", "orochi.models", ["fus_decoder"]),
        ("IR_decoder", "orochi.models", ["IR_decoder"]),

        # Losses
        ("Losses module", "orochi.losses", None),
        ("get_loss_function", "orochi.losses", ["get_loss_function"]),
        ("SSIM2D", "orochi.losses", ["SSIM2D"]),
        ("SSIM3D", "orochi.losses", ["SSIM3D"]),
        ("Grad", "orochi.losses", ["Grad"]),
        ("Grad3d", "orochi.losses", ["Grad3d"]),
        ("NCC_vxm", "orochi.losses", ["NCC_vxm"]),
        ("DiceLoss", "orochi.losses", ["DiceLoss"]),
        ("DisplacementRegularizer", "orochi.losses", ["DisplacementRegularizer"]),
        ("MutualInformation", "orochi.losses", ["MutualInformation"]),

        # Data
        ("Data module", "orochi.data", None),
        ("get_dataset", "orochi.data", ["get_dataset"]),
        ("PretrainDataset", "orochi.data", ["PretrainDataset"]),
        ("IXIBrainDataset", "orochi.data", ["IXIBrainDataset"]),
        ("IXIBrainInferDataset", "orochi.data", ["IXIBrainInferDataset"]),
        ("CollateFn", "orochi.data", ["CollateFn"]),
        ("random_crop", "orochi.data", ["random_crop"]),
        ("get_dataloader", "orochi.data", ["get_dataloader"]),

        # Data utilities
        ("Data utils module", "orochi.data.data_utils", None),
        ("pkload", "orochi.data.data_utils", ["pkload"]),
        ("init_fn", "orochi.data.data_utils", ["init_fn"]),
        ("get_all_coords", "orochi.data.data_utils", ["get_all_coords"]),
        ("gen_feats", "orochi.data.data_utils", ["gen_feats"]),
        ("normalize_intensity", "orochi.data.data_utils", ["normalize_intensity"]),

        # Metrics
        ("Metrics module", "orochi.metrics", None),
        ("get_metric", "orochi.metrics", ["get_metric"]),

        # Utils
        ("Utils module", "orochi.utils", None),
        ("Utils helpers", "orochi.utils.helpers", None),

        # Training
        ("Training module", "orochi.training", None),

        # Inference
        ("Inference module", "orochi.inference", None),

        # CLI
        ("CLI module", "orochi.cli", None),
    ]

    # Run tests
    for description, module_path, items in test_cases:
        success, error = test_import(module_path, items)

        if success:
            print(f"{GREEN}✓{RESET} {description:<50} {GREEN}PASS{RESET}")
            passed += 1
        else:
            print(f"{RED}✗{RESET} {description:<50} {RED}FAIL{RESET}")
            print(f"  {YELLOW}Error: {error}{RESET}")
            failed += 1
            errors.append((description, error))

    # Summary
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}Summary{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    print(f"Total tests: {passed + failed}")
    print(f"{GREEN}Passed: {passed}{RESET}")
    print(f"{RED}Failed: {failed}{RESET}")

    if failed > 0:
        print(f"\n{RED}{'='*70}{RESET}")
        print(f"{RED}Failed Tests Details:{RESET}")
        print(f"{RED}{'='*70}{RESET}")
        for desc, error in errors:
            print(f"\n{RED}✗ {desc}{RESET}")
            print(f"  {error}")

    print(f"\n{BLUE}{'='*70}{RESET}\n")

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
