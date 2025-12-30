"""Constants for loss functions.

This module defines all magic numbers used in loss calculations,
providing clear documentation for each constant's purpose.
"""

# SSIM Constants
SSIM_WINDOW_SIZE = 11  # Standard Gaussian window size for SSIM
SSIM_C1 = (0.01) ** 2  # Stability constant for luminance comparison
SSIM_C2 = (0.03) ** 2  # Stability constant for contrast comparison
SSIM_SIGMA = 1.5  # Standard deviation for Gaussian window

# Gradient Loss Constants
GRADIENT_PENALTY = "l2"  # Default gradient penalty type
GRADIENT_LOSS_TYPE = "l2"  # Default type for gradient computation

# Dice Loss Constants
DICE_SMOOTH = 1e-5  # Smoothing constant to avoid division by zero

# Mutual Information Constants
MI_NUM_BINS = 32  # Number of bins for histogram computation
MI_SIGMA_RATIO = 0.5  # Sigma ratio for Parzen window

# Surface Distance Constants
SURFACE_MAX_DIST = 20.0  # Maximum distance for surface computations (mm)

# Registration Loss Constants
NCC_WINDOW_SIZE = 9  # Window size for local NCC computation

# Default device
DEFAULT_DEVICE = "cuda"  # Default device for loss computations
