"""Utility functions for biomedical image processing and registration.

This module provides various utility functions for:
- Visualization of logits and registration results
- Spatial transformations and image warping
- Metric computation (Dice score, Jacobian determinant)
- Uncertainty estimation and calibration
- General helper functions for training

Classes:
    AverageMeter: Tracks and computes running averages
    SpatialTransformer: N-D spatial transformer for image warping
    register_model: Simple registration model wrapper

Functions:
    visualize_logits: Visualize model outputs with overlays
    check_nan, check_grad_nan: NaN detection utilities
    flatten_loss_dict: Flatten nested loss dictionaries
    pad_image: Pad images to target size
    dice_val, dice_val_VOI, dice_val_substruct, dice: Dice score computation
    jacobian_determinant_vxm: Compute Jacobian determinant of displacement field
    smooth_seg: Apply Gaussian smoothing to segmentations
    get_mc_preds: Monte Carlo predictions for uncertainty
    uncert_regression_gal: Uncertainty quantification
    uceloss: Uncertainty Calibration Error loss
"""

# Standard library imports
import math
import os
import re

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pystrum.pynd.ndutils as nd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch import nn


def visualize_logits(
    logits, output_folder=None, slice_idx=None, figsize=(20, 20), verbose=True
):
    """Visualize model logits including registration, fusion, and SR results.

    Creates comprehensive visualizations of model outputs including raw images,
    processed results, flow fields, and difference overlays.

    Args:
        logits: Dictionary containing model outputs with structure:
            - 'raw': Raw input image
            - Task-specific keys ('reg', 'fus', 'SR', 'IR') with nested dicts
        output_folder: Directory to save visualizations. If None, shows plots. Default: None
        slice_idx: Index of slice to visualize. If None, uses middle slice. Default: None
        figsize: Figure size for matplotlib. Default: (20, 20)
        verbose: If True, print logits structure. Default: True
    """

    def print_dict_structure(d, indent=0):
        """Recursively print nested dictionary structure."""
        for key, value in d.items():
            print("  " * indent + str(key))
            if isinstance(value, dict):
                print_dict_structure(value, indent + 1)
            else:
                print(
                    "  " * (indent + 1)
                    + f"{type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}"
                )

    if verbose:
        print("Logits structure:")
        print_dict_structure(logits)
        print("\n")

    def get_slice(img, idx):
        """Extract a 2D slice from multi-dimensional image."""
        if img.ndim == 5:  # (B, C, D, H, W)
            if img.shape[1] == 3:  # Flow data
                return img[0, :, idx].transpose(1, 2, 0)
            else:
                return img[0, 0, idx]
        elif img.ndim == 4:  # (B, D, H, W)
            return img[0, idx]
        elif img.ndim == 3:  # (D, H, W)
            return img[idx]
        return img

    def visualize_flow(flow, ax):
        """Visualize flow field as magnitude or RGB."""
        if flow.ndim == 2:  # (H, W)
            magnitude = np.linalg.norm(flow, axis=-1)
            ax.imshow(magnitude, cmap="viridis")
            ax.set_title("Flow Magnitude")
        elif flow.ndim == 3 and flow.shape[-1] == 3:  # (H, W, 3)
            flow_norm = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
            flow_rgb = (flow_norm + 1) / 2
            ax.imshow(flow_rgb)
            ax.set_title("Flow (RGB)")
        else:
            print(f"Unexpected flow shape: {flow.shape}")

    def create_misalignment_overlay(img1, img2, enhance_diff=True):
        """Create RGB overlay highlighting misalignment between two images.

        Args:
            img1: First image
            img2: Second image
            enhance_diff: If True, enhance differences. Default: True

        Returns:
            RGB overlay where red=img1, blue=img2, green=difference
        """
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())

        diff = np.abs(img1_norm - img2_norm)

        if enhance_diff:
            diff = np.power(diff, 0.5)
            threshold = 0.1
            diff[diff < threshold] = 0

        overlay = np.zeros((img1.shape[0], img1.shape[1], 3))
        overlay[:, :, 0] = img1_norm
        overlay[:, :, 2] = img2_norm
        overlay[:, :, 1] = diff  # Green channel shows misalignment

        return overlay

    def create_complementary_image(raw_img, mask_a, mask_b):
        """Create complementary image by combining two masked regions."""
        complementary = np.where(
            mask_a > 0, raw_img, np.where(mask_b > 0, raw_img, 0)
        )
        return complementary

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    if slice_idx is None:
        slice_idx = logits["raw"].shape[2] // 2  # Middle slice of depth dimension

    raw_slice = get_slice(logits["raw"], slice_idx)

    for key, value in logits.items():
        if isinstance(value, dict):
            n_images = 1  # Start with 1 for the raw image
            for sub_key in value.keys():
                if sub_key not in ["flow", "inverted_flow"]:
                    n_images += 1
                else:
                    n_images += 1  # Count flow images

            # Add overlay images
            if "registered" in value:
                n_images += 1
            if "deformed" in value:
                n_images += 1
            if "masked_A" in value and "masked_B" in value:
                n_images += 1  # Add one for complementary image

            # Calculate optimal grid layout
            n_cols = int(np.ceil(np.sqrt(n_images)))
            n_rows = int(np.ceil(n_images / n_cols))

            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            fig.suptitle(f"Visualization of {key}", fontsize=16)

            # Plot raw image first
            axes[0].imshow(raw_slice, cmap="gray")
            axes[0].set_title("Raw")
            axes[0].axis("off")

            ax_index = 1
            for sub_key, img in value.items():
                if ax_index < len(axes):
                    ax = axes[ax_index]
                    try:
                        if sub_key in ["deformed_grid", "restored_grid"]:
                            grid_slice = get_slice(img, slice_idx)
                            ax.imshow(grid_slice, cmap="gray", vmin=0, vmax=1)
                            ax.set_title(f"{sub_key} (Grid Only)")
                        elif sub_key in ["flow", "inverted_flow"]:
                            visualize_flow(get_slice(img, slice_idx), ax)
                        else:
                            ax.imshow(get_slice(img, slice_idx), cmap="gray")

                        ax.set_title(sub_key)
                        ax.axis("off")
                        ax_index += 1
                    except Exception as e:
                        print(f"Error visualizing {sub_key}: {e}")

            # Add overlay of raw and registered images
            if "registered" in value and ax_index < len(axes):
                ax = axes[ax_index]
                raw_img = get_slice(logits["raw"], slice_idx)
                reg_img = get_slice(value["registered"], slice_idx)
                overlay = create_misalignment_overlay(raw_img, reg_img, enhance_diff=True)
                ax.imshow(overlay)
                ax.set_title("Misalignment (Raw: Red, Registered: Blue)\nGreen: Difference")
                ax.axis("off")
                ax_index += 1

            # Add overlay of raw and deformed images
            if "deformed" in value and ax_index < len(axes):
                ax = axes[ax_index]
                raw_img = get_slice(logits["raw"], slice_idx)
                deformed_img = get_slice(value["deformed"], slice_idx)
                overlay = create_misalignment_overlay(
                    raw_img, deformed_img, enhance_diff=True
                )
                ax.imshow(overlay)
                ax.set_title("Misalignment (Raw: Red, Deformed: Blue)\nGreen: Difference")
                ax.axis("off")
                ax_index += 1

            # Add complementary image of masked_A and masked_B
            if "masked_A" in value and "masked_B" in value and ax_index < len(axes):
                ax = axes[ax_index]
                raw_slice = get_slice(logits["raw"], slice_idx)
                mask_a = get_slice(value["masked_A"], slice_idx)
                mask_b = get_slice(value["masked_B"], slice_idx)
                complementary_img = create_complementary_image(raw_slice, mask_a, mask_b)
                ax.imshow(complementary_img, cmap="gray")
                ax.set_title("Complementary Image\n(A and B combined)")
                ax.axis("off")
                ax_index += 1

            # Remove extra subplots
            for i in range(ax_index, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()

            if output_folder:
                output_path = os.path.join(output_folder, f"{key}.png")
                plt.savefig(output_path)
                plt.close()
            else:
                plt.show()


def check_nan(tensor, name):
    """Check if a tensor contains NaN values.

    Args:
        tensor: Tensor to check
        name: Name for logging

    Returns:
        True if NaN detected, False otherwise
    """
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False


def check_grad_nan(model):
    """Check if any gradients in model contain NaN values.

    Args:
        model: PyTorch model to check

    Returns:
        True if NaN detected in any gradient, False otherwise
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                return True
    return False


def flatten_loss_dict(loss_dict, parent_key="", sep="_"):
    """Flatten a nested loss dictionary into a single-level dict.

    Args:
        loss_dict: Nested dictionary of losses
        parent_key: Parent key for recursion. Default: ""
        sep: Separator for concatenating keys. Default: "_"

    Returns:
        Flattened dictionary

    Example:
        >>> nested = {'mse': {'reg': 0.1, 'fus': 0.2}, 'ncc': {'reg': 0.3}}
        >>> flatten_loss_dict(nested)
        {'mse_reg': 0.1, 'mse_fus': 0.2, 'ncc_reg': 0.3}
    """
    items = []
    for k, v in loss_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_loss_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AverageMeter(object):
    """Computes and stores the average, current value, and standard deviation.

    Useful for tracking metrics during training.

    Attributes:
        val: Current value
        avg: Running average
        sum: Cumulative sum
        count: Number of updates
        vals: List of all values
        std: Standard deviation of all values
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        """Update statistics with new value.

        Args:
            val: New value to add
            n: Weight for the update. Default: 1
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def pad_image(img, target_size):
    """Pad image to target size.

    Args:
        img: Input image of shape (B, C, D, H, W)
        target_size: Target size tuple (D, H, W)

    Returns:
        Padded image
    """
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img


class SpatialTransformer(nn.Module):
    """N-D Spatial Transformer for image warping.

    Applies spatial transformations to images using flow fields.
    Supports both 2D and 3D images.

    Args:
        size: Size of the image (D, H, W) or (H, W)
        mode: Interpolation mode. Default: 'bilinear'

    Note:
        Grid is registered as a buffer, which includes it in state_dict.
        This increases model file size but ensures correct device placement.
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # Register grid as buffer
        # Note: This adds it to state_dict, increasing model file size
        # See: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        """Apply spatial transformation.

        Args:
            src: Source image to transform
            flow: Displacement field

        Returns:
            Transformed image
        """
        # Compute new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channels dim to last position
        # Note: Channels need to be reversed for grid_sample
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class register_model(nn.Module):
    """Simple registration model wrapper.

    Wraps SpatialTransformer for easy registration.

    Args:
        img_size: Size of images. Default: (64, 256, 256)
        mode: Interpolation mode. Default: 'bilinear'
    """

    def __init__(self, img_size=(64, 256, 256), mode="bilinear"):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        """Apply registration.

        Args:
            x: List of [image, flow]

        Returns:
            Warped image
        """
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out


def dice_val(y_pred, y_true, num_clus):
    """Compute Dice score for multi-class segmentation.

    Args:
        y_pred: Predicted labels of shape (B, 1, D, H, W)
        y_true: Ground truth labels of shape (B, 1, D, H, W)
        num_clus: Number of classes

    Returns:
        Mean Dice score across all classes and batch
    """
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.0 * intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))


def dice_val_VOI(y_pred, y_true):
    """Compute Dice score for volumes of interest (VOI).

    Uses FreeSurfer label indices for brain structures.

    Args:
        y_pred: Predicted labels
        y_true: Ground truth labels

    Returns:
        Mean Dice score across all VOI labels
    """
    VOI_lbls = [
        1,
        2,
        3,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        18,
        20,
        21,
        22,
        23,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        34,
        36,
    ]
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.0 * intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        idx += 1
    return np.mean(DSCs)


def jacobian_determinant_vxm(disp):
    """Compute Jacobian determinant of a displacement field.

    Uses numerical gradients to compute the determinant.
    Reference: VoxelMorph implementation

    Args:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        Jacobian determinant (scalar field)
    """
    # Check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), "flow has to be 2D or 3D"

    # Compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # Compute gradients
    J = np.gradient(disp + grid)

    # 3D flow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # Compute Jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # 2D
        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]


def process_label():
    """Process FreeSurfer labeling information.

    Reads FreeSurfer label information and creates a lookup table.
    Note: Uses hard-coded path - should be updated for different systems.

    Returns:
        Dictionary mapping label indices to names
    """
    # FreeSurfer segmentation table
    seg_table = [
        0,
        2,
        3,
        4,
        5,
        7,
        8,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        24,
        26,
        28,
        30,
        31,
        41,
        42,
        43,
        44,
        46,
        47,
        49,
        50,
        51,
        52,
        53,
        54,
        58,
        60,
        62,
        63,
        72,
        77,
        80,
        85,
        251,
        252,
        253,
        254,
        255,
    ]

    # TODO: Update this path for different systems
    file1 = open("/root/daigaole/data/IXI_data/label_info.txt", "r")
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(" +", " ", line).split(" ")
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict


def write2csv(line, name):
    """Write a line to CSV file.

    Args:
        line: String to write
        name: Base name of CSV file (without .csv extension)
    """
    with open(name + ".csv", "a") as file:
        file.write(line)
        file.write("\n")


def dice_val_substruct(y_pred, y_true, std_idx):
    """Compute per-structure Dice scores and format as CSV line.

    Args:
        y_pred: Predicted labels
        y_true: Ground truth labels
        std_idx: Subject index for CSV formatting

    Returns:
        CSV-formatted string with Dice scores for 46 structures
    """
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = "p_{}".format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.0 * intersection) / (union + 1e-5)
        line = line + "," + str(dsc)
    return line


def dice(y_pred, y_true):
    """Compute Dice score for binary segmentation.

    Args:
        y_pred: Predicted binary mask (numpy array)
        y_true: Ground truth binary mask (numpy array)

    Returns:
        Dice score
    """
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.0 * intersection) / (union + 1e-5)
    return dsc


def smooth_seg(binary_img, sigma=1.5, thresh=0.4):
    """Apply Gaussian smoothing to binary segmentation.

    Args:
        binary_img: Binary segmentation mask
        sigma: Gaussian kernel standard deviation. Default: 1.5
        thresh: Threshold for binarization after smoothing. Default: 0.4

    Returns:
        Smoothed binary mask
    """
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img


def get_mc_preds(net, inputs, mc_iter: int = 25):
    """Perform Monte Carlo sampling for uncertainty estimation.

    Runs multiple forward passes with dropout/stochasticity enabled.

    Args:
        net: Neural network model (with dropout or other stochasticity)
        inputs: Input to network
        mc_iter: Number of MC samples. Default: 25

    Returns:
        Tuple of (img_list, flow_list) containing MC samples
    """
    img_list = []
    flow_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
    return img_list, flow_list


def calc_uncert(tar, img_list):
    """Calculate uncertainty as mean squared error across MC samples.

    Args:
        tar: Target image
        img_list: List of predicted images from MC sampling

    Returns:
        Mean uncertainty map
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar) ** 2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def calc_error(tar, img_list):
    """Calculate prediction error as mean squared error across MC samples.

    Note: This function is identical to calc_uncert for backward compatibility.

    Args:
        tar: Target image
        img_list: List of predicted images from MC sampling

    Returns:
        Mean error map
    """
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar) ** 2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert


def get_mc_preds_w_errors(net, inputs, target, mc_iter: int = 25):
    """Perform MC sampling and compute per-sample errors.

    Args:
        net: Neural network model
        inputs: Input to network
        target: Target image for error computation
        mc_iter: Number of MC samples. Default: 25

    Returns:
        Tuple of (img_list, flow_list, err) where err is list of MSE values
    """
    img_list = []
    flow_list = []
    MSE = nn.MSELoss()
    err = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, flow = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            err.append(MSE(img, target).item())
    return img_list, flow_list, err


def get_diff_mc_preds(net, inputs, mc_iter: int = 25):
    """Perform MC sampling for diffeomorphic registration.

    Args:
        net: Neural network model
        inputs: Input to network
        mc_iter: Number of MC samples. Default: 25

    Returns:
        Tuple of (img_list, flow_list, disp_list) containing MC samples
    """
    img_list = []
    flow_list = []
    disp_list = []
    with torch.no_grad():
        for _ in range(mc_iter):
            img, _, flow, disp = net(inputs)
            img_list.append(img)
            flow_list.append(flow)
            disp_list.append(disp)
    return img_list, flow_list, disp_list


def uncert_regression_gal(img_list, reduction="mean"):
    """Compute aleatoric and epistemic uncertainty using Gal's method.

    Separates uncertainty into aleatoric (data) and epistemic (model) components.

    Args:
        img_list: List of predictions with uncertainty estimates
        reduction: Reduction method ('mean', 'sum', or None). Default: 'mean'

    Returns:
        Tuple of (aleatoric, epistemic, total_uncertainty)
        If reduction is None, returns tensors instead of scalars
    """
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:, :-1].mean(dim=0, keepdim=True)
    ale = img_list[:, -1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:, :-1], dim=0, keepdim=True)
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == "mean":
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == "sum":
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()


def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    """Compute Uncertainty Calibration Error (UCE).

    Measures calibration between predicted uncertainty and actual errors.

    Args:
        errors: Prediction errors
        uncert: Predicted uncertainties
        n_bins: Number of bins for calibration. Default: 15
        outlier: Minimum proportion of samples in bin to include. Default: 0.0
        range: Tuple of (min, max) for bin boundaries. If None, uses data range.

    Returns:
        Tuple of (uce, err_in_bin, avg_uncert_in_bin, prop_in_bin)
    """
    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(
            uncert.min().item(), uncert.max().item(), n_bins + 1, device=device
        )
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculate |uncertainty - error| in each bin
        in_bin = uncert.gt(bin_lower.item()) * uncert.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()  # |Bm| / n
        prop_in_bin_list.append(prop_in_bin)
        if prop_in_bin.item() > outlier:
            errors_in_bin = errors[in_bin].float().mean()  # err()
            avg_uncert_in_bin = uncert[in_bin].mean()  # uncert()
            uce += torch.abs(avg_uncert_in_bin - errors_in_bin) * prop_in_bin

            errors_in_bin_list.append(errors_in_bin)
            avg_uncert_in_bin_list.append(avg_uncert_in_bin)

    err_in_bin = torch.tensor(errors_in_bin_list, device=device)
    avg_uncert_in_bin = torch.tensor(avg_uncert_in_bin_list, device=device)
    prop_in_bin = torch.tensor(prop_in_bin_list, device=device)

    return uce, err_in_bin, avg_uncert_in_bin, prop_in_bin
