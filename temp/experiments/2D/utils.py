import math
import numpy as np
import torch.nn.functional as F
import torch, sys
from torch import nn
import torch.distributed as dist
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import pystrum.pynd.ndutils as nd
from scipy.ndimage import gaussian_filter
from microssim import MicroSSIM, MicroMS3IM # 导入类本身

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import math
import socket
import shutil
import glob

def psnr(img1, img2, data_range=None):
    if data_range is None:
        data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

def carepsnr(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Calculate CARE-PSNR

    Args:
        img1 (torch.Tensor): Predict image
        img2 (torch.Tensor): Real image
        eps (float)
    """

    def _zero_mean(x: torch.Tensor) -> torch.Tensor:
        return x - torch.mean(x, dim=1, keepdim=True)

    def _fix_range(gt: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        a = torch.sum(gt * x, dim=1, keepdim=True) / (torch.sum(x * x, dim=1, keepdim=True) + eps)
        return x * a

    def _fix(gt: torch.Tensor, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        gt_ = _zero_mean(gt)
        return _fix_range(gt_, _zero_mean(x), eps)

    def _psnr_internal(gt: torch.Tensor, pred: torch.Tensor, range_: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        mse = torch.mean((gt - pred) ** 2, dim=1)
        return 20 * torch.log10(range_ / torch.sqrt(mse + eps))
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)

    img1 = img1.float()
    img2 = img2.float()

    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 2:
        img2 = img2.unsqueeze(0)

    assert len(img1.shape) == 3, f"Input should be (B, H, W) or (H, W), yet we get: {img1.shape}"
    assert img1.shape == img2.shape, f"Shape should be same {img1.shape} vs {img2.shape}"

    pred_flat = img1.view(img1.shape[0], -1)
    gt_flat = img2.view(img2.shape[0], -1)
    
    gt_std_squeezed = torch.std(gt_flat, dim=1)
    ra = (torch.max(gt_flat, dim=1).values - torch.min(gt_flat, dim=1).values) / (gt_std_squeezed + eps)
    
    gt_std_unsqueezed = gt_std_squeezed.unsqueeze(1)
    gt_ = _zero_mean(gt_flat) / (gt_std_unsqueezed + eps)
    
    pred_fixed = _fix(gt_, pred_flat, eps)

    psnr_values = _psnr_internal(gt_, pred_fixed, ra, eps)

    return psnr_values.mean().item()

def microssim(pred_img, gt_img):

    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    assert pred_img.shape == gt_img.shape, \
        f"Shape should be same: {pred_img.shape} vs {gt_img.shape}"

    shape = pred_img.shape
    if len(shape) == 4:
        B, C, H, W = shape
        pred_stack = pred_img.reshape(B * C, H, W)
        gt_stack = gt_img.reshape(B * C, H, W)
    elif len(shape) == 3:
        pred_stack = pred_img
        gt_stack = gt_img
    elif len(shape) == 2:
        pred_stack = pred_img[np.newaxis, ...]
        gt_stack = gt_img[np.newaxis, ...]
    else:
        raise ValueError(f"Wrong shape: {shape}")

    microssim_evaluator = MicroSSIM()
    
    microssim_evaluator.fit(gt_stack, pred_stack)
    
    scores = []
    for i in range(len(gt_stack)):
        score = microssim_evaluator.score(gt_stack[i], pred_stack[i])
        scores.append(score)
        
    return np.mean(scores)


def microms3im(pred_img, gt_img):

    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(gt_img, torch.Tensor):
        gt_img = gt_img.detach().cpu().numpy()

    assert pred_img.shape == gt_img.shape, \
        f"Shape should be same: {pred_img.shape} vs {gt_img.shape}"
        
    shape = pred_img.shape
    if len(shape) == 4:
        B, C, H, W = shape
        pred_stack = pred_img.reshape(B * C, H, W)
        gt_stack = gt_img.reshape(B * C, H, W)
    elif len(shape) == 3:
        pred_stack = pred_img
        gt_stack = gt_img
    elif len(shape) == 2:
        pred_stack = pred_img[np.newaxis, ...]
        gt_stack = gt_img[np.newaxis, ...]
    else:
        raise ValueError(f"Wrong shape: {shape}")

    microms3im_evaluator = MicroMS3im()
    microms3im_evaluator.fit(gt_stack, pred_stack)
    
    scores = []
    for i in range(len(gt_stack)):
        score = microms3im_evaluator.score(gt_stack[i], pred_stack[i]).item()
        scores.append(score)
        
    return np.mean(scores) 

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.log = open(os.path.join(save_dir, "logfile.log"), "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

class WarmupCosineSchedule(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1, warmup_start_factor=0.01):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.warmup_start_factor = warmup_start_factor
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * ((self.warmup_start_factor - 1) * (self.warmup_steps - self.last_epoch) / self.warmup_steps + 1)
                    for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_steps) / (self.t_total - self.warmup_steps)
            return [base_lr * (0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress))) for base_lr in self.base_lrs]

def save_checkpoint(state, config, is_best, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'model_best.pth.tar'))
    all_checkpoints = glob.glob(os.path.join(save_dir, '*.pth.tar'))
    all_checkpoints = [ckpt for ckpt in all_checkpoints if not ckpt.endswith('model_best.pth.tar')]
    if len(all_checkpoints) > max_model_num:
        oldest_file = min(all_checkpoints, key=os.path.getctime)
        os.remove(oldest_file)

def visualize_logits(logits, output_folder=None, figsize=(20, 20), verbose=True):
    def print_dict_structure(d, indent=0):
        for key, value in d.items():
            print('  ' * indent + str(key))
            if isinstance(value, dict):
                print_dict_structure(value, indent+1)
            else:
                print('  ' * (indent+1) + f"{type(value)}, Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
    
    if verbose:
        print("Logits structure:")
        print_dict_structure(logits)
        print("\n")

    def get_image(img):
        if img.ndim == 4:  # (B, C, H, W)
            if img.shape[1] == 2:  # Flow data
                return img[0].transpose(1, 2, 0)
            else:
                return img[0, 0]
        elif img.ndim == 3:  # (C, H, W)
            return img.transpose(1, 2, 0) if img.shape[0] == 2 else img[0]
        return img

    def visualize_flow(flow, ax):
        if flow.ndim == 2:  # (H, W)
            magnitude = np.linalg.norm(flow, axis=-1)
            ax.imshow(magnitude, cmap='viridis')
            ax.set_title('Flow Magnitude')
        elif flow.ndim == 3 and flow.shape[-1] == 2:  # (H, W, 2)
            flow_norm = flow / (np.linalg.norm(flow, axis=-1, keepdims=True) + 1e-8)
            flow_rgb = np.zeros((flow.shape[0], flow.shape[1], 3))
            flow_rgb[..., :2] = (flow_norm + 1) / 2
            ax.imshow(flow_rgb)
            ax.set_title('Flow (RGB)')
        else:
            print(f"Unexpected flow shape: {flow.shape}")

    def create_misalignment_overlay(img1, img2, enhance_diff=True):
        img1_norm = (img1 - img1.min()) / (img1.max() - img1.min())
        img2_norm = (img2 - img2.min()) / (img2.max() - img2.min())
        
        diff = np.abs(img1_norm - img2_norm)
        
        if enhance_diff:
            diff = np.power(diff, 0.5)
            threshold = 0.1
            diff[diff < threshold] = 0
        
        overlay = np.zeros((img1.shape[0], img1.shape[1], 3))
        overlay[:,:,0] = img1_norm
        overlay[:,:,2] = img2_norm
        overlay[:,:,1] = diff  # Green channel now shows misalignment
        
        return overlay

    def create_complementary_image(raw_img, mask_a, mask_b):
        complementary = np.where(mask_a > 0, raw_img, np.where(mask_b > 0, raw_img, 0))
        return complementary
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    raw_image = get_image(logits['raw'])

    for key, value in logits.items():
        if isinstance(value, dict):
            n_images = 1  # Start with 1 for the raw image
            for sub_key in value.keys():
                if sub_key not in ['flow', 'inverted_flow']:
                    n_images += 1
                else:
                    n_images += 1  # Count flow images

            # Add overlay images
            if 'registered' in value:
                n_images += 1
            if 'deformed' in value:
                n_images += 1
            if 'masked_A' in value and 'masked_B' in value:
                n_images += 1  # Add one for complementary image

            # Calculate optimal grid layout
            n_cols = int(np.ceil(np.sqrt(n_images)))
            n_rows = int(np.ceil(n_images / n_cols))
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            fig.suptitle(f"Visualization of {key}", fontsize=16)
            
            # Plot raw image first
            axes[0].imshow(raw_image, cmap='gray')
            axes[0].set_title('Raw')
            axes[0].axis('off')
            
            ax_index = 1
            for sub_key, img in value.items():
                if ax_index < len(axes):
                    ax = axes[ax_index]
                    try:
                        if sub_key in ['deformed_grid', 'restored_grid']:
                            grid_image = get_image(img)
                            ax.imshow(grid_image, cmap='gray', vmin=0, vmax=1)
                            ax.set_title(f'{sub_key} (Grid Only)')
                        elif sub_key in ['flow', 'inverted_flow']:
                            visualize_flow(get_image(img), ax)
                        else:
                            ax.imshow(get_image(img), cmap='gray')
                        
                        ax.set_title(sub_key)
                        ax.axis('off')
                        ax_index += 1
                    except Exception as e:
                        print(f"Error visualizing {sub_key}: {e}")
            
            # Add overlay of raw and registered images
            if 'registered' in value and ax_index < len(axes):
                ax = axes[ax_index]
                reg_img = get_image(value['registered'])
                overlay = create_misalignment_overlay(raw_image, reg_img, enhance_diff=True)
                ax.imshow(overlay)
                ax.set_title('Misalignment (Raw: Red, Registered: Blue)\nGreen: Difference')
                ax.axis('off')
                ax_index += 1
            
            # Add overlay of raw and deformed images
            if 'deformed' in value and ax_index < len(axes):
                ax = axes[ax_index]
                deformed_img = get_image(value['deformed'])
                overlay = create_misalignment_overlay(raw_image, deformed_img, enhance_diff=True)
                ax.imshow(overlay)
                ax.set_title('Misalignment (Raw: Red, Deformed: Blue)\nGreen: Difference')
                ax.axis('off')
                ax_index += 1

            # Add complementary image of masked_A and masked_B
            if 'masked_A' in value and 'masked_B' in value and ax_index < len(axes):
                ax = axes[ax_index]
                mask_a = get_image(value['masked_A'])
                mask_b = get_image(value['masked_B'])
                complementary_img = create_complementary_image(raw_image, mask_a, mask_b)
                ax.imshow(complementary_img, cmap='gray')
                ax.set_title('Complementary Image\n(A and B combined)')
                ax.axis('off')
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

# Usage
# visualize_logits(logits, output_folder='logits_visualization')

def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def check_grad_nan(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient detected in {name}")
                return True
    return False

def flatten_loss_dict(loss_dict, parent_key='', sep='_'):
    items = []
    for k, v in loss_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_loss_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

def dice_val(y_pred, y_true, num_clus):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))

def dice_val_VOI(y_pred, y_true):
    VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36]
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
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] =dsc
        idx += 1
    return np.mean(DSCs)

def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]

import re
def process_label():
    #process labeling information for FreeSurfer
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]


    file1 = open('./data/IXI_data/label_info.txt', 'r')
    Lines = file1.readlines()  
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
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
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        line = line+','+str(dsc)
    return line

def dice(y_pred, y_true, ):
    intersection = y_pred * y_true
    intersection = np.sum(intersection)
    union = np.sum(y_pred) + np.sum(y_true)
    dsc = (2.*intersection) / (union + 1e-5)
    return dsc

def smooth_seg(binary_img, sigma=1.5, thresh=0.4):
    binary_img = gaussian_filter(binary_img.astype(np.float32()), sigma=sigma)
    binary_img = binary_img > thresh
    return binary_img

def get_mc_preds(net, inputs, mc_iter: int = 25):
    """Convenience fn. for MC integration for uncertainty estimation.
    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
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
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert

def calc_error(tar, img_list):
    sqr_diffs = []
    for i in range(len(img_list)):
        sqr_diff = (img_list[i] - tar)**2
        sqr_diffs.append(sqr_diff)
    uncert = torch.mean(torch.cat(sqr_diffs, dim=0)[:], dim=0, keepdim=True)
    return uncert

def get_mc_preds_w_errors(net, inputs, target, mc_iter: int = 25):
    """Convenience fn. for MC integration for uncertainty estimation.
    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
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
    """Convenience fn. for MC integration for uncertainty estimation.
    Args:
        net: DIP model (can be standard, MFVI or MCDropout)
        inputs: input to net
        mc_iter: number of MC samples
        post_processor: process output of net before computing loss (e.g. downsampler in SR)
        mask: multiply output and target by mask before computing loss (for inpainting)
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

def uncert_regression_gal(img_list, reduction = 'mean'):
    img_list = torch.cat(img_list, dim=0)
    mean = img_list[:,:-1].mean(dim=0, keepdim=True)
    ale = img_list[:,-1:].mean(dim=0, keepdim=True)
    epi = torch.var(img_list[:,:-1], dim=0, keepdim=True)
    #if epi.shape[1] == 3:
    epi = epi.mean(dim=1, keepdim=True)
    uncert = ale + epi
    if reduction == 'mean':
        return ale.mean().item(), epi.mean().item(), uncert.mean().item()
    elif reduction == 'sum':
        return ale.sum().item(), epi.sum().item(), uncert.sum().item()
    else:
        return ale.detach(), epi.detach(), uncert.detach()

def uceloss(errors, uncert, n_bins=15, outlier=0.0, range=None):
    device = errors.device
    if range == None:
        bin_boundaries = torch.linspace(uncert.min().item(), uncert.max().item(), n_bins + 1, device=device)
    else:
        bin_boundaries = torch.linspace(range[0], range[1], n_bins + 1, device=device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    errors_in_bin_list = []
    avg_uncert_in_bin_list = []
    prop_in_bin_list = []

    uce = torch.zeros(1, device=device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |uncertainty - error| in each bin
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