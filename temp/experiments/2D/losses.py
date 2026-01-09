import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
import math
import torch.nn as nn
import lpips
from torchmetrics import Dice

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size,
                                                                  window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _ssim_3D(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv3d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM2D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM2D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim(img1, img2, window, self.window_size, channel, self.size_average)


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return 1-_ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def ssim3D(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

class LogMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(LogMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, input, target):
        mse_loss = self.mse(input, target)
        log_mse_loss = torch.log(mse_loss + self.eps)  
        return log_mse_loss

class LpipsLoss(nn.Module):
    def __init__(self,net='vgg', use_gpu=True):
        super(LpipsLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net=net)  
        if use_gpu:
            self.lpips_loss = self.lpips_loss.cuda()

    def forward(self, input, target):
        lpips_loss_value = self.lpips_loss(input, target)
        return lpips_loss_value.mean()  
    
# class DiceLoss(nn.Module):
#     def __init__(self):
#         super(DiceLoss, self).__init__()
#         self.metric = Dice()

#     def forward(self, input, target):
#         dice_loss_value =  1 - self.metric(input, target)
#         return dice_loss_value

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input = input.float()
        # input = torch.softmax(input, dim=1)
        # input = torch.sigmoid(input)
        target = target.float()

        intersection = (input * target).sum()
        total_area = input.sum() + target.sum()

        dice = (2. * intersection + self.smooth) / (total_area + self.smooth)

        return 1 - dice

class S3IMLoss(nn.Module):
    def __init__(self, B: int = 1024, M: int = 10, window_size: int = 4):
        super().__init__()
        self.B = B
        self.M = M
        self.window_size = window_size
        
        self.patch_dim = int(B**0.5)
        if self.patch_dim * self.patch_dim != B:
            raise ValueError(f"B (number of sampled pixels) should be a squared number, yet we get {B}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_batch, n_channels, height, width = input.shape
        input_flat = input.view(n_batch * n_channels, height * width)
        target_flat = target.view(n_batch * n_channels, height * width)
        
        num_pixels_per_image = height * width
        
        ssim_values = []
        
        window = create_window(self.window_size, 1).to(input.device)

        for _ in range(self.M):
            rand_indices = torch.randperm(num_pixels_per_image, device=input.device)[:self.B]
            
            sampled_input = torch.index_select(input_flat, 1, rand_indices)
            sampled_target = torch.index_select(target_flat, 1, rand_indices)
            
            input_patch = sampled_input.view(n_batch * n_channels, 1, self.patch_dim, self.patch_dim)
            target_patch = sampled_target.view(n_batch * n_channels, 1, self.patch_dim, self.patch_dim)

            ssim_val = _ssim(input_patch, target_patch, window, self.window_size, channel=1, size_average=True)
            ssim_values.append(ssim_val)
            
        avg_s3im = torch.mean(torch.stack(ssim_values))
        
        return 1.0 - avg_s3im

class S3IMLossStitched(nn.Module):
    """
    S3IM Loss, a variant where the input is a batch of pre-divided patches.
    The loss function randomly selects a subset of these patches, stitches them
    into a pseudo-image, and computes SSIM.
    """
    def __init__(self, patches_to_stitch: int = 16, M: int = 10, window_size: int = 4):
        super().__init__()
        self.patches_to_stitch = patches_to_stitch
        self.M = M
        self.window_size = window_size
        
        self.grid_dim = int(self.patches_to_stitch**0.5)
        if self.grid_dim * self.grid_dim != self.patches_to_stitch:
            raise ValueError(f"patches_to_stitch should be a squared number, yet we get {self.patches_to_stitch}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        num_available_patches, n_channels, patch_h, patch_w = input.shape
        
        if num_available_patches < self.patches_to_stitch:
            raise ValueError(f" num_available_patche ({num_available_patches}) is less than self.patches_to_stitch ({self.patches_to_stitch})")

        ssim_values = []
        
        window = create_window(self.window_size, 1).to(input.device)

        for _ in range(self.M):
            rand_indices = torch.randperm(num_available_patches, device=input.device)[:self.patches_to_stitch]
            
            selected_input_patches = input[rand_indices] 
            selected_target_patches = target[rand_indices]


            list_of_input_patches = torch.chunk(selected_input_patches, chunks=self.patches_to_stitch, dim=0)
            list_of_target_patches = torch.chunk(selected_target_patches, chunks=self.patches_to_stitch, dim=0)

            input_rows = [torch.cat(list_of_input_patches[i*self.grid_dim : (i+1)*self.grid_dim], dim=3) for i in range(self.grid_dim)]
            target_rows = [torch.cat(list_of_target_patches[i*self.grid_dim : (i+1)*self.grid_dim], dim=3) for i in range(self.grid_dim)]
            
            stitched_input = torch.cat(input_rows, dim=2)
            stitched_target = torch.cat(target_rows, dim=2)


            stitched_input_reshaped = stitched_input.view(n_channels, 1, 
                                                          self.grid_dim * patch_h, 
                                                          self.grid_dim * patch_w)
            stitched_target_reshaped = stitched_target.view(n_channels, 1, 
                                                            self.grid_dim * patch_h, 
                                                            self.grid_dim * patch_w)

            ssim_val = _ssim(stitched_input_reshaped, stitched_target_reshaped, 
                             window, self.window_size, channel=1, size_average=True)
            ssim_values.append(ssim_val)
            
        avg_s3im = torch.mean(torch.stack(ssim_values))
        
        return 1.0 - avg_s3im

class RangeInvariantPSNRLoss(nn.Module):

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _zero_mean(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.mean(x, dim=1, keepdim=True)

    def _fix_range(self, gt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = torch.sum(gt * x, dim=1, keepdim=True) / (torch.sum(x * x, dim=1, keepdim=True) + self.eps)
        return x * a

    def _fix(self, gt: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gt_ = self._zero_mean(gt)
        return self._fix_range(gt_, self._zero_mean(x))
    
    def _psnr_internal(self, gt: torch.Tensor, pred: torch.Tensor, range_: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((gt - pred) ** 2, dim=1)
        return 20 * torch.log10(range_ / torch.sqrt(mse + self.eps))
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        assert len(input.shape) > 1, "Input tensor must have at least 2 dimensions"
        original_shape = input.shape
        input_flat = input.view(original_shape[0], -1)
        target_flat = target.view(original_shape[0], -1)

        target_std = torch.std(target_flat, dim=1, keepdim=True)
        gt_ = self._zero_mean(target_flat) / (target_std + self.eps)
        
        pred_fixed = self._fix(gt_, input_flat)

        ra = (torch.max(target_flat, dim=1).values - torch.min(target_flat, dim=1).values) / (target_std.squeeze() + self.eps)
        
        psnr = self._psnr_internal(gt_, pred_fixed, ra)

        return -psnr.mean()    

class Grad2d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad2d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad

class Grad3DiTV(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self):
        super(Grad3DiTV, self).__init__()
        a = 1

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, :-1, 1:, 1:])
        dx = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, :-1, 1:])
        dz = torch.abs(y_pred[:, :, 1:, 1:, 1:] - y_pred[:, :, 1:, 1:, :-1])
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz
        d = torch.mean(torch.sqrt(dx+dy+dz+1e-6))
        grad = d / 3.0
        return grad

class DisplacementRegularizer(torch.nn.Module):
    def __init__(self, energy_type):
        super().__init__()
        self.energy_type = energy_type

    def gradient_dx(self, fv): return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(self, fv): return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(self, fv): return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(self, Txyz, fn):
        return torch.stack([fn(Txyz[:,i,...]) for i in [0, 1, 2]], dim=1)

    def compute_gradient_norm(self, displacement, flag_l1=False):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        if flag_l1:
            norms = torch.abs(dTdx) + torch.abs(dTdy) + torch.abs(dTdz)
        else:
            norms = dTdx**2 + dTdy**2 + dTdz**2
        return torch.mean(norms)/3.0

    def compute_bending_energy(self, displacement):
        dTdx = self.gradient_txyz(displacement, self.gradient_dx)
        dTdy = self.gradient_txyz(displacement, self.gradient_dy)
        dTdz = self.gradient_txyz(displacement, self.gradient_dz)
        dTdxx = self.gradient_txyz(dTdx, self.gradient_dx)
        dTdyy = self.gradient_txyz(dTdy, self.gradient_dy)
        dTdzz = self.gradient_txyz(dTdz, self.gradient_dz)
        dTdxy = self.gradient_txyz(dTdx, self.gradient_dy)
        dTdyz = self.gradient_txyz(dTdy, self.gradient_dz)
        dTdxz = self.gradient_txyz(dTdx, self.gradient_dz)
        return torch.mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2)

    def forward(self, disp, _):
        if self.energy_type == 'bending':
            energy = self.compute_bending_energy(disp)
        elif self.energy_type == 'gradient-l2':
            energy = self.compute_gradient_norm(disp)
        elif self.energy_type == 'gradient-l1':
            energy = self.compute_gradient_norm(disp, flag_l1=True)
        else:
            raise Exception('Not recognised local regulariser!')
        return energy

class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

class MIND_loss(torch.nn.Module):
    """
        Local (over window) normalized cross correlation loss.
        """

    def __init__(self, win=None):
        super(MIND_loss, self).__init__()
        self.win = win

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).cuda()
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = nn.ReplicationPad3d(dilation)
        rpad2 = nn.ReplicationPad3d(radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind /= mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

class MutualInformation(torch.nn.Module):
    """
    Mutual Information
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        print(sigma)

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def forward(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class localMutualInformation(torch.nn.Module):
    """
    Local Mutual Information for non-overlapping patches
    """

    def __init__(self, sigma_ratio=1, minval=0., maxval=1., num_bin=32, patch_size=5):
        super(localMutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False).cuda()
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
        self.patch_size = patch_size

    def local_mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        """Reshape bin centers"""
        o = [1, 1, np.prod(self.vol_bin_centers.shape)]
        vbc = torch.reshape(self.vol_bin_centers, o).cuda()

        """Making image paddings"""
        if len(list(y_pred.size())[2:]) == 3:
            ndim = 3
            x, y, z = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            z_r = -z % self.patch_size
            padding = (z_r // 2, z_r - z_r // 2, y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        elif len(list(y_pred.size())[2:]) == 2:
            ndim = 2
            x, y = list(y_pred.size())[2:]
            # compute padding sizes
            x_r = -x % self.patch_size
            y_r = -y % self.patch_size
            padding = (y_r // 2, y_r - y_r // 2, x_r // 2, x_r - x_r // 2, 0, 0, 0, 0)
        else:
            raise Exception('Supports 2D and 3D but not {}'.format(list(y_pred.size())))
        y_true = F.pad(y_true, padding, "constant", 0)
        y_pred = F.pad(y_pred, padding, "constant", 0)

        """Reshaping images into non-overlapping patches"""
        if ndim == 3:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 3, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size,
                                                  (z + z_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 6, 3, 5, 7)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 3, 1))
        else:
            y_true_patch = torch.reshape(y_true, (y_true.shape[0], y_true.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_true_patch = y_true_patch.permute(0, 1, 2, 4, 3, 5)
            y_true_patch = torch.reshape(y_true_patch, (-1, self.patch_size ** 2, 1))

            y_pred_patch = torch.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1],
                                                  (x + x_r) // self.patch_size, self.patch_size,
                                                  (y + y_r) // self.patch_size, self.patch_size))
            y_pred_patch = y_pred_patch.permute(0, 1, 2, 4, 3, 5)
            y_pred_patch = torch.reshape(y_pred_patch, (-1, self.patch_size ** 2, 1))

        """Compute MI"""
        I_a_patch = torch.exp(- self.preterm * torch.square(y_true_patch - vbc))
        I_a_patch = I_a_patch / torch.sum(I_a_patch, dim=-1, keepdim=True)

        I_b_patch = torch.exp(- self.preterm * torch.square(y_pred_patch - vbc))
        I_b_patch = I_b_patch / torch.sum(I_b_patch, dim=-1, keepdim=True)

        pab = torch.bmm(I_a_patch.permute(0, 2, 1), I_b_patch)
        pab = pab / self.patch_size ** ndim
        pa = torch.mean(I_a_patch, dim=1, keepdim=True)
        pb = torch.mean(I_b_patch, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()

    def forward(self, y_true, y_pred):
        return -self.local_mi(y_true, y_pred)

#for fusion

class SobelxyRGB3D(nn.Module):
    def __init__(self, isSignGrad=True):
        super(SobelxyRGB3D, self).__init__()
        self.isSignGrad = isSignGrad
        kernelx = [[[0.2, 0, -0.2], [1, 0, -1], [0.2, 0, -0.2]],
                   [[1, 0, -1], [4, 0, -4], [1, 0, -1]],
                   [[0.2, 0, -0.2], [1, 0, -1], [0.2, 0, -0.2]]]
        kernely = [[[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]],
                   [[1, 4, 1], [0, 0, 0], [-1, -4, -1]],
                   [[0.2, 1, 0.2], [0, 0, 0], [-0.2, -1, -0.2]]]
        kernelz = [[[0.2, 1, 0.2], [1, 4, 1], [0.2, 1, 0.2]],
                   [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                   [[-0.2, -1, -0.2], [-1, -4, -1], [-0.2, -1, -0.2]]]
        self.weightx = nn.Parameter(torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1,1), requires_grad=False)
        self.weighty = nn.Parameter(torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1,1), requires_grad=False)
        self.weightz = nn.Parameter(torch.FloatTensor(kernelz).unsqueeze(0).unsqueeze(0).repeat(1,3,1,1,1), requires_grad=False)

    def forward(self, x):
        sobelx = F.conv3d(x, self.weightx, padding=1)
        sobely = F.conv3d(x, self.weighty, padding=1)
        sobelz = F.conv3d(x, self.weightz, padding=1)
        if self.isSignGrad:
            return sobelx + sobely + sobelz
        else:
            return torch.abs(sobelx) + torch.abs(sobely) + torch.abs(sobelz)

class MaxGradLoss3D(nn.Module):
    def __init__(self, loss_weight=1.0, isSignGrad=True):
        super(MaxGradLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.sobelconv = SobelxyRGB3D(isSignGrad)
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None):
        if im_fusion.shape[1] == 1:
            im_fusion = im_fusion.repeat(1, 3, 1, 1, 1)
        if im_rgb.shape[1] == 1:
            im_rgb = im_rgb.repeat(1, 3, 1, 1, 1)
        if im_tir is not None and im_tir.shape[1] == 1:
            im_tir = im_tir.repeat(1, 3, 1, 1, 1)
        if im_tir is not None:
            rgb_grad = self.sobelconv(im_rgb)
            tir_grad = self.sobelconv(im_tir)

            mask = torch.ge(torch.abs(rgb_grad), torch.abs(tir_grad))
            max_grad_joint = tir_grad.masked_fill_(mask, 0) + rgb_grad.masked_fill_(~mask, 0)
            
            generate_img_grad = self.sobelconv(im_fusion)

            sobel_loss = self.L1_loss(generate_img_grad, max_grad_joint)
            loss_grad = self.loss_weight * sobel_loss
        else:
            rgb_grad = self.sobelconv(im_rgb)
            generate_img_grad = self.sobelconv(im_fusion)
            sobel_loss = self.L1_loss(generate_img_grad, rgb_grad)
            loss_grad = self.loss_weight * sobel_loss

        return loss_grad

def to_gray3d(img):
    r, g, b = img.unbind(dim=-4)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-4)
    return l_img

class MaxPixelLoss3D(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaxPixelLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None):
        if im_tir is not None:
            pixel_max = torch.max(im_rgb, im_tir).detach()
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, pixel_max)
        else:
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, im_rgb)
        
        return pixel_loss

class PixelLoss3D(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PixelLoss3D, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()

    def forward(self, im_fusion, im_rgb, im_tir=None):
        if im_tir is not None:
            pixel_mean = (im_rgb + im_tir) / 2.0
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, pixel_mean)
        else:
            pixel_loss = self.loss_weight * self.L1_loss(im_fusion, im_rgb)
        
        return pixel_loss

class MaxGradTokenSelect3D(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(MaxGradTokenSelect3D, self).__init__()
        self.sobelconv = SobelxyRGB3D()

    def forward(self, im_rgb, im_tir):
        im_rgb_gray = to_gray3d(im_rgb)
        im_tir_gray = to_gray3d(im_tir)
        rgb_grad = self.sobelconv(im_rgb_gray)
        tir_grad = self.sobelconv(im_tir_gray)

        im_rgb, info = self.patchify3d(im_rgb)
        im_tir, _ = self.patchify3d(im_tir)
        rgb_grad_patch, info_grad = self.patchify3d(rgb_grad)
        tir_grad_patch, _ = self.patchify3d(tir_grad)

        rgb_grad, _ = torch.max(rgb_grad_patch, -1)
        tir_grad, _ = torch.max(tir_grad_patch, -1)

        AB_mask = (rgb_grad >= tir_grad).unsqueeze(dim=-1)
        out = torch.where(AB_mask, im_rgb, im_tir)

        out = self.unpatchify3d(out, info)

        return out

    def patchify3d(self, imgs):
        p = 16
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        C, D, H, W = imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
        x = imgs.reshape(shape=(imgs.shape[0], C, D//p, p, H//p, p, W//p, p))
        x = torch.einsum('ncdphqwr->ndhwpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], D//p * H//p * W//p, p*p*p*C))
        return x, (D, H, W, C)

    def unpatchify3d(self, x, shape):
        p = 16
        D, H, W, C = shape
        assert D * H * W // (p**3) == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], D//p, H//p, W//p, p, p, p, C))
        x = torch.einsum('ndhwpqrc->ncdphqwr', x)
        imgs = x.reshape(shape=(x.shape[0], C, D, H, W))
        return imgs