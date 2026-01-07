"""Vision Transformer U-Net model for biomedical image processing.

This module implements a U-Net architecture using Vision Transformer (ViT) encoder
instead of Mamba encoder. It maintains compatibility with all existing decoders
and training infrastructure.

The model supports the same tasks as MambaULight:
- Image registration
- Image fusion
- Super-resolution (SR)
- Isotropic restoration (IR)

Classes:
    ViTULight: Main ViT U-Net model with multiple task-specific decoders
"""

# Standard library imports
import random

# Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import losses
from src.encoder_factory import create_encoder, print_encoder_info
from src.ours_mamba import (
    reg_decoder,
    fus_decoder,
    SR_decoder,
    IR_decoder,
    SpatialTransformer,
)
from src.bottleneck import BottleneckFFN, HierarchicalBottleneck


class ViTULight(nn.Module):
    """Main Vision Transformer U-Net model for multiple image processing tasks.

    Combines a hierarchical Vision Transformer encoder with multiple task-specific
    decoders for registration, fusion, super-resolution, and isotropic restoration.

    This model is architecturally comparable to MambaULight but uses standard
    transformer self-attention instead of Mamba state-space models.

    Args:
        config: Configuration object containing all model parameters including:
            - img_size: Input image size [D, H, W]
            - patch_size: Size of image patches
            - in_chans: Number of input channels
            - embed_dim: Base embedding dimension
            - depths: List of depths for each stage
            - num_heads: Number of attention heads (ViT-specific)
            - mlp_ratio: MLP expansion ratio (ViT-specific)
            - grid_size: Size for spatial transformation grid
            - And other encoder/decoder configuration parameters
    """

    def __init__(self, config):
        super(ViTULight, self).__init__()
        self.if_convskip = config.if_convskip
        self.if_transskip = config.if_transskip
        self.embed_dim = config.embed_dim
        self.img_size = config.img_size
        self.task = getattr(config, 'task', 'multi_task')  # 'ir', 'reg', 'fus', 'sr', or 'multi_task'

        # Encoder configuration
        encoder_type = getattr(config, 'encoder_type', 'vit')  # 'mamba', 'vit', '3dino', 'huggingface'
        pretrained_path = getattr(config, 'pretrained_encoder_path', None)
        freeze_encoder = getattr(config, 'freeze_encoder', False)

        # Create encoder using factory
        print(f"\n🔧 Creating {encoder_type.upper()} encoder...")
        self.encoder = create_encoder(
            config,
            encoder_type=encoder_type,
            pretrained_path=pretrained_path,
            freeze=freeze_encoder
        )
        print_encoder_info(self.encoder)

        # Spatial transformation
        self.grid_size = config.grid_size
        self.spatial_trans = SpatialTransformer(config.grid_size)
        self.grid_img = self.create_grid_image()

        # Create decoder config with patch_size=4 (what pretrained decoders expect)
        # The ViT encoder uses patch_size=16 for memory efficiency, but interpolates
        # outputs to match patch_size=4 dimensions
        import copy
        decoder_config = copy.deepcopy(config)
        decoder_config.patch_size = 4  # Match pretrained Mamba decoder expectations

        # Optional: Bottleneck for dimension adaptation (e.g., 3DINO-ViT → Mamba decoders)
        # This allows training with frozen encoder + frozen decoders, only training bottleneck
        self.use_bottleneck = getattr(config, 'use_bottleneck', False)
        self.encoder_dim = getattr(config, 'encoder_dim', config.embed_dim)  # May differ from embed_dim
        self.decoder_dim = config.embed_dim  # Decoders expect this dimension

        if self.use_bottleneck and self.encoder_dim != self.decoder_dim:
            print(f"🔧 Creating bottleneck: {self.encoder_dim} → {self.decoder_dim}")

            # Get hierarchical dimensions
            encoder_dims = self._get_encoder_dims(config)
            decoder_dims = self._get_decoder_dims(config)

            # Create hierarchical bottleneck for all encoder levels
            self.bottleneck = HierarchicalBottleneck(
                encoder_dims=encoder_dims,
                decoder_dims=decoder_dims,
                hidden_ratio=getattr(config, 'bottleneck_hidden_ratio', 2.0),
                dropout=getattr(config, 'bottleneck_dropout', 0.1),
                activation=getattr(config, 'bottleneck_activation', 'gelu')
            )

            bottleneck_params = sum(p.numel() for p in self.bottleneck.parameters())
            print(f"   Bottleneck parameters: {bottleneck_params:,}")
        else:
            self.bottleneck = None

        # Task-specific decoders (reuse from ours_mamba)
        self.reg_decoder = reg_decoder(decoder_config)
        self.fus_decoder = fus_decoder(decoder_config)
        self.SR_decoder = SR_decoder(decoder_config)
        self.IR_decoder = IR_decoder(decoder_config)

        # Loss functions
        self.mse = nn.MSELoss()
        self.ncc = losses.NCC_vxm()
        self.grad = losses.Grad3d(penalty="l2")
        self.ssim = losses.SSIM3D()

    def _get_encoder_dims(self, config):
        """Get encoder output dimensions at each hierarchical level."""
        depths = config.depths
        encoder_type = getattr(config, 'encoder_type', 'vit')

        if encoder_type == 'huggingface' or encoder_type == '3dino':
            # HuggingFace/3DINO ViT outputs same dimension at all levels
            # (no hierarchical feature pyramid)
            dims = [self.encoder_dim] * len(depths)
        else:
            # For Mamba/custom ViT encoder, dimensions scale with depth
            base_dim = self.encoder_dim
            dims = [base_dim * (2 ** i) for i in range(len(depths))]
        return dims

    def _get_decoder_dims(self, config):
        """Get decoder input dimensions at each hierarchical level."""
        # Decoders expect standard Mamba dimensions
        base_dim = self.decoder_dim
        depths = config.depths
        dims = [base_dim * (2 ** i) for i in range(len(depths))]
        return dims

    def freeze_encoder(self):
        """Freeze encoder parameters for bottleneck-only training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("❄️  Encoder frozen")

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("🔥 Encoder unfrozen")

    def freeze_decoders(self):
        """Freeze decoder parameters."""
        for decoder in [self.reg_decoder, self.fus_decoder, self.SR_decoder, self.IR_decoder]:
            for param in decoder.parameters():
                param.requires_grad = False
        print("❄️  Decoders frozen")

    def unfreeze_decoders(self):
        """Unfreeze decoder parameters."""
        for decoder in [self.reg_decoder, self.fus_decoder, self.SR_decoder, self.IR_decoder]:
            for param in decoder.parameters():
                param.requires_grad = True
        print("🔥 Decoders unfrozen")

    def freeze_bottleneck(self):
        """Freeze bottleneck parameters."""
        if self.bottleneck is not None:
            for param in self.bottleneck.parameters():
                param.requires_grad = False
            print("❄️  Bottleneck frozen")

    def unfreeze_bottleneck(self):
        """Unfreeze bottleneck parameters."""
        if self.bottleneck is not None:
            for param in self.bottleneck.parameters():
                param.requires_grad = True
            print("🔥 Bottleneck unfrozen")

    def print_trainable_status(self):
        """Print which components are trainable."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        decoder_params = sum(p.numel() for p in [self.reg_decoder, self.fus_decoder,
                                                   self.SR_decoder, self.IR_decoder]
                            for p in decoder.parameters() if p.requires_grad)
        bottleneck_params = sum(p.numel() for p in self.bottleneck.parameters() if p.requires_grad) if self.bottleneck else 0
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n📊 Trainable Parameters:")
        print(f"   Encoder:    {encoder_params:>12,} {'✓ trainable' if encoder_params > 0 else '❄️  frozen'}")
        print(f"   Bottleneck: {bottleneck_params:>12,} {'✓ trainable' if bottleneck_params > 0 else '❄️  frozen'}")
        print(f"   Decoders:   {decoder_params:>12,} {'✓ trainable' if decoder_params > 0 else '❄️  frozen'}")
        print(f"   ─────────────────────────────")
        print(f"   Total:      {trainable_params:>12,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)\n")

    def forward(self, raw):
        """Forward pass through selected task(s).

        Applies synthetic degradations and restores them using task-specific decoders.
        Supports single-task or multi-task training based on self.task.

        Args:
            raw: Input image of shape (B, C, D, H, W)

        Returns:
            Tuple of (logits, aux_loss) where:
                - logits: Dictionary containing degraded and restored images for each task
                - aux_loss: Dictionary containing loss values for each task
        """
        logits = {"raw": raw.detach().cpu().numpy()}
        aux_loss = {"mse": {}, "ncc": {}, "grad": {}}

        # Map full task names to short names for internal use
        task_mapping = {
            'isotropic_restoration': 'ir',
            'registration': 'reg',
            'fusion': 'fus',
            'super_resolution': 'sr',
            'multi_task': 'multi_task'
        }

        # Determine which tasks to run
        task_key = task_mapping.get(self.task, self.task)

        if task_key == 'multi_task':
            tasks_to_run = ['reg', 'fus', 'sr', 'ir']
        else:
            tasks_to_run = [task_key]

        # DEBUG: Print task info (remove after debugging)
        # print(f"DEBUG: self.task={self.task}, task_key={task_key}, tasks_to_run={tasks_to_run}")

        # Registration task
        if 'reg' in tasks_to_run:
            reg_source, reg_flow = self.deform(raw)
            x = torch.cat([reg_source, raw], dim=1)
            out_feats = self.encoder(x)
            # Apply bottleneck projection if enabled
            if self.bottleneck is not None:
                out_feats = self.bottleneck(out_feats)
            reg_inv_flow = self.reg_decoder(out_feats)
            reged = self.spatial_trans(reg_source, reg_inv_flow)

            logits["reg"] = {
                "deformed": reg_source.detach().cpu().numpy(),
                "registered": reged.detach().cpu().numpy(),
            }
            aux_loss["mse"]["reg"] = self.mse(reged, raw)
            aux_loss["ncc"]["reg"] = self.ncc(reged, raw)
            aux_loss["grad"]["reg"] = self.grad(reg_inv_flow, raw)

        # Fusion task
        if 'fus' in tasks_to_run:
            fus_source_A = self.mask(raw)
            fus_source_B = self.mask(raw)
            x = torch.cat([fus_source_A, fus_source_B], dim=1)
            out_feats = self.encoder(x)
            # Apply bottleneck projection if enabled
            if self.bottleneck is not None:
                out_feats = self.bottleneck(out_feats)
            fused = self.fus_decoder(out_feats)

            logits["fus"] = {
                "masked_A": fus_source_A.detach().cpu().numpy(),
                "masked_B": fus_source_B.detach().cpu().numpy(),
                "fused": fused.detach().cpu().numpy(),
            }
            aux_loss["mse"]["fus"] = self.mse(fused, raw)

        # Super-resolution task
        if 'sr' in tasks_to_run:
            SR_source = self.downsample(raw)
            x = torch.cat([SR_source, SR_source], dim=1)
            out_feats = self.encoder(x)
            # Apply bottleneck projection if enabled
            if self.bottleneck is not None:
                out_feats = self.bottleneck(out_feats)
            SRed = self.SR_decoder(out_feats)

            logits["SR"] = {
                "downsampled": SR_source.detach().cpu().numpy(),
                "super_resolution": SRed.detach().cpu().numpy(),
            }
            aux_loss["mse"]["SR"] = self.mse(SRed, raw)

        # Isotropic restoration task
        if 'ir' in tasks_to_run:
            IR_source = self.noise(raw)
            x = torch.cat([IR_source, IR_source], dim=1)
            out_feats = self.encoder(x)
            # Apply bottleneck projection if enabled
            if self.bottleneck is not None:
                out_feats = self.bottleneck(out_feats)
            IRed = self.IR_decoder(out_feats)

            logits["IR"] = {
                "noisy": IR_source.detach().cpu().numpy(),
                "restored": IRed.detach().cpu().numpy(),
            }
            aux_loss["mse"]["IR"] = self.mse(IRed, raw)

        return logits, aux_loss

    def create_grid_image(self, grid_spacing=4, line_width=1):
        """Create a 3D grid image for visualizing deformations.

        Args:
            grid_spacing: Spacing between grid lines. Default: 4
            line_width: Width of grid lines. Default: 1

        Returns:
            3D grid image tensor
        """
        depth, height, width = self.grid_size
        grid = torch.zeros((1, 1, depth, height, width), dtype=torch.float32)

        # Create horizontal lines
        for y in range(0, height, grid_spacing):
            grid[:, :, :, y : y + line_width, :] = 1

        # Create vertical lines
        for x in range(0, width, grid_spacing):
            grid[:, :, :, :, x : x + line_width] = 1

        # Create depth lines
        for z in range(0, depth, grid_spacing):
            grid[:, :, z : z + line_width, :, :] = 1

        return grid

    def deform(self, image):
        """Apply synthetic deformation to image.

        Generates natural-looking deformation fields using multi-scale Perlin noise
        and applies spatial transformation.

        Args:
            image: Input image of shape (B, C, D, H, W)

        Returns:
            Tuple of (deformed_image, flow_field)
        """
        b, c, d, h, w = image.shape

        # Generate low-resolution deformation field
        lowres_d, lowres_h, lowres_w = d // 2, h // 2, w // 2

        flow = self.generate_natural_deformation_field(
            b, lowres_d, lowres_h, lowres_w, device=image.device
        )

        # Apply nonlinear transformation for natural deformation
        flow = torch.tanh(flow) * 0.6

        # Apply spatially-varying Gaussian filtering
        sigma_range = [1.5, 3.5]
        flow = self.spatially_varying_gaussian_filter(flow, sigma_range)

        # Upsample to original resolution
        flow = F.interpolate(flow, size=(d, h, w), mode="trilinear", align_corners=True)

        return self.spatial_trans(image, flow), flow

    def generate_natural_deformation_field(self, b, d, h, w, device):
        """Generate natural deformation field using multi-scale Perlin noise.

        Args:
            b: Batch size
            d, h, w: Dimensions of deformation field
            device: Device to create tensors on

        Returns:
            3-channel deformation field
        """

        def perlin_noise(coords, octaves=4, persistence=0.5):
            """Generate Perlin noise with multiple octaves."""
            noise = torch.zeros(b, d, h, w, device=device)
            frequency = 1
            amplitude = 1
            for _ in range(octaves):
                noise += amplitude * self.simplex_noise(frequency * coords)
                frequency *= 2
                amplitude *= persistence
            return noise

        coords = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(-1, 1, d),
                    torch.linspace(-1, 1, h),
                    torch.linspace(-1, 1, w),
                    indexing="ij",
                ),
                dim=-1,
            )
            .to(device)
        )
        coords = coords.unsqueeze(0).expand(b, -1, -1, -1, -1)

        flow = torch.stack(
            [perlin_noise(coords), perlin_noise(coords), perlin_noise(coords)], dim=1
        )

        return flow - flow.mean(dim=(2, 3, 4), keepdim=True)

    def simplex_noise(self, x):
        """Generate simplex noise for natural deformation patterns.

        Args:
            x: Input coordinates of shape (B, D, H, W, 3)

        Returns:
            Simplex noise values
        """
        b, d, h, w, _ = x.shape
        x = x.view(-1, 3)

        dot = lambda a, b: torch.sum(a * b, dim=-1)

        corners = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            device=x.device,
        )

        noise = torch.zeros(x.shape[0], device=x.device)
        for corner in corners:
            grid = x.floor() + corner
            P = x - grid
            N = torch.exp(-dot(P, P) / 0.5)
            grad = torch.randn_like(grid)
            grad = grad / grad.norm(dim=-1, keepdim=True)
            noise += N * dot(grad, P)

        return noise.view(b, d, h, w)

    def spatially_varying_gaussian_filter(self, input, sigma_range):
        """Apply spatially-varying Gaussian filtering.

        Args:
            input: Input tensor
            sigma_range: [min_sigma, max_sigma] for Gaussian kernel

        Returns:
            Filtered tensor
        """

        def gaussian_kernel_1d(sigma, kernel_size):
            """Create 1D Gaussian kernel."""
            x = torch.arange(kernel_size) - (kernel_size - 1) / 2
            return torch.exp(-(x**2) / (2 * sigma**2))

        b, c, d, h, w = input.shape

        # Generate different sigma values for each spatial location
        sigma_map = (
            torch.rand(b, 1, d, h, w, device=input.device)
            * (sigma_range[1] - sigma_range[0])
            + sigma_range[0]
        )

        # Create 3D kernel with maximum kernel size
        max_kernel_size = int(4 * sigma_range[1] + 1)
        kernel_x = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_y = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_z = gaussian_kernel_1d(sigma_range[1], max_kernel_size).to(input.device)
        kernel_3d = (
            kernel_x.view(-1, 1, 1) * kernel_y.view(1, -1, 1) * kernel_z.view(1, 1, -1)
        )
        kernel_3d = kernel_3d.view(1, 1, *kernel_3d.shape)

        # Apply different Gaussian filtering to each location
        output = torch.zeros_like(input)
        for i in range(c):
            channel_input = input[:, i : i + 1]
            channel_output = F.conv3d(
                F.pad(channel_input, (max_kernel_size // 2,) * 6, mode="reflect"),
                kernel_3d.expand(1, -1, -1, -1, -1),
                groups=1,
            )
            # Adjust output based on sigma_map
            output[:, i : i + 1] = channel_input + (
                channel_output - channel_input
            ) * (sigma_map - sigma_range[0]) / (sigma_range[1] - sigma_range[0])

        return output

    def mask(self, image):
        """Apply random masking degradation.

        Args:
            image: Input image

        Returns:
            Masked image with 50% random pixels set to zero
        """
        mask = torch.rand_like(image) < 0.5
        return image * mask

    def downsample(self, image):
        """Apply downsampling degradation.

        Args:
            image: Input image

        Returns:
            Downsampled and upsampled image (blurred)
        """
        sigma_down = [1, 2, 2]
        self.spatially_varying_gaussian_filter(image, sigma_down)
        # Downsample
        down = F.interpolate(
            image, scale_factor=0.5, mode="trilinear", align_corners=True
        )
        # Upsample back to original size
        up = F.interpolate(
            down, size=image.shape[2:], mode="trilinear", align_corners=True
        )
        return up

    def noise(self, image):
        """Apply noise degradation simulating low-light conditions.

        Combines Gaussian noise and Poisson noise to simulate photon noise.

        Args:
            image: Input image with values in [0, 1]

        Returns:
            Noisy image clipped to [0, 1]
        """
        # Add Gaussian noise (simulate photon noise in low light)
        noise_level = random.uniform(0.05, 0.15)
        noise = torch.randn_like(image) * noise_level
        noisy = image + noise
        # Ensure noisy >= 0 to avoid negative lambda_poisson
        noisy = torch.clamp(noisy, min=0.0)
        # Simulate Poisson distribution noise for photon counting
        lambda_poisson = noisy * 255  # Convert from [0,1] to [0,255]
        noisy = torch.poisson(lambda_poisson) / 255.0
        return torch.clamp(noisy, 0, 1)


def print_model_details(model):
    """Print detailed model statistics.

    Args:
        model: PyTorch model to analyze
    """
    print(model)
    params_dict = {}
    for name, param in model.named_parameters():
        params_dict[name] = param.numel()
    total_params = sum(params_dict.values())
    trainable_params_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params_dict[name] = param.numel()
    trainable_params = sum(trainable_params_dict.values())
    print(f"Total parameters: {total_params}")
    print(
        f"Trainable parameters: {trainable_params}, Ratio: {trainable_params/total_params*100:.2f}%"
    )
    print(
        f"Top 5 largest layers: {sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]}, "
        f"Ratio: {[x[1]/total_params*100 for x in sorted(params_dict.items(), key=lambda x: x[1], reverse=True)[:5]]}"
    )
    print(
        f"Top 5 largest trainable layers: {sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]}, "
        f"Ratio: {[x[1]/trainable_params*100 for x in sorted(trainable_params_dict.items(), key=lambda x: x[1], reverse=True)[:5]]}"
    )
