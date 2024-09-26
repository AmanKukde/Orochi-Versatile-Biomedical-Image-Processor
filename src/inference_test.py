import torch
import ml_collections
import ours_mamba as MODELS
import os
from torch.utils.data import Dataset, DataLoader
import tifffile
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.nn.functional as F

class HIPSCDataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.img_size = img_size  # (D, H, W)
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        return [f for f in os.listdir(self.root_dir) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        
        # 读取预处理后的图像
        raw_image = tifffile.imread(sample_path)
        
        # 上采样和随机裁剪到指定大小
        raw_image = self.upsample_and_crop(raw_image)
        
        return raw_image

    def upsample_and_crop(self, image):
        d, h, w = image.shape
        td, th, tw = self.img_size

        # 转换为 PyTorch tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

        # 计算缩放因子
        d_factor = max(1, td / d)
        h_factor = max(1, th / h)
        w_factor = max(1, tw / w)

        # 如果需要上采样
        if d_factor > 1 or h_factor > 1 or w_factor > 1:
            # 使用 PyTorch 的插值函数进行上采样
            image_tensor = F.interpolate(image_tensor, 
                                         size=(int(d*d_factor), int(h*h_factor), int(w*w_factor)),
                                         mode='trilinear', 
                                         align_corners=False)

        # 随机裁剪
        d, h, w = image_tensor.shape[2:]  # 更新尺寸
        if d > td:
            start_d = torch.randint(0, d - td + 1, (1,)).item()
            image_tensor = image_tensor[:, :, start_d:start_d+td]
        if h > th:
            start_h = torch.randint(0, h - th + 1, (1,)).item()
            image_tensor = image_tensor[:, :, :, start_h:start_h+th]
        if w > tw:
            start_w = torch.randint(0, w - tw + 1, (1,)).item()
            image_tensor = image_tensor[:, :, :, :, start_w:start_w+tw]

        # 确保输出尺寸正确
        image_tensor = F.interpolate(image_tensor, size=self.img_size, mode='trilinear', align_corners=False)

        return image_tensor.squeeze(0)  # 移除批次维度，保留通道维度

def get_dataloader(config):
    dataset = HIPSCDataset(config.data_dir, img_size=config.img_size)
    return DataLoader(dataset, 
                      batch_size=config.batch_size, 
                      shuffle=True, 
                      num_workers=config.num_workers,
                      pin_memory=True)

def visualize_results(raw, reg_source, fus_source, SR_source, IR_source, reged, fused, SRed, IRed):
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    
    # 选择中间帧
    mid_frame = raw.shape[2] // 2  # 注意：这里改成了2，因为现在的shape是(B, C, D, H, W)
    
    # 原始输入
    axs[0, 0].imshow(raw[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[0, 0].set_title('Original Input')
    
    # 退化后的图像
    axs[0, 1].imshow(reg_source[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[0, 1].set_title('Deformed')
    
    axs[0, 2].imshow(fus_source[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[0, 2].set_title('Masked')
    
    axs[1, 0].imshow(SR_source[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[1, 0].set_title('Downsampled')
    
    axs[1, 1].imshow(IR_source[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[1, 1].set_title('Noisy')
    
    # 模型预测的图像
    axs[1, 2].imshow(reged[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[1, 2].set_title('Registered')
    
    axs[2, 0].imshow(fused[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[2, 0].set_title('Fused')
    
    axs[2, 1].imshow(SRed[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[2, 1].set_title('Super-resolutioned')
    
    axs[2, 2].imshow(IRed[0, 0, mid_frame].cpu().numpy(), cmap='gray')
    axs[2, 2].set_title('Restored')
    
    for ax in axs.flatten():
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.savefig('/root/daigaole/outputs/foundation_mamba_biomed/results_visualization.png')
    plt.close()

def get_mamba_B_config():
    '''
    Trainable params: 15,201,579
    '''
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.out_indices = (0, 1, 2, 3)
    config.decoder_head_chan = 16
    
    config.img_size = (160, 192, 224)
    config.grid_size = (160, 192, 224)
    config.patch_size = 4
    config.pat_merg_rf = 2
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (4, 4, 4, 4)
    config.window_size = (5, 6, 7)
    config.drop_rate = 0
    config.drop_path_rate = 0.2
    config.ssm_cfg=None
    config.norm_epsilon=1e-5
    config.initializer_cfg=None
    config.fused_add_norm=True
    config.rms_norm=True
    config.residual_in_fp32=True
    config.bimamba=True
    config.patch_norm = True      
    config.use_checkpoint = False
    
    config.data_dir = '/root/daigaole/data/HIPSC/split/'
    config.batch_size = 2
    config.use_checkpoint = False
    config.checkpoint_num=0
    config.device=None
    config.dtype=None
    config.num_workers = 4  # 添加这一行来指定num_workers
    return config


if __name__ == "__main__":
    start_time = time.time()
    config = get_mamba_B_config() 
    model = MODELS.MambaULight(config).cuda()
    MODELS.print_model_details(model)
    print(f'create model time: {time.time() - start_time:.4f} seconds')
    dataloader = get_dataloader(config)
    print(f'create dataloader time: {time.time() - start_time:.4f} seconds')
    source = next(iter(dataloader)).cuda()
    print(f'load data time: {time.time() - start_time:.4f} seconds')
    model.eval()
    
    with torch.no_grad():
        reg_source, fus_source, SR_source, IR_source, reged, fused, SRed, IRed, aux_loss = model(source)
    print(f'First inference time: {time.time() - start_time:.4f} seconds')
    
    with torch.no_grad():
        reg_source, fus_source, SR_source, IR_source, reged, fused, SRed, IRed, aux_loss = model(source)
    end_time = time.time()
    print(f"Normal inference time: {end_time - start_time:.4f} seconds")
    
    # 可视化结果
    visualize_results(source, reg_source, fus_source, SR_source, IR_source, reged, fused, SRed, IRed)