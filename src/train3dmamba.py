import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

import os, utils, glob, math
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import ours_mamba as MODELS
from tqdm import tqdm
from natsort import natsorted

import socket
import ml_collections
import time

# 定义顶层的 CollateFn 类
class CollateFn:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, batch):
        processed_batch = [random_crop(item, self.target_size) for item in batch]
        return torch.stack(processed_batch, dim=0)

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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

def save_checkpoint(state, config, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    state['config'] = config
    torch.save(state, os.path.join(save_dir, filename))
    model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(os.path.join(save_dir, '*')))

def cleanup():
    dist.destroy_process_group()

class HIPSCDataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.img_size = img_size  # (D, H, W)
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        return [f for f in os.listdir(self.root_dir) if (f.endswith('.tiff') or f.endswith('.npy')) and 'c4' not in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        
        # 读取预处理后的图像
        if sample_path.endswith('.tiff'):
            raw_image = tifffile.imread(sample_path)
        elif sample_path.endswith('.npy'):
            raw_image = np.load(sample_path)
        else:
            raise ValueError(f"不支持的文件格式: {sample_path}")
        
        # 转换为 PyTorch tensor 并添加通道维度
        image_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)
        
        # 执行上采样
        image_tensor = self.upsample(image_tensor)
        
        return image_tensor

    def upsample(self, image_tensor):
        d, h, w = image_tensor.shape[1:]  # 注意：现在shape是(C, D, H, W)
        td, th, tw = self.img_size

        # 计算缩放因子
        d_factor = max(1, td / d)
        h_factor = max(1, th / h)
        w_factor = max(1, tw / w)

        # 如果需要上采样
        if d_factor > 1 or h_factor > 1 or w_factor > 1:
            image_tensor = F.interpolate(image_tensor.unsqueeze(0), 
                                         size=(int(d*d_factor), int(h*h_factor), int(w*w_factor)),
                                         mode='trilinear', 
                                         align_corners=False).squeeze(0)

        return image_tensor

def random_crop(image_tensor, target_size):
    _, d, h, w = image_tensor.shape
    td, th, tw = target_size

    # 随机裁剪
    if d > td:
        start_d = torch.randint(0, d - td + 1, (1,)).item()
        image_tensor = image_tensor[:, start_d:start_d+td]
    if h > th:
        start_h = torch.randint(0, h - th + 1, (1,)).item()
        image_tensor = image_tensor[:, :, start_h:start_h+th]
    if w > tw:
        start_w = torch.randint(0, w - tw + 1, (1,)).item()
        image_tensor = image_tensor[:, :, :, start_w:start_w+tw]

    # 确保输出尺寸正确
    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=target_size, mode='trilinear', align_corners=False).squeeze(0)

    return image_tensor

# 移除原来的 create_collate_fn 函数

def get_dataloader(config, is_train=True):
    dataset = HIPSCDataset(config.data_dir, img_size=config.img_size)
    sampler = DistributedSampler(dataset) if is_train else None
    
    # 使用顶层定义的 CollateFn 类
    collate = CollateFn(config.img_size)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate
    )

def train(rank, world_size, gpu_ids, config, port):
    setup(rank, world_size, port)
    
    torch.cuda.set_device(gpu_ids[rank])
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    
    save_dir = config.save_dir
    
    if rank == 0:
        if not os.path.exists(save_dir+'experiments/'):
            os.makedirs(save_dir+'experiments/')
        if not os.path.exists(save_dir+'experiments/'+'logs/'):
            os.makedirs(save_dir+'experiments/'+'logs/')
        if not os.path.exists(save_dir+'experiments/'+'visualizations/'):
            os.makedirs(save_dir+'experiments/'+'visualizations/')
        logger = Logger(save_dir+'experiments/'+'logs/')
        sys.stdout = logger
        sys.stderr = logger
    
    model = MODELS.MambaULight(config).to(device)
    model = DDP(model, device_ids=[gpu_ids[rank]], output_device=gpu_ids[rank], find_unused_parameters=True)
    
    if rank == 0:
        print(f'Configuration: {config}')
        MODELS.print_model_details(model)
    
    train_loader = get_dataloader(config, is_train=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.max_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = WarmupCosineSchedule(optimizer, 
                                     warmup_steps=warmup_steps, 
                                     t_total=total_steps, 
                                     warmup_start_factor=config.warmup_start_factor)
    
    writer = SummaryWriter(log_dir=save_dir+'experiments/'+'logs/') if rank == 0 else None
    save_dir = save_dir+'experiments/'

    global_step = 0
    for epoch in range(config.max_epoch):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        loss_all = utils.AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.max_epoch}', disable=rank != 0)

        for idx, source in enumerate(progress_bar):
            source = source.to(device)
            optimizer.zero_grad()
            
            logits, aux_loss = model(source)
            flat_aux_loss = utils.flatten_loss_dict(aux_loss)
            loss = sum(flat_aux_loss.values())  # 此处已假设 aux_loss 已经是扁平化字典
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            loss_all.update(loss.item(), source.numel())
            
            if rank == 0:
                postfix_dict = {
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                }
                for key, value in flat_aux_loss.items():
                    postfix_dict[key] = f'{value.item():.4f}'
                progress_bar.set_postfix(postfix_dict)

                if global_step % config.save_steps == 0:
                    vis_save_path = os.path.join(save_dir, 'visualizations', f'epoch_{epoch}_iter_{global_step}')
                    utils.visualize_logits(logits, output_folder=vis_save_path, verbose=False)

            if idx % 50 == 0:
                torch.cuda.empty_cache()

            global_step += 1

        if rank == 0:
            writer.add_scalar('Loss/train', loss_all.avg, epoch)
            print(f'Epoch {epoch} loss {loss_all.avg:.4f}')
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, config, save_dir=save_dir, filename=f'MambaULight_epoch_{epoch+1}.pth.tar')

    if rank == 0:
        writer.close()
        sys.stdout.close()
    
    cleanup()

def get_mamba_B_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.out_indices = (0, 1, 2, 3)
    config.decoder_head_chan = 64
    
    config.img_size = (32, 256, 256)
    config.grid_size = (32, 256, 256)
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
    
    config.gpu_ids = [2, 3, 4, 5, 6, 7]
    config.checkpoint_dir = None
    config.data_dir = '/root/daigaole/data/HIPSC_1T/'
    config.save_dir = f'/root/daigaole/outputs/foundation_mamba_biomed/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}/'

    config.batch_size = 4
    config.lr = 0.0001
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.warmup_start_factor = 0.01
    config.max_epoch = 50
    config.num_workers = 8
    config.save_steps = 10
    return config

def main():
    config = get_mamba_B_config()
    num_gpus = len(config.gpu_ids)  #使用所有选定的卡
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # 或者您系统中实际的网络接口名称
    port = get_free_port()
    
    # 注意：这里 gpu_ids 参数传递的是 [0, 1, 2, 3, 4, 5]，因为 CUDA_VISIBLE_DEVICES 会重新映射设备ID
    mp.spawn(train, args=(num_gpus, list(range(num_gpus)), config, port), nprocs=num_gpus, join=True)

if __name__ == "__main__":
    main()