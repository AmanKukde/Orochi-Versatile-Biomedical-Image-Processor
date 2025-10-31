import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import tifffile
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

import os
import utils, losses
import glob
import math
import datasets
import sys
import numpy as np
import matplotlib.pyplot as plt
import ours_mamba as MODELS
from tqdm import tqdm
from natsort import natsorted

import socket
import multiprocessing
import shutil
import ml_collections
import time


def extract_patches(image, patch_size):
    C, H, W = image.shape
    ph, pw = patch_size
    
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    
    if pad_h > 0 or pad_w > 0:
        padding = (0, pad_w, 0, pad_h)  # left, right, top, bottom
        image = F.pad(image, padding, mode='reflect')
    
    _, H_pad, W_pad = image.shape
    
    patches = []
    positions = []
    for y in range(0, H_pad, ph):
        for x in range(0, W_pad, pw):
            patch = image[:, y:y+ph, x:x+pw]
            patches.append(patch)
            positions.append((y, x))
    
    return patches, positions, (H_pad, W_pad)

def reconstruct_from_patches(patches, positions, original_size, padded_size):
    C, h, w = patches[0].shape
    H_pad, W_pad = padded_size
    H, W = original_size
    
    reconstructed = torch.zeros((C, H_pad, W_pad), device=patches[0].device)
    count = torch.zeros((C, H_pad, W_pad), device=patches[0].device)
    
    for patch, (y, x) in zip(patches, positions):
        reconstructed[:, y:y+h, x:x+w] += patch
        count[:, y:y+h, x:x+w] += 1
    
    reconstructed = reconstructed / (count + 1e-6)
    
    reconstructed = reconstructed[:, :H, :W]
    return reconstructed

class ISOLiverDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.data_dir = config.data_dir
        self.is_train = is_train
        self.img_size = config.img_size
        
        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        if is_train:
            self.original_shape = None
        else:
            self.original_shape = info['original_test_shape']
        
        if is_train:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'train', 'slice_*.npz')))
        else:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'test', 'xz_slice_*.npz')))
            
        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")
        
        sample_data = np.load(self.file_list[0])
        source_shape = sample_data['source'].shape
        print(f"Original data shape: {source_shape}")
        if is_train:
            print(f"Target resize shape: {self.img_size}")

    def resize_2d(self, img):
        img_resized = F.interpolate(
            img.unsqueeze(0),
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        return img_resized

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        source = torch.from_numpy(data['source']).float()
        target = torch.from_numpy(data['target']).float()
        
        if self.is_train:
            source = self.resize_2d(source)
            target = self.resize_2d(target)
            return source, target
        else:
            position = data['position']
            return source, target, position

    def __len__(self):
        return len(self.file_list)

def inference_and_visualize(model, test_dataset, test_loader, config, device, epoch, global_step):
    model.eval()
    all_psnr = []
    all_psnr_noisy = []
    original_shape = test_dataset.original_shape  # (Z, Y, X)
    is_final_epoch = (epoch) == config.max_epoch
    
    if is_final_epoch:
        reconstructed_volume = np.zeros(original_shape, dtype=np.float32)
        gt_volume = np.zeros(original_shape, dtype=np.float32)
    
    vis_dir = os.path.join(config.save_dir, 'experiments/visualizations', f'epoch_{epoch}_iter_{global_step}')
    patches_dir = os.path.join(vis_dir, 'patches')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (source, target, position) in enumerate(tqdm(test_loader)):
            y_pos = position[0].item()  
            
            if not is_final_epoch and y_pos >= 50:
                continue
            
            source = source.to(device)  # (1, C, Z, X)
            target = target.to(device)  # (1, C, Z, X)
            
            source_patches, patch_positions, padded_size = extract_patches(source.squeeze(0), config.img_size)
            target_patches, _, _ = extract_patches(target.squeeze(0), config.img_size)
            
            restored_patches = []
            unrestored_patches = []
            for idx, (source_patch, target_patch) in enumerate(zip(source_patches, target_patches)):
                source_patch = source_patch.unsqueeze(0)
                target_patch = target_patch.unsqueeze(0)
                
                logits, _ = model(source_patch, target_patch)
                restored_patch = logits['restored']
                restored_patches.append(restored_patch.squeeze(0))
                unrestored_patches.append(source_patch.squeeze(0))
                
                if y_pos < 50 and idx % 5 == 0:
                    plt.figure(figsize=(15, 5))
                    plt.subplot(131)
                    plt.imshow(source_patch[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                    plt.title('Source Patch')
                    plt.colorbar()
                    
                    plt.subplot(132)
                    plt.imshow(restored_patch[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                    plt.title('Restored Patch')
                    plt.colorbar()
                    
                    plt.subplot(133)
                    plt.imshow(target_patch[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                    plt.title('Target Patch')
                    plt.colorbar()
                    
                    plt.savefig(os.path.join(patches_dir, f'slice_{y_pos:04d}_patch_{idx:04d}.png'))
                    plt.close()
            
            restored_slice = reconstruct_from_patches(
                restored_patches, 
                patch_positions,
                (source.shape[2], source.shape[3]),
                padded_size
            )
            unrestored_slice = reconstruct_from_patches(
                unrestored_patches, 
                patch_positions,
                (source.shape[2], source.shape[3]),
                padded_size
            )
            
            psnr_noisy = utils.psnr(source.squeeze(0).cpu().numpy(), target.squeeze(0).cpu().numpy())
            psnr_val = utils.psnr(target.squeeze(0).cpu().numpy(), restored_slice.cpu().numpy())
            all_psnr.append(psnr_val)
            all_psnr_noisy.append(psnr_noisy)
            
            if is_final_epoch:
                reconstructed_volume[:, y_pos, :] = restored_slice[0].cpu().numpy()
                gt_volume[:, y_pos, :] = target.squeeze(0)[0].cpu().numpy()
            
            if y_pos < 50:
                plt.figure(figsize=(10, 40))
                
                plt.subplot(411)
                plt.imshow(source.squeeze(0)[0].cpu().numpy(), cmap='hot', vmin=0, vmax=1) 
                psnr_noisy = utils.psnr(target.squeeze(0).cpu().numpy(), source.squeeze(0).cpu().numpy())
                plt.title(f'Input (PSNR: {psnr_noisy:.2f})')
                plt.colorbar()
                
                plt.subplot(412)
                plt.imshow(restored_slice[0].cpu().numpy(), cmap='hot', vmin=0, vmax=1) 
                plt.title(f'Restored (PSNR: {psnr_val:.2f})')
                plt.colorbar()
                
                plt.subplot(413)
                plt.imshow(target.squeeze(0)[0].cpu().numpy(), cmap='hot', vmin=0, vmax=1) 
                plt.title('Target')
                plt.colorbar()
                
                plt.subplot(414)
                diff = np.abs(restored_slice[0].cpu().numpy() - target.squeeze(0)[0].cpu().numpy())
                plt.imshow(diff, cmap='hot', vmin=0, vmax=1)  
                plt.title('Difference')
                plt.colorbar()
                
                plt.tight_layout()  
                plt.savefig(os.path.join(vis_dir, f'slice_{y_pos:04d}_full.png'))
                plt.close()
    
    if is_final_epoch:
        plt.figure(figsize=(15, 5))
        mid_y = original_shape[1] // 2
        
        plt.subplot(131)
        plt.imshow(reconstructed_volume[:, mid_y, :].T, cmap='hot', vmin=0, vmax=1)
        plt.title('Reconstructed')
        plt.colorbar()
        
        plt.subplot(132)
        plt.imshow(gt_volume[:, mid_y, :].T, cmap='hot', vmin=0, vmax=1)
        plt.title('Ground Truth')
        plt.colorbar()
        
        plt.subplot(133)
        diff = np.abs(reconstructed_volume[:, mid_y, :] - gt_volume[:, mid_y, :]).T
        plt.imshow(diff, cmap='hot', vmin=0, vmax=1)
        plt.title('Difference')
        plt.colorbar()
        
        plt.savefig(os.path.join(vis_dir, 'full_volume_middle_slice_comparison.png'))
        plt.close()
    
    avg_psnr = np.mean(all_psnr)
    print(f'Average PSNR: {avg_psnr:.2f}, Average Noisy PSNR: {np.mean(all_psnr_noisy):.2f}')
    
    if is_final_epoch:
        return avg_psnr, reconstructed_volume
    else:
        return avg_psnr, None

def train(rank, world_size, gpu_ids, config, port):
    utils.setup(rank, world_size, port)
    torch.cuda.set_device(gpu_ids[rank])
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    save_dir = config.save_dir
    model = MODELS.Orochi_Finetune(config).to(device)
    if rank == 0:
        os.makedirs(save_dir+'experiments/logs/', exist_ok=True)
        os.makedirs(save_dir+'experiments/visualizations/', exist_ok=True)
        logger = utils.Logger(save_dir+'experiments/logs/')
        sys.stdout = logger
        sys.stderr = logger

    start_epoch = 0
    best_psnr = 0
    global_step = 0

    train_set = ISOLiverDataset(config, is_train=True)
    test_set = ISOLiverDataset(config, is_train=False)

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set,
                            batch_size=config.batch_size,
                            sampler=train_sampler,
                            num_workers=config.num_workers,
                            pin_memory=True)

    val_loader = DataLoader(test_set,
                        batch_size=1, 
                        shuffle=False,
                        num_workers=config.num_workers,
                        pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.max_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = utils.WarmupCosineSchedule(optimizer, 
                                    warmup_steps=warmup_steps, 
                                    t_total=total_steps, 
                                    warmup_start_factor=config.warmup_start_factor)

    if config.if_resume:
        if os.path.isfile(config.checkpoint_dir):
            print(f"=> loading checkpoint '{config.checkpoint_dir}'")
            checkpoint = torch.load(config.checkpoint_dir, map_location=device)
            start_epoch = checkpoint['epoch']
            best_psnr = checkpoint['best_PSNR']
            global_step = start_epoch * len(train_loader)
            
            state_dict = checkpoint['state_dict']
            if "module." in list(state_dict.keys())[0] and "module." not in list(model.state_dict().keys())[0]:
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            elif "module." not in list(state_dict.keys())[0] and "module." in list(model.state_dict().keys())[0]:
                state_dict = {f"module.{k}": v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f"=> loaded checkpoint '{config.checkpoint_dir}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config.checkpoint_dir}'")
    elif config.checkpoint_dir:
        print("##################Checkpoint found##################")
        checkpoint = torch.load(config.checkpoint_dir, map_location=device)
        checkpoint_state_dict = checkpoint['state_dict']

        if "module." in list(checkpoint_state_dict.keys())[0] and "module." not in list(model.state_dict().keys())[0]:
            checkpoint_state_dict = {k.replace("module.", ""): v for k, v in checkpoint_state_dict.items()}
        elif "module." not in list(checkpoint_state_dict.keys())[0] and "module." in list(model.state_dict().keys())[0]:
            checkpoint_state_dict = {f"module.{k}": v for k, v in checkpoint_state_dict.items()}

        selected_state_dict = {k: v for k, v in checkpoint_state_dict.items() 
                            if any(module in k for module in config.load_modules['load'])}

        model.load_state_dict(selected_state_dict, strict=False)
        print(f"Loaded {config.load_modules['load']} from checkpoint: {config.checkpoint_dir}") 

        for name, param in model.named_parameters():
            for froze_module in config.load_modules['froze']:
                for unfroze_module in config.load_modules['unfroze_from_froze']:
                    if froze_module in name:
                        param.requires_grad = False
                    if unfroze_module in name:
                        param.requires_grad = True
        print(f"froze {config.load_modules['froze']}") 
        print(f"unfroze from froze {config.load_modules['unfroze_from_froze']}")  
        del checkpoint
        print("##################Checkpoint loaded##################") 

    model = DDP(model, device_ids=[gpu_ids[rank]], output_device=gpu_ids[rank], find_unused_parameters=True)

    if rank == 0:
        print(f'Configuration: {config}')
        MODELS.print_model_details(model)
        if config.wandb_key:
            wandb.login(key=config.wandb_key)
            wandb.init(project=config.wandb_project,
                    config=config,
                    save_code=True,
                    group="finetune",
                    job_type="train",
                    name=str(config.save_dir.split('/')[-1]),
                    )
            wandb.watch(model, log="all", log_freq=100, log_graph=True)
            print(f"Wandb initialized with run name: {wandb.run.name}")

    writer = SummaryWriter(log_dir=save_dir+'experiments/logs/') if rank == 0 else None
    save_dir = save_dir+'experiments/'

    for epoch in range(start_epoch, config.max_epoch):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        loss_all = utils.AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.max_epoch}', disable=rank != 0)

        for idx, data in enumerate(progress_bar):
            try:
                source, target = [t.to(device) for t in data]
                logits, aux_loss = model(source, target)  
                
                optimizer.zero_grad()
                flat_aux_loss = utils.flatten_loss_dict(aux_loss)
                loss = sum(flat_aux_loss.values())
                loss.backward()
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

                    if config.wandb_key:
                        wandb.log({
                            'Loss/train': loss.item(),
                            'LearningRate': optimizer.param_groups[0]['lr'],
                            **{f'Loss/{key}': value.item() for key, value in flat_aux_loss.items()}
                        }, step=global_step)

                    writer.add_scalar('Loss/train', loss.item(), global_step)
                    for key, value in flat_aux_loss.items():
                        writer.add_scalar(f'Loss/{key}', value.item(), global_step)
                    writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], global_step)

                global_step += 1

            except Exception as e:
                print(f"Error in training loop: {e}")
                continue

        if rank == 0 and epoch % config.save_steps == 0:
            model.eval()
            avg_psnr, reconstructed_volume = inference_and_visualize(
                model, test_set, val_loader,
                config, device, epoch, global_step)
            
            if config.wandb_key:
                wandb.log({
                    'Validation/PSNR': avg_psnr,
                }, step=global_step)

            is_best = avg_psnr > best_psnr
            best_psnr = max(avg_psnr, best_psnr)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_PSNR': best_psnr,
            }, config, is_best, save_dir=save_dir, 
            filename=f'Orochi_epoch_{epoch+1}_PSNR_{avg_psnr:.2f}.pth.tar')
            
            print(f'Epoch {epoch}, Best PSNR {best_psnr:.2f}, Current PSNR {avg_psnr:.2f}')

    if rank == 0:
        writer.close()
        sys.stdout.close()
        if config.wandb_key:
            wandb.finish()

    utils.cleanup()    

def get_orochi_B_config():
    config = ml_collections.ConfigDict()
    #encoder
    config.img_size = (128, 128) 
    config.patch_size = 4
    config.pat_merg_rf = 2
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (4, 4, 4, 4)
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.if_convskip = True
    config.out_indices = (0, 1, 2, 3)
    #mamba
    config.ssm_cfg=None
    config.norm_epsilon=1e-5
    config.initializer_cfg=None
    config.fused_add_norm=True
    config.rms_norm=True
    config.residual_in_fp32=True
    config.patch_norm = True
    config.use_checkpoint = False
    #decoder
    config.decoder_bn = False
    config.decoder_depthseparable = True # This means the decoder is light, if set False, the decoder is dense.
    config.decoder_mode = '2d'
    config.decoder_head_chan = 64
    config.head_sparsity = 0.0

    #training
    config.if_resume = False
    config.finetune_mode = 'IR'
    config.load_modules = {
        # This config is to just finetue decoder.
        'load': ['encoder', 'decoder'],
        'froze': ['encoder'],
        'unfroze_from_froze': ['norm', 'bias'],
        # This config is to finetue all, referring to "full" finetune in paper.
        # 'froze': [],
        # 'unfroze_from_froze': []
    }  
    config.batch_size = 128
    config.lr = 0.0001
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.warmup_start_factor = 0.01
    config.max_epoch = 300
    config.gpu_ids = [0] # Set for gpu id, usually 0 if you only have one gpu.
    config.num_workers = min(multiprocessing.cpu_count() * 2, 16)
    config.save_steps = 5
    config.losses = {
        "mse": (nn.MSELoss(), 1.0),
        "ssim": (losses.SSIM2D(), 1.0), 
        "lpips": (losses.LpipsLoss(config), 1.0),
    }

    #path
    config.checkpoint_dir = './checkpoints/2D/IR/checkpoint.pth.tar'
    config.data_dir = './data_UniFMIR/preprocessed2D/'
    config.save_dir = f'./Experiment/ISOIR2D/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}/'

    # Config for wandb if needed
    config.wandb_key = None
    config.wandb_project = "Orochi"

    return config

def main():
    config = get_orochi_B_config()
    num_gpus = len(config.gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    port = utils.get_free_port()

    mp.spawn(train, args=(num_gpus, list(range(num_gpus)), config, port), nprocs=num_gpus, join=True)
    
if __name__ == "__main__":
    main()