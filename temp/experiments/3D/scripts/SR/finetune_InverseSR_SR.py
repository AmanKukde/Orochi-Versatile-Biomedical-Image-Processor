import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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


def inference_and_visualize(model, test_loader, config, device, epoch, global_step):
    model.eval()
    all_psnr = []
    
    vis_dir = os.path.join(config.save_dir, 'experiments/visualizations', f'epoch_{epoch}_iter_{global_step}')
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (source, target) in enumerate(tqdm(test_loader)):
            source = source.to(device)
            target = target.to(device)

            logits, _ = model(source, target)
            restored = logits['restored']
            
            for i in range(source.size(0)):
                psnr_val = utils.psnr(target[i].cpu().numpy(), restored[i].cpu().numpy())
                all_psnr.append(psnr_val)
                
                plt.figure(figsize=(15, 5))
                mid_slice = config.img_size[0] // 2
                
                plt.subplot(131)
                plt.imshow(source[i, 0, mid_slice].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                plt.title('Source')
                plt.colorbar()
                
                plt.subplot(132)
                plt.imshow(restored[i, 0, mid_slice].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                plt.title(f'Restored (PSNR: {psnr_val:.2f})')
                plt.colorbar()
                
                plt.subplot(133)
                plt.imshow(target[i, 0, mid_slice].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                plt.title('Target')
                plt.colorbar()
                
                sample_idx = batch_idx * config.batch_size + i
                plt.savefig(os.path.join(vis_dir, f'sample_{sample_idx}_psnr_{psnr_val:.2f}.png'))
                plt.close()
    avg_psnr = np.mean(all_psnr)
    print(f'Average PSNR: {avg_psnr:.2f}')
    torch.cuda.empty_cache()

    return avg_psnr, avg_psnr

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
    print(f"Cached memory in loading model: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    start_epoch = 0
    best_psnr = 0
    global_step = 0

    train_set = datasets.InverseSRData(config, is_train=True)
    test_set = datasets.InverseSRData(config, is_train=False)

    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set,
                            batch_size=config.batch_size,
                            sampler=train_sampler,
                            num_workers=config.num_workers,
                            pin_memory=True)

    # val_sampler = DistributedSampler(test_set)
    val_loader = DataLoader(test_set,
                        batch_size=1,
                        # sampler=val_sampler,
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
                _, aux_loss = model(source, target)  

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
            # Free memory from train_loader, and load val_loader
            model.eval()
            torch.cuda.empty_cache()
            print(f"Cached memory after empty cache: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
            
            avg_psnr, _ = inference_and_visualize(
                model, val_loader,
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
    config.img_size = (224, 160, 160)
    config.patch_size = 4
    config.pat_merg_rf = 2
    config.in_chans = 2
    config.embed_dim = 128
    config.depths = (4, 4, 4, 4)
    config.drop_rate = 0
    config.drop_path_rate = 0.2
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
    config.decoder_mode = '3d'
    config.decoder_head_chan = 64
    config.head_sparsity = 0.0


    #training
    config.if_resume = False
    config.finetune_mode = 'SR' # 'reg', 'fuse', 'SR', 'IR'
    config.load_modules = {
        # This config is to just finetue decoder.
        'load': ['encoder', 'decoder'],
        'froze': ['encoder'],
        'unfroze_from_froze': ['norm', 'bias'],
        # This config is to finetue all, referring to "full" finetune in paper.
        # 'froze': [],
        # 'unfroze_from_froze': []
    }  
    config.batch_size = 1
    config.lr = 0.0001
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.warmup_start_factor = 0.01
    config.max_epoch = 301
    config.gpu_ids = [0] # Set for gpu id, usually 0 if you only have one gpu.
    config.num_workers = min(multiprocessing.cpu_count() * 2, 16)
    config.save_steps = 1 
    config.losses = {
        "mse": (nn.MSELoss(), 1.0),
        "ssim": (losses.SSIM3D(),1.0), 
    }

    config.train_length = None
    config.val_length = None

    #path
    config.checkpoint_dir = './checkpoints/3D/SR/checkpoint.pth.tar'
    config.data_dir = './data_inverseSR' 
    config.save_dir = f'./Experiment/InverseSR/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}/'
    config.down_factor = 8

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