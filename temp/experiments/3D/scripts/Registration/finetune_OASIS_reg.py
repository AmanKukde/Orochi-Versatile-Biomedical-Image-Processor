import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
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
    best_dsc = 0
    global_step = 0

    train_composed = transforms.Compose([
                                         datasets.NumpyType((np.float32, np.int16)),
                                         ],)

    val_composed = transforms.Compose([
                                         datasets.NumpyType((np.float32, np.int16)),
                                         ],)

    train_set = datasets.OASISBrainDataset(glob.glob(config.train_dir + '*.pkl'),
                                           transforms=train_composed)
    val_set = datasets.OASISBrainInferDataset(glob.glob(config.val_dir + '*.pkl'),
                                              transforms=val_composed)
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=config.batch_size,
                              sampler=train_sampler, 
                              num_workers=config.num_workers,
                              pin_memory=True,
                              prefetch_factor=2)
    val_loader = DataLoader(val_set,
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
            best_dsc = checkpoint['best_DSC']
            global_step = start_epoch * len(train_loader)  # 估算 global_step
            
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
                            if any(module in k for module in config.load_modules)}

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
    
    reg_model = utils.register_model(config.img_size, 'nearest').to(device)
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear').to(device)

    writer = SummaryWriter(log_dir=save_dir+'experiments/logs/') if rank == 0 else None
    save_dir = save_dir+'experiments/'

    for epoch in range(start_epoch, config.max_epoch):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        loss_all = utils.AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.max_epoch}', disable=rank != 0)

        for idx, data in enumerate(progress_bar):
            try:
                data = [t.to(device) for t in data]
                source, target, source_seg, target_seg = data
                
                source_seg_oh = F.one_hot(source_seg.long(), num_classes=36).squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
                logits, aux_loss = model(source, target)
                def_segs = []
                for i in range(36):
                    def_seg = model.module.spatial_trans(source_seg_oh[:, i:i + 1, ...].float(),
                                                  logits["flow"].float())
                    def_segs.append(def_seg)
                def_seg = torch.cat(def_segs, dim=1)
                aux_loss["dice"] = config.losses["dice"][0](def_seg, target_seg.long())    
                optimizer.zero_grad()
                flat_aux_loss = utils.flatten_loss_dict(aux_loss)
                loss = sum(flat_aux_loss.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                traget_seg_oh = F.one_hot(target_seg.long(), num_classes=36).squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
                logits, aux_loss = model(target, source)
                def_segs = []
                for i in range(36):
                    def_seg = model.module.spatial_trans(traget_seg_oh[:, i:i + 1, ...].float(),
                                                  logits["flow"].float())
                    def_segs.append(def_seg)
                def_seg = torch.cat(def_segs, dim=1)
                aux_loss["dice"] = config.losses["dice"][0](def_seg, source_seg.long())    
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
            eval_dsc = utils.AverageMeter()
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc=f'Validation Epoch {epoch}', leave=False)
                for idx, data in enumerate(val_progress):
                    data = [t.to(device) for t in data]
                    source, target, source_seg, target_seg = data
                    logits, aux_loss = model(source, target)
                    deformed_move, flow = logits['registered'], logits['flow']
                    grid_img = utils.mk_grid_img(8, 1, config.img_size).to(device)
                    deformed_seg = reg_model([source_seg.float(), flow])
                    deformed_grid = reg_model_bilin([grid_img.float(), flow])
                    dsc = utils.dice_val_VOI(deformed_seg.long(), target_seg.long())
                    eval_dsc.update(dsc.item(), target.size(0))
                    val_progress.set_postfix({'DSC': f'{dsc.item():.4f}', 'Avg DSC': f'{eval_dsc.avg:.4f}'})
                    # 保存图像
                    plt.figure(figsize=(20, 16))
                    plt.subplot(141)
                    plt.imshow(source_seg[0, 0, 48].cpu().numpy(), cmap='gray')
                    plt.title('Move seg')
                    plt.subplot(142)
                    plt.imshow(target_seg[0, 0, 48].cpu().numpy(), cmap='gray')
                    plt.title('Fix seg')
                    plt.subplot(143)
                    plt.imshow(deformed_seg[0, 0, 48].cpu().numpy(), cmap='gray')
                    plt.title('Deformed seg')
                    plt.subplot(144)
                    plt.imshow(deformed_grid[0, 0, 48].cpu().numpy(), cmap='gray')
                    plt.title('Deformed Grid')
                    plt.subplot(341)
                    plt.imshow(flow[0, 0, 48].cpu().numpy(), cmap='gray')
                    plt.title('flow')
                    plt.subplot(342)
                    plt.imshow(source[0, 0, 48].cpu().numpy(), cmap='hot')
                    plt.title('Move')
                    plt.subplot(343)
                    plt.imshow(target[0, 0, 48].cpu().numpy(), cmap='hot')
                    plt.title('Fix')
                    plt.subplot(344)
                    plt.imshow(deformed_move[0, 0, 48].cpu().numpy(), cmap='hot')
                    plt.title('Deformed Move')
                    vis_save_path = os.path.join(save_dir,
                                                    'visualizations',
                                                    f'epoch_{epoch}_iter_{global_step}',
                                                    f'DSC_{dsc.item():.4f}_idx_{idx}.png')
                    if not os.path.exists(os.path.dirname(vis_save_path)):
                        os.makedirs(os.path.dirname(vis_save_path))
                    plt.savefig(vis_save_path)
                    plt.close()
            print(f'Epoch {epoch}, Best DSC {best_dsc}, Avg DSC {eval_dsc.avg}')
            
            if config.wandb_key:
                wandb.log({
                    'Validation/Avg_DSC': eval_dsc.avg,
                    'Validation/Best_DSC': best_dsc
                }, step=global_step)

            is_best = eval_dsc.avg > best_dsc
            best_dsc = max(eval_dsc.avg, best_dsc)
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_DSC': best_dsc,
            }, config, is_best, save_dir=save_dir, filename=f'Orochi_epoch_{epoch+1}_DSC_{eval_dsc.avg:.4f}.pth.tar')

    if rank == 0:
        writer.close()
        sys.stdout.close()
        if config.wandb_key:
            wandb.finish()
    
    utils.cleanup()

def get_orochi_B_config():
    config = ml_collections.ConfigDict()
    #encoder
    config.img_size = (160, 192, 224)
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
    #training
    config.if_resume = False
    config.finetune_mode = 'reg' # 'reg', 'fus', 'SR', 'IR'
    config.load_modules = {
        # This config is to just finetue decoder.
        'load': ['encoder', 'decoder'],
        'froze': ['encoder'],
        'unfroze_from_froze': ['norm', 'bias'],
        # This config is to finetue all, referring to "full" finetune in paper.
        # 'froze': [],
        # 'unfroze_from_froze': []
    }  
    config.batch_size = 2
    config.lr = 0.0001
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.warmup_start_factor = 0.01
    config.max_epoch = 301
    config.gpu_ids = [0]
    num_cpus = multiprocessing.cpu_count()
    config.num_workers = min(num_cpus * 2, 16)
    config.save_steps = 5
    config.losses = {
        "mse": (nn.MSELoss(), 1.0),
        "ssim": (losses.SSIM3D(), 0.0), 
        "ncc": (losses.NCC_vxm(), 1.0),
        "grad": (losses.Grad3d(penalty='l2'), 1.0),
        "dice": (losses.DiceLoss(), 1.0),
    }
    #path
    # config.checkpoint_dir = '/root/daigaole/checkpoints/HIPSC_HIPCT_50epoch.tar'
    config.checkpoint_dir = "./checkpoints/3D/Registration/checkpoint.pth.tar"
    config.train_dir = './data_OASISReg/OASIS_L2R_2021_task03/All/'
    config.atlas_dir = './data_OASISReg/IXI_data/atlas.pkl'
    config.val_dir = './data_OASISReg/OASIS_L2R_2021_task03/Test/'
    config.test_dir = './data_OASISReg/OASIS_L2R_2021_task03/Test/'
    config.save_dir = f'./Experiment/OASISREG/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}/'
    
    # wandb 配置
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