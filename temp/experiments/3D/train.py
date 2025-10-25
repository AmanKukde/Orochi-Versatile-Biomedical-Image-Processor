import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

import os, utils, glob, math, datasets
import sys
import ours_mamba as MODELS
from tqdm import tqdm
from natsort import natsorted

import socket
import multiprocessing
import shutil
import ml_collections
import time
import wandb

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

def save_checkpoint(state, config, is_best, save_dir='models', filename='checkpoint.pth.tar', max_model_num=4):
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(save_dir, 'model_best.pth.tar'))
    if len(glob.glob(os.path.join(save_dir, '*.pth.tar'))) > max_model_num:
        oldest_file = min(glob.glob(os.path.join(save_dir, '*.pth.tar')), key=os.path.getctime)
        os.remove(oldest_file)

def cleanup():
    dist.destroy_process_group()
    
def train(rank, world_size, gpu_ids, config, port):
    setup(rank, world_size, port)
    
    torch.cuda.set_device(gpu_ids[rank])
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    
    save_dir = config.save_dir
    
    if rank == 0:
        os.makedirs(save_dir+'experiments/logs/', exist_ok=True)
        os.makedirs(save_dir+'experiments/visualizations/', exist_ok=True)
        logger = Logger(save_dir+'experiments/logs/')
        sys.stdout = logger
        sys.stderr = logger
    
    model = MODELS.Orochi_Pretrain(config).to(device)
    model = DDP(model, device_ids=[gpu_ids[rank]], output_device=gpu_ids[rank], find_unused_parameters=True)
    
    if rank == 0:
        print(f'Configuration: {config}')
        MODELS.print_model_details(model)
        # 初始化 wandb
        if config.wandb_key:
            wandb.login(key=config.wandb_key)
            wandb.init(project=config.wandb_project,
                       config=config,
                       save_code=True,
                       group="pretrain",
                       job_type="train",
                       name=str(config.save_dir.split('/')[-1]),
                       )
            wandb.watch(model, log="all", log_freq=100, log_graph=True)
            print(f"Wandb initialized with run name: {wandb.run.name}")
    
    
    train_loader = datasets.get_dataloader(config, is_train=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    total_steps = len(train_loader) * config.max_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = WarmupCosineSchedule(optimizer, 
                                    warmup_steps=warmup_steps, 
                                    t_total=total_steps, 
                                    warmup_start_factor=config.warmup_start_factor)

    
    writer = SummaryWriter(log_dir=save_dir+'experiments/logs/') if rank == 0 else None
    save_dir = save_dir+'experiments/'

    global_step = 0
    best_loss = float('inf')

    for epoch in range(config.max_epoch):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        loss_all = utils.AverageMeter()
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.max_epoch}', disable=rank != 0)

        for idx, source in enumerate(progress_bar):
            try:
                source = source.to(device)
                
                optimizer.zero_grad()
                logits, aux_loss = model(source)
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

                    if global_step % config.save_steps == 0:
                        vis_save_path = os.path.join(save_dir, 'visualizations', f'epoch_{epoch}_iter_{global_step}_loss_{loss}')
                        utils.visualize_logits(logits, output_folder=vis_save_path, verbose=False)

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

        if rank == 0:
            print(f'Epoch {epoch} loss {loss_all.avg:.4f}')
            
            is_best = loss_all.avg < best_loss
            best_loss = min(loss_all.avg, best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_loss': best_loss,
            }, config, is_best, save_dir=save_dir, filename=f'Orochi_epoch_{epoch+1}_loss_{loss_all.avg:.4f}.pth.tar')

    if rank == 0:
        writer.close()
        sys.stdout.close()
    
    cleanup()


def get_orochi_B_config():
    config = ml_collections.ConfigDict()
    #encoder
    config.img_size = (32, 224, 224)
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
    config.decoder_depthseparable = False
    config.decoder_mode = '3d'
    config.decoder_head_chan = 64
    #training
    config.batch_size = 3
    config.lr = 0.0005
    config.weight_decay = 0.01
    config.warmup_ratio = 0.1
    config.warmup_start_factor = 0.01
    config.max_epoch = 50
    config.gpu_ids = [0,1,2,3,4,5,6,7]
    num_cpus = multiprocessing.cpu_count()
    config.num_workers = min(num_cpus * 2, 16)
    config.save_steps = 10
    #path
    config.checkpoint_dir = None
    config.data_dir = './data/HIPSC_1T/'
    config.save_dir = f'./outputs/{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}/'
    config.wandb_key = None
    config.wandb_project = "Orochi"
    
    return config

def main():
    config = get_orochi_B_config()
    num_gpus = len(config.gpu_ids) 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.gpu_ids))
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  
    port = get_free_port()
    
    mp.spawn(train, args=(num_gpus, list(range(num_gpus)), config, port), nprocs=num_gpus, join=True)

if __name__ == "__main__":
    main()