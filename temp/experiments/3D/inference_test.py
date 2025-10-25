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
    
    config.data_dir = './data/HIPSC/split/'
    config.batch_size = 2
    config.use_checkpoint = False
    config.checkpoint_num=0
    config.device=None
    config.dtype=None
    config.num_workers = 4 
    return config

if __name__ == "__main__":
    start_time = time.time()
    config = get_mamba_B_config() 
    model = MODELS.Orochi_Pretrain(config).cuda()
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