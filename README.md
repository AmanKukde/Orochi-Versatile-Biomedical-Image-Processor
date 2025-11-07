# Orochi
Official Implementation for "Orochi: Versatile Biomedical Image Processor"

Please cite our paper if you use the code and data we provide

```bash
@article{dai2025orochi,
  title={Orochi: Versatile Biomedical Image Processor},
  author={Dai, Gaole and Zhou, Chenghao and Zhou, Yu and Zhang, Rongyu and Zhang, Yuan and Hou, Chengkai and Huang, Tiejun and Chen, Jianxu and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2509.22583},
  year={2025}
}
```

## Environment

### 1. Create the environment

You can clone this repo via git
```bash
git clone git@github.com:jnjnnjzch/foundation-mamba-biomed.git
```
Then create the environment via this command:
```bash
conda env create -f environment.yaml
```
### 2. Modify the codes of specific package

2.1 You might get the site-package path by running this command in the created python enviroment.

```bash
python -c "import site; print(site.getsitepackages())"
```

2.2 Then, you may need to add codes at the 777th line in `[site-package path]/mamba_ssm/ops/triton/ssd_combined.py`. 
```python
        assert zxbcdt.shape == (batch, seqlen, 2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads)
        assert dt_bias.shape == (nheads,)
        assert A.shape == (nheads,)
        zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)
        xBC = xBC.contiguous() # <------------- Add this line of code!!!
        seq_idx = seq_idx.contiguous() if seq_idx is not None else None
        xBC_conv = rearrange(
            causal_conv1d_cuda.causal_conv1d_fwd(rearrange(xBC, "b s d -> b d s"),
                                                 conv1d_weight, conv1d_bias, seq_idx, None, None, activation in ["silu", "swish"]),
            "b d s -> b s d"
        )
```

## Pretrain Orochi

### 🧬 Dataset
- **hiPSC 2D**: [eternalaudrey/hipsc_2d](https://huggingface.co/datasets/eternalaudrey/hipsc_2d)  
- **hiPSC 3D**: [eternalaudrey/hipsc_3d](https://huggingface.co/datasets/eternalaudrey/hipsc_3d)  
- **HiP-CT 2D**: [eternalaudrey/hipct_2d](https://huggingface.co/datasets/eternalaudrey/hipct_2d)  
- **IDR 2D**: [eternalaudrey/idr_2d](https://huggingface.co/datasets/eternalaudrey/idr_2d)  
- **IDR Raw**:  
  - [eternalaudrey/idr-01](https://huggingface.co/datasets/eternalaudrey/idr-01)  
  - [eternalaudrey/idr-02](https://huggingface.co/datasets/eternalaudrey/idr-02)  
  - [eternalaudrey/idr-03](https://huggingface.co/datasets/eternalaudrey/idr-03)  
  - [eternalaudrey/idr-04](https://huggingface.co/datasets/eternalaudrey/idr-04)

### 💾 Checkpoints
- **2D Pretrained Model**: [eternalaudrey/mamba-fm-2d-ckpt](https://huggingface.co/eternalaudrey/mamba-fm-2d-ckpt)  
- **3D Pretrained Model**: [eternalaudrey/mamba-fm-3d-ckpt](https://huggingface.co/eternalaudrey/mamba-fm-3d-ckpt)



### 1. Prepare the datasets

For finetune, you can download dataset according to different tasks.

The dataset used in our finetune process can be download as follows:
- Super resolution
    - 2D: [Unifmir Zendo repository](https://zenodo.org/records/8401470), download the data with "BioSR" prefix and unzip into the corresponding folders.
    - 3D: InverseSR

- Isotropic restoration
    - 2D: [Unifmir Zendo repository](https://zenodo.org/records/8401470), download the `Isotropic_Liver.tgz` and unzip into the corresponding folders. You may need to use the provided preprocess code in `./data_preprocess` to manage the data.

- Fusion
    - 2D: Download datasets form: http://www.med.harvard.edu/aanlib/, and copy to the corresponding folders. You may need to use the provided preprocess code in `./data_preprocess` to manage the data. ([this link](https://github.com/xianming-gu/ASFE-Fusion).)

- Registration
    - 3D: [Transmorph githubrepository](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git) Downlaod datasets from https://drive.google.com/uc?export=download&id=1BdEaylMDpeXtyuX5QH8l_Ut4OgenKss4


### 2. Finetune models

You can run the codes in `./experiments/[2D or 3D]/scripts/[Task names]` folder to finetune the pretrained models. 

### 3. Test models

You can run the jupyter notebooks in `./experiments/[2D or 3D]/scripts/[Task names]` folder to evaluate the finetuned models.

