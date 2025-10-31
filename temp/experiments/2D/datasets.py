import os
import numpy as np
import torch
import tifffile
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from PIL import Image
from torchvision import transforms
import random




class CollateFn:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, batch):
        processed_batch = [random_crop(item, self.target_size) for item in batch]
        return torch.stack(processed_batch, dim=0)

class Pretrain_Dataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.img_size = img_size  # (H, W)
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        return [f for f in os.listdir(self.root_dir) if (f.endswith('.tiff') or f.endswith('.npy'))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = os.path.join(self.root_dir, self.samples[idx])
        
        if sample_path.endswith('.tiff'):
            raw_image = tifffile.imread(sample_path)
        elif sample_path.endswith('.npy'):
            raw_image = np.load(sample_path)
        else:
            raise ValueError(f"Unsupported: {sample_path}")
        
        if raw_image.ndim == 3:
            raw_image = raw_image[raw_image.shape[0]//2]
        elif raw_image.ndim > 3:
            raise ValueError(f"Unexpected image dimensions: {raw_image.shape}")
        
        image_tensor = torch.from_numpy(raw_image).float().unsqueeze(0)
        
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        
        image_tensor = self.upsample(image_tensor)
        
        return image_tensor

    def upsample(self, image_tensor):
        h, w = image_tensor.shape[1:]
        th, tw = self.img_size

        h_factor = max(1, th / h)
        w_factor = max(1, tw / w)

        if h_factor > 1 or w_factor > 1:
            image_tensor = F.interpolate(image_tensor.unsqueeze(0), 
                                         size=(int(h*h_factor), int(w*w_factor)),
                                         mode='bilinear', 
                                         align_corners=False).squeeze(0)

        return image_tensor

def random_crop(image_tensor, target_size):
    _, h, w = image_tensor.shape
    th, tw = target_size

    if h > th:
        start_h = torch.randint(0, h - th + 1, (1,)).item()
        image_tensor = image_tensor[:, start_h:start_h+th]
    if w > tw:
        start_w = torch.randint(0, w - tw + 1, (1,)).item()
        image_tensor = image_tensor[:, :, start_w:start_w+tw]

    image_tensor = F.interpolate(image_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

    return image_tensor

def get_dataloader(config, is_train=True):
    dataset = Pretrain_Dataset(config.data_dir, img_size=config.img_size)
    sampler = DistributedSampler(dataset) if is_train else None
    
    collate = CollateFn(config.img_size)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(sampler is None and is_train),
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=True,
    )

class ISOLiverDataset(Dataset):
    def __init__(self, config, is_train=True):

        self.data_dir = config.data_dir
        self.is_train = is_train
        self.img_size = config.img_size
        
        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        if is_train:
            self.original_shape = info['original_train_shape']
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
        print(f"Target resize shape: {self.img_size}")

    def resize_2d(self, img):
        # img shape: (C, H, W)
        c, h, w = img.shape
        img_resized = F.interpolate(
            img.unsqueeze(0),  
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0) 
        return img_resized

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        source = torch.from_numpy(data['source']).float()  # (C, H, W)
        target = torch.from_numpy(data['target']).float()  # (C, H, W)
        
        if self.is_train:
            source = self.resize_2d(source)
            target = self.resize_2d(target)
            return source, target
        else:
            # source = self.resize_2d(source)
            # target = self.resize_2d(target)
            position = data['position']
            return source, target, position

    def __len__(self):
        return len(self.file_list)
    
class ProjFlywingDataset(Dataset):
    def __init__(self, config, is_train=True, condition=0):

        self.config = config
        self.is_train = is_train
        self.condition = condition
        self.iso = ['Projection_Flywing']

        if self.is_train:
            self._load_train_data()
        else:
            self._load_test_data()

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {self.lenth} samples")

    def _load_train_data(self):
        datapath = f"{self.config.data_dir}/train_data/my_training_data.npz"
        datapath2 = f"{self.config.data_dir}/train_data/data_label.npz"

        # X1, Y1 = self._split_patches(*self._load_npz_data(datapath))
        X1, Y1 = self._load_npz_data(datapath)
        # X2, Y2 = self._load_training_data(datapath2, axes='SCZYX')

        # self.nm_lr = np.concatenate([X1, X2], axis=0)
        # self.nm_hr = np.concatenate([Y1, Y2], axis=0)

        self.nm_lr = X1
        self.nm_hr = Y1
        self.lenth = len(self.nm_lr)

    def _load_test_data(self):
        dir_lr = f"{self.config.data_dir}/test_data/"
        self.nm_lr = sorted(glob.glob(f"{dir_lr}Input/C{self.condition}/*.tif"))
        self.nm_hr = sorted(glob.glob(f"{dir_lr}GT/C{self.condition}/*.tif"))
        self.lenth = len(self.nm_lr)

    def _load_npz_data(self, path):
        """ .npz (784, 1, 50, 128, 128), SCZYX"""
        data = np.load(path)
        return data['X'], data['Y']

    def _split_patches(self, X, Y, patch_size=64):
        X_patches, Y_patches = [], []
        for n in range(len(X)):
            for i in range(0, X.shape[3], patch_size):
                for j in range(0, X.shape[4], patch_size):
                    X_patches.append(X[n][:, j:j + patch_size, i:i + patch_size, :])
                    Y_patches.append(Y[n][:, j:j + patch_size, i:i + patch_size, :])
        return np.array(X_patches), np.array(Y_patches)

    def _load_training_data(self, file, axes):
        """ (784, 1, 1, 128, 128) """
        data = np.load(file)
        X, Y = data['X'], data['Y']

        return X, Y

    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.is_train:
            lr, hr = self.nm_lr[idx], self.nm_hr[idx]
        else:
            lr = np.float32(imread(self.nm_lr[idx]))
            hr = np.expand_dims(np.float32(imread(self.nm_hr[idx])), 0)

        lr = torch.from_numpy(np.ascontiguousarray(lr * self.config.rgb_range)).float()
        hr = torch.from_numpy(np.ascontiguousarray(hr * self.config.rgb_range)).float()

        return lr, hr

    def __len__(self):
        return self.lenth
    
from pathlib import Path
import glob

from torch.utils.data import Dataset

class BSAFusionDataset2D(Dataset):
    def __init__(self, config, is_train = True):
        data_path = Path(config.data_dir)
        self.modularities = config.modularities # Can be "CT-MRI", "PET-MRI", and "SPECT-MRI"
        modularity1 = self.modularities.split('-')[0]
        modularity2 = self.modularities.split('-')[1]
        if is_train:
            image_dir = data_path / self.modularities / "train"
            mod1_images = glob.glob(str(image_dir / modularity1 / "*.png"))
            mod2_images = glob.glob(str(image_dir / modularity2 / "*.png"))

        else:
            image_dir = data_path / self.modularities / "test"
            mod1_images = glob.glob(str(image_dir / modularity1 / "*.png"))
            mod2_images = glob.glob(str(image_dir / modularity2 / "*.png"))

        # Check if the name of image matches
        mod1_images = sorted(mod1_images)
        mod2_images = sorted(mod2_images)
        assert len(mod1_images) == len(mod2_images), "The number of images in two modalities should be the same."
        for i in range(len(mod1_images)):
            assert Path(mod1_images[i]).name == Path(mod2_images[i]).name, "The name of images in two modalities should be the same."
                    
        self.mod1_images = mod1_images
        self.mod2_images = mod2_images

        self.length = len(mod1_images)

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        source1_pil = Image.open(self.mod1_images[idx]).convert('RGB')
        source1 = transforms.ToTensor()(source1_pil)  # 自动归一化到 [0,1] 并转为 (3, H, W)
        source2_pil = Image.open(self.mod2_images[idx]).convert('L')
        source2 = transforms.ToTensor()(source2_pil)  # 转为 (1, H, W) + 归一化到 [0,1]
        source2 = source2.expand(3, -1, -1)  # (3, H, W)

        target = source2  # Fake target
        return source1, source2, target
    

class UnifmirSRDataset2D(Dataset):
    def __init__(self, config, is_train = True):
        self.data_dir = Path(config.data_dir)
        self.data_types = config.data_types # A list, contains ['CCPs', 'F-actin', 'ER', 'Microtubules']

        self.is_train = is_train
        if self.is_train:
            self.data_length = {'CCPs': 19440, 'F-actin': 19584, 'ER': 19584, 'Microtubules': 19800}
        else:
            self.data_length = {'CCPs': 100, 'F-actin': 100, 'ER': 100, 'Microtubules': 100}

        self.img_size = config.img_size

    def resize_2d(self, img):
        # img shape: (H, W)
        h, w = img.shape
        img_resized = F.interpolate(
            img.unsqueeze(0).unsqueeze(0), 
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        )  
        return img_resized.squeeze()
    
    def __len__(self):
        length = 0
        for data_type in self.data_types:
            length += self.data_length[data_type]
        self.length = length
        return self.length

    def __getitem__(self, idx):
        # Compute data_path accrording to idx
        data_length = self.data_length
        data_type = None
        for key in self.data_types:
            if idx < data_length[key]:
                data_type = key
                break
            else:
                idx -= data_length[key]
        if self.is_train:
            data_path = self.data_dir / 'train' / data_type
            source = np.load(data_path / f"preprocessed/X_{idx}.npy").squeeze() # (128, 128)
            target = np.load(data_path / f"preprocessed/Y_{idx}.npy").squeeze() # (256, 256)
        else:
            data_path = self.data_dir / 'test' / data_type
            source = tifffile.imread(data_path / 'LR' / f"im{idx+1}_LR.tif").squeeze() # (128, 128)
            target = tifffile.imread(data_path / 'GT' / f"im{idx+1}_GT.tif").squeeze() # (256, 256)

        source = (source - source.min()) / (source.max() - source.min()) if source.max() != source.min() else np.zeros_like(source)
        target = (target - target.min()) / (target.max() - target.min()) if target.max() != target.min() else np.zeros_like(target)
        source = torch.tensor(source, dtype=torch.float32)  #  (1, C, H, W)
        target = torch.tensor(target, dtype=torch.float32)

        source = self.resize_2d(source).unsqueeze(0)
        target = self.resize_2d(target).unsqueeze(0)
        return source, target
    
class MedSAMSegDataset2D(Dataset):
    LABEL_MAPPING = {
        1: "Liver",
        2: "Right Kidney",
        3: "Spleen",
        4: "Pancreas",
        5: "Aorta",
        6: "Inferior Vena Cava (IVC)",
        7: "Right Adrenal Gland (RAG)",
        8: "Left Adrenal Gland (LAG)",
        9: "Gallbladder",
        10: "Esophagus",
        11: "Stomach",
        12: "Duodenum",
        13: "Left Kidney"
    }
    
    ORGAN_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

    # def __init__(self, data_root, bbox_shift=20, include_labels=None, is_train=True):
    def __init__(self, config, is_train=True):

        if is_train:
            self.data_dir = os.path.join(config.data_dir, "train")
        else:
            self.data_dir = os.path.join(config.data_dir, "val")
        self.gt_path = os.path.join(self.data_dir, "gts")
        self.img_path = os.path.join(self.data_dir, "imgs")
        self.bbox_shift = config.bbox_shift if config.bbox_shift is not None else 20

        self.include_labels = self._process_include_labels(config.include_labels)
        
        self.samples = self._preprocess_samples()
        
        self.index_map = []
        for sample_idx, labels in enumerate(self.samples):
            self.index_map.extend([(sample_idx, label) for label in labels["valid_labels"]])
        
        print(f"Total instances: {len(self)}")
        print(f"Class distribution: {self.get_class_distribution()}")

    def _process_include_labels(self, include_labels):
        if include_labels is None:
            valid_labels = set(self.LABEL_MAPPING.keys())
        else:
            valid_labels = set()
            for item in include_labels:
                if isinstance(item, str):
                    if item in self.ORGAN_TO_LABEL:
                        valid_labels.add(self.ORGAN_TO_LABEL[item])
                    else:
                        raise ValueError(f"Invalid organ name: {item}")
                elif isinstance(item, int):
                    if item in self.LABEL_MAPPING:
                        valid_labels.add(item)
                    else:
                        raise ValueError(f"Invalid label ID: {item}")
                else:
                    raise TypeError("Labels must be str or int")
        
        valid_labels.discard(0) # Remove the background label if present
        return valid_labels

    def _preprocess_samples(self):
        samples = []
        gt_files = sorted(glob.glob(os.path.join(self.gt_path, "**/*.npy"), recursive=True))
        
        for gt_file in gt_files:
            img_name = os.path.basename(gt_file)
            img_file = os.path.join(self.img_path, img_name)
            
            if not os.path.exists(img_file):
                continue
            
            gt = np.load(gt_file, 'r', allow_pickle=True)
            present_labels = set(np.unique(gt))
            
            valid_labels = []
            for label in present_labels:
                if label == 0:
                    continue
                if self.include_labels is None or label in self.include_labels:
                    valid_labels.append(label)
            
            if valid_labels:
                samples.append({
                    "img_path": img_file,
                    "gt_path": gt_file,
                    "valid_labels": valid_labels
                })
        
        return samples

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        sample_idx, label_id = self.index_map[idx]
        sample = self.samples[sample_idx]
        
        img_1024 = np.load(sample["img_path"], 'r', allow_pickle=True)  # (H, W, C)
        img_1024 = img_1024[:, :, 0]  # (H, W, C)
        # img_1024 = np.transpose(img_1024, (2, 0, 1))  # (C, H, W)
        
        assert 0.0 <= img_1024.min() and img_1024.max() <= 1.0, "Image values out of [0,1] range"
        
        gt = np.load(sample["gt_path"], 'r', allow_pickle=True)
        gt2D = np.uint8(gt == label_id) 
        
        if np.sum(gt2D) == 0:
            print(f"Warning: Empty mask for label {label_id} in {sample['gt_path']}")
        
        y_indices, x_indices = np.where(gt2D > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            x_min, x_max = 0, gt2D.shape[1]
            y_min, y_max = 0, gt2D.shape[0]
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        H, W = gt2D.shape
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        
        bboxes = np.array([x_min, y_min, x_max, y_max])
        
        return (
            torch.tensor(img_1024).unsqueeze(0).float(),
            torch.tensor(gt2D).unsqueeze(0).float(), 
            # torch.tensor(gt2D).long(),  
            torch.tensor(bboxes).float(),
            torch.tensor(label_id).long(),
            self.LABEL_MAPPING[label_id], 
            os.path.basename(sample["img_path"])
        )

    def get_class_distribution(self):
        label_counts = {}
        for sample in self.samples:
            for label in sample["valid_labels"]:
                organ_name = self.LABEL_MAPPING[label]
                label_counts[organ_name] = label_counts.get(organ_name, 0) + 1
        return label_counts

    @classmethod
    def get_available_labels(cls):
        return list(cls.LABEL_MAPPING.items())
    
class MicroSplitDataset2D(Dataset):
    def __init__(self, config, is_train=True):
        self.data_dir = Path(config.data_dir)
        self.is_train = is_train
        self.img_size = config.img_size
        
        if is_train:
            self.file_list = sorted(glob.glob(str(self.data_dir / "**" / 'train' / '*.npz'), recursive=True))
        else:
            self.file_list = sorted(glob.glob(str(self.data_dir / "**" / 'test' / '*.npz'), recursive=True))
        
        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        source = torch.from_numpy(data['source']).float().unsqueeze(0) # (C, H, W)
        target1 = torch.from_numpy(data['target1']).float().unsqueeze(0) # (C, H, W)
        target2 = torch.from_numpy(data['target2']).float().unsqueeze(0) # (C, H, W)

        # Resize to target size
        source = F.interpolate(source.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        target1 = F.interpolate(target1.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        target2 = F.interpolate(target2.unsqueeze(0), size=self.img_size, mode='bilinear', align_corners=False).squeeze(0)
        target = torch.cat((target1, target2), dim=0)
        return source, target
    

class MedicalSAMSegDataset2D(Dataset):
    def random_click(self, mask, point_label = 1):
        max_label = max(set(mask.flatten()))
        if round(max_label) == 0:
            point_label = round(max_label)
        indices = np.argwhere(mask == max_label) 
        return point_label, indices[np.random.randint(len(indices))]
    
    def __init__(self, config, is_train):
        self.data_path = config.data_dir
        self.img_size = config.img_size
        self.prompt = 'click'
        self.mask_size = self.img_size[0]
        if is_train:
            self.mode = 'Training'
        else:
            self.mode = 'Test'

        self.subfolders = [f.path for f in os.scandir(os.path.join(self.data_path, self.mode + '-400')) if f.is_dir()]
        # self.transform = transform
        self.transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.subfolders)

    def __getitem__(self, index):

        """Get the images"""
        subfolder = self.subfolders[index]
        name = subfolder.split('/')[-1]

        # raw image and raters path
        img_path = os.path.join(subfolder, name + '_cropped.jpg')
        multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '_cropped.jpg') for i in range(1, 8)]

        # img_path = os.path.join(subfolder, name + '.jpg')
        # multi_rater_cup_path = [os.path.join(subfolder, name + '_seg_cup_' + str(i) + '.png') for i in range(1, 8)]

        # raw image and rater images
        # img = Image.open(img_path).convert('RGB')
        img = Image.open(img_path).convert('L')
        multi_rater_cup = [Image.open(path).convert('L') for path in multi_rater_cup_path]

        # apply transform
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            multi_rater_cup = [torch.as_tensor((self.transform(single_rater) >=0.5).float(), dtype=torch.float32) for single_rater in multi_rater_cup]
            multi_rater_cup = torch.stack(multi_rater_cup, dim=0)

            torch.set_rng_state(state)

        # find init click and apply majority vote
        # if self.prompt == 'click':

        point_label_cup, pt_cup = self.random_click(np.array((multi_rater_cup.mean(axis=0)).squeeze(0)), point_label = 1)
        
        selected_rater_mask_cup_ori = multi_rater_cup.mean(axis=0)
        selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 


        selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
        selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        # # Or use any specific rater as GT
        # point_label_cup, pt_cup = random_click(np.array(multi_rater_cup[0, :, :, :].squeeze(0)), point_label = 1)
        # selected_rater_mask_cup_ori = multi_rater_cup[0,:,:,:]
        # selected_rater_mask_cup_ori = (selected_rater_mask_cup_ori >= 0.5).float() 

        # selected_rater_mask_cup = F.interpolate(selected_rater_mask_cup_ori.unsqueeze(0), size=(self.mask_size, self.mask_size), mode='bilinear', align_corners=False).mean(dim=0) # torch.Size([1, mask_size, mask_size])
        # selected_rater_mask_cup = (selected_rater_mask_cup >= 0.5).float()


        # image_meta_dict = {'filename_or_obj':name}
        # return {
        #     'image':img,
        #     'multi_rater': multi_rater_cup, 
        #     'p_label': point_label_cup,
        #     'pt':pt_cup, 
        #     'mask': selected_rater_mask_cup, 
        #     'mask_ori': selected_rater_mask_cup_ori,
        #     'image_meta_dict':image_meta_dict,
        # }
        
        # return image, mask, pt, p_label
        return img, selected_rater_mask_cup, torch.tensor(pt_cup), torch.tensor(point_label_cup)
