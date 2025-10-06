import os, random
import numpy as np
import torch
import tifffile
import pickle
import glob
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from collections.abc import Sequence



class CollateFn:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, batch):
        processed_batch = [random_crop(item, self.target_size) for item in batch]
        return torch.stack(processed_batch, dim=0)

class Pretrain_Dataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.root_dir = root_dir
        self.img_size = img_size  # (D, H, W)
        self.samples = self._create_sample_list()

    def _create_sample_list(self):
        return [f for f in os.listdir(self.root_dir) if (f.endswith('.tiff') or f.endswith('.npy'))]

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
        
        # 归一化到 0-1
        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min())
        
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

########################################################

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

Identity = Base

class RandomFlip(Base):
    # mirror flip across all x,y,z
    def __init__(self,axis=0):
        # assert axis == (1,2,3) # For both data and label, it has to specify the axis.
        self.axis = (1,2,3)
        self.x_buffer = None
        self.y_buffer = None
        self.z_buffer = None

    def sample(self, *shape):
        self.x_buffer = np.random.choice([True,False])
        self.y_buffer = np.random.choice([True,False])
        self.z_buffer = np.random.choice([True,False])
        return list(shape) # the shape is not changed

    def tf(self,img,k=0): # img shape is (1, 240, 240, 155, 4)
        if self.x_buffer:
            img = np.flip(img,axis=self.axis[0])
        if self.y_buffer:
            img = np.flip(img,axis=self.axis[1])
        if self.z_buffer:
            img = np.flip(img,axis=self.axis[2])
        return img

class Seg_norm(Base):
    def __init__(self, ):
        a = None
        self.seg_table = np.array([0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255])
    def tf(self, img, k=0):
        if k == 0:
            return img
        img_out = np.zeros_like(img)
        for i in range(len(self.seg_table)):
            img_out[img == self.seg_table[i]] = i
        return img_out

class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)
    
class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg


    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class OASISBrainDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class OASISBrainInferDataset(Dataset):
    def __init__(self, data_path, transforms):
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)

class ISOLiverDataset(Dataset):
    def __init__(self, config, is_train=True):
        """
        Args:
            config: 配置对象
            is_train: 是否为训练模式
        """
        self.data_dir = config.data_dir
        self.is_train = is_train
        
        # 加载数据集信息
        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        self.original_shape = info['original_test_shape'] if not is_train else None
        
        # 获取文件列表
        if is_train:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'train', 'cluster_*.npz')))
        else:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'test', 'patch_*.npz')))
            
        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")
        
        # 加载第一个文件以获取数据形状
        sample_data = np.load(self.file_list[0])
        source_shape = sample_data['source'].shape
        print(f"Original data shape: {source_shape}")
        
        if not is_train:
            # 仅测试集需要patch坐标信息
            self.patch_coords = []
            for file_path in self.file_list:
                data = np.load(file_path)
                self.patch_coords.append(data['position'])

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        source = data['source']
        target = data['target']
            
        # 转换为tensor并添加维度
        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()
        
    
        if self.is_train:
            return source, target
        else:
            position = data['position']
            source = source.squeeze(0)
            target = target.squeeze(0)
            return source, target, position

    def __len__(self):
        return len(self.file_list)
    
class FATFusionDataset(Dataset):
    def __init__(self, config, is_train = True):
        if is_train:
            self.data = np.load(os.path.join(config.data_dir, 'train_data.npy'))
        else:
            self.data = np.load(os.path.join(config.data_dir, 'test_data.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]  #  (3, H, W, D)
        source1 = item[0, :, :, :] 
        source2 = item[1, :, :, :] 
        target = item[2, :, :, :] 
        
        # 归一化
        source1 = (source1 - source1.min()) / (source1.max() - source1.min()) if source1.max() != source1.min() else np.zeros_like(source1)
        source2 = (source2 - source2.min()) / (source2.max() - source2.min()) if source2.max() != source2.min() else np.zeros_like(source2)
        target = (target - target.min()) / (target.max() - target.min()) if target.max() != target.min() else np.zeros_like(target)

        # 调整维度顺序从 (H, W, D) -> (D, H, W)
        source1 = np.transpose(source1, (2, 0, 1))  # 变成 (D, H, W)
        source2 = np.transpose(source2, (2, 0, 1))  # 变成 (D, H, W)
        target = np.transpose(target, (2, 0, 1))    # 变成 (D, H, W)

        source1 = torch.tensor(source1, dtype=torch.float32).unsqueeze(0)  # 变成 (1, D, H, W)
        source2 = torch.tensor(source2, dtype=torch.float32).unsqueeze(0)  # 变成 (1, D, H, W)
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)    # 变成 (1, D, H, W)

        
        # source1 = source1.repeat(1,4,1,1)
        # source2 = source2.repeat(1,4,1,1)
        # target = target.repeat(1,4,1,1)
        # 返回三部分数据
        return source1, source2, target

class InverseSRData(Dataset):
    def __init__(self, config, is_train=True):
        if config.down_factor is None:
            config.down_factor = 4
        self.down_factor = config.down_factor
        self.data_dir = config.data_dir
    
        self.is_train = is_train

        if self.is_train:
            self.length = 464 if config.train_length is None else config.train_length
        else:
            self.length = 117 if config.val_length is None else config.val_length

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.is_train:
            item = np.load(os.path.join(self.data_dir, f'train_data_{self.down_factor}x_{idx}.npy'))
        else:
            item = np.load(os.path.join(self.data_dir, f'val_data_{self.down_factor}x_{idx}.npy'))

        # 在 NumPy 中完成维度调整
        source = item[0].transpose(1, 0, 2)[np.newaxis, :, :, :]  # (1, D, H, W)
        target = item[1].transpose(1, 0, 2)[np.newaxis, :, :, :]  # (1, D, H, W)

        # 最后转换为 PyTorch 张量
        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()

        return source, target
    # def __getitem__(self, idx):
    #     if self.is_train:
    #         item = np.load(os.path.join(self.config.data_dir, f'train_data_{self.config.down_factor}x_{idx}.npy'))
    #     else:
    #         item = np.load(os.path.join(self.config.data_dir, f'val_data_{self.config.down_factor}x_{idx}.npy'))
    #     # item = self.data[idx]  #  (2, H, W, D)
    #     source = torch.tensor(item[0, :, :, :])
    #     target = torch.tensor(item[1, :, :, :])

    #     # 调整维度顺序从 (H, W, D) -> (D, H, W)
    #     source = source.permute(2, 0, 1)  # 变成 (D, H, W)
    #     target = target.permute(2, 0, 1)  # 变成 (D, H, W)

    #     source = source.float().unsqueeze(0)  # 确保数据类型，并变成 (1, D, H, W)
    #     target = target.float().unsqueeze(0)  # 确保数据类型，并变成 (1, D, H, W)

    #     # 返回数据
    #     return source, target

class ProjFlywingDataset(Dataset):
    def __init__(self, config, is_train=True):
        """
        Args:
            config: 配置对象，包含数据路径、补丁大小等信息
            is_train: 是否为训练模式
            condition: 测试数据的条件编号
        """
        self.config = config
        self.is_train = is_train
        self.condition = config.condition
        self.iso = ['Projection_Flywing']

        # 数据加载
        if self.is_train:
            self._load_train_data()
        else:
            self._load_test_data()

        print(f"{'Train' if is_train else 'Test'} dataset initialized with {self.lenth} samples")

    def _load_train_data(self):
        """加载训练数据"""
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
        """加载测试数据"""
        dir_lr = f"{self.config.data_dir}/test_data/"
        self.nm_lr = sorted(glob.glob(f"{dir_lr}Input/C{self.condition}/*.tif"))
        self.nm_hr = sorted(glob.glob(f"{dir_lr}GT/C{self.condition}/*.tif"))
        self.lenth = len(self.nm_lr)

    def _load_npz_data(self, path):
        """加载 .npz 格式数据 (784, 1, 50, 128, 128), SCZYX"""
        data = np.load(path)
        return data['X'], data['Y']

    def _split_patches(self, X, Y, patch_size=64):
        """将数据切分为小补丁"""
        X_patches, Y_patches = [], []
        for n in range(len(X)):
            for i in range(0, X.shape[3], patch_size):
                for j in range(0, X.shape[4], patch_size):
                    X_patches.append(X[n][:, j:j + patch_size, i:i + patch_size, :])
                    Y_patches.append(Y[n][:, j:j + patch_size, i:i + patch_size, :])
        return np.array(X_patches), np.array(Y_patches)

    def _load_training_data(self, file, axes):
        """加载并预处理训练数据 (784, 1, 1, 128, 128) """
        data = np.load(file)
        X, Y = data['X'], data['Y']

        return X, Y

    def __getitem__(self, idx):
        idx = idx % self.lenth
        if self.is_train:
            lr, hr = self.nm_lr[idx], self.nm_hr[idx]
        else:
            lr = np.float32(tifffile.imread(self.nm_lr[idx]))
            hr = np.expand_dims(np.float32(tifffile.imread(self.nm_hr[idx])), 0)

        lr = torch.from_numpy(np.ascontiguousarray(lr)).float()  # (1, D, H, W)
        # Add an all zero layer to the first and last, become (1, D+2, H, W)
        H = lr.shape[2]
        W = lr.shape[3]
        lr = torch.cat([torch.zeros(1, 7, H, W), lr, torch.zeros(1, 7, H, W)], dim=1) # (1, 64, 128, 128)
        hr = torch.from_numpy(np.ascontiguousarray(hr)).float()
        # Repeat hr to be (1, 64, 128, 128), now is (1, 1, 128, 128)
        hr = hr.repeat(1, 64, 1, 1)


        # print(f"lr shape: {lr.shape}, hr shape: {hr.shape}")
        return lr, hr

    def __len__(self):
        return self.lenth
    
# class DenoiseLiverDataset(Dataset):
#     def __init__(self, config, is_train=True):
#         """
#         Args:
#             config: 配置对象
#             is_train: 是否为训练模式
#         """
#         self.data_dir = config.data_dir
#         self.is_train = is_train
#         self.denoisegt = ['Denoising_Planaria', 'Denoising_Tribolium']
#         c = config.condition

#         if is_train:
#             self._scandenoisenpy()
#         else:
#             self._scandenoisetif(c)
        
#         self.lenthdenoise = len(self.nm_lrdenoise)
#         self.lenth = self.lenthdenoise

#         if is_train:
#             print('++ ++ ++ ++ ++ ++ length of training images = ', self.lenth, '++ ++ ++ ++ ++ ++')
#         else:
#             print('++ ++ ++ ++ ++ ++ length of test images = ', self.lenth, '++ ++ ++ ++ ++ ++')
    
#     def _scandenoisenpy(self):
#         hr = []
#         lr = []
#         datapath = self.data_dir # '/home/zhenghanfang/Project/3D/data/Denoising/'
#         for i in self.denoisegt:
#             # Planaria: X/Y  (17005, 16, 64, 64, 1)(895, 16, 64, 64, 1)  float32
#             # Tr  (14725, 16, 64, 64, 1) (775, 16, 64, 64, 1)
#             train_data = np.load(datapath + '/' + i + '/' + '/train_data/data_label.npz')
#             X = train_data['X']# S, C, D, H, W
#             Y = train_data['Y'] # S, C, D, H, W
#             print('Dataset:', i, 'np.isnan(X).any(), np.isnan(Y).any()', np.isnan(X).any(), np.isnan(Y).any())
#             print('X.shape, Y.shape = ', X.shape, Y.shape)
#             assert len(X) == len(Y)
#             hr.extend(Y)
#             lr.extend(X)
#         self.nm_hrdenoise, self.nm_lrdenoise = hr, lr

#     def _scandenoisetif(self, c=1):
#         lr = []
#         datapath = self.data_dir
#         lr.extend(sorted(glob.glob(datapath + '/%s/test_data/condition_%d/*.tif' % (self.denoisegt[0], c))))
#         self.hrpath = datapath + '/%s/test_data/GT/' % self.denoisegt[0]
#         lr.sort()
#         self.nm_lrdenoise = lr

#     def __getitem__(self, idx):
#         # data = np.load(self.file_list[idx])
#         if self.is_train:
#             target = self.nm_hrdenoise[idx]
#             source = self.nm_lrdenoise[idx]
#             # 转换为tensor并添加维度
#             source = torch.from_numpy(source).float()
#             target = torch.from_numpy(target).float()
#             return source, target
#         else:
#             filename, fmt = os.path.splitext(os.path.basename(self.nm_lrdenoise[idx]))
#             target = np.float32(tifffile.imread(self.hrpath + filename + fmt))  # / 65535
#             source = np.float32(tifffile.imread(self.nm_lrdenoise[idx]))
#             # print('Test Denoise, ----> rgblr.max/min', rgblr.max(), rgblr.min(), rgblr.shape)
#             source = torch.from_numpy(np.ascontiguousarray(source * 255)).float().unsqueeze(0)
#             target = torch.from_numpy(np.ascontiguousarray(target * 255)).float().unsqueeze(0)
#             return source, target

#     def __len__(self):
#         return self.length

class DenoisePlanriaDataset(Dataset):
    def __init__(self, config, is_train=True):
        """
        Args:
            config: 配置对象
            is_train: 是否为训练模式
        """
        self.data_dir = config.data_dir
        # Path('/home/zch/Documents/foundation_mamba_biomed/data_unifmir/Denoise/Denoising_Planaria/preprocessed')
        self.is_train = is_train
        self.condition = config.condition # one in ['condition_1', 'condition_2', 'condition_3', 'GT']
        
        # 加载数据集信息
        info = np.load(os.path.join(self.data_dir, 'dataset_info.npy'), allow_pickle=True).item()
        self.original_shape = info['original_test_shape'] if not is_train else None
        self.target_num = info['data_number'] if not is_train else None
        # 获取文件列表
        if is_train:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'train', 'cluster_*.npz')))
        else:
            self.file_list = sorted(glob.glob(os.path.join(self.data_dir, 'test', 'patch_*.npz')))
            
        print(f"{'Train' if is_train else 'Test'} dataset initialized with {len(self.file_list)} samples")
        
        # 加载第一个文件以获取数据形状
        sample_data = np.load(self.file_list[0])
        if is_train:
            source_shape = sample_data['source'].shape
        else:
            source_shape = sample_data[self.condition].shape
        print(f"Original data shape: {source_shape}")
        
        if not is_train:
            # 仅测试集需要patch坐标信息
            self.patch_coords = []
            for file_path in self.file_list:
                data = np.load(file_path)
                self.patch_coords.append(data['start_point'])

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        if self.is_train:
            source = data['source']
            target = data['target']
        else:
            source = data[self.condition]
            target = data['GT']
            
        # 转换为tensor并添加维度
        
        source = source / source.max()

        source = torch.from_numpy(source).float()
        target = torch.from_numpy(target).float()
        
        if self.is_train:
            # vstack source and target, become 32*64*64
            # Resize to 32*64*64
            source = torch.vstack([source, source])
            target = torch.vstack([target, target])
            return source.unsqueeze(0), target.unsqueeze(0)
        else:
            position = data['start_point']
            data_index = data['data_index'].item()

            source = torch.vstack([source, source])
            target = torch.vstack([target, target])
            source = source.unsqueeze(0)
            target = target.unsqueeze(0)
            return source, target, position, data_index

    def __len__(self):
        return len(self.file_list)
