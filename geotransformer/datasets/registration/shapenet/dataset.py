from typing import Optional
import os
import glob
import numpy as np
import torch.utils.data
import open3d as o3d
import pathlib

from geotransformer.utils.pointcloud import random_sample_transform, apply_transform, inverse_transform
from geotransformer.transforms.functional import (
    random_shuffle_points,
    random_crop_point_cloud_with_plane_num,
)

def load_data(dataset_dir):
    train_sample_list = []
    for cls_name in os.listdir(dataset_dir):
        cls_dir = os.path.join(dataset_dir, cls_name, 'train')
        pcd_files = glob.glob(os.path.join(cls_dir, '*template*.pcd'))
        train_sample_list.extend(pcd_files)
    return train_sample_list

def load_data_train(dataset_dir,cls_name):
    train_sample_list = []
    cls_dir = os.path.join(dataset_dir, cls_name, 'train')
    pcd_files = glob.glob(os.path.join(cls_dir, '*template*.pcd'))
    train_sample_list.extend(pcd_files)
    return train_sample_list

def load_data_test(dataset_dir,cls_name):
    test_sample_list = glob.glob(str(os.path.join(dataset_dir, cls_name, 'test')) + '/*.pcd')
    train_dir = os.path.join(dataset_dir, cls_name, 'train')
    pcd_files = glob.glob(os.path.join(train_dir, '*template*.pcd'))
    pcd_files = sorted(pcd_files)
    template_path = pcd_files[0]
    template = read_point_cloud(template_path)
    return test_sample_list, template

def load_data_bank(dataset_dir,cls_name):
    train_dir = os.path.join(dataset_dir, cls_name, 'train')
    pcd_files = glob.glob(os.path.join(train_dir, '*template*.pcd'))
    pcd_files = sorted(pcd_files)
    template_path = pcd_files[0]
    template = read_point_cloud(template_path)
    return pcd_files, template

def read_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    pointcloud = np.array(pcd.points)
    return pointcloud

def generate_data_list(train_sample_list, num_points, subset):
    data_list = []
    for file_path in train_sample_list:
        pointcloud = read_point_cloud(file_path)
        pointcloud, _ = norm_pcd(pointcloud, num_points)
        data_list.append(pointcloud)
    data_list = data_list * 25
    return data_list

def generate_data_list_train(train_sample_list, num_points, subset):
    data_list = []
    voxel_list = []
    for file_path in train_sample_list:
        pointcloud = read_point_cloud(file_path)
        pointcloud, _, voxel_size = norm_pcd_train(pointcloud, num_points)
        data_list.append(pointcloud)
        voxel_list.append(voxel_size)
    data_list = data_list * 25
    voxel_list = voxel_list * 25
    return data_list, voxel_list

def generate_data_list_test(train_sample_list):
    data_list = []
    for file_path in train_sample_list:
        pointcloud = read_point_cloud(file_path)
        data_list.append(pointcloud)
    return data_list

def get_downsample_pcd(source_data, voxel_size):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_data)
    source_down = source.voxel_down_sample(voxel_size)
    return np.asarray(source_down.points)
    
def binary_search_voxel_size(source_data, target_num, tol_rate):
    tol = target_num * tol_rate
    low, high = 0.01, 2.0
    max_iter = 100
    for _ in range(max_iter):
        mid = (low + high) / 2
        downsampled_points = get_downsample_pcd(source_data, mid)
        num_points = downsampled_points.shape[0]
            
        if num_points - target_num <= tol and num_points >= target_num:
            return mid, downsampled_points
            
        if num_points > target_num:
            low = mid
        else:
            high = mid
    return low, downsampled_points 

def preprocess_pcd(pcd, norm_size):
    pcd = pcd / norm_size
    _, down_pcd = binary_search_voxel_size(pcd, 2048, 0.01)

    center = np.average(down_pcd,axis=0)
    norm_pcd = pcd - np.expand_dims(center,axis=0)

    return norm_pcd

def norm_pcd(template, num_points):
    
    _, down_pcd = binary_search_voxel_size(template, 2048, 0.01)

    norm_center = np.average(down_pcd,axis=0)
    norm_points = template-np.expand_dims(norm_center,axis=0)
    norm_size = np.max(np.sqrt(np.sum(norm_points**2, axis=1)))

    template_norm = preprocess_pcd(template, norm_size)  

    return template_norm, norm_size

def norm_pcd_train(template, num_points):
    
    _, down_pcd = binary_search_voxel_size(template, 2048, 0.05)

    norm_center = np.average(down_pcd,axis=0)
    norm_points = template-np.expand_dims(norm_center,axis=0)
    norm_size = np.max(np.sqrt(np.sum(norm_points**2, axis=1)))

    template_norm = preprocess_pcd(template, norm_size)  

    voxel_size, _ = binary_search_voxel_size(template_norm, num_points, 0.05)

    return template_norm, norm_size, voxel_size

class ShapeNetPairDataset_train(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root: str,
        class_name: str,
        subset: str,
        voxel_size: float = 0.05,
        num_points: int = 1024,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        noise_magnitude: Optional[float] = None,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super(ShapeNetPairDataset_train, self).__init__()

        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.class_name = class_name
        self.subset = subset

        self.num_points = num_points
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.noise_magnitude = noise_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index
        
        train_sample_list = load_data(self.dataset_root)
        data_list = generate_data_list(train_sample_list, self.num_points, self.subset)
        self.voxel_size = voxel_size
        self.data_list = data_list

    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index
        
        raw_points = self.data_list[index]
        voxel_size = self.voxel_size

        # set deterministic 
        if self.deterministic:
            np.random.seed(index)

        # split reference and source point cloud
        ref_points = raw_points.copy()
        src_points = ref_points.copy()

        # random transform to source point cloud
        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        inv_transform = inverse_transform(transform)
        src_points = apply_transform(src_points, inv_transform)
        
        # downsample
        ref_points = get_downsample_pcd(ref_points, voxel_size)     
        src_points = get_downsample_pcd(src_points, voxel_size) 
        
        # crop
        #'''
        num_points = ref_points.shape[0]
        lower_bound = int(num_points * 0.25)
        upper_bound = int(num_points * 1.0)
        random_num = np.random.randint(lower_bound, upper_bound + 1)
        src_points = random_crop_point_cloud_with_plane_num(src_points, random_num)
        #'''
        
        # random shuffle
        ref_points = random_shuffle_points(ref_points)
        src_points = random_shuffle_points(src_points)

        new_data_dict = {
            'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'voxel_size': voxel_size,
            'index': int(index),
        }

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

        return new_data_dict

    def __len__(self):
        return len(self.data_list)

class ShapeNetPairDataset_test(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root: str,
        class_name: str,
        subset: str,
        voxel_size: float = 0.05,
        num_points: int = 1024,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super(ShapeNetPairDataset_test, self).__init__()

        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.class_name = class_name
        self.subset = subset
         
        self.num_points = num_points
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index

        test_sample_list, template = load_data_test(self.dataset_root, self.class_name)
        
        self.test_sample_list = test_sample_list
        template_norm, self.norm_size = norm_pcd(template, self.num_points)
        self.voxel_size = voxel_size
        self.template = get_downsample_pcd(template_norm, self.voxel_size)

    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index
        
        path = self.test_sample_list[index]
        voxel_size = self.voxel_size
        
        if 'positive' in path:
            point_cloud = read_point_cloud(path)
            mask = np.zeros((point_cloud.shape[0]))
            label = 0
        else:
            gt_path = str(os.path.join(self.dataset_root, self.class_name, 'GT'))
            filename = pathlib.Path(path).stem
            txt_path = os.path.join(gt_path, filename + '.txt')
            pcd = np.genfromtxt(txt_path, delimiter=",")
            point_cloud = pcd[:, :3]
            mask = pcd[:, 3]
            label = 1

        if self.deterministic:
            np.random.seed(index)

        src_points = point_cloud.copy()
        ref_points = self.template.copy()
        
        # norm
        src_points = preprocess_pcd(src_points, self.norm_size)
        # down
        raw_points = src_points.copy()
        src_points = get_downsample_pcd(src_points, voxel_size)

        # random shuffle
        ref_points = random_shuffle_points(ref_points)
        src_points = random_shuffle_points(src_points)

        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        
        new_data_dict = {
            'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'voxel_size': voxel_size,
            'index': int(index),
            'path': path,
            'mask': mask,
            'label': label,
            'class_name': self.class_name,
        }

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

        return new_data_dict

    def __len__(self):
        return len(self.test_sample_list)


class ShapeNetDataset_train(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root: str,
        class_name: str,
        subset: str,
        voxel_size: float = 0.05,
        num_points: int = 1024,
        rotation_magnitude: float = 45.0,
        translation_magnitude: float = 0.5,
        keep_ratio: float = 0.7,
        crop_method: str = 'plane',
        deterministic: bool = False,
        twice_sample: bool = False,
        twice_transform: bool = False,
        return_normals: bool = True,
        return_occupancy: bool = False,
        min_overlap: Optional[float] = None,
        max_overlap: Optional[float] = None,
        estimate_normal: bool = False,
        overfitting_index: Optional[int] = None,
    ):
        super(ShapeNetDataset_train, self).__init__()

        assert subset in ['train', 'val', 'test']
        assert crop_method in ['plane', 'point']

        self.dataset_root = dataset_root
        self.class_name = class_name
        self.subset = subset
         
        self.num_points = num_points
        self.rotation_magnitude = rotation_magnitude
        self.translation_magnitude = translation_magnitude
        self.keep_ratio = keep_ratio
        self.crop_method = crop_method
        self.deterministic = deterministic
        self.twice_sample = twice_sample
        self.twice_transform = twice_transform
        self.return_normals = return_normals
        self.return_occupancy = return_occupancy
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.check_overlap = self.min_overlap is not None or self.max_overlap is not None
        self.estimate_normal = estimate_normal
        self.overfitting_index = overfitting_index

        test_sample_list, template = load_data_bank(self.dataset_root, self.class_name)
        
        self.test_sample_list = test_sample_list
        template_norm, self.norm_size = norm_pcd(template, self.num_points)
        self.voxel_size = voxel_size
        self.template = get_downsample_pcd(template_norm, self.voxel_size)

    def __getitem__(self, index):
        if self.overfitting_index is not None:
            index = self.overfitting_index
        
        path = self.test_sample_list[index]
        point_cloud = read_point_cloud(path)
        voxel_size = self.voxel_size
 
        if self.deterministic:
            np.random.seed(index)

        src_points = point_cloud.copy()
        ref_points = self.template.copy()
        
        # norm
        src_points = preprocess_pcd(src_points, self.norm_size)
        # down
        raw_points = src_points.copy()
        src_points = get_downsample_pcd(src_points, self.voxel_size) 

        # random shuffle
        ref_points = random_shuffle_points(ref_points)
        src_points = random_shuffle_points(src_points)

        transform = random_sample_transform(self.rotation_magnitude, self.translation_magnitude)
        
        mask = np.zeros((raw_points.shape[0]))
        label = 0

        new_data_dict = {
            'raw_points': raw_points.astype(np.float32),
            'ref_points': ref_points.astype(np.float32),
            'src_points': src_points.astype(np.float32),
            'transform': transform.astype(np.float32),
            'voxel_size': voxel_size,
            'index': int(index),
            'mask': mask,
            'label': label,
            'class_name': self.class_name,
        }

        if self.return_occupancy:
            new_data_dict['ref_feats'] = np.ones_like(ref_points[:, :1]).astype(np.float32)
            new_data_dict['src_feats'] = np.ones_like(src_points[:, :1]).astype(np.float32)

        return new_data_dict

    def __len__(self):
        return len(self.test_sample_list)