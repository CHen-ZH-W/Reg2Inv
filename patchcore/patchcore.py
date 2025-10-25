"""PatchCore and PatchCore detection methods."""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.common
import patchcore.sampler

import argparse
from sklearn.cluster import KMeans
import open3d as o3d
from utils.cpu_knn import fill_missing_values, KNN, average_pooling
from geotransformer.utils.torch import to_cuda
from geotransformer.modules.ops import apply_transform
from model import create_model
from sklearn.decomposition import PCA
from utils.riconv_utils import index_points

LOGGER = logging.getLogger(__name__)

class PatchCore(torch.nn.Module):
    def __init__(self, device):
        """PatchCore anomaly detection class."""
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        device,
        target_embed_dimension,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        self.device = device
        self.forward_modules = torch.nn.ModuleDict({})
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )

        _ = preadapt_aggregator.to(self.device)

        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        self.featuresampler = featuresampler
        self.dataloader_count = 0
        self.deep_feature_extractor = None
        self.deep_feature_extractor_pmae = None
        self.knn = KNN(64)
        #self.pca = PCA(n_components=10, whiten=True)
        
    def set_deep_feature_extractor(self, cfg, snapshot, voxel_size, neighbor_limits):
        self.deep_feature_extractor = create_model(cfg, voxel_size, neighbor_limits).cuda()
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        assert 'model' in state_dict, 'No model can be loaded.'
        self.deep_feature_extractor.load_state_dict(state_dict['model'], strict=True)
        self.deep_feature_extractor.eval()

    def set_dataloadercount(self, dataloader_count):
        self.dataloader_count = dataloader_count

    def build_memory_bank(self, features, limit_size):
        features, _ = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features
    
    def predict_score(self, all_features, all_xyz_sampled, all_point_clouds):
        scores = []
        masks = []
        for features, xyz_sampled, point_cloud in zip(all_features, all_xyz_sampled, all_point_clouds):
            patch_scores = self.anomaly_scorer.predict([features])[0]
            full_scores = fill_missing_values(xyz_sampled, patch_scores, point_cloud, k=16)
            pooling_scores = average_pooling(xyz_sampled, patch_scores, xyz_sampled, k=64)
            image_scores = np.max(pooling_scores)

            scores.extend([image_scores])
            masks.extend([mask for mask in full_scores])
        return scores, masks

    def fit_with_limit_size(self, train_loader, limit_size):
        _ = self.forward_modules.eval()

        def _image_to_features(data_dict):
            with torch.no_grad():
                data_dict = to_cuda(data_dict)
                output_dict = self.deep_feature_extractor(data_dict)
                est_transform = output_dict['estimated_transform']

                src_points = output_dict['src_points_f']
                est_src_points = apply_transform(src_points, est_transform)
                position_feature = est_src_points.cpu().numpy()
                shape_feature = output_dict['src_feats_i'].cpu().numpy()
                features = np.concatenate([position_feature, shape_feature], axis=1) 
                
                return features

        features = []
        total_iterations = len(train_loader)
        pbar = tqdm.tqdm(enumerate(train_loader), total=total_iterations)

        for iteration, data_dict in pbar:
            features.append(_image_to_features(data_dict))

        features = np.concatenate(features, axis=0)

        position_feature = features[:,:3]
        shape_feature = features[:,3:]
        position_norm = max(np.linalg.norm(position_feature, axis=1))
        shape_norm = max(np.linalg.norm(shape_feature, axis=1))
        position_feature = position_feature / position_norm
        shape_feature = shape_feature / shape_norm 
        features = np.concatenate([position_feature, shape_feature], axis=1)

        features, _ = self.featuresampler.run_with_limit_memory(features, limit_size)
        self.anomaly_scorer.fit(detection_features=[features])
        print(features.shape)
        return features, position_norm, shape_norm
    
    def predict(self, data, position_norm, shape_norm, dataset_name):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data, position_norm, shape_norm, dataset_name)
        return self._predict(data, position_norm, shape_norm, dataset_name)

    def _predict_dataloader(self, test_loader, position_norm, shape_norm, dataset_name):

        _ = self.forward_modules.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = list()
        
        total_iterations = len(test_loader)
        pbar = tqdm.tqdm(enumerate(test_loader), total=total_iterations)

        for iteration, data_dict in pbar:
            label = data_dict['label']
            mask = data_dict['mask']
            labels_gt.append(label)
            masks_gt.extend(mask.flatten().numpy())
            _scores, _masks = self._predict(data_dict, iteration, position_norm, shape_norm, dataset_name)
            scores.extend(_scores)
            masks.extend(_masks)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, data_dict, iteration, position_norm, shape_norm, dataset_name):
        with torch.no_grad():

            data_dict = to_cuda(data_dict)
            output_dict = self.deep_feature_extractor(data_dict)
            input_pointcloud = data_dict['raw_points']
            point_cloud = input_pointcloud.cpu()
            est_transform = output_dict['estimated_transform']

            src_points = output_dict['src_points_f']    
            xyz_sampled = output_dict['src_points_f'].cpu()       
            est_src_points = apply_transform(src_points, est_transform)
            position_feature = est_src_points.cpu().numpy()  

            shape_feature = output_dict['src_feats_i'].cpu().numpy() 
            position_feature = position_feature / position_norm
            shape_feature = shape_feature / shape_norm
            features = np.concatenate([position_feature, shape_feature], axis=1)

            src_points = output_dict['src_points_f']   
            check_points = src_points.cpu().unsqueeze(0)
            
            _, group_idx = self.knn(check_points, check_points)           
            neighborhood = index_points(check_points, group_idx).squeeze(0)
            check_points = check_points.squeeze(0)
            group_centers = torch.mean(neighborhood, dim=1)
            centers_distances = torch.norm(neighborhood - group_centers.unsqueeze(1), dim=2)
            _, top_indices = torch.topk(centers_distances, k=1, dim=1, largest=False)
            center_in_top = torch.any(top_indices == 0, dim=1)
            xyz_sampled = src_points[center_in_top].cpu()
            features = features[center_in_top.cpu().numpy()]
            features = np.asarray(features,order='C').astype('float32')
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]
            full_scores = fill_missing_values(xyz_sampled, patch_scores, point_cloud, k=1)

            pooling_scores =  average_pooling(xyz_sampled, patch_scores, xyz_sampled, k=32)
            image_scores = np.max(pooling_scores)
            
        return [image_scores], [mask for mask in full_scores]