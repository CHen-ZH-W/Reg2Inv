import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from geotransformer.utils.data import precompute_data_stack_mode_two
from geotransformer.utils.torch import to_cuda
from backbone import RIKPConvFPN


class GeoTransformer(nn.Module):
    def __init__(self, cfg, init_voxel_size, neighbor_limits=None):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.neighbor_limits = neighbor_limits   
        
        self.backbone = RIKPConvFPN(
            cfg.backbone.nsample,
            cfg.backbone.in_channel,
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.base_radius*init_voxel_size,
            cfg.backbone.base_sigma*init_voxel_size,
            cfg.backbone.group_norm,
        )
    
        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )    
        
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            num_points = cfg.ransac.num_points,
            num_iterations = cfg.ransac.num_iterations,
        )

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)
        self.optimal_transport_i = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        transform = data_dict['transform'].detach()
        voxel_size = data_dict['voxel_size']
        
        if self.training:
            ref_points_list = data_dict['ref_points']
            ref_points_c = ref_points_list[-1].detach()
            ref_points_f = ref_points_list[0].detach()
            ref_points_m = ref_points_list[1].detach()
            ref_neighbors_list = data_dict['ref_neighbors']
            ref_subsampling_list = data_dict['ref_subsampling']
            ref_upsampling_list = data_dict['ref_upsampling']
            
            src_points_list = data_dict['src_points']
            src_points_c = src_points_list[-1].detach()
            src_points_f = src_points_list[0].detach()
            src_points_m = src_points_list[1].detach()
            src_neighbors_list = data_dict['src_neighbors']
            src_subsampling_list = data_dict['src_subsampling']
            src_upsampling_list = data_dict['src_upsampling']
        else:
            ref_points = data_dict['ref_points'].cpu()
            src_points = data_dict['src_points'].cpu()
            num_stages = 3
            search_radius = voxel_size * 2.5
            ref_lengths = torch.LongTensor([ref_points.shape[0]])
            src_lengths = torch.LongTensor([src_points.shape[0]])
            input_dict = precompute_data_stack_mode_two(ref_points,src_points,ref_lengths,src_lengths,num_stages,voxel_size,search_radius,self.neighbor_limits)
            input_dict = to_cuda(input_dict)
            
            ref_points_list = input_dict['ref_points']
            ref_points_c = input_dict['ref_points'][-1].detach()
            ref_points_f = input_dict['ref_points'][0].detach()
            ref_points_m = input_dict['ref_points'][1].detach()
            ref_neighbors_list = input_dict['ref_neighbors']
            ref_subsampling_list = input_dict['ref_subsampling']
            ref_upsampling_list = input_dict['ref_upsampling']
            
            src_points_list = input_dict['src_points']
            src_points_c = input_dict['src_points'][-1].detach()
            src_points_f = input_dict['src_points'][0].detach()
            src_points_m = input_dict['src_points'][1].detach()
            src_neighbors_list = input_dict['src_neighbors']
            src_subsampling_list = input_dict['src_subsampling']
            src_upsampling_list = input_dict['src_upsampling']
        
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points_m'] = ref_points_m
        output_dict['src_points_m'] = src_points_m
        output_dict['ref_points'] = ref_points_f
        output_dict['src_points'] = src_points_f

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            voxel_size,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )
        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. RIKPFCNN Encoder
        ref_feats_list, _ = self.backbone(ref_points_list, ref_neighbors_list, ref_subsampling_list, ref_upsampling_list)
        src_feats_list, group_idx = self.backbone(src_points_list, src_neighbors_list, src_subsampling_list, src_upsampling_list)

        ref_feats_c = ref_feats_list[-1]
        src_feats_c = src_feats_list[-1]
        ref_feats_f = ref_feats_list[0]
        src_feats_f = src_feats_list[0]
        ref_feats_i = ref_feats_list[1]
        src_feats_i = src_feats_list[1]

        # 3. Conditional Transformer 
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1) 

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f
        output_dict['ref_feats_i'] = ref_feats_i
        output_dict['src_feats_i'] = src_feats_i
        output_dict['group_idx'] = group_idx

        # 5. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 6. Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )
                
        # 7 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        ref_padded_feats_i = torch.cat([ref_feats_i, torch.zeros_like(ref_feats_i[:1])], dim=0)
        src_padded_feats_i = torch.cat([src_feats_i, torch.zeros_like(src_feats_i[:1])], dim=0)
        ref_node_corr_knn_feats_i = index_select(ref_padded_feats_i, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats_i = index_select(src_padded_feats_i, src_node_corr_knn_indices, dim=0)  # (P, K, C)
        
        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / ref_feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        output_dict['matching_scores'] = matching_scores
       
        matching_scores_i = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats_i, src_node_corr_knn_feats_i)  # (P, K, K)
        matching_scores_i = matching_scores_i / ref_feats_i.shape[1] ** 0.5
        matching_scores_i = self.optimal_transport_i(matching_scores_i, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        output_dict['matching_scores_i'] = matching_scores_i
        
        # 9. Generate final transform during testing
        if not self.training:
            with torch.no_grad():
                if not self.fine_matching.use_dustbin:
                    matching_scores = matching_scores[:, :-1, :-1]
                
                estimated_transform = self.fine_matching(
                    ref_node_corr_knn_masks,
                    src_node_corr_knn_masks,
                    ref_node_corr_knn_indices,
                    src_node_corr_knn_indices,
                    matching_scores,
                    src_points_f,
                    ref_points_f,
                    voxel_size,
                )
                output_dict['estimated_transform'] = estimated_transform
        
        return output_dict
        


def create_model(cfg, voxel_size, neighbor_limits=None):
    model = GeoTransformer(cfg, voxel_size, neighbor_limits)
    return model

