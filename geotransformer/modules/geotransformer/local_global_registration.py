from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from geotransformer.modules.ops import apply_transform
from geotransformer.modules.registration import WeightedProcrustes
from geotransformer.utils.open3d import registration_with_ransac_from_correspondences, registration_with_ransac_from_feats, registration_with_ransac_from_points


class LocalGlobalRegistration(nn.Module):
    def __init__(
        self,
        k: int,
        mutual: bool = True,
        confidence_threshold: float = 0.05,
        use_dustbin: bool = False,
        num_points: int = 3,
        num_iterations: int = 50000,
    ):
        super(LocalGlobalRegistration, self).__init__()
        self.k = k
        self.mutual = mutual
        self.confidence_threshold = confidence_threshold
        self.use_dustbin = use_dustbin
        self.num_points = num_points
        self.num_iterations = num_iterations

    def compute_correspondence_matrix(self, score_mat, ref_knn_masks, src_knn_masks):
        r"""Compute matching matrix and score matrix for each patch correspondence."""
        mask_mat = torch.logical_and(ref_knn_masks.unsqueeze(2), src_knn_masks.unsqueeze(1))

        batch_size, ref_length, src_length = score_mat.shape
        batch_indices = torch.arange(batch_size).cuda()
        
        # correspondences from reference side
        ref_topk_scores, ref_topk_indices = score_mat.topk(k=self.k, dim=2)  # (B, N, K)
        ref_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, ref_length, self.k)  # (B, N, K)
        ref_indices = torch.arange(ref_length).cuda().view(1, ref_length, 1).expand(batch_size, -1, self.k)  # (B, N, K)
        ref_score_mat = torch.zeros_like(score_mat)
        ref_score_mat[ref_batch_indices, ref_indices, ref_topk_indices] = ref_topk_scores
        ref_corr_mat = torch.gt(ref_score_mat, self.confidence_threshold)

        # correspondences from source side
        src_topk_scores, src_topk_indices = score_mat.topk(k=self.k, dim=1)  # (B, K, N)
        src_batch_indices = batch_indices.view(batch_size, 1, 1).expand(-1, self.k, src_length)  # (B, K, N)
        src_indices = torch.arange(src_length).cuda().view(1, 1, src_length).expand(batch_size, self.k, -1)  # (B, K, N)
        src_score_mat = torch.zeros_like(score_mat)
        src_score_mat[src_batch_indices, src_topk_indices, src_indices] = src_topk_scores
        src_corr_mat = torch.gt(src_score_mat, self.confidence_threshold)

        # merge results from two sides
        if self.mutual:
            corr_mat = torch.logical_and(ref_corr_mat, src_corr_mat)
        else:
            corr_mat = torch.logical_or(ref_corr_mat, src_corr_mat)

        if self.use_dustbin:
            corr_mat = corr_mat[:, -1:, -1]

        corr_mat = torch.logical_and(corr_mat, mask_mat)
        return corr_mat

    def ransac_registration_from_correspondences(self, corr_mat, ref_knn_indices, src_knn_indices, src_points_f, ref_points_f, distance_threshold):
        batch_indices, ref_indices, src_indices = torch.nonzero(corr_mat, as_tuple=True)

        global_ref_corr_indices = ref_knn_indices[batch_indices, ref_indices]
        global_src_corr_indices = src_knn_indices[batch_indices, src_indices]

        src_np = src_points_f.cpu().numpy()
        ref_np = ref_points_f.cpu().numpy()
        src_corr_indices_np = global_src_corr_indices.cpu().numpy()
        ref_corr_indices_np = global_ref_corr_indices.cpu().numpy()
        correspondences = np.stack([src_corr_indices_np, ref_corr_indices_np], axis=1)
        
        estimated_transform = registration_with_ransac_from_correspondences(
                    src_np,
                    ref_np,
                    correspondences = correspondences,
                    distance_threshold = distance_threshold,
                    ransac_n = self.num_points,
                    num_iterations = self.num_iterations,
                )
        estimated_transform = torch.from_numpy(estimated_transform.astype(np.float32)).cuda()

        return estimated_transform

    def forward(
        self,
        ref_knn_masks,
        src_knn_masks,
        ref_knn_indices,
        src_knn_indices,
        score_mat,
        src_points_f,
        ref_points_f,
        distance_threshold,
    ):
        score_mat = torch.exp(score_mat)
        corr_mat = self.compute_correspondence_matrix(score_mat, ref_knn_masks, src_knn_masks) 
        
        estimated_transform = self.ransac_registration_from_correspondences(
            corr_mat, ref_knn_indices, src_knn_indices, src_points_f, ref_points_f, distance_threshold
        )

        return estimated_transform
