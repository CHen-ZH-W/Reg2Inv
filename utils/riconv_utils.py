import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import open3d as o3d
from knn_cuda import KNN

def compute_LRA(xyz, weighting=True, nsample = 32):
    with torch.no_grad():
        dists = torch.cdist(xyz, xyz)

        dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
        dists = dists.unsqueeze(-1)

        group_xyz = index_points(xyz, idx)
        group_xyz = group_xyz - xyz.unsqueeze(2)

        if weighting:
            dists_max, _ = dists.max(dim=2, keepdim=True)
            dists = dists_max - dists
            dists_sum = dists.sum(dim=2, keepdim=True)
            weights = dists / dists_sum
            weights[weights != weights] = 1.0
            M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
        else:
            M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

        #eigen_values, vec = M.symeig(eigenvectors=True)
        eigen_values, vec = torch.linalg.eigh(M)

        LRA = vec[:,:,:,0]
        LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
        LRA = LRA / LRA_length
    return LRA # B N 3

def compute_norms(xyz):
    device = xyz.device
    B, N, _ = xyz.shape
    norms = torch.zeros_like(xyz)
    
    for i in range(B):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz[i].cpu().numpy())
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        pcd.orient_normals_consistent_tangent_plane(k=20)
        
        norms[i] = torch.from_numpy(np.asarray(pcd.normals)).to(device)
    return norms

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(
        sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return group_idx

def sample_and_group(nsample, xyz, norm):
    """
    Input:
        nsample: number of samples for each new point
        xyz: input points position data, [B, N, 3]
        norm: input points normal data, [B, N, 3]
    Return:
        ri_feat: sampled ri attributes, [B, npoint, nsample, 8]
        idx_ordered: ordered index of the sample position data, [B, npoint, nsample]
    """
    xyz = xyz.contiguous()
    norm = norm.contiguous()
    
    knn = KNN(k=nsample, transpose_mode=True)
    _, idx = knn(xyz, xyz)
    #idx = knn_point(nsample, xyz, xyz.contiguous())
    
    ri_feat, idx_ordered = RI_features(xyz, norm, xyz, norm, idx)

    return ri_feat, idx_ordered, idx

def sample_and_group_down(nsample, xyz, norm, new_xyz, new_norm):
    """
    Input:
        nsample: number of samples for each new point
        xyz: input points position data, [B, N, 3]
        norm: input points normal data, [B, N, 3]
    Return:
        ri_feat: sampled ri attributes, [B, npoint, nsample, 8]
        idx_ordered: ordered index of the sample position data, [B, npoint, nsample]
    """
    xyz = xyz.contiguous()
    norm = norm.contiguous()
    
    idx = knn_point(nsample, xyz, new_xyz.contiguous())
    
    ri_feat, idx_ordered = RI_features(xyz, norm, new_xyz, new_norm, idx)

    return ri_feat, idx_ordered, idx

def order_index_group(grouped_xyz, new_xyz, new_norm, idx):
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered

    # project and order
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane*new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / proj_xyz_length
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # set nan to zero

    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1,1,1,3)) # corresponds to the largest length
    
    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(projected_xyz_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1))
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.  # the first is the center point itself, just set sign as 1 to differ from ref_vec 
    dots = sign*dots - (1-sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))

    return dots_sorted, idx_ordered

def order_index(xyz, new_xyz, new_norm, idx):
    B, S, C = new_xyz.shape
    nsample = idx.shape[2]
    grouped_xyz = index_points(xyz, idx)
    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered

    # project and order
    dist_plane = torch.matmul(grouped_xyz_local, new_norm)
    proj_xyz = grouped_xyz_local - dist_plane*new_norm.view(B, S, 1, C)
    proj_xyz_length = torch.norm(proj_xyz, dim=-1, keepdim=True)
    projected_xyz_unit = proj_xyz / proj_xyz_length
    projected_xyz_unit[projected_xyz_unit != projected_xyz_unit] = 0  # set nan to zero

    length_max_idx = torch.argmax(proj_xyz_length, dim=2)
    vec_ref = projected_xyz_unit.gather(2, length_max_idx.unsqueeze(-1).repeat(1,1,1,3)) # corresponds to the largest length
    
    dots = torch.matmul(projected_xyz_unit, vec_ref.view(B, S, C, 1))
    sign = torch.cross(projected_xyz_unit, vec_ref.view(B, S, 1, C).repeat(1, 1, nsample, 1))
    sign = torch.matmul(sign, new_norm)
    sign = torch.sign(sign)
    sign[:, :, 0, 0] = 1.  # the first is the center point itself, just set sign as 1 to differ from ref_vec 
    dots = sign*dots - (1-sign)
    dots_sorted, indices = torch.sort(dots, dim=2, descending=True)
    idx_ordered = idx.gather(2, indices.squeeze_(-1))

    return dots_sorted, idx_ordered

def RI_features_group(xyz, norm, new_xyz, new_norm, idx, neighborhood):
    B, S, C = new_xyz.shape

    new_norm = new_norm.unsqueeze(-1)
    dots_sorted, idx_ordered = order_index_group(neighborhood, new_xyz, new_norm, idx)

    epsilon=1e-7
    grouped_xyz = index_points(xyz, idx_ordered)  # [B, npoint, nsample, C]

    grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)
    grouped_xyz_length = torch.norm(grouped_xyz_local, dim=-1, keepdim=True) # nn lengths
    grouped_xyz_unit = grouped_xyz_local / grouped_xyz_length
    grouped_xyz_unit[grouped_xyz_unit != grouped_xyz_unit] = 0  # set nan to zero
    grouped_xyz_norm = index_points(norm, idx_ordered) # nn neighbor normal vectors
    
    grouped_xyz_angle_0 = torch.matmul(grouped_xyz_unit, new_norm)
    grouped_xyz_angle_1 =  (grouped_xyz_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_angle_norm = torch.matmul(grouped_xyz_norm, new_norm)
    grouped_xyz_angle_norm = torch.acos(torch.clamp(grouped_xyz_angle_norm, -1 + epsilon, 1 - epsilon))  #
    D_0 = (grouped_xyz_angle_0 < grouped_xyz_angle_1)
    D_0[D_0 ==0] = -1
    grouped_xyz_angle_norm = D_0.float() * grouped_xyz_angle_norm

    grouped_xyz_inner_vec = grouped_xyz_local - torch.roll(grouped_xyz_local, 1, 2)
    grouped_xyz_inner_length = torch.norm(grouped_xyz_inner_vec, dim=-1, keepdim=True) # nn lengths
    grouped_xyz_inner_unit = grouped_xyz_inner_vec / grouped_xyz_inner_length
    grouped_xyz_inner_unit[grouped_xyz_inner_unit != grouped_xyz_inner_unit] = 0  # set nan to zero
    grouped_xyz_inner_angle_0 = (grouped_xyz_inner_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_1 = (grouped_xyz_inner_unit * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = (grouped_xyz_norm * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = torch.acos(torch.clamp(grouped_xyz_inner_angle_2, -1 + epsilon, 1 - epsilon))
    D_1 = (grouped_xyz_inner_angle_0 < grouped_xyz_inner_angle_1)
    D_1[D_1 ==0] = -1
    grouped_xyz_inner_angle_2 = D_1.float() * grouped_xyz_inner_angle_2

    proj_inner_angle_feat = dots_sorted - torch.roll(dots_sorted, 1, 2)
    proj_inner_angle_feat[:,:,0,0] = (-3) - dots_sorted[:,:,-1,0]

    ri_feat = torch.cat([grouped_xyz_length, 
                            proj_inner_angle_feat,
                            grouped_xyz_angle_0,
                            grouped_xyz_angle_1,
                            grouped_xyz_angle_norm,
                            grouped_xyz_inner_angle_0,
                            grouped_xyz_inner_angle_1,
                            grouped_xyz_inner_angle_2], dim=-1)

    return ri_feat, idx_ordered

def RI_features(xyz, norm, new_xyz, new_norm, idx, group_all=False):
    B, S, C = new_xyz.shape

    new_norm = new_norm.unsqueeze(-1)
    dots_sorted, idx_ordered = order_index(xyz, new_xyz, new_norm, idx)

    epsilon=1e-7
    grouped_xyz = index_points(xyz, idx_ordered)  # [B, npoint, nsample, C]
    if not group_all:
        grouped_xyz_local = grouped_xyz - new_xyz.view(B, S, 1, C)  # centered
    else:
        grouped_xyz_local = grouped_xyz  # treat orgin as center
    grouped_xyz_length = torch.norm(grouped_xyz_local, dim=-1, keepdim=True) # nn lengths
    grouped_xyz_unit = grouped_xyz_local / grouped_xyz_length
    grouped_xyz_unit[grouped_xyz_unit != grouped_xyz_unit] = 0  # set nan to zero
    grouped_xyz_norm = index_points(norm, idx_ordered) # nn neighbor normal vectors
    
    grouped_xyz_angle_0 = torch.matmul(grouped_xyz_unit, new_norm)
    grouped_xyz_angle_1 =  (grouped_xyz_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_angle_norm = torch.matmul(grouped_xyz_norm, new_norm)
    grouped_xyz_angle_norm = torch.acos(torch.clamp(grouped_xyz_angle_norm, -1 + epsilon, 1 - epsilon))  #
    D_0 = (grouped_xyz_angle_0 < grouped_xyz_angle_1)
    D_0[D_0 ==0] = -1
    grouped_xyz_angle_norm = D_0.float() * grouped_xyz_angle_norm

    grouped_xyz_inner_vec = grouped_xyz_local - torch.roll(grouped_xyz_local, 1, 2)
    grouped_xyz_inner_length = torch.norm(grouped_xyz_inner_vec, dim=-1, keepdim=True) # nn lengths
    grouped_xyz_inner_unit = grouped_xyz_inner_vec / grouped_xyz_inner_length
    grouped_xyz_inner_unit[grouped_xyz_inner_unit != grouped_xyz_inner_unit] = 0  # set nan to zero
    grouped_xyz_inner_angle_0 = (grouped_xyz_inner_unit * grouped_xyz_norm).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_1 = (grouped_xyz_inner_unit * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = (grouped_xyz_norm * torch.roll(grouped_xyz_norm, 1, 2)).sum(-1, keepdim=True)
    grouped_xyz_inner_angle_2 = torch.acos(torch.clamp(grouped_xyz_inner_angle_2, -1 + epsilon, 1 - epsilon))
    D_1 = (grouped_xyz_inner_angle_0 < grouped_xyz_inner_angle_1)
    D_1[D_1 ==0] = -1
    grouped_xyz_inner_angle_2 = D_1.float() * grouped_xyz_inner_angle_2

    proj_inner_angle_feat = dots_sorted - torch.roll(dots_sorted, 1, 2)
    proj_inner_angle_feat[:,:,0,0] = (-3) - dots_sorted[:,:,-1,0]

    ri_feat = torch.cat([grouped_xyz_length, 
                            proj_inner_angle_feat,
                            grouped_xyz_angle_0,
                            grouped_xyz_angle_1,
                            grouped_xyz_angle_norm,
                            grouped_xyz_inner_angle_0,
                            grouped_xyz_inner_angle_1,
                            grouped_xyz_inner_angle_2], dim=-1)

    return ri_feat, idx_ordered

class RIConv2SetAbstraction(nn.Module):
    def __init__(self, nsample, in_channel, mlp):
        super(RIConv2SetAbstraction, self).__init__()
        self.nsample = nsample
        self.prev_mlp_convs = nn.ModuleList()
        self.prev_mlp_bns = nn.ModuleList()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        
        in_channel_0 = 8
        mlp_0 = [32, 64]
        last_channel = in_channel_0
        for out_channel in mlp_0:
            self.prev_mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.prev_mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, norm, points):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            norm: input normal vector, [B, N, 3]
            points: input points (feature) data, [B, N, C]
        Return:
            ri_feat: created ri features, [B, N, C]
        """
        B, N, C = xyz.shape

        ri_feat, idx_ordered, idx = sample_and_group(self.nsample, xyz, norm)

        # lift
        ri_feat = ri_feat.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.prev_mlp_convs):
            bn = self.prev_mlp_bns[i]
            ri_feat =  F.relu(bn(conv(ri_feat)))

        # concat previous layer features
        if points is not None:
            if idx_ordered is not None:
                grouped_points = index_points(points, idx_ordered)
            else:
                grouped_points = points.view(B, 1, N, -1)
            grouped_points = grouped_points.permute(0, 3, 2, 1)
            new_points = torch.cat([ri_feat, grouped_points], dim=1)
        else:
            new_points = ri_feat

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        #ri_feat = torch.max(new_points, 2)[0]     # maxpooling
        ri_feat = torch.mean(new_points, 2)      # averagepooling
        ri_feat = ri_feat.permute(0, 2, 1)
        return ri_feat, idx