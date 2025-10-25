import torch.nn as nn
from sklearn.neighbors import KNeighborsRegressor,NearestNeighbors
import numpy as np
import torch

class KNN(nn.Module):

    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k

    def forward(self, ref, query):  #B N 3  B 1024 3
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            D, I = [], []
            for bi in range(batch_size):
                point_cloud = ref[bi]
                sample_points = query[bi]
                point_cloud = point_cloud.detach().cpu()
                sample_points = sample_points.detach().cpu()
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(point_cloud.float(), point_cloud.float())
                distances, indices = knn.kneighbors(sample_points, n_neighbors=self.k)

                D.append(distances)
                I.append(indices)
            D = torch.from_numpy(np.array(D))
            I = torch.from_numpy(np.array(I))
        return D, I

class KNN_CHECK(nn.Module):

    def __init__(self, k):
        super(KNN_CHECK, self).__init__()
        self.k = k

    def forward(self, xyz, center, idx):
        assert xyz.size(0) == center.size(0), "ref.shape={} != query.shape={}".format(xyz.shape, center.shape)
        with torch.no_grad():
            batch_size = xyz.size(0)
            C, I1, I2 = [], [], []
            for bi in range(batch_size):
                point_cloud = xyz[bi]
                point_cloud = point_cloud.detach().cpu()
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(point_cloud.float(), point_cloud.float())
                
                
                sample_points = center[bi]
                sample_points = sample_points.detach().cpu()
                sample_idx    = idx[bi]
                sample_idx    = sample_idx.detach().cpu()
                
                center_list =[]
                center_idx_list = []
                group_idx_list = []

                while sample_idx.numel() > 0:
                    _, indices = knn.kneighbors(sample_points, n_neighbors=self.k)
                    indices = torch.from_numpy(indices)
                    neighborhood = point_cloud[indices, :]
                    group_centers = torch.mean(neighborhood, dim=1)
                    centers_distances = torch.norm(neighborhood - group_centers.unsqueeze(1), dim=2)
                    _, top_indices = torch.topk(centers_distances, k=5, dim=1, largest=False)
                    center_in_top = torch.any(top_indices == 0, dim=1)

                    selected_points = sample_points[center_in_top]
                    selected_idx = sample_idx[center_in_top]
                    selected_indices = indices[center_in_top]
                    
                    center_list.append(selected_points)
                    center_idx_list.append(selected_idx)
                    group_idx_list.append(selected_indices)

                    new_center_idx = top_indices[~center_in_top, 0]
                    indices = indices[~center_in_top]
                    sample_idx = indices[torch.arange(indices.size(0)), new_center_idx]
                    sample_points = point_cloud[sample_idx,:]
                c = torch.cat(center_list, dim=0).numpy()
                idx1 = torch.cat(center_idx_list, dim=0).numpy()
                idx2 = torch.cat(group_idx_list, dim=0).numpy()
                C.append(c)
                I1.append(idx1)
                I2.append(idx2)
            C  = torch.from_numpy(np.array(C))
            I1 = torch.from_numpy(np.array(I1))
            I2 = torch.from_numpy(np.array(I2))
        return C, I1, I2

def fill_missing_values(x_data,x_label,y_data, k=1):
    # 创建最近邻居模型
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)
    # 找到每个点的最近邻居
    distances, indices = nn.kneighbors(y_data)
    avg_values = np.mean(x_label[indices], axis=1)
    return avg_values

def average_pooling(x_data,x_label,y_data, k=32):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(x_data)
    # 找到每个点的最近邻居
    distances, indices = nn.kneighbors(y_data)
    avg_values = np.mean(x_label[indices], axis=1)
    return avg_values