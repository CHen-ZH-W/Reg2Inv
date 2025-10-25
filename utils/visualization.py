import open3d as o3d
import numpy as np
import cv2
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scienceplots
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
from tqdm import tqdm


def vis_pointcloud(path=None):
    pcd = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([pcd])
    
def vis_pointcloud_withcoord(path=None):
    pcd = o3d.io.read_point_cloud(path)
    FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd,FOR1])

def vis_pointcloud_np(xyz=None):
    colors = np.repeat([[0.4, 0.4, 0.4]], xyz.shape[0], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def vis_pointcloud_gt(path=None):
    gt_pc = np.loadtxt(path)
    gt = gt_pc[:,3]
    new_colors = np.zeros_like(gt_pc[:,:3])
    anomaly_pos = gt==1
    normal_pos = gt==0
    new_colors[normal_pos] = [0.4,0.4,0.4]
    new_colors[anomaly_pos] = [1,0,0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_pc[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    o3d.visualization.draw_geometries([pcd])
    
def vis_pointcloud_gt_voxel_down(path=None,voxel_size=0.5):
    gt_pc = np.loadtxt(path)
    gt = gt_pc[:,3]
    new_colors = np.zeros_like(gt_pc[:,:3])
    anomaly_pos = gt==1
    normal_pos = gt==0
    new_colors[normal_pos] = [0.4,0.4,0.4]
    new_colors[anomaly_pos] = [1,0,0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_pc[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    pcd_new.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd_new])

def cv2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def vis_pointcloud_anomalymap(point_cloud, anomaly_map):
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd])
    
def vis_pointcloud_anomalymap_pcdpath(pcd_path, anomaly_map):
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    o3d.visualization.draw_geometries([pcd])

def save_anomalymap(pcd_path, anomaly_map, target_path):
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd_points = np.asarray(pcd.points)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    o3d.io.write_point_cloud(target_path, pcd)

def save_anomalymap_points(points, anomaly_map, target_path):
    heatmap = cv2heatmap(anomaly_map*255)/255
    heatmap = heatmap.squeeze()
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(heatmap)
    o3d.io.write_point_cloud(target_path, pcd)

def norm_pcd(pcd):
    points_coord = np.asarray(pcd.points)
    center = np.average(points_coord,axis=0)
    new_points = points_coord-np.expand_dims(center,axis=0)
    pcd.points = o3d.utility.Vector3dVector(new_points)
    return pcd

def down_sample_voxel(pcd,voxel_size):
    pcd_new = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size)
    return pcd_new

def plot_attentions(xyz, attention_map, path,  num_patches=100):
    path_list = path[0].split('/')
    save_root_dir = './result/attentions/'
    if( not os.path.exists(save_root_dir+path_list[-3])):
        os.makedirs(save_root_dir+path_list[-3])
    target_path = os.path.join(save_root_dir,path_list[-3],path_list[-1])

    nh = attention_map.shape[1]
    attention_map = attention_map[0, 0, :, :].cpu().numpy()

    for i in range(num_patches):
        # Randomly select a patch
        patch_idx = np.random.randint(0, attention_map.shape[1])
        attention = attention_map[patch_idx]

        min_score, max_score = np.min(attention), np.max(attention)
        norm = Normalize(vmin=min_score, vmax=max_score)
        normalized_scores = norm(attention)
        colormap = plt.get_cmap('magma')
        s = colormap(normalized_scores)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(s)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        #o3d.visualization.draw_geometries([pcd])
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1080, height=1080, visible=False)
        vis.add_geometry(pcd)
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.clear_geometries()
        
        image = (np.asarray(image) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
        new_path = target_path[:-4] + f'_{i}.png' 
        cv2.imwrite(new_path, image)

def plot_average_attentions(xyz, attention_map, path,  num_patches=100):
    path_list = path[0].split('/')
    save_root_dir = './result/average_attentions/'
    if( not os.path.exists(save_root_dir+path_list[-3])):
        os.makedirs(save_root_dir+path_list[-3])
    target_path = os.path.join(save_root_dir,path_list[-3],path_list[-1])

    nh = attention_map.shape[1]
    attention_map = attention_map[0, :, :, :].cpu().numpy()
    attention_map = np.mean(attention_map, axis=0)

    for i in range(num_patches):
        # Randomly select a patch
        patch_idx = np.random.randint(0, attention_map.shape[1])
        attention = attention_map[patch_idx]

        min_score, max_score = np.min(attention), np.max(attention)
        norm = Normalize(vmin=min_score, vmax=max_score)
        normalized_scores = norm(attention)
        colormap = plt.get_cmap('magma')
        s = colormap(normalized_scores)[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(s)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        #o3d.visualization.draw_geometries([pcd])
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1080, height=1080, visible=False)
        vis.add_geometry(pcd)
        image = vis.capture_screen_float_buffer(do_render=True)
        vis.clear_geometries()
        
        image = (np.asarray(image) * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
        new_path = target_path[:-4] + f'_{i}.png' 
        cv2.imwrite(new_path, image)

def plot_attention(xyz, attention_map, path):
    path_list = path[0].split('/')
    save_root_dir = './result/attention/'
    if( not os.path.exists(save_root_dir+path_list[-3])):
        os.makedirs(save_root_dir+path_list[-3])
    target_path = os.path.join(save_root_dir,path_list[-3],path_list[-1])
    
    nh = attention_map.shape[1]
    attention_map = attention_map[0, :, 0, :].reshape(nh, -1).cpu().numpy()
    attention_0 = attention_map[0]

    min_score, max_score = np.min(attention_0), np.max(attention_0)
    norm = Normalize(vmin=min_score, vmax=max_score)
    normalized_scores = norm(attention_0)
    colormap = plt.get_cmap('magma')
    s = colormap(normalized_scores)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(s)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    
    image = (np.asarray(image) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    new_path = target_path[:-4] + f'.png' 
    cv2.imwrite(new_path, image)

def plot_average_attention(xyz, attention_map, path):
    path_list = path[0].split('/')
    save_root_dir = './result/average_attention/'
    if( not os.path.exists(save_root_dir+path_list[-3])):
        os.makedirs(save_root_dir+path_list[-3])
    target_path = os.path.join(save_root_dir,path_list[-3],path_list[-1])
    
    nh = attention_map.shape[1]
    attention_map = attention_map[0, :, 0, :].reshape(nh, -1).cpu().numpy()
    average_attention = np.mean(attention_map, axis=0)

    min_score, max_score = np.min(average_attention), np.max(average_attention)
    norm = Normalize(vmin=min_score, vmax=max_score)
    normalized_scores = norm(average_attention)
    colormap = plt.get_cmap('magma')
    s = colormap(normalized_scores)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(s)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    
    image = (np.asarray(image) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    new_path = target_path[:-4] + f'.png' 
    cv2.imwrite(new_path, image)

def plot_pca(xyz, pca_feature, path):
    path_list = path[0].split('/')
    save_root_dir = './result/pca/'
    if( not os.path.exists(save_root_dir+path_list[-3])):
        os.makedirs(save_root_dir+path_list[-3])
    target_path = os.path.join(save_root_dir,path_list[-3],path_list[-1])
    
    # 提取第一主成分
    first_component = pca_feature[:, 0]
    mean = np.mean(first_component)
    std = np.std(first_component)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    first_component_clipped = np.clip(first_component, lower_bound, upper_bound)
    first_component_normalized = (first_component_clipped - lower_bound) / (upper_bound - lower_bound)
    colormap = plt.get_cmap('coolwarm')
    s = colormap(first_component_normalized)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(s)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image = (np.asarray(image) * 255).astype(np.uint8)
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    second_component = pca_feature[:, 1]
    mean = np.mean(second_component)
    std = np.std(second_component)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    second_component_clipped = np.clip(second_component, lower_bound, upper_bound)
    second_component_normalized = (second_component_clipped - lower_bound) / (upper_bound - lower_bound)
    colormap = plt.get_cmap('coolwarm')
    s = colormap(second_component_normalized)[:, :3]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(xyz)
    pcd2.colors = o3d.utility.Vector3dVector(s)
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd2)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image = (np.asarray(image) * 255).astype(np.uint8)
    image2 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    third_component = pca_feature[:, 2]
    mean = np.mean(third_component)
    std = np.std(third_component)
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    third_component_clipped = np.clip(third_component, lower_bound, upper_bound)
    third_component_normalized = (third_component_clipped - lower_bound) / (upper_bound - lower_bound)
    colormap = plt.get_cmap('coolwarm')
    s = colormap(third_component_normalized)[:, :3]
    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(xyz)
    pcd3.colors = o3d.utility.Vector3dVector(s)
    pcd3.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd3)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image = (np.asarray(image) * 255).astype(np.uint8)
    image3 = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    height, width, channels = image.shape
    combined_image = np.zeros((height, 3 * width, channels), dtype=np.uint8)
    combined_image[:, 0:width, :] = image1
    combined_image[:, width:2*width, :] = image2
    combined_image[:, 2*width:3*width, :] = image3
    new_path = target_path[:-4] + f'.png' 
    cv2.imwrite(new_path, combined_image)

def plot_norm(xyz, feature, path):
    path_list = path[0].split('/')
    save_root_dir = './result/norm/'
    if( not os.path.exists(save_root_dir+path_list[-3])):
        os.makedirs(save_root_dir+path_list[-3])
    target_path = os.path.join(save_root_dir,path_list[-3],path_list[-1])
    
    # 计算 L2 范数
    l2_norm = np.linalg.norm(feature, axis=1)
    min_score, max_score = np.min(l2_norm), np.max(l2_norm)

    # 归一化到 [0, 1] 范围
    norm = Normalize(vmin=min_score, vmax=max_score)
    normalized_scores = norm(l2_norm)

    # 使用 matplotlib 的 magma 颜色映射
    colormap = plt.get_cmap('magma')
    s = colormap(normalized_scores)[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(s)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    #o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    
    image = (np.asarray(image) * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    new_path = target_path[:-4] + f'_{max_score:.3f}_{min_score:.3f}.png' 
    cv2.imwrite(new_path, image)

def plot_sample_o3d(pcds, paths, scores, gts, min_score, max_score, img_scores):
    total_number = len(pcds)
    
    color_map_scores = list()

    for idx in range(total_number):
        map_score = scores[idx]
        map_score = (map_score - min_score) / (max_score - min_score) * 255
        map_score = map_score.astype(np.uint8)
        s = cv2.applyColorMap(map_score, cv2.COLORMAP_JET)
        s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
        s = s / 255.
        color_map_scores.append(s)
    
    for idx, (pcd, color_map_score, gt, path, img_score) in enumerate(zip(pcds, color_map_scores, gts, paths, img_scores)):
        save_combinemap_2D(pcd, color_map_score, gt, path, img_score)        

def plot_sample_o3d_single(pcds, paths, scores, gts, img_scores):
    total_number = len(pcds)
    
    color_map_scores = list()

    for idx in range(total_number):
        map_score = scores[idx]
        min_score = min(map_score)
        max_score = max(map_score)
        map_score = (map_score - min_score) / (max_score - min_score) * 255
        map_score = map_score.astype(np.uint8)
        s = cv2.applyColorMap(map_score, cv2.COLORMAP_JET)
        s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
        s = s / 255.
        color_map_scores.append(s)
    for idx, (pcd, color_map_score, gt, path, img_score) in tqdm(enumerate(zip(pcds, color_map_scores, gts, paths, img_scores)), total=len(pcds), desc="Plot_image...", leave=False):
        save_combinemap_2D(pcd, color_map_score, gt, path, img_score)

def save_combinemap_2D(point_cloud, anomaly_map, mask, target_path, score):
    #heatmap = cv2heatmap(anomaly_map*255)/255
    #heatmap = heatmap.squeeze()
    anomaly_map = anomaly_map.squeeze()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(anomaly_map)

    #o3d.io.write_point_cloud(target_path, pcd)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd)
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)
    new_path = target_path[:-4] + '_scores.png'
    cv2.imwrite(new_path, image)
    #mask = mask.flatten().numpy()
    mask = mask.flatten()
    new_colors = np.zeros((mask.shape[0], 3))
    anomaly_pos = mask==1
    normal_pos = mask==0
    new_colors[normal_pos] = [0.4,0.4,0.4]
    new_colors[anomaly_pos] = [1,0,0]
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(point_cloud)
    pcd2.colors = o3d.utility.Vector3dVector(new_colors)
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd2)
    image2 = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image2 = cv2.cvtColor(np.asarray(image2) * 255, cv2.COLOR_RGB2BGR)
    new_path = target_path[:-4] + '_gt.png'
    cv2.imwrite(new_path, image2)
    '''
    new_path = target_path[:-4] + f'_{score:.5f}_combined.png'  
    height, width, channels = image.shape
    combined_image = np.zeros((height, 2 * width, channels), dtype=np.uint8)
    combined_image[:, 0:width, :] = image2
    combined_image[:, width:2*width, :] = image
    cv2.imwrite(new_path, combined_image)
    '''

def plot_anomaly_score_distributions(scores, ground_truths_list, save_paths):

    ground_truths = np.stack(ground_truths_list, axis=0)

    N_COUNT = 5000000

    layer_score = np.stack(scores, axis=0)
    normal_score = layer_score[ground_truths == 0]
    abnormal_score = layer_score[ground_truths != 0]

    plt.clf()
    plt.figure(figsize=(2, 1.5))
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    with plt.style.context(['science', 'ieee', 'no-latex']):
        sns.histplot(np.random.choice(normal_score, N_COUNT), color="green", bins=50, label='${d(p_n)}$',
                    stat='probability', alpha=.75)
        sns.histplot(np.random.choice(abnormal_score, N_COUNT), color="red", bins=50, label='${d(p_a)}$',
                    stat='probability', alpha = .75)

    save_path = os.path.join(os.path.dirname(save_paths[0]), f'distributions.jpg')

    plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_projection_pcd(point_cloud,target_path,data_idx):
    '''
    new_colors = np.full((point_cloud.shape[0], 3), [0.4,0.4,0.4])
    for idx in cloud_indices:
        new_colors[idx] = [1,0,0]
    '''
    colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
    new_colors = np.zeros((point_cloud.shape[0], 3))
    for i, cloud_id in enumerate(data_idx):
        if cloud_id < len(colors):
            new_colors[i] = colors[cloud_id]
            
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(new_colors)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd)
    vis.get_render_option().light_on = False 
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)
    cv2.imwrite(target_path, image)
    #o3d.io.write_point_cloud(target_path, pcd)

def vis_pointcloud_np_two(xyz1=None,xyz2=None):
    colors1 = np.repeat([[1, 0, 0]], xyz1.shape[0], axis=0)
    colors2 = np.repeat([[0, 1, 0]], xyz2.shape[0], axis=0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(xyz1)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(xyz2)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)

    o3d.visualization.draw_geometries([pcd1,pcd2])

def save_registration_result(src, tgt, src_trans, name, save_path):

    colors1 = np.repeat([[0.98824, 0.70589, 0.19216]], src.shape[0], axis=0)
    colors2 = np.repeat([[0.4, 0.60392, 0.72941]], tgt.shape[0], axis=0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(tgt)
    pcd2.colors = o3d.utility.Vector3dVector(colors2)
    pcd2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    vis.get_render_option().light_on = False 
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)

    pcd3 = o3d.geometry.PointCloud()
    pcd3.points = o3d.utility.Vector3dVector(src_trans)
    pcd3.colors = o3d.utility.Vector3dVector(colors1)
    pcd3.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd3)
    vis.add_geometry(pcd2)
    vis.get_render_option().light_on = False 
    image2 = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    image2 = cv2.cvtColor(np.asarray(image2) * 255, cv2.COLOR_RGB2BGR)
    
    height, width, channels = image.shape
    combined_image = np.zeros((height, 2 * width, channels), dtype=np.uint8)
    combined_image[:, 0:width, :] = image
    combined_image[:, width:2*width, :] = image2
    save_name = save_path + f'{name}.png'
    cv2.imwrite(save_name, image2)

def save_image(src, name, num, save_path):

    colors1 = np.repeat([[0.98824, 0.70589, 0.19216]], src.shape[0], axis=0)
    #colors1 = np.repeat([[0.4, 0.60392, 0.72941]], src.shape[0], axis=0)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(src)
    pcd1.colors = o3d.utility.Vector3dVector(colors1)
    pcd1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1080, height=1080, visible=False)
    vis.add_geometry(pcd1)
    
    #vis.get_render_option().light_on = False 
    image = vis.capture_screen_float_buffer(do_render=True)
    vis.clear_geometries()
    
    image = cv2.cvtColor(np.asarray(image) * 255, cv2.COLOR_RGB2BGR)

    save_name = save_path + f'{name}_{num}.png'
    cv2.imwrite(save_name, image)