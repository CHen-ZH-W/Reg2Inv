import contextlib
import logging
import os
import sys
import click
import numpy as np
import torch
import tqdm
import patchcore.sampler
from sklearn.metrics import roc_auc_score, average_precision_score
import time
from utils.visualization import plot_sample_o3d,plot_anomaly_score_distributions,plot_sample_o3d_single
from utils.utils import set_torch_device,fix_seeds
import pandas as pd
from dataset_shapenet import test_data_loader, bank_data_loader
from config_shapenet import make_cfg
from geotransformer.utils.torch import to_cuda
from model import create_model
from geotransformer.modules.ops import apply_transform
from utils.riconv_utils import index_points
from utils.cpu_knn import KNN

LOGGER = logging.getLogger(__name__)


@click.group(chain=True)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--memory_size", type=int, default=10000, show_default=True)
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
@click.option("--snapshots", default=None)
@click.option("--dataset_name", default='airplane')
@click.option("--result_path", default='./results/shapenet/')
@click.option("--vis",is_flag=True, default=False)
@click.option("--faiss_on_gpu", is_flag=True, default=True)
@click.option("--faiss_num_workers", type=int, default=8)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    gpu,
    seed,
    memory_size,
    anomaly_scorer_num_nn,
    vis,
    faiss_on_gpu,
    faiss_num_workers,
    snapshots,
    dataset_name,
    result_path,

):
    methods = {key: item for (key, item) in methods}

    device = set_torch_device(gpu)
    
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    
    cfg = make_cfg()

    DATASET_NAME = [dataset_name]

    df = pd.DataFrame(DATASET_NAME, columns=['Category'])

    if( not os.path.exists(result_path+dataset_name)):
        os.makedirs(result_path+dataset_name)
    fix_seeds(seed, device)
    
    cfg.data.class_name = dataset_name

    train_loader, neighbor_limits, voxel_size = bank_data_loader(cfg)
    test_loader, _, _ = test_data_loader(cfg)

    with device_context:

        torch.cuda.empty_cache()
        snapshot = snapshots + 'snapshots/snapshot.pth.tar'
        deep_feature_extractor = create_model(cfg, voxel_size, neighbor_limits).cuda()
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))
        deep_feature_extractor.load_state_dict(state_dict['model'], strict=True)
        deep_feature_extractor.eval()

        train_features = []
        total_iterations = len(train_loader)
        pbar = tqdm.tqdm(enumerate(train_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            with torch.no_grad():
                data_dict = to_cuda(data_dict)
                output_dict = deep_feature_extractor(data_dict)
                est_transform = output_dict['estimated_transform']
                
                src_points = output_dict['src_points_f']
                est_src_points = apply_transform(src_points, est_transform)
                position_feature = est_src_points.cpu().numpy()
                shape_feature = output_dict['src_feats_i'].cpu().numpy()
                features = np.concatenate([position_feature, shape_feature], axis=1)
  
                train_features.append(features)
                torch.cuda.empty_cache()
        
        train_features = np.concatenate(train_features, axis=0)
        position_feature = train_features[:,:3]
        shape_feature = train_features[:,3:]
        position_norm = max(np.linalg.norm(position_feature, axis=1))
        shape_norm = max(np.linalg.norm(shape_feature, axis=1))
        position_feature = position_feature / position_norm
        shape_feature = shape_feature / shape_norm 
        train_features = np.concatenate([position_feature, shape_feature], axis=1)
        
        labels_gt = []
        masks_gt = list()
        
        all_features = []
        all_xyz_sampled = []
        all_point_clouds = []

        total_iterations = len(test_loader)
        pbar = tqdm.tqdm(enumerate(test_loader), total=total_iterations)
        for iteration, data_dict in pbar:
            label = data_dict['label']
            mask = data_dict['mask']
            labels_gt.append(label)
            masks_gt.extend(mask.flatten().numpy())

            with torch.no_grad():
                data_dict = to_cuda(data_dict)
                output_dict = deep_feature_extractor(data_dict)
                input_pointcloud = data_dict['raw_points']
                point_cloud = input_pointcloud.cpu()
                est_transform = output_dict['estimated_transform']

                src_points = output_dict['src_points_f']        
                est_src_points = apply_transform(src_points, est_transform)
                position_feature = est_src_points.cpu().numpy()  
                shape_feature = output_dict['src_feats_i'].cpu().numpy() 
                position_feature = position_feature / position_norm
                shape_feature = shape_feature / shape_norm
                features = np.concatenate([position_feature, shape_feature], axis=1)

                src_points = output_dict['src_points_f']   
                check_points = src_points.cpu().unsqueeze(0)
                knn = KNN(64)          
                _, group_idx = knn(check_points, check_points) 
                neighborhood = index_points(check_points, group_idx).squeeze(0)
                check_points = check_points.squeeze(0)
                group_centers = torch.mean(neighborhood, dim=1)
                centers_distances = torch.norm(neighborhood - group_centers.unsqueeze(1), dim=2)
                _, top_indices = torch.topk(centers_distances, k=1, dim=1, largest=False)
                center_in_top = torch.any(top_indices == 0, dim=1)
                xyz_sampled = src_points[center_in_top].cpu()
                features = features[center_in_top.cpu().numpy()]
                features = np.asarray(features,order='C').astype('float32')

                all_features.append(features)
                all_xyz_sampled.append(xyz_sampled)
                all_point_clouds.append(point_cloud)

        import patchcore.common
        import patchcore.patchcore
        import patchcore.metrics

        sampler = methods["get_sampler"](
            device,
        )
        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
        PatchCore = patchcore.patchcore.PatchCore(device)
        PatchCore.load(
            device=device,
            target_embed_dimension=1024,
            featuresampler=sampler,
            anomaly_scorer_num_nn=anomaly_scorer_num_nn,
            nn_method=nn_method
        )
        
        memory_feature = PatchCore.build_memory_bank(train_features, memory_size)

        aggregator = {"scores": [], "segmentations": []}
        scores, segmentations = PatchCore.predict_score(
            all_features, all_xyz_sampled, all_point_clouds
        )
        aggregator["scores"].append(scores)
        scores = np.array(aggregator["scores"])
        min_scores = scores.min(axis=-1).reshape(-1, 1)
        max_scores = scores.max(axis=-1).reshape(-1, 1)
        scores = (scores - min_scores) / (max_scores - min_scores)
        scores = np.mean(scores, axis=0)
        ap_seg = np.asarray(segmentations)
        ap_seg = ap_seg.flatten()
        min_seg = np.min(ap_seg)
        max_seg = np.max(ap_seg)
        ap_seg = (ap_seg-min_seg)/(max_seg-min_seg)

        LOGGER.info("Computing evaluation metrics.")

        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            scores, labels_gt
        )["auroc"]
        img_ap = average_precision_score(labels_gt,scores)

        ap_mask = np.stack(masks_gt, axis=0)
        ap_mask = ap_mask.flatten().astype(np.int32)

        pixel_ap = average_precision_score(ap_mask,ap_seg)
        full_pixel_auroc = roc_auc_score(ap_mask,ap_seg)

        auroc = round(auroc, 3)
        full_pixel_auroc = round(full_pixel_auroc, 3)
        img_ap = round(img_ap, 3)
        pixel_ap = round(pixel_ap, 3)

        print('Task:{}, image_auc:{}, pixel_auc:{}, image_ap:{}, pixel_ap:{}'.format
                (dataset_name,auroc,full_pixel_auroc,img_ap,pixel_ap))
        
        df['image_rocaucs'] = auroc
        df['pixel_rocaucs'] = full_pixel_auroc
        df['image_apaucs'] = img_ap
        df['pixel_apaucs'] = pixel_ap

        if(vis):
            cur_pc_idx = 0
            idx = 0
            img_scores = list()
            pcds = list()
            paths = list()
            labels = list()
            gts = []
            predictions = []

            total_iterations = len(test_loader)
            pbar = tqdm.tqdm(enumerate(test_loader), total=total_iterations)
            for iteration, data_dict in pbar:
                    pointcloud = data_dict['raw_points']
                    mask = data_dict['mask']
                    sample_path = data_dict['path']

                    pc_length = pointcloud.shape[0]
                    anomaly_cur = ap_seg[cur_pc_idx:cur_pc_idx+pc_length]
                    path_list = sample_path.split('/')
                    target_path = os.path.join(result_path,dataset_name,path_list[-1])
                
                    img_scores.append(scores[idx])
                    labels.extend(mask.flatten().numpy())
                    predictions.append(anomaly_cur.flatten())
                    pcds.append(pointcloud.squeeze(0).cpu().numpy())
                    paths.append(target_path)
                    gts.append(mask.flatten().numpy())
                    idx = idx + 1
                    cur_pc_idx = cur_pc_idx+pc_length
            plot_sample_o3d_single(pcds, paths, predictions, gts, img_scores)
            plot_anomaly_score_distributions(ap_seg, labels, paths)

    file_path = result_path + 'results.csv'
    if os.path.exists(file_path):
        mode = 'a'
        header = False
    else:
        mode = 'a'
        header = True
    df.to_csv(file_path, mode=mode, index=False, header=header)

@main.command("sampler")
@click.argument("name", type=str, default="approx_greedy_coreset")
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()