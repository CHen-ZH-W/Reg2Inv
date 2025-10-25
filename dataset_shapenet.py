from geotransformer.datasets.registration.shapenet.dataset import ShapeNetPairDataset_train,ShapeNetPairDataset_test,ShapeNetDataset_train
from geotransformer.utils.data import (
    registration_collate_fn_stack_mode,
    calibrate_neighbors_stack_mode,
    build_dataloader_stack_mode,
)


def train_valid_data_loader(cfg, distributed):
    train_dataset = ShapeNetPairDataset_train(
        cfg.data.dataset_root,
        cfg.data.class_name,
        "train",
        voxel_size=cfg.data.voxel_size,
        num_points=cfg.data.num_points,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.train.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    neighbor_limits, voxel_size = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
    )
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed,
    )

    valid_dataset = ShapeNetPairDataset_train(
        cfg.data.dataset_root,
        cfg.data.class_name,
        "val",
        voxel_size=cfg.data.voxel_size,
        num_points=cfg.data.num_points,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.train.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        deterministic=True,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    valid_loader = build_dataloader_stack_mode(
        valid_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        distributed=distributed,
    )

    return train_loader, valid_loader, neighbor_limits, voxel_size


def test_data_loader(cfg):
    train_dataset = ShapeNetPairDataset_train(
        cfg.data.dataset_root,
        cfg.data.class_name,
        "train",
        voxel_size=cfg.data.voxel_size,
        num_points=cfg.data.num_points,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.train.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    neighbor_limits, voxel_size = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
    )
    test_dataset = ShapeNetPairDataset_test(
        cfg.data.dataset_root,
        cfg.data.class_name,
        "test",
        voxel_size=cfg.data.voxel_size,
        num_points=cfg.data.num_points,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        deterministic=True,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        precompute_data=False
    )

    return test_loader, neighbor_limits, voxel_size

def bank_data_loader(cfg):
    train_dataset = ShapeNetPairDataset_train(
        cfg.data.dataset_root,
        cfg.data.class_name,
        "train",
        voxel_size=cfg.data.voxel_size,
        num_points=cfg.data.num_points,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        noise_magnitude=cfg.train.noise_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        deterministic=False,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    neighbor_limits, voxel_size = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
    )
    test_dataset = ShapeNetDataset_train(
        cfg.data.dataset_root,
        cfg.data.class_name,
        "test",
        voxel_size=cfg.data.voxel_size,
        num_points=cfg.data.num_points,
        rotation_magnitude=cfg.data.rotation_magnitude,
        translation_magnitude=cfg.data.translation_magnitude,
        keep_ratio=cfg.data.keep_ratio,
        crop_method=cfg.data.crop_method,
        deterministic=True,
        twice_sample=cfg.data.twice_sample,
        twice_transform=cfg.data.twice_transform,
        return_normals=False,
        return_occupancy=True,
    )
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        precompute_data=False
    )

    return test_loader, neighbor_limits, voxel_size