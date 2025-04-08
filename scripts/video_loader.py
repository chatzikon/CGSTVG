from JEPA.src.datasets.data_manager import init_data
from JEPA.evals.video_classification_frozen.utils import make_transforms
from JEPA.src.utils.distributed import init_distributed



def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=8,
    subset_file=None
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader



def video_loader(logger,cfg):

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')



    train_loader = make_dataloader(
            dataset_type=cfg.data.dataset_type,
            root_path=cfg.data.dataset_train,
            resolution=cfg.data.resolution,
            frames_per_clip=cfg.pretrain.frames_per_clip,
            frame_step=cfg.data.frame_step,
            eval_duration=cfg.pretrain.clip_duration,
            num_segments=cfg.data.num_segments if cfg.optimization.attend_across_segments else 1,
            num_views_per_segment=1,
            allow_segment_overlap=True,
            batch_size=cfg.optimization.batch_size,
            world_size=world_size,
            rank=rank,
            training=True)

    val_loader = make_dataloader(
        dataset_type=cfg.data.dataset_type,
        root_path=cfg.data.dataset_val,
        resolution=cfg.data.resolution,
        frames_per_clip=cfg.pretrain.frames_per_clip,
        frame_step=cfg.data.frame_step,
        num_segments=cfg.data.num_segments,
        eval_duration=cfg.pretrain.clip_duration,
        num_views_per_segment=cfg.data.num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=cfg.optimization.batch_size,
        world_size=world_size,
        rank=rank,
        training=False)
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    return train_loader, val_loader