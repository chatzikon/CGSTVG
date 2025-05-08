from JEPA.src.datasets.data_manager import init_data
from timm.data import create_transform as timm_make_transforms
import torchvision.transforms as transforms
from JEPA.src.utils.distributed import init_distributed


def make_dataloader(
    logger,
    dataset_name,
    root_path,
    image_folder,
    batch_size,
    world_size,
    rank,
    resolution=224,
    training=False,
    subset_file=None
):
    normalization = ((0.485, 0.456, 0.406),
                     (0.229, 0.224, 0.225))
    if training:
        logger.info('implementing auto-agument strategy')
        transform = timm_make_transforms(
            input_size=resolution,
            is_training=training,
            auto_augment='original',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=normalization[0],
            std=normalization[1])
    else:
        transform = transforms.Compose([
            transforms.Resize(size=int(resolution * 256/224)),
            transforms.CenterCrop(size=resolution),
            transforms.ToTensor(),
            transforms.Normalize(normalization[0], normalization[1])])

    data_loader, _ = init_data(
        data=dataset_name,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        root_path=root_path,
        image_folder=image_folder,
        training=training,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader


def image_loader(logger,cfg):

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    train_loader = make_dataloader(
        logger,
            dataset_name=cfg.data.dataset_name,
            root_path=cfg.data.root_path,
            resolution=cfg.data.resolution,
            image_folder=cfg.data.image_folder,
            batch_size=cfg.optimization.batch_size,
            world_size=world_size,
            rank=rank,
            training=True)

    val_loader = make_dataloader(
        logger,
        dataset_name=cfg.data.dataset_name,
        root_path=cfg.data.root_path,
        resolution=cfg.data.resolution,
        image_folder=cfg.data.image_folder,
        batch_size=cfg.optimization.batch_size,
        world_size=world_size,
        rank=rank,
        training=False)

    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    return train_loader, val_loader