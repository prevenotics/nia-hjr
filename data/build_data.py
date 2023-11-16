import os
import torch.distributed as dist
import torch
from data.hjr_dataset import HJRDataset, HJRDataset_for_test


def build_data_loader(dataset_name, mode, csv_path, imgtype, sample_point, batch_size, num_workers, local_rank, patch, band_patch, band, shuffle=True):
    
    if dataset_name =='hjr' and mode =='train':
        dataset = HJRDataset(csv_file=csv_path, imgtype=imgtype, sample_point=sample_point, patch=patch, band_patch=band_patch, band = band)
    elif dataset_name =='hjr' and mode =='test':
        dataset = HJRDataset_for_test(csv_file=csv_path, imgtype=imgtype, sample_point=sample_point, patch=patch, band_patch=band_patch, band = band)
    
    
    # local_rank = os.environ['LOCAL_RANK']
    print(f"local rank {local_rank} / global rank {dist.get_rank()} successfully build dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return data_loader