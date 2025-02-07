"""DataLoader creation for document rotation datasets.

This module handles the creation of DataLoader instances for training, validation,
and testing, with proper dataset combinations, augmentations, and preprocessing.
"""

from typing import List, Optional, Tuple
from torch.utils.data import DataLoader, ConcatDataset
from .dataset import *

def get_dataset_class(name: str) -> type:
    """Get dataset class by name.
    
    Args:
        name: Dataset name
        
    Returns:
        Dataset class
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    datasets = {
        "rvl-cdip": RVLCDIPDataset,
        "publaynet": PubLayNetDataset,
        "midv500": MIDV500Dataset,
        "sroie": SROIEDataset,
        "chartqa": ChartQADataset,
        "plotqa": PlotQADataset,
        "cord": CORDDataset,
        "tablebench": TableBenchDataset,
    }
    
    if name not in datasets:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available datasets: {list(datasets.keys())}"
        )
    
    return datasets[name]


def create_dataloaders(
    batch_size: int,
    num_workers: int,
    train_datasets: List[str],
    val_datasets: List[str],
    test_datasets: Optional[List[str]] = None,
    image_size: int = 384,
    val_split: float = 0.1,
    random_seed: int = 42,
    persistent_workers: bool = True,
    prefetch_factor: int = 2
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create training, validation, and test dataloaders.
    
    Args:
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes for data loading
        train_datasets: List of dataset names for training
        val_datasets: List of dataset names for validation
        test_datasets: Optional list of dataset names for testing
        image_size: Target image size
        val_split: Fraction of data to use for validation
        random_seed: Random seed for reproducible splits
        persistent_workers: Whether to maintain worker processes between iterations
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader will be None if test_datasets is None
    """
    train_datasets_list = []
    val_datasets_list = []
    
    # Create train and validation splits for each dataset
    for name in train_datasets:
        dataset_class = get_dataset_class(name)
        train_dataset, val_dataset = dataset_class.create_splits(
            root_dir=f"data/{name}",  # Assuming standard data directory structure
            img_size=image_size,
            val_split=val_split,
            random_seed=random_seed
        )
        train_datasets_list.append(train_dataset)
        val_datasets_list.append(val_dataset)
    
    # Add additional validation datasets if specified
    for name in val_datasets:
        if name not in train_datasets:  # Skip if already added from train split
            dataset_class = get_dataset_class(name)
            _, val_dataset = dataset_class.create_splits(
                root_dir=f"data/{name}",
                img_size=image_size,
                val_split=val_split,
                random_seed=random_seed
            )
            val_datasets_list.append(val_dataset)
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets_list)
    val_dataset = ConcatDataset(val_datasets_list)
    
    # Create test datasets if specified
    test_dataset = None
    if test_datasets:
        test_datasets_list = []
        for name in test_datasets:
            dataset_class = get_dataset_class(name)
            test_dataset = dataset_class(
                root_dir=f"data/{name}",
                split="test",
                img_size=image_size
            )
            test_datasets_list.append(test_dataset)
        test_dataset = ConcatDataset(test_datasets_list)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor
        )
    
    return train_loader, val_loader, test_loader 