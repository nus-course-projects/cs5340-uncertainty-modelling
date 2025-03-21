import os

from utils.dataset import StreamingVideoDataset
from torch.utils.data import DataLoader


def get_video_dataloader(binary_file: str, index_file: str, batch_size: int = 4, num_workers: int = 2, shuffle: bool = True):
    """
    Returns a PyTorch DataLoader for the video dataset.

    Args:
        binary_file (str): Path to the binary file containing packed videos.
        index_file (str): Path to the index JSON file.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker threads for data loading.
        shuffle (bool): Whether to shuffle the dataset.

    Returns:
        DataLoader: A PyTorch DataLoader that yields (video, label, metadata).
    """
    dataset = StreamingVideoDataset(binary_file, index_file, label_threshold=100)  # Adjust threshold if needed
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader