from io import BytesIO
import json
import os
import struct
from typing import Iterable, Tuple, Dict
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import av
from tqdm import tqdm
from IPython.display import HTML
from utils.metadata import MetadataDict
import csv
import numpy as np

class KeypointDataset(torch.utils.data.Dataset):
    """Dataset class for loading keypoint data from npz files"""
    
    def __init__(self, npz_path: str, label_map: dict = None):
        """
        Initialize KeypointDataset
        
        Args:
            npz_path (str): Path to .npz file containing keypoint data
            label_map (dict, optional): Mapping from original labels to consecutive integers
        """
        # Load data from npz file
        data = np.load(npz_path)
        self.keypoints = data['keypoints']  # Shape: (N, T, K, 2) 
        self.keypoint_scores = data['keypoint_scores']  # Shape: (N, T, K, 1)
        self.labels = data['labels']  # Shape: (N,)
        
        # Apply label mapping if provided
        if label_map is not None:
          # Filter to only include labels in the map
          mask = np.isin(self.labels, list(label_map.keys()))
          self.keypoints = self.keypoints[mask]
          self.keypoint_scores = self.keypoint_scores[mask]
          self.labels = np.array([label_map[label] for label in self.labels[mask]])
            
        # Convert to torch tensors
        self.keypoints = torch.from_numpy(self.keypoints).float()
        self.keypoint_scores = torch.from_numpy(self.keypoint_scores).float()
        self.labels = torch.from_numpy(self.labels).long()
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx (int): Index of sample to get
            
        Returns:
            tuple: (features, label) where features combines keypoints and scores
        """
        keypoints = self.keypoints[idx] / 224.0  # Shape: (T, K, 2)
        scores = self.keypoint_scores[idx]  # Shape: (T, K, 1) 
        label = self.labels[idx]
        
        # Combine keypoints and scores into features
        features = torch.cat([keypoints, scores], dim=-1)  # Shape: (T, K, 3)
        
        return features, label

def load_keypoints(data_dir: str, top_k_labels: int = None):
    """
    Load keypoints data from separate test, train, and validation files
    
    Args:
        data_dir (str): Directory containing keypoint data files
        top_k_labels (int, optional): Only include the top k most frequent labels
        
    Returns:
        tuple: (test_dataset, train_dataset, validation_dataset)
    """
    # Load separate datasets for test, train, and validation
    test_path = os.path.join(data_dir, "keypoints_test.npz")
    train_path = os.path.join(data_dir, "keypoints_train.npz")
    val_path = os.path.join(data_dir, "keypoints_validation.npz")
    
    # First, determine the label mapping from the training set
    temp_data = np.load(train_path)
    train_labels = temp_data['labels']
    
    # Create label map based on top_k most frequent labels in training set
    label_map = None
    if top_k_labels is not None:
        unique_labels, counts = np.unique(train_labels, return_counts=True)
        top_k_indices = np.argsort(counts)[-top_k_labels:]
        top_k_label_values = unique_labels[top_k_indices]
        label_map = {label: i for i, label in enumerate(top_k_label_values)}
    
    # Create dataset objects for each split using the same label map
    train_dataset = KeypointDataset(train_path, label_map)
    test_dataset = KeypointDataset(test_path, label_map)
    val_dataset = KeypointDataset(val_path, label_map)
    
    return test_dataset, train_dataset, val_dataset

class VideoPackerWithIndex:
  def __init__(self, videos_folder: str, metadata_file: str, output_file: str, index_file: str) -> None:
    self.videos_folder = videos_folder
    self.metadata_file = metadata_file
    self.output_file = output_file
    self.index_file = index_file

    with open(self.metadata_file, 'r', encoding='utf-8') as f:
      self.metadata = json.load(f)

    self.index: list[dict] = []

  def pack(self) -> None:
    with open(self.output_file, 'wb') as f:
      f.write(struct.pack('I', len(self.metadata)))
      for entry in tqdm(self.metadata):
        video_path = os.path.join(self.videos_folder, f"{entry['filename']}.mp4")
        with open(video_path, 'rb') as vf:
          video_bytes = vf.read()

        metadata_str = json.dumps(entry)
        metadata_bytes = metadata_str.encode('utf-8')

        offset = f.tell()
        f.write(struct.pack('I', len(video_bytes)))
        f.write(video_bytes)
        f.write(struct.pack('I', len(metadata_bytes)))
        f.write(metadata_bytes)

        self.index.append({
          'filename': f"{entry['filename']}.mp4",
          'offset': offset,
          'video_len': len(video_bytes),
          'metadata_len': len(metadata_bytes),
          'label': entry['label']
        })
    with open(self.index_file, 'w', encoding='utf-8') as f:
      json.dump(self.index, f, indent=2)
    print(f"Packed into {self.output_file} with index saved at {self.index_file}")


class StreamingVideoDataset(torch.utils.data.Dataset):
  def __init__(self, binary_file: str, index_file: str, label_map: dict = None) -> None:
    self.binary_file = binary_file
    
    with open(index_file, 'r', encoding='utf-8') as f:
      self.full_index = json.load(f)
    
    if label_map is not None:
      # Filter index to only include entries with labels in the provided label map
      self.index = [entry for entry in self.full_index if entry['label'] in label_map]
      self.label_map = label_map
    else:
      self.index = self.full_index
      self.label_map = {label: i for i, label in enumerate(set(entry['label'] for entry in self.full_index))}

  def __len__(self) -> int:
    return len(self.index)

  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
    entry = self.index[idx]
    with open(self.binary_file, 'rb') as f:
      f.seek(entry['offset'])
      video_len = struct.unpack('I', f.read(4))[0]
      video_bytes = f.read(video_len)
      metadata_len = struct.unpack('I', f.read(4))[0]
      metadata_bytes = f.read(metadata_len)
      metadata = json.loads(metadata_bytes.decode('utf-8'))

    video = self.decode_video(video_bytes)
    original_label = metadata['label']
    # Map the original label to the new label index
    label = self.label_map.get(original_label, -1)  # -1 for labels not in the map (shouldn't happen)
    return video, label, metadata

  @staticmethod
  def decode_video(video_bytes: bytes) -> torch.Tensor:
    video_stream = BytesIO(video_bytes)
    container = av.open(video_stream)
    frames = []
    for frame in container.decode(video=0):  # type: ignore
      frame_rgb = frame.to_rgb().to_ndarray()
      frame_tensor = torch.tensor(frame_rgb, dtype=torch.uint8).permute(2, 0, 1)
      frames.append(frame_tensor)

    return torch.stack(frames)

  def iterate_metadata(self) -> Iterable[MetadataDict]:
    with open(self.binary_file, 'rb') as f:
      for entry in self.index:
        f.seek(entry['offset'])
        video_len = struct.unpack('I', f.read(4))[0]
        f.seek(video_len, os.SEEK_CUR)  # Skip over video data

        metadata_len = struct.unpack('I', f.read(4))[0]
        metadata_bytes = f.read(metadata_len)
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        yield metadata

  def show_video(self, index: int) -> None:
    video, _, _ = self[index]
    video_numpy = video.permute(0, 2, 3, 1).numpy()  # Convert to (Frames, Height, Width, Channels)

    fig, ax = plt.subplots(figsize=(video_numpy.shape[2] / 100, video_numpy.shape[1] / 100))
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    frame_display = ax.imshow(video_numpy[0])
    ax.axis('off')

    def update(frame_idx):
      frame_display.set_data(video_numpy[frame_idx])
      return frame_display,

    anim = animation.FuncAnimation(fig, update, frames=len(video_numpy), interval=50, blit=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def load_msasl(data_dir: str, top_k_labels: int = None) -> Tuple[StreamingVideoDataset, StreamingVideoDataset, StreamingVideoDataset]:
  # First load all labels from the training dataset to determine the label mapping
  train_index_file = os.path.join(data_dir, "train", "index.json")
  with open(train_index_file, 'r', encoding='utf-8') as f:
    train_index = json.load(f)
  
  # Create label map based on top_k_labels
  label_map = None
  if top_k_labels is not None:
    # Get the top k most frequent labels
    label_counts = {}
    for entry in train_index:
      label = entry['label']
      label_counts[label] = label_counts.get(label, 0) + 1
    
    # Sort labels by frequency (descending) and take top k
    top_labels = sorted(label_counts.keys(), key=lambda x: label_counts[x], reverse=True)[:top_k_labels]
    label_map = {label: i for i, label in enumerate(top_labels)}
  
  # Create datasets using the same label map
  train_binary_file = os.path.join(data_dir, "train", "train.bin")
  train_dataset = StreamingVideoDataset(train_binary_file, train_index_file, label_map)
  print(f"[TRAIN] Loaded {len(train_dataset)} videos" + 
        (f" with top {top_k_labels} labels" if top_k_labels else ""))
  
  test_binary_file = os.path.join(data_dir, "test", "test.bin")
  test_index_file = os.path.join(data_dir, "test", "index.json")
  test_dataset = StreamingVideoDataset(test_binary_file, test_index_file, label_map)
  print(f"[TEST] Loaded {len(test_dataset)} videos" + 
        (f" with top {top_k_labels} labels" if top_k_labels else ""))

  validation_binary_file = os.path.join(data_dir, "validation", "validation.bin")
  validation_index_file = os.path.join(data_dir, "validation", "index.json")
  validation_dataset = StreamingVideoDataset(validation_binary_file, validation_index_file, label_map)
  print(f"[VALIDATION] Loaded {len(validation_dataset)} videos" + 
        (f" with top {top_k_labels} labels" if top_k_labels else ""))

  return test_dataset, train_dataset, validation_dataset


class ASLCitizenDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading ASL Citizen data from a CSV file.
    
    Expected CSV columns:
      - Participant ID: Identifier for the participant
      - Video file: Name (or relative path) of the video file (e.g., 'video1.mp4')
      - Gloss: The ASL sign gloss, used as the video label
      - ASL-LEX Code: Additional metadata (e.g., ASL-LEX identifier)
      
    Any additional columns in the CSV will be stored in the metadata.
    
    Args:
        csv_path (str): Path to the CSV file.
        videos_folder (str): Path to the folder containing the video files.
        label_map (dict, optional): A mapping from the original gloss (label) to integer indices.
                                    If provided, rows with Gloss not in the mapping will be skipped.
    """
    def __init__(self, csv_path: str, videos_folder: str, label_map: Dict[str, int] = None):
        self.videos_folder = videos_folder
        # Read CSV entries into a list of dictionaries
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.entries = [row for row in reader]
            
        # If a label mapping is provided, filter and remap entries to only include valid gloss labels
        if label_map is not None:
            filtered_entries = []
            for entry in self.entries:
                if entry['Gloss'] in label_map:
                    entry['mapped_label'] = label_map[entry['Gloss']]
                    filtered_entries.append(entry)
            self.entries = filtered_entries
        
        # If no label_map is provided, create one automatically from unique gloss labels
        if label_map is None:
            unique_labels = sorted({entry['Gloss'] for entry in self.entries})
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
        else:
            self.label_map = label_map

    def __len__(self) -> int:
        return len(self.entries)

    def decode_video(self, video_path: str) -> torch.Tensor:
        """
        Decodes a video from the given file path using PyAV.
        
        Returns:
            torch.Tensor: A tensor of shape (Frames, Channels, Height, Width)
        """
        container = av.open(video_path)
        frames = []
        for frame in container.decode(video=0):
            frame_rgb = frame.to_rgb().to_ndarray()  # (H, W, 3)
            # Convert to tensor and permute to (C, H, W)
            frame_tensor = torch.tensor(frame_rgb, dtype=torch.uint8).permute(2, 0, 1)
            frames.append(frame_tensor)
        return torch.stack(frames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, dict]:
        """
        Returns:
            tuple: (video, label, metadata)
              - video: Tensor of shape (Frames, Channels, Height, Width)
              - label: The integer label corresponding to the Gloss
              - metadata: Dictionary with extra metadata (e.g., Participant ID, Video file, Gloss, ASL-LEX Code, etc.)
        """
        entry = self.entries[idx]
        video_file = entry['Video file']
        video_filepath = os.path.join(self.videos_folder, video_file)
        video = self.decode_video(video_filepath)
        # Use the mapped label if available; otherwise, compute from label_map
        label = entry.get('mapped_label', self.label_map.get(entry['Gloss'], -1))
        return video, label, entry

    def show_video(self, idx: int) -> HTML:
        """
        Displays the video at the given index as an animation in Jupyter.
        
        Args:
            idx (int): Index of the video to display.
        
        Returns:
            IPython.display.HTML: HTML animation of the video.
        """
        video, _, _ = self[idx]
        # Convert video tensor to NumPy array with shape (Frames, Height, Width, Channels)
        video_numpy = video.permute(0, 2, 3, 1).numpy()
        fig, ax = plt.subplots(figsize=(video_numpy.shape[2] / 100, video_numpy.shape[1] / 100))
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        frame_display = ax.imshow(video_numpy[0])
        ax.axis('off')

        def update(frame_idx):
            frame_display.set_data(video_numpy[frame_idx])
            return frame_display,

        anim = animation.FuncAnimation(fig, update, frames=len(video_numpy), interval=50, blit=True)
        plt.close(fig)
        return HTML(anim.to_jshtml())

def load_asl_citizen(data_dir: str, top_k_labels: int = None) -> Tuple[ASLCitizenDataset, ASLCitizenDataset, ASLCitizenDataset]:
    """
    Loads the ASL Citizen dataset for train, test, and validation splits from CSV files.
    
    Assumes the following CSV files are located in data_dir:
      - train.csv
      - test.csv
      - val.csv
      
    Optionally, the dataset can be limited to only the top_k most frequent gloss labels (from train.csv).
    
    Args:
        data_dir (str): Directory containing the CSV files and video files.
        top_k_labels (int, optional): If provided, only include the top_k most frequent gloss labels.
        
    Returns:
        tuple: (test_dataset, train_dataset, val_dataset)
    """
    train_csv = os.path.join(data_dir, "splits/train.csv")
    test_csv = os.path.join(data_dir, "splits/test.csv")
    val_csv = os.path.join(data_dir, "splits/val.csv")
    videos_dir = os.path.join(data_dir, "videos")
    # Read training CSV entries to determine the label mapping based on Gloss
    with open(train_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        train_entries = [row for row in reader]
    
    label_map = None
    if top_k_labels is not None:
        label_counts = {}
        for entry in train_entries:
            gloss = entry['Gloss']
            label_counts[gloss] = label_counts.get(gloss, 0) + 1
        # Determine the top k labels by frequency
        top_labels = sorted(label_counts, key=lambda x: label_counts[x], reverse=True)[:top_k_labels]
        label_map = {label: i for i, label in enumerate(top_labels)}
    
    # Create dataset objects using the same label mapping across splits
    train_dataset = ASLCitizenDataset(train_csv, videos_folder=videos_dir, label_map=label_map)
    test_dataset = ASLCitizenDataset(test_csv, videos_folder=videos_dir, label_map=label_map)
    val_dataset = ASLCitizenDataset(val_csv, videos_folder=videos_dir, label_map=label_map)
    
    print(f"[TRAIN] Loaded {len(train_dataset)} videos" +
          (f" with top {top_k_labels} labels" if top_k_labels else ""))
    print(f"[TEST] Loaded {len(test_dataset)} videos" +
          (f" with top {top_k_labels} labels" if top_k_labels else ""))
    print(f"[VALIDATION] Loaded {len(val_dataset)} videos" +
          (f" with top {top_k_labels} labels" if top_k_labels else ""))
    
    return test_dataset, train_dataset, val_dataset