from io import BytesIO
import json
import os
import struct
from typing import Iterable, Tuple
import torch
import cv2
import matplotlib.pyplot as plt
import av
from tqdm import tqdm
from utils.metadata import MetadataDict


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
  def __init__(self, binary_file: str, index_file: str, label_threshold: int) -> None:
    self.binary_file = binary_file
    self.label_threshold = label_threshold

    with open(index_file, 'r', encoding='utf-8') as f:
      self.full_index = json.load(f)
    self.index = [entry for entry in self.full_index if entry['label'] < label_threshold]

  def __len__(self) -> int:
    return len(self.index)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
    entry = self.index[idx]
    with open(self.binary_file, 'rb') as f:
      f.seek(entry['offset'])
      video_len = struct.unpack('I', f.read(4))[0]
      video_bytes = f.read(video_len)
      metadata_len = struct.unpack('I', f.read(4))[0]
      metadata_bytes = f.read(metadata_len)
      metadata = json.loads(metadata_bytes.decode('utf-8'))

    video = self.decode_video(video_bytes)
    label = metadata['label']
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


def show_video_frames(video_tensor: torch.Tensor) -> None:
  video_numpy = video_tensor.permute(0, 2, 3, 1).numpy()
  for frame in video_numpy:
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def load_msasl(data_dir: str, label_threshold: int) -> Tuple[StreamingVideoDataset, StreamingVideoDataset, StreamingVideoDataset]:
  test_binary_file = os.path.join(data_dir, "test", "test.bin")
  test_index_file = os.path.join(data_dir, "test", "index.json")
  test_dataset = StreamingVideoDataset(test_binary_file, test_index_file, label_threshold)
  print(f"[TEST] Loaded {len(test_dataset)} videos with label < {label_threshold}")

  train_binary_file = os.path.join(data_dir, "train", "train.bin")
  train_index_file = os.path.join(data_dir, "train", "index.json")
  train_dataset = StreamingVideoDataset(train_binary_file, train_index_file, label_threshold)
  print(f"[TRAIN] Loaded {len(train_dataset)} videos with label < {label_threshold}")

  validation_binary_file = os.path.join(data_dir, "validation", "validation.bin")
  validation_index_file = os.path.join(data_dir, "validation", "index.json")
  validation_dataset = StreamingVideoDataset(validation_binary_file, validation_index_file, label_threshold)
  print(f"[VALIDATION] Loaded {len(validation_dataset)} videos with label < {label_threshold}")

  return test_dataset, train_dataset, validation_dataset
