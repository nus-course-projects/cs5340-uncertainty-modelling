import subprocess
import os
import uuid
import json
import pickle
import yt_dlp  # type: ignore
from utils.metadata import Metadata


def contains_file(folder_path: str, filename: str) -> bool:
  return os.path.isfile(os.path.join(folder_path, filename))


def fix_url(url: str) -> str:
  if url.startswith('https://'):
    return url
  return f"https://{url}"


def seconds_to_hmsms(seconds: float) -> str:
  # Calculate hours, minutes, seconds, and milliseconds
  hours = int(seconds // 3600)
  minutes = int((seconds % 3600) // 60)
  seconds = seconds % 60
  milliseconds = int((seconds % 1) * 1000)

  # Format the time as HH:MM:SS.ms
  return f"{hours:02}:{minutes:02}:{int(seconds):02}.{milliseconds:03}"


def download_video(url: str, output_dir: str, output_filename: str, start_time: str, duration: float) -> None:
  video_id = url.split('?v=')[1]
  download_video_filename = f"{video_id}.mp4"
  base_output_path = os.path.join(output_dir, download_video_filename)
  if not contains_file(output_dir, f"{video_id}.mp4"):
    print(f"Downloading new video from {url}")
    ydl_opts = {
      'format': 'bestvideo[ext=mp4]',
      'outtmpl': base_output_path,
      'quiet': True,
      'postprocessors': []
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      ydl.download([url])

  print(f"Cropping video {video_id}")
  trimmed_path = os.path.join(output_dir, "trimmed", f"{output_filename}.mp4")
  subprocess.run([
    "mpv", base_output_path,
    f"--start={start_time}",
    f"--length={duration:.3f}",
    f"--o={trimmed_path}",
  ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_dataset(json_file: str, data_dir: str) -> None:
  with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

  metadata_dict = {}
  metadata_json = []
  error_count = 0
  for i, entry in enumerate(data):
    video_id = str(uuid.uuid4())
    filename = f"{entry['clean_text']}_{video_id}"
    fixed_url = fix_url(entry['url'])
    start_time = seconds_to_hmsms(float(entry['start_time']))
    duration = float(entry['end_time']) - float(entry['start_time'])
    try:
      print(f"{i}/{len(data)}")
      download_video(fixed_url, data_dir, filename, start_time, duration)
    except Exception as e:
      error_count += 1
      print(f"[{error_count}] Error building {filename}: {e}")
      continue
    metadata_dict[video_id] = Metadata(
      id=video_id,
      org_text=entry['org_text'],
      clean_text=entry['clean_text'],
      start_time=float(entry['start_time']),
      end_time=float(entry['end_time']),
      signer_id=int(entry['signer_id']),
      signer=int(entry['signer']),
      start=int(entry['start']),
      end=int(entry['end']),
      file=entry['file'],
      label=int(entry['label']),
      height=float(entry['height']),
      width=float(entry['width']),
      fps=float(entry['fps']),
      url=entry['url'],
      text=entry['text'],
      box=[float(box_entry) for box_entry in entry["box"]],
      filename=filename,
    )
    metadata_json.append(metadata_dict[video_id].as_dict())

  updated_json_path = os.path.join(data_dir, "metadata.json")
  with open(updated_json_path, 'w', encoding='utf-8') as f:
    json.dump(metadata_json, f, indent=2)

  pickle_path = os.path.join(data_dir, "metadata.pkl")
  with open(pickle_path, "wb") as f:
    pickle.dump(metadata_dict, f, pickle.HIGHEST_PROTOCOL)

  print(f"Processing complete. Updated metadata saved at {updated_json_path}")
  print(f"Pickle file saved at {pickle_path}")


class DatasetDownloader:
  def __init__(self, input_json_dir: str, output_data_dir: str) -> None:
    self.input_test_file = os.path.join(input_json_dir, "MSASL_test.json")
    self.input_train_file = os.path.join(input_json_dir, "MSASL_train.json")
    self.input_validation_file = os.path.join(input_json_dir, "MSASL_val.json")

    self.output_test_dir = os.path.join(output_data_dir, "test")
    trimmed_test_dir = os.path.join(self.output_test_dir, "trimmed")
    self.output_train_dir = os.path.join(output_data_dir, "train")
    trimmed_train_dir = os.path.join(self.output_train_dir, "trimmed")
    self.output_validation_dir = os.path.join(output_data_dir, "validation")
    trimmed_validation_dir = os.path.join(self.output_validation_dir, "trimmed")

    os.makedirs(trimmed_test_dir, exist_ok=True)
    os.makedirs(trimmed_train_dir, exist_ok=True)
    os.makedirs(trimmed_validation_dir, exist_ok=True)

  def download(self) -> None:
    build_dataset(self.input_test_file, self.output_test_dir)
    build_dataset(self.input_train_file, self.output_train_dir)
    build_dataset(self.input_validation_file, self.output_validation_dir)
