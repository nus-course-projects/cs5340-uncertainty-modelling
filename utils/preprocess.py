import concurrent.futures
import json
import os
import cv2
import av
import numpy as np
from tqdm import tqdm


MAX_RETRIES = 5


def write_video(directory: str, filename: str, frames, fps: int = 25) -> None:
    height, width, _ = frames[0].shape

    output_filename = os.path.join(directory, f"{filename}.mp4")
    output_container = av.open(output_filename, "w")

    stream = output_container.add_stream("libx264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        # Convert the frame to the appropriate format (YUV420p)
        frame_yuv = av.VideoFrame.from_ndarray(frame, format='rgb24')
        packet = stream.encode(frame_yuv)
        output_container.mux(packet)

    packet = stream.encode()
    if packet:
        output_container.mux(packet)
    output_container.close()


def already_processed(directory: str, filename: str, extension: str = "mp4") -> bool:
    output_file = os.path.join(directory, f"{filename}.{extension}")
    return os.path.isfile(output_file)


def process_entry(entry, videos_dir: str, processed_dir: str):
    filename = entry["filename"]
    y1, x1, y2, x2 = entry["box"]

    video_file_path = os.path.join(videos_dir, f"{filename}.mp4")

    processed_metadata = entry.copy()
    processed_metadata['fps'] = 25
    processed_metadata.pop('box', None)
    processed_metadata.pop('start_time', None)
    processed_metadata.pop('start', None)
    processed_metadata.pop('end_time', None)
    processed_metadata.pop('end', None)
    processed_metadata.pop('height', None)
    processed_metadata.pop('width', None)

    if already_processed(processed_dir, filename):
        result = [processed_metadata]
        return result

    container = av.open(video_file_path)
    frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
    height, width, _ = frames[0].shape
    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
    cropped_frames = [frame[y1:y2, x1:x2] for frame in frames]
    resized_frames = [cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR) for frame in cropped_frames]

    # Loop if less than 64 frames
    if len(resized_frames) < 64:
        resized_frames = (resized_frames * (64 // len(resized_frames))) + resized_frames[:64]
    elif len(resized_frames) > 64:
        _indices = np.random.choice(len(resized_frames), 64, replace=False)
        _indices.sort()
        resized_frames = [resized_frames[i] for i in _indices]

    write_video(processed_dir, filename, resized_frames[:64])

    return [processed_metadata]


def preprocess(
    metadata_json: str,
    videos_dir: str,
    processed_dir: str,
    limit: int = -1,
    batch_size: int = 100,
    max_workers: int = 8
) -> None:

    with open(metadata_json, 'r', encoding='utf-8') as f:
        _metadata = json.load(f)

    if limit > 0:
        _metadata = _metadata[:limit]

    new_metadata = []
    output_metadata = os.path.join(processed_dir, "metadata.json")
    output_videos_dir = os.path.join(processed_dir, "videos")
    os.makedirs(output_videos_dir, exist_ok=True)

    for i in range(0, len(_metadata), batch_size):
        end = len(_metadata) if i + batch_size > len(_metadata) else i + batch_size
        print(f"Processing batch {i + 1} - {end}/{len(_metadata)}")
        batch = _metadata[i:i + batch_size]

        retries = 0
        while retries < MAX_RETRIES:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(process_entry, entry, videos_dir, output_videos_dir) for entry in batch]
                    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                        new_metadata.extend(future.result(timeout=30))

                    with open(output_metadata, 'w', encoding='utf-8') as outfile:
                        json.dump(new_metadata, outfile, indent=4)
                    break
            except concurrent.futures.TimeoutError:
                print(f"Batch {i} timed out. Retrying...({retries + 1}/{MAX_RETRIES})")
                retries += 1
            except Exception as e:
                print(f"Error processing batch {i}: {e}. Retrying...({retries + 1}/{MAX_RETRIES})")
                retries += 1

        if retries == MAX_RETRIES:
            print(f"Failed to process batch {i} after {MAX_RETRIES} retries. Skipping...")
