# Bayesian I3D for Word Level ASL

This repository hosts the bayesian augmentation for I3D to model uncertainity in word level ASL classification

## Getting Started

Create a virtual environment
```shell
python -m venv venv
```

Activate the virtual environment
- For Linux
```shell
source venv/bin/activate
```

- For Windows
```shell
.\venv\Scripts\activate
```

We use Poetry for dependency management as it let's us lock dependencies using a lockfile.
```shell
pip install poetry
```

Then, install the project dependencies
```shell
poetry install
```


## Setting up for CUDA
If you see the dependencies in [pyproject.toml](./pyproject.toml), you'll see that the dependencies installed are specified for cuda 12.6. If you need a different version, you should uninstall the existing versions and install the appropriate version from the respective repositories.

### OpenCV
The default package for opencv available on pypi is compiled for CPU only. To use opencv with CUDA support, we need to build it from source.

Clone the [opencv](https://github.com/opencv/opencv) and [opencv_contrib](https://github.com/opencv/opencv_contrib) repositories into a subfolder called tools

In the [tools](./tools/) directory, you will see the following structure:
```
tools
  ├── opencv
  └── opencv_contrib
```

We build both from source using the script [build-opencv.sh](./build-opencv.sh)

#### Pre-requisites
- You must have Nvidia CUDA Developer Toolkit installed - [Downloads](https://developer.nvidia.com/cuda-downloads)
- You must have cudnn installed - [Downloads](https://developer.nvidia.com/cudnn)
- You must have Nvidia Video Codex SDK Installed - [Downloads](https://developer.nvidia.com/video-codec-sdk)
- Run the below to install build dependencies
- Create and activate a virtual env where you will install opencv after build
- Install project dependencies
```shell
poetry install
``` 
- Install OpenCV Build Dependencies
```shell
sudo apt update && sudo apt upgrade -y
```
```shell
sudo apt install -y build-essential cmake g++ wget unzip pkg-config \
    libgtk-3-dev libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libv4l-dev libxvidcore-dev libx264-dev freeglut3-dev
```
- Find the architecture number for your nvidia gpu (for example, for GTX 1650 it is 7.5)
```shell
nvidia-smi --query-gpu=compute_cap --format=csv
```

- Run the shell script to build from source (this will take a while. Around 65 mins for me)
```shell
./build-opencv.sh <ARCH_NUM>
```
For example, on the GTX 1650 the above command will be `./build-opencv.sh 75`

- Install OpenCV in the virtual environment
```shell
./install-opencv.sh
```


## Using the Dataset

You can download the dataset from Google Drive under [Uncertainty Modeling Dataset](https://drive.google.com/drive/u/0/folders/1mzAql9-bdX59mUN7_TDTzLQ8Sb68ELqi).
There are two main folders: `bin` and `zip`. If you wish to download the dataset as a single binary file, use the bin folder. If you wish to have all the preprocessed videos as separate files, use the zip folder. Personally I feel the single binary file should be fine too. As the file is opened in `'rb'` mode, the OS does not lock it and multiple child processes can read from the file, plus frequence accesses to the same file can be cached so we might see slight benefits on the IO imo. If you are constrained by your system's IO, use the single binary file, else use the zipped mp4 files.

### Dataset Loading for the binary file

The binary file acts as a streaming loader. i.e., it does not put the entire dataset into memory at once. You can use the below as a starting point for reading the dataset as a binary file.

Assuming you downloaded the test, train, and validation folders into a directory called `data`:

```python
from utils.dataset import load_msasl

# If you want to load the full MSASL Dataset with all 1000 classes
test_dataset, train_dataset, validation_dataset = load_msasl("data", 1000)

# If you want to load a subset of the MSASL Dataset with 500 classes
test_dataset, train_dataset, validation_dataset = load_msasl("data", 500)

# If you want to load a subset of the MSASL Dataset with n classes
test_dataset, train_dataset, validation_dataset = load_msasl("data", n)


# Iterating over the dataset
# When you call dataset[idx], you are returned a tuple of the following format (video_tensor: Tensor, label: int, metadata: dict)
video, label, metadata = train_dataset[0]
```

### Dataset Loading from individual files

You can use Pytorch's video loader classes to load videos from the extracted directory in conjunction with the metadata.json file