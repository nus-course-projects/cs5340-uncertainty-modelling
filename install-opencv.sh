#!/bin/bash

# Check if the virtual environment is activated and VIRTUAL_ENV is set
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: You must activate a virtual environment first."
    echo "Usage: source <path_to_virtualenv>/bin/activate"
    exit 1
fi

cd ./tools/opencv

# Check if the build directory exists
if [ ! -d "build" ]; then
    echo "Error: Build directory not found. Please run the build script first."
    exit 1
fi

# Navigate to the build directory
cd build

# Install OpenCV in the virtual environment
echo "Installing OpenCV in the virtual environment..."
cmake --install .

PYTHON_VERSION=$(python3 -c 'import sys; print(sys.version[:4])')
ln -s $VIRTUAL_ENV/lib/opencv/lib/python$PYTHON_VERSION/site-packages/cv2 $VIRTUAL_ENV/lib/python$PYTHON_VERSION/site-packages/cv2

echo "OpenCV installation completed in the virtual environment: $VIRTUAL_ENV"