#!/usr/bin/env bash
set -e

# Setup script for Video-Search repository

# Determine whether sudo is available
SUDO=""
if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
fi

# Install system packages needed for video processing and Tk support
$SUDO apt-get update
$SUDO apt-get install -y python3-venv python3-dev ffmpeg python3-tk

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

echo -e "\nSetup complete! Activate the environment with 'source venv/bin/activate'." \
         "Then run 'python embed_videos.py' to generate embeddings or" \
         "'python search_videos.py' to search videos."

