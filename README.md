# Video Search

A Python application that allows you to search through videos using natural language queries. The application uses the SigLIP model to generate embeddings of video frames and enables semantic search across your video collection.

## Features

- Extract one frame per second from videos
- Generate embeddings using the SigLIP 2 model
- Search videos using natural language queries
- GPU acceleration for faster processing
- Support for common video formats (mp4, avi, mov, mkv)

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (required for efficient processing)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/JackB-4/Video-Search.git
cd Video-Search
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers opencv-python pillow tqdm numpy
```

## Usage

1. Add your videos to the `videos` directory. The application supports common video formats (mp4, avi, mov, mkv).

2. Process your videos to generate embeddings:
```bash
python embed_videos.py
```
This will:
- Extract one frame per second from each video
- Generate embeddings using the SigLIP model
- Save the embeddings to `video_embeddings.json`

3. Search through your videos:
```bash
python search_videos.py
```
When prompted, enter your search query (e.g., "a person riding a bicycle" or "a red car on the road").

The application will return the most relevant video segments based on your query, including:
- The video file path
- The frame number
- The timestamp in the video
- A similarity score

## Example Queries

- "a person walking on the street"
- "a red car driving on the road"
- "someone playing basketball"
- "a dog running in the park"

## Notes

- The application requires a CUDA-capable GPU for efficient processing
- Processing time depends on the number and length of videos
- Embeddings are stored in `video_embeddings.json` and can be reused for multiple searches
- The `videos` directory is empty by default - add your videos before running the application
