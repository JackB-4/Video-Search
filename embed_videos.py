import os
import torch
from transformers import SiglipModel, SiglipProcessor
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path

# Set the model name and device
model_name = "google/siglip-large-patch16-384"
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    raise RuntimeError("CUDA is not available. This script requires a GPU to run efficiently.")

# Load the SigLIP model and processor in FP16 mode
model = (SiglipModel.from_pretrained(model_name, torch_dtype=torch.float16)
         .to(device)
         .eval())  # Set to eval mode for inference
processor = SiglipProcessor.from_pretrained(model_name)

print(f"Model loaded on {device} using FP16")

# Define the path for storing embeddings
EMBEDDINGS_FILE = "video_embeddings.json"

def get_video_info(video_path):
    """Get basic video information."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    return fps, total_frames, duration

def extract_frames(video_path):
    """
    Extract exactly one frame per second from the video.
    Returns a list of (frame_number, timestamp) tuples.
    """
    fps, total_frames, duration = get_video_info(video_path)
    
    # Calculate frame numbers for each second
    frames = []
    for second in range(int(duration)):
        frame_number = int(second * fps)
        if frame_number < total_frames:
            frames.append((frame_number, second))
    
    print(f"Extracting {len(frames)} frames (1 per second) from video of length {duration:.2f}s")
    return frames

def get_frame(video_path, frame_number):
    """
    Extract a specific frame from the video and convert it to PIL Image.
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

def embed_frame(frame):
    """
    Embed a single frame using SigLIP.
    Returns a normalized embedding.
    """
    inputs = processor(images=frame, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.squeeze().cpu().numpy()

def load_embeddings():
    """Load embeddings from JSON file."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'r') as f:
            data = json.load(f)
            # Convert embedding lists back to numpy arrays
            for item in data:
                item['embedding'] = np.array(item['embedding'])
            return data
    return []

def save_embeddings(embeddings):
    """Save embeddings to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    json_data = []
    for item in embeddings:
        json_item = item.copy()
        json_item['embedding'] = json_item['embedding'].tolist()
        json_data.append(json_item)
    
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump(json_data, f)

def process_video(video_path, embeddings):
    """
    Process a video file:
    1. Extract one frame per second
    2. Embed each frame
    3. Store in embeddings list
    """
    print(f"\nProcessing video: {video_path}")
    
    # Extract frames
    frames = extract_frames(video_path)
    
    # Process each frame
    for frame_number, timestamp in tqdm(frames, desc="Processing frames"):
        frame = get_frame(video_path, frame_number)
        if frame is None:
            continue
        
        try:
            # Embed the frame
            embedding = embed_frame(frame)
            
            # Store the embedding
            doc = {
                "video_path": video_path,
                "frame_number": int(frame_number),
                "timestamp": float(timestamp),
                "embedding": embedding
            }
            embeddings.append(doc)
            
        except Exception as e:
            print(f"Error processing frame {frame_number} from {video_path}: {str(e)}")

def process_directory(directory="videos"):
    """
    Process all video files in the specified directory.
    """
    # Load existing embeddings
    embeddings = load_embeddings()
    
    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_files = [f for f in os.listdir(directory) 
                  if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in {directory}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Process each video
    for video_file in video_files:
        video_path = os.path.join(directory, video_file)
        # Skip if video is already processed
        if not any(e['video_path'] == video_path for e in embeddings):
            process_video(video_path, embeddings)
    
    # Save updated embeddings
    save_embeddings(embeddings)
    print("\nProcessing complete!")
    print(f"Total frames stored: {len(embeddings)}")

def enhance_prompt(query):
    """
    Format the query following SigLIP/CLIP pattern.
    """
    # Basic template following CLIP/SigLIP pattern
    template = "This is a photo of {}."
    
    # Clean up the query
    query = query.lower().strip()
    
    # Handle specific types of queries
    if any(word in query for word in ["car", "vehicle", "truck"]):
        if any(color in query for color in ["red", "blue", "green", "yellow", "white", "black"]):
            # For colored vehicles, maintain the color-first pattern
            template = "This is a photo of a {} on a road."
        else:
            template = "This is a photo of a {} driving on a road."
    elif "person" in query or "people" in query:
        template = "This is a photo of {} in the scene."
    
    enhanced = template.format(query)
    print(f"Enhanced prompt: '{enhanced}'")
    return enhanced

def search_videos(query_text, num_results=5):
    """
    Search for video frames using a text query.
    Returns video paths, frame numbers, and timestamps.
    """
    # Load embeddings
    embeddings = load_embeddings()
    if not embeddings:
        print("No embeddings found. Please process videos first.")
        return []
    
    # Enhance the prompt using CLIP/SigLIP style template
    enhanced_query = enhance_prompt(query_text)
    
    # Prepare the text query
    inputs = processor(
        text=enhanced_query,
        padding="max_length",
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    query_embedding = text_features.squeeze().cpu().numpy()
    
    # Calculate similarities and sort results
    results = []
    for item in embeddings:
        similarity = np.dot(query_embedding, item['embedding'])
        results.append({
            "_source": {
                "video_path": item['video_path'],
                "frame_number": item['frame_number'],
                "timestamp": item['timestamp']
            },
            "_score": float(similarity)
        })
    
    # Sort by similarity score
    results.sort(key=lambda x: x["_score"], reverse=True)
    return results[:num_results]

if __name__ == "__main__":
    # Create videos directory if it doesn't exist
    if not os.path.exists("videos"):
        os.makedirs("videos")
        print("Created 'videos' directory. Please add your video files there.")
    else:
        process_directory("videos") 