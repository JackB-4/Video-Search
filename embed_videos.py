import os
import torch
from transformers import AutoModel, AutoProcessor
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import json
from pathlib import Path
import gc
from collections import defaultdict

# Set the model name and device
model_name = "google/siglip2-giant-opt-patch16-384"  # SigLIP 2 Giant
device = "cuda" if torch.cuda.is_available() else "cpu"

if device != "cuda":
    print("WARNING: CUDA not found. Embedding will be very slow on CPU.")
    # Or raise RuntimeError if GPU is strictly required

# Clear GPU memory before loading model
torch.cuda.empty_cache()

print(f"Loading SigLIP 2 Giant model from {model_name}...")

# Initialize model with GPU memory settings
model_kwargs = {
    "torch_dtype": torch.float16,
    "low_cpu_mem_usage": True,
    "ignore_mismatched_sizes": True # Useful if fine-tuning or experimenting
}
if device == "cuda":
    model_kwargs["device_map"] = "auto"

model = AutoModel.from_pretrained(model_name, **model_kwargs)
model.eval()  # Set to eval mode for inference

print(f"Model loaded on {device} using FP16")
print(f"Model size: 1.87B parameters")

# Initialize processor with proper configuration
processor = AutoProcessor.from_pretrained(model_name)

# Define paths for storing different embedding versions
EMBEDDINGS_FILE = "video_embeddings_siglip2_giant.json"
DEFAULT_VIDEOS_DIR = "videos"

def get_video_info(video_path):
    """Get FPS, total frames, and duration of a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            return None, None, None
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
    finally:
        if cap and cap.isOpened():
            cap.release()
    return fps, total_frames, duration

def extract_frames(video_path):
    """
    Extract frames at roughly one-second intervals from a video.
    Returns a list of (frame_number, timestamp) tuples.
    """
    fps, total_frames, duration = get_video_info(video_path)
    if fps is None or total_frames is None:
        return []

    frames_to_extract = []
    # Ensure we handle potential float inaccuracies and edge cases
    for second in range(int(np.ceil(duration))):
        frame_number = int(round(second * fps))
        # Check frame number is within valid range
        if 0 <= frame_number < total_frames:
            # Avoid duplicates if FPS is very low or rounding causes clashes
            if not frames_to_extract or frames_to_extract[-1][0] != frame_number:
                frames_to_extract.append((frame_number, float(second)))

    print(f"Extracting {len(frames_to_extract)} frames (~1/sec) from {os.path.basename(video_path)} ({duration:.2f}s)")
    return frames_to_extract

def resize_with_padding(image, target_size):
    """
    Resize PIL Image to target size, maintaining aspect ratio by adding padding.
    """
    width, height = image.size
    target_aspect = 1.0 # Target is square
    img_aspect = width / height

    if img_aspect > target_aspect: # Wider than target: fit width
        new_width = target_size
        new_height = int(new_width / img_aspect)
    else: # Taller than target: fit height
        new_height = target_size
        new_width = int(new_height * img_aspect)

    # Calculate padding
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    pad_right = target_size - new_width - pad_left
    pad_bottom = target_size - new_height - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # Resize and pad
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    result = Image.new(image.mode, (target_size, target_size), (0, 0, 0))
    result.paste(image, (pad_left, pad_top))
    
    return result

def get_frame(video_path, frame_number):
    """Extract a specific frame, convert to PIL Image, and resize."""
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            return None
    finally:
        if cap and cap.isOpened():
            cap.release()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Resize image to match model input size (384 for SigLIP Giant 384)
    input_size = 384 # Use the known size directly
    resized_image = resize_with_padding(pil_image, input_size)
    return resized_image

def embed_frame(frame):
    """Embed a single PIL image frame using the loaded SigLIP model."""
    inputs = processor(images=frame, return_tensors="pt").to(device)
    
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # L2 Normalize the features
        norm = torch.norm(image_features, p=2, dim=1, keepdim=True)
        normalized_features = image_features / norm
        
    features_np = normalized_features.cpu().numpy().astype(np.float32).squeeze()
    # Explicitly delete tensors and clear cache to manage GPU memory
    del inputs, image_features, norm, normalized_features
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect() # Suggest garbage collection
    return features_np

def embed_text(text):
    """Embed a text query using the loaded SigLIP model."""
    # SigLIP expects lowercase text, optionally with a template
    text = text.lower().strip()
    # Apply the recommended template for SigLIP
    formatted_text = f"this is a photo of {text}" 

    # SigLIP docs suggest specific padding for optimal performance
    inputs = processor(
        text=formatted_text, # Use templated text
        return_tensors="pt",
        padding="max_length",
        max_length=64 # As per SigLIP paper/docs
    ).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        norm = torch.norm(text_features, p=2, dim=1, keepdim=True)
        normalized_features = text_features / norm
        
    features_np = normalized_features.cpu().numpy().astype(np.float32).squeeze()
    del inputs, text_features, norm, normalized_features
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return features_np

def enhance_prompt(query):
    """
    Clean up query text (lowercase) for SigLIP embedding.
    Note: The specific template like "this is a photo of..." is often handled 
    internally by the embedding function or processor, but basic cleaning is good.
    """
    return query.lower().strip()

def search_videos(query_text, num_videos=3, num_timestamps_per_video=5, timestamp_threshold=0.05):
    """
    Search through embeddings for frames matching the query text.
    Returns the top N unique videos and a list of relevant timestamp info for each.
    """
    embeddings = load_embeddings()
    if not embeddings:
        print(f"ERROR: No embeddings found in {EMBEDDINGS_FILE}. Run embedding process first.")
        return []
    
    query_embedding = embed_text(query_text)
    # Get embedding dimension directly from the computed query embedding
    if query_embedding is None or query_embedding.ndim == 0:
        print("ERROR: Failed to generate query embedding.")
        return []
    embedding_dim = query_embedding.shape[0]
    # --- Calculate Similarities ---
    all_results = []
    for item in embeddings:
        item_embedding = np.array(item['embedding'], dtype=np.float32)
        if item_embedding.shape[0] != embedding_dim:
            print(f"Warning: Skipping item with wrong embedding dimension ({item_embedding.shape[0]}) for {item['video_path']}")
            continue
            
        # Cosine similarity = dot product of normalized vectors
        similarity = float(np.dot(query_embedding, item_embedding))
        
        all_results.append({
            "video_path": item['video_path'],
            "frame_number": int(item['frame_number']),
            "timestamp": float(item['timestamp']),
            "score": similarity
        })
    
    if not all_results:
        print("No similarity scores calculated (check embeddings?).")
        return []

    # --- Group by Video and Rank ---
    all_results.sort(key=lambda x: x["score"], reverse=True)
    
    video_groups = defaultdict(list)
    for result in all_results:
        video_groups[result['video_path']].append(result)
        
    # Find the best score for each video
    video_best_scores = []
    for video_path, frames in video_groups.items():
        best_frame = frames[0] # Highest score frame due to sort
        video_best_scores.append({
            'video_path': video_path,
            'best_score': best_frame['score'],
            'best_match_frame': best_frame
        })
        
    # Sort videos by their best score
    video_best_scores.sort(key=lambda x: x['best_score'], reverse=True)
    
    # --- Select Top Videos and Timestamps ---
    top_videos = video_best_scores[:num_videos]
    final_results = []
    print("\n--- Top Video Matches ---")
    for video_info in top_videos:
        video_path = video_info['video_path']
        all_frames_for_video = video_groups[video_path]
        best_match_frame = video_info['best_match_frame']
        
        # Get timestamps meeting threshold
        valid_timestamps = [f for f in all_frames_for_video if f['score'] >= timestamp_threshold]
        
        # Ensure the absolute best frame is always included
        if best_match_frame['frame_number'] not in {f['frame_number'] for f in valid_timestamps}:
            valid_timestamps.append(best_match_frame)
            
        # Sort by score to pick the top N overall for this video
        valid_timestamps.sort(key=lambda x: x['score'], reverse=True)
        top_timestamps_for_display = valid_timestamps[:num_timestamps_per_video]

        # Sort final list chronologically for display purposes
        top_timestamps_for_display.sort(key=lambda x: x['timestamp'])
        
        if top_timestamps_for_display:
            final_results.append({
                'video_path': video_path,
                'best_match': best_match_frame, 
                'timestamps': top_timestamps_for_display 
            })
            
            # Print summary to console
            print(f"Video: {os.path.basename(video_path)} (Best Score: {video_info['best_score']:.4f}) @ {best_match_frame['timestamp']:.1f}s")
            print(f"  Selected Timestamps (Score >= {timestamp_threshold}, max {num_timestamps_per_video}):")
            for ts in top_timestamps_for_display:
                print(f"    - {ts['timestamp']:.1f}s (Score: {ts['score']:.4f})")
        else:
            print(f"Video: {os.path.basename(video_path)} - No timestamps met threshold {timestamp_threshold} (Best score was {video_info['best_score']:.4f})")

    return final_results

def load_embeddings():
    """Load embeddings from the JSON file."""
    if not os.path.exists(EMBEDDINGS_FILE):
        return []
    try:
        with open(EMBEDDINGS_FILE, 'r') as f:
            data = json.load(f)
            # Convert lists back to numpy arrays
            for item in data:
                if isinstance(item.get('embedding'), list):
                    item['embedding'] = np.array(item['embedding'], dtype=np.float32)
            return data
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {EMBEDDINGS_FILE}")
        return []
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return []

def save_embeddings(embeddings):
    """Save embeddings list to the JSON file."""
    json_data = []
    for item in embeddings:
        # Ensure embedding is serializable (list)
        if isinstance(item.get('embedding'), np.ndarray):
            save_item = item.copy()
            save_item['embedding'] = save_item['embedding'].tolist()
            json_data.append(save_item)
        else:
            print(f"Warning: Skipping item with non-numpy embedding: {item.get('video_path')}")

    try:
        with open(EMBEDDINGS_FILE, 'w') as f:
            json.dump(json_data, f, indent=2) # Add indent for readability
        print(f"Embeddings saved to {EMBEDDINGS_FILE}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def process_video(video_path, existing_embeddings_map):
    """Generate and return embeddings for a single video file."""
    print(f"\nProcessing video: {video_path}")
    
    frames_to_process = extract_frames(video_path)
    if not frames_to_process:
        return []
    
    new_embeddings = []
    for frame_number, timestamp in tqdm(frames_to_process, desc=f"Embedding {os.path.basename(video_path)}"):
        # Check if frame already exists (optional, adds overhead)
        # if (video_path, frame_number) in existing_embeddings_map: continue
        
        frame_image = get_frame(video_path, frame_number)
        if frame_image is None:
            print(f"Warning: Could not read frame {frame_number} from {video_path}")
            continue
        
        try:
            embedding = embed_frame(frame_image)
            new_embeddings.append({
                "video_path": video_path,
                "frame_number": int(frame_number),
                "timestamp": float(timestamp),
                "embedding": embedding
            })
        except Exception as e:
            print(f"ERROR embedding frame {frame_number} from {video_path}: {str(e)}")
    
    return new_embeddings

def process_directory(directory=DEFAULT_VIDEOS_DIR):
    """Process all video files in a directory, updating embeddings."""
    if not os.path.isdir(directory):
        print(f"ERROR: Directory not found: {directory}")
        return
    
    embeddings = load_embeddings()
    # Create a quick lookup map of existing processed frames
    # existing_map = {(e['video_path'], e['frame_number']) for e in embeddings}
    processed_videos = {e['video_path'] for e in embeddings}
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')
    video_files = sorted([os.path.join(directory, f) 
                         for f in os.listdir(directory) 
                         if f.lower().endswith(video_extensions)])
    
    if not video_files:
        print(f"No video files found in '{directory}'")
        return
    
    print(f"Found {len(video_files)} videos in '{directory}'")
    
    newly_processed_count = 0
    for video_path in video_files:
        # Skip if video path already fully represented in embeddings
        if video_path in processed_videos:
            print(f"Skipping already processed video: {os.path.basename(video_path)}")
            continue
        
        video_embeddings = process_video(video_path, None) # Pass None for map for now
        if video_embeddings:
            embeddings.extend(video_embeddings)
            newly_processed_count += 1
            # Optimization: Save periodically? Might slow down if many small videos.
            # if newly_processed_count % 5 == 0: save_embeddings(embeddings)

    if newly_processed_count > 0:
        save_embeddings(embeddings)
        print(f"\nProcessing complete! Added embeddings for {newly_processed_count} new videos.")
        print(f"Total embeddings stored: {len(embeddings)}")
    else:
        print("\nNo new videos processed.")

if __name__ == "__main__":
    # Create videos directory if it doesn't exist
    if not os.path.exists(DEFAULT_VIDEOS_DIR):
        os.makedirs(DEFAULT_VIDEOS_DIR)
        print(f"Created '{DEFAULT_VIDEOS_DIR}' directory. Please add video files there.")
    
    print("\nThis script processes videos to create embeddings for searching.")
    print(f"Embeddings will be saved to: {EMBEDDINGS_FILE}")
    print("NOTE: If using a new model (like SigLIP 2), previous embeddings must be regenerated.")
    
    run_process = input("Process videos in '{}' directory now? (y/n): ".format(DEFAULT_VIDEOS_DIR))
    if run_process.lower() == 'y':
        regenerate = input("Regenerate all embeddings (delete existing file)? (y/n): ")
        if regenerate.lower() == 'y':
            if os.path.exists(EMBEDDINGS_FILE):
                try:
                    os.remove(EMBEDDINGS_FILE)
                    print(f"Removed existing embeddings file: {EMBEDDINGS_FILE}")
                except OSError as e:
                    print(f"Error removing file {EMBEDDINGS_FILE}: {e}")
        
        process_directory(DEFAULT_VIDEOS_DIR)
    else:
        print("Video processing skipped.") 