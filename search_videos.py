import cv2
import matplotlib
matplotlib.use('TkAgg') # Force TkAgg backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from embed_videos import search_videos, enhance_prompt, get_frame
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import os
from collections import defaultdict

print("\nInitializing Video Search...")

# --- Model Config ---
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Loading (Lazy) ---
dino_processor = None
dino_model = None

def load_dino_model():
    """Load the Grounding DINO model only when needed to save memory."""
    global dino_processor, dino_model
    if dino_model is None:
        print(f"Loading Grounding DINO model ({model_id}) on {device}...")
        dino_processor = AutoProcessor.from_pretrained(model_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        print("Grounding DINO model loaded.")

# --- Object Detection ---
def detect_objects(frame, query):
    """
    Detect objects in a single frame using Grounding DINO based on a text query.
    Returns a dictionary containing boxes, scores, and labels.
    """
    if dino_model is None:
        load_dino_model()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Clean query for Grounding DINO (remove potential CLIP prefixes)
    clean_query = query.lower()
    if clean_query.startswith("this is a photo of "):
        clean_query = clean_query[len("this is a photo of "):]
    
    # Format query with periods for multi-object detection (Grounding DINO syntax)
    formatted_query = clean_query.replace(",", ".").replace(" and ", ".")
    text_prompt = [[formatted_query]]
    
    inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    target_sizes = [(image.height, image.width)]
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.35,      # Object detection confidence
        text_threshold=0.25, # Text-box alignment confidence
        target_sizes=target_sizes
    )
    
    return results[0] 

# --- Drawing Utilities ---
def draw_detections(ax, frame_shape, detections):
    """Draws bounding boxes and labels on a Matplotlib axis, clearing previous ones."""
    # Clear previous Rectangle patches added by this function
    # Iterate over a copy [:] because remove() modifies the list
    for patch in ax.patches[:]: 
        patch.remove()

    # Clear previous Text annotations for detections (identified by GID)
    # Iterate over a copy [:] because remove() modifies the list
    for txt in ax.texts[:]:
        if txt.get_gid() and txt.get_gid().startswith("det_"):
            txt.remove()

    colors = plt.cm.get_cmap('tab10', 10).colors

    # Check if detections are valid before proceeding
    if not (
        detections is not None and 
        isinstance(detections, dict) and 
        'boxes' in detections and 
        isinstance(detections['boxes'], torch.Tensor) and 
        detections['boxes'].nelement() > 0 and 
        'scores' in detections and 
        isinstance(detections['scores'], torch.Tensor) and
        'text_labels' in detections and
        isinstance(detections['text_labels'], list) and
        len(detections['scores']) == len(detections['boxes']) and 
        len(detections['text_labels']) == len(detections['boxes'])
        ):
        return # No valid detections to draw

    # Draw new detections
    for i, (box, score, label) in enumerate(zip(detections["boxes"], detections["scores"], detections["text_labels"])):
        box = box.cpu().numpy()
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]
        
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, 
                           linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        conf_str = f"{label}: {score.item():.2f}"
        ax.text(x1, y1 - 5, conf_str, color='white', fontsize=8,
                 bbox=dict(facecolor=color, alpha=0.7), 
                 gid=f"det_{i}") # GID helps identify detection text for clearing

# --- Interactive Plotting --- 
plot_data = defaultdict(dict) # Stores data needed for click events

def on_click(event):
    """Handles clicks on timestamp annotations (Text objects with GID starting 'ts_')."""
    global plot_data
    
    clicked_artist = event.artist
    is_relevant_click = (
        event.mouseevent.inaxes and 
        isinstance(clicked_artist, plt.Text) and 
        clicked_artist.get_gid() and 
        clicked_artist.get_gid().startswith("ts_")
    )

    if not is_relevant_click:
        return

    clicked_ax = event.mouseevent.inaxes
    ax_index = -1
    # Find which subplot's text was clicked
    for i, data_entry in plot_data.items():
        if data_entry['ax'] == clicked_ax:
            ax_index = i
            break
    
    if ax_index == -1:
        print("Warning: Clicked timestamp text not associated with a known subplot.")
        return
            
    # Extract timestamp index from the clicked text GID (e.g., "ts_1_2" -> 2)
    try:
        ts_index = int(clicked_artist.get_gid().split('_')[2]) 
    except (IndexError, ValueError):
        print(f"Warning: Could not parse timestamp index from GID: {clicked_artist.get_gid()}")
        return

    # Retrieve data needed for update
    data = plot_data[ax_index]
    ax = data['ax']
    video_path = data['video_path']
    if ts_index >= len(data['timestamps']):
        print(f"Warning: Timestamp index {ts_index} out of range for this video.")
        return
    timestamp_info = data['timestamps'][ts_index]
    query = data['query']
    img_display = data['image_display']
    frame_number = timestamp_info['frame_number']
    
    # --- Update Frame and Detections ---
    new_frame = get_frame(video_path, frame_number)
    if new_frame is None:
        print(f"ERROR: Error reading frame {frame_number} from {video_path}")
        return

    new_frame_cv = cv2.cvtColor(np.array(new_frame), cv2.COLOR_RGB2BGR)
    new_frame_rgb = cv2.cvtColor(new_frame_cv, cv2.COLOR_BGR2RGB)

    try:
        detections = detect_objects(new_frame_cv, query)
        draw_detections(ax, new_frame_rgb.shape, detections) # Clears old, draws new
    except Exception as e:
        print(f"ERROR: Object detection error on frame {frame_number}: {str(e)}")
        draw_detections(ax, new_frame_rgb.shape, None) # Clear detections on error
        
    img_display.set_data(new_frame_rgb)
    event.canvas.draw_idle()

def display_results(results, query):
    """
    Display the top N (max 3) video results interactively in Matplotlib.
    Shows the best matching frame initially, with clickable timestamps below.
    """
    global plot_data
    plot_data.clear() 
    
    num_results_found = len(results)
    if num_results_found == 0:
        print("No results found.")
        return

    results_to_display = results[:3]
    num_results_display = len(results_to_display)

    fig, axes = plt.subplots(1, num_results_display, figsize=(6 * num_results_display, 7.5)) 
    if num_results_display == 1:
        axes = [axes] 
        
    fig.suptitle(f'Top {num_results_display} Video Matches for: "{query}"\n(Click timestamp to view frame)', fontsize=14)
    
    if dino_model is None:
        load_dino_model()
    
    for idx, video_result in enumerate(results_to_display):
        ax = axes[idx]
        video_path = video_result['video_path']
        best_match = video_result['best_match']
        timestamps = video_result['timestamps'] 
        
        # --- Initial Frame ---
        initial_frame_num = best_match['frame_number']
        initial_timestamp = best_match['timestamp']
        initial_score = best_match['score']
        
        print(f"Displaying initial match: {os.path.basename(video_path)} @ {initial_timestamp:.1f}s (Frame {initial_frame_num}) Score: {initial_score:.4f}")
        initial_frame = get_frame(video_path, initial_frame_num)
        
        if initial_frame is None:
            ax.set_title(f"{os.path.basename(video_path)}\nError loading frame {initial_frame_num}")
            ax.axis('off')
            continue
            
        initial_frame_cv = cv2.cvtColor(np.array(initial_frame), cv2.COLOR_RGB2BGR)
        initial_frame_rgb = cv2.cvtColor(initial_frame_cv, cv2.COLOR_BGR2RGB)
        
        img_display = ax.imshow(initial_frame_rgb)
        ax.axis('off')
        video_name = os.path.basename(video_path)
        ax.set_title(f'{video_name}\nBest Match: {initial_timestamp:.1f}s (Score: {initial_score:.4f})', fontsize=10)

        # --- Initial Detections ---
        try:
            detections = detect_objects(initial_frame_cv, query.lower().strip())
            draw_detections(ax, initial_frame_rgb.shape, detections)
        except Exception as e:
            print(f"Initial object detection error: {str(e)}")
            draw_detections(ax, initial_frame_rgb.shape, None)

        # --- Clickable Timestamps ---
        y_pos = 0.01 # Start just inside bottom edge
        dy = 0.055   # Vertical spacing
        for ts_idx, ts_info in enumerate(timestamps):
            ts_text = f"{ts_info['timestamp']:.1f}s (Score: {ts_info['score']:.4f})"
            ax.text(0.5, y_pos, ts_text, 
                    ha='center', va='bottom', 
                    transform=ax.transAxes, 
                    fontsize=9, 
                    color='blue', 
                    picker=5, # Click tolerance
                    gid=f"ts_{idx}_{ts_idx}") # GID links text to subplot & index
            y_pos += dy 
            
        plot_data[idx] = {
            'ax': ax,
            'video_path': video_path,
            'timestamps': timestamps, 
            'query': query, 
            'image_display': img_display,
        }
    
    fig.canvas.mpl_connect('pick_event', on_click)
    plt.tight_layout(rect=[0, 0.01, 1, 0.94]) 
    plt.subplots_adjust(bottom=0.05, top=0.90, hspace=0.1)
    plt.show()

if __name__ == "__main__":
    print("\nVideo Search with Grounding DINO Object Detection")
    print("===================================================")
    print("Tips: Use simple, specific descriptions (lowercase). Try objects, colors, actions.")
    print("Type 'quit' to exit.")
    print("===================================================\n")
    
    from embed_videos import EMBEDDINGS_FILE
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"ERROR: Embeddings file not found ({EMBEDDINGS_FILE})")
        print("Please run embed_videos.py first.")
        exit(1)
    
    while True:
        query = input("\nEnter search query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        
        results = search_videos(query) 
        
        if results:
            print(f"Found {len(results)} matching videos. Displaying top {len(results)}...")
            display_results(results, query) # Pass raw query for detection 
        else:
            print("No matching frames found.") 