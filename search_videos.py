import cv2
import matplotlib.pyplot as plt
from embed_videos import search_videos, enhance_prompt
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load Grounding DINO model and processor
# Using base model which is a good balance between performance and memory usage for 8GB GPU
model_id = "IDEA-Research/grounding-dino-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize the model and processor
dino_processor = None
dino_model = None

def load_dino_model():
    """Load the Grounding DINO model only when needed to save memory"""
    global dino_processor, dino_model
    if dino_model is None:
        print(f"Loading Grounding DINO model on {device}...")
        dino_processor = AutoProcessor.from_pretrained(model_id)
        dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        print("Model loaded successfully")

def detect_objects(frame, query):
    """
    Detect objects in a frame using Grounding DINO.
    Returns boxes, scores, and text labels for detected objects.
    """
    # Ensure model is loaded
    if dino_model is None:
        load_dino_model()
    
    # Convert OpenCV frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    
    # Process image and text
    # Format query as list of strings separated by periods for multiple object detection
    formatted_query = query.replace(",", ".").replace(" and ", ".")
    text_prompt = [[formatted_query]]
    
    inputs = dino_processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    # Post-process results
    target_sizes = [(image.height, image.width)]
    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        threshold=0.35,      # Detection confidence threshold
        text_threshold=0.25, # Text confidence threshold
        target_sizes=target_sizes
    )
    
    return results[0]  # Return results for the first (and only) image

def display_results(results, query, enhanced_query, rows=2):
    """
    Display the search results in a grid with video information and bounding boxes
    """
    num_results = len(results)
    if num_results == 0:
        print("No results found.")
        return
        
    cols = (num_results + rows - 1) // rows
    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle(f'Search results for: "{query}"\nEnhanced query: "{enhanced_query}"', fontsize=16)
    
    # Prompt user to load model if needed
    if dino_model is None:
        load_dino_model()
    
    for idx, hit in enumerate(results):
        source = hit["_source"]
        video_path = source["video_path"]
        frame_number = source["frame_number"]
        timestamp = source["timestamp"]
        score = hit["_score"]
        
        # Extract the frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            plt.subplot(rows, cols, idx + 1)
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            plt.imshow(frame_rgb)
            plt.axis('off')
            
            # Run object detection on the frame
            detections = detect_objects(frame, query)
            
            # Draw bounding boxes
            colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
            for i, (box, score, label) in enumerate(zip(detections["boxes"], detections["scores"], detections["text_labels"])):
                # Convert box tensor to numpy array
                box = box.cpu().numpy()
                x1, y1, x2, y2 = box
                
                # Choose color (cycle through the colors list)
                color = colors[i % len(colors)]
                
                # Create rectangle patch and add it to the plot
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    linewidth=2, edgecolor=color, facecolor='none')
                plt.gca().add_patch(rect)
                
                # Add label with confidence score
                conf_str = f"{label}: {score.item():.2f}"
                plt.text(x1, y1-5, conf_str, color='white', fontsize=8,
                         bbox=dict(facecolor=color, alpha=0.7))
            
            # Create detailed title with video info
            title = f'Video: {video_path.split("/")[-1]}\n'
            title += f'Time: {timestamp:.1f}s\n'
            title += f'Score: {score:.4f}'
            plt.title(title, fontsize=10)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\nVideo Frame Search")
    print("==================")
    print("Tips for better results:")
    print("- Use simple, clear descriptions")
    print("- Describe what you see (e.g., 'a person walking', 'a red car')")
    print("- The model works best with straightforward descriptions")
    print("- For object detection, try specific terms (e.g., 'person', 'car', 'dog')")
    print("- Type 'quit' to exit")
    print("==================\n")
    
    print("Grounding DINO will be loaded when you make your first search")
    
    while True:
        query = input("\nEnter text query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
            
        # Get enhanced query for display
        enhanced_query = enhance_prompt(query)
        
        # Search with more candidates for better results
        results = search_videos(query, num_results=6)  # Get 6 results for 2x3 grid
        
        if results:
            print("\nTop matching frames:")
            for hit in results:
                source = hit["_source"]
                print(f"\nVideo: {source['video_path']}")
                print(f"Time: {source['timestamp']:.1f}s")
                print(f"Score: {hit['_score']:.4f}")
            
            print("\nApplying object detection with Grounding DINO...")
            display_results(results, query, enhanced_query) 