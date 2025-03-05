import cv2
import matplotlib.pyplot as plt
from embed_videos import search_videos, enhance_prompt
import numpy as np
from PIL import Image

def display_results(results, query, enhanced_query, rows=2):
    """
    Display the search results in a grid with video information
    """
    num_results = len(results)
    if num_results == 0:
        print("No results found.")
        return
        
    cols = (num_results + rows - 1) // rows
    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle(f'Search results for: "{query}"\nEnhanced query: "{enhanced_query}"', fontsize=16)
    
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
    print("- Type 'quit' to exit")
    print("==================\n")
    
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
            
            display_results(results, query, enhanced_query) 