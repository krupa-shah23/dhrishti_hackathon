import cv2
import numpy as np
import os
import sys

from src.outputs_eval.heatmap import accumulate_mask, generate_heatmap, overlay_heatmap, save_heatmap

def run_test():
    height, width = 480, 640
    accumulator = np.zeros((height, width), dtype=np.float32)
    
    # Simulate a moving object across 100 frames
    print("Simulating 100 frames of motion masks...")
    for i in range(100):
        mask = np.zeros((height, width), dtype=np.uint8)
        # Moving 50x50 box
        x = int((i / 100) * (width - 50))
        y = height // 2 - 25
        mask[y:y+50, x:x+50] = 255
        
        accumulator = accumulate_mask(accumulator, mask)
        
    print(f"Accumulator max value: {accumulator.max()}")
    
    heatmap = generate_heatmap(accumulator)
    
    # Generate a dummy background frame
    frame = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    overlay = overlay_heatmap(frame, heatmap, alpha=0.6)
    
    os.makedirs("outputs", exist_ok=True)
    save_heatmap(heatmap, "outputs/test_heatmap.png")
    cv2.imwrite("outputs/test_heatmap_overlay.png", overlay)
    print("Test completed. Heatmap saved to outputs/test_heatmap.png and outputs/test_heatmap_overlay.png")

if __name__ == "__main__":
    run_test()
