import cv2
import numpy as np
import pandas as pd
import time

# Load the model parameters
params = np.load("mnist_model_params.npy", allow_pickle=True).item()

def forward_prop(params, X):
    cache = {}
    L = len(params) // 2
   
    cache['Z1'] = params['W1'].dot(X) + params['b1']
    cache['A1'] = ReLU(cache['Z1'])
   
    for i in range(2, L):
        cache[f'Z{i}'] = params[f'W{i}'].dot(cache[f'A{i-1}']) + params[f'b{i}']
        cache[f'A{i}'] = ReLU(cache[f'Z{i}'])
       
    cache[f'Z{L}'] = params[f'W{L}'].dot(cache[f'A{L-1}']) + params[f'b{L}']
    cache[f'A{L}'] = softmax(cache[f'Z{L}'])
   
    return cache

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

# Create a 28x28 canvas with 5x magnification
GRID_SIZE = 28
SCALE = 10
DISPLAY_SIZE = GRID_SIZE * SCALE
BRUSH_SIZE = 2  # Brush radius for paintbrush effect

# Initialize canvas and display
canvas = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
display_canvas = np.ones((400, DISPLAY_SIZE + 350), dtype=np.uint8) * 255  # Extra 200px for UI

# Create gaussian brush kernel for paintbrush effect
def create_brush(size, sigma=1.0):
    """Create a gaussian brush kernel"""
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * 
                     np.exp(-((x-size/2)**2 + (y-size/2)**2)/(2*sigma**2)), 
        (size, size)
    )
    return kernel / kernel.max()

brush = create_brush(2*BRUSH_SIZE+1, sigma=BRUSH_SIZE/2)

drawing = False
last_update = time.time()
prediction_scores = np.zeros(10)

def apply_brush(x, y, canvas, intensity=1.0):
    """Apply brush at position (x,y) with given intensity"""
    brush_size = BRUSH_SIZE
    # Get region where the brush will be applied
    y_min = max(0, y - brush_size)
    y_max = min(GRID_SIZE, y + brush_size + 1)
    x_min = max(0, x - brush_size)
    x_max = min(GRID_SIZE, x + brush_size + 1)
    
    # Get the corresponding part of the brush
    brush_y_min = brush_size - (y - y_min)
    brush_y_max = brush_size + (y_max - y)
    brush_x_min = brush_size - (x - x_min)
    brush_x_max = brush_size + (x_max - x)
    
    brush_part = brush[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
    
    # Apply the brush (darker = more ink)
    # We're working with a canvas where 1.0 is white and 0.0 is black
    # So we subtract the brush effect (multiplied by intensity)
    canvas[y_min:y_max, x_min:x_max] = np.minimum(
        canvas[y_min:y_max, x_min:x_max],
        1.0 - brush_part * intensity
    )

def draw(event, x, y, flags, param):
    global drawing, canvas, display_canvas, last_update, prediction_scores
    
    # Convert display coordinates to grid coordinates (only for the drawing area)
    if x >= DISPLAY_SIZE:  # If in the UI area, ignore
        return
    
    grid_x = x // SCALE
    grid_y = y // SCALE
    
    # Ensure within bounds
    if grid_x >= GRID_SIZE or grid_y >= GRID_SIZE or grid_x < 0 or grid_y < 0:
        return
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        # Apply brush effect
        apply_brush(grid_x, grid_y, canvas, intensity=0.8)
        last_update = 0  # Force update
        
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Apply brush effect
        apply_brush(grid_x, grid_y, canvas, intensity=0.8)
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

def update_display():
    """Update the display with current canvas and predictions"""
    global display_canvas, canvas, prediction_scores
    
    # Update the drawing area
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            # Convert float intensity (0.0-1.0) to uint8 (0-255)
            color = int(canvas[y, x] * 255)
            cv2.rectangle(
                display_canvas, 
                (x * SCALE, y * SCALE), 
                ((x + 1) * SCALE - 1, (y + 1) * SCALE - 1),
                color, -1
            )
    
    # Draw grid lines
    for i in range(0, DISPLAY_SIZE + 1, SCALE):
        cv2.line(display_canvas, (i, 0), (i, DISPLAY_SIZE), 200, 1)
        cv2.line(display_canvas, (0, i), (DISPLAY_SIZE, i), 200, 1)
    
    # Clear UI area
    display_canvas[:, DISPLAY_SIZE:] = 255
    
    # Draw UI section
    cv2.line(display_canvas, (DISPLAY_SIZE, 0), (DISPLAY_SIZE, DISPLAY_SIZE), 0, 2)
    
    # Draw title
    cv2.putText(display_canvas, "MNIST Predictor", 
                (DISPLAY_SIZE + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Draw predictions
    cv2.putText(display_canvas, "Digit  Confidence", 
                (DISPLAY_SIZE + 10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw prediction bars
    max_bar_width = 150
    for i in range(10):
        y_pos = 80 + i * 20
        # Digit label
        cv2.putText(display_canvas, f"{i}", 
                    (DISPLAY_SIZE + 20, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Bar background
        cv2.rectangle(display_canvas, 
                     (DISPLAY_SIZE + 40, y_pos - 10), 
                     (DISPLAY_SIZE + 40 + max_bar_width, y_pos - 2), 
                     (220, 220, 220), -1)
        
        # Confidence bar
        bar_width = int(prediction_scores[i] * max_bar_width)
        bar_color = (0, 0, 0)  # Default black
        if i == np.argmax(prediction_scores):
            bar_color = (0, 0, 180)  # Red for highest confidence
        
        cv2.rectangle(display_canvas, 
                     (DISPLAY_SIZE + 40, y_pos - 10), 
                     (DISPLAY_SIZE + 40 + bar_width, y_pos - 2), 
                     bar_color, -1)
        
        # Confidence percentage
        cv2.putText(display_canvas, f"{prediction_scores[i]*100:.1f}%", 
                    (DISPLAY_SIZE + 40 + max_bar_width + 5, y_pos - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Instructions
    cv2.putText(display_canvas, "Controls:", 
                (DISPLAY_SIZE + 10, y_pos + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(display_canvas, "r - Reset canvas", 
                (DISPLAY_SIZE + 20, y_pos + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.putText(display_canvas, "q - Quit", 
                (DISPLAY_SIZE + 20, y_pos + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

def predict():
    """Make prediction on current canvas"""
    global canvas, prediction_scores
    
    # Convert canvas to input format (1.0 is white, 0.0 is black in our canvas)
    img = (1.0 - canvas).reshape(784, 1)  # Invert and flatten
    
    # Run forward prop
    output = forward_prop(params, img)
    prediction_scores = output[f"A{len(params)//2}"].flatten()
    
    return prediction_scores

# Create window and set callback
cv2.namedWindow("MNIST Drawing")
cv2.setMouseCallback("MNIST Drawing", draw)

# First update
update_display()

while True:
    # Show the canvas
    cv2.imshow("MNIST Drawing", display_canvas)
    
    # Check if we need to update the prediction (every 0.1 seconds)
    current_time = time.time()
    if current_time - last_update > 0.1:
        predict()
        update_display()
        last_update = current_time
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):
        # Reset canvas
        canvas = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        prediction_scores = np.zeros(10)
        update_display()
            
    elif key == ord('q'):
        break

cv2.destroyAllWindows()