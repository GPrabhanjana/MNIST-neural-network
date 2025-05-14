import tkinter as tk
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from scipy import ndimage

class MnistViewer:
    def __init__(self, root, data_path):
        self.root = root
        self.data = pd.read_csv(data_path)
        self.current_index = 0
        self.show_augmented = False
        self.augmented_image = None
        
        # Set up UI
        self.root.title("MNIST Dataset Viewer with Augmentation")
        self.root.geometry("600x700")  # Increased window size
        
        # Frame for the original image
        original_frame = tk.Frame(root, width=280, height=280, bd=2, relief=tk.SUNKEN)
        original_frame.grid(row=0, column=0, padx=10, pady=10)
        original_frame.grid_propagate(False)
        
        # Frame for the augmented image
        augmented_frame = tk.Frame(root, width=280, height=280, bd=2, relief=tk.SUNKEN)
        augmented_frame.grid(row=0, column=1, padx=10, pady=10)
        augmented_frame.grid_propagate(False)
        
        # Create UI elements
        self.original_label = tk.Label(original_frame)
        self.original_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.augmented_label = tk.Label(augmented_frame)
        self.augmented_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.digit_label = tk.Label(root, text="Label: ", font=("Arial", 16))
        self.digit_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        self.index_label = tk.Label(root, text="")
        self.index_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Navigation buttons
        button_frame = tk.Frame(root)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        tk.Button(button_frame, text="Previous", command=lambda: self.navigate(-1)).grid(row=0, column=0, padx=10)
        tk.Button(button_frame, text="Next", command=lambda: self.navigate(1)).grid(row=0, column=1, padx=10)
        tk.Button(button_frame, text="Augment", command=self.toggle_augmentation).grid(row=0, column=2, padx=10)
        
        # Control frame for augmentation parameters
        control_frame = tk.Frame(root)
        control_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Augmentation parameter sliders (limited to 20%)
        tk.Label(control_frame, text="Max Rotation (°):").grid(row=0, column=0, sticky="e")
        self.rotation_var = tk.DoubleVar(value=20.0)
        tk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, 
                 variable=self.rotation_var, resolution=1, 
                 command=lambda x: self.update_augmentation()).grid(row=0, column=1)
        
        tk.Label(control_frame, text="Max Scale (%):").grid(row=1, column=0, sticky="e")
        self.scale_var = tk.DoubleVar(value=20.0)
        tk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, 
                 variable=self.scale_var, resolution=1,
                 command=lambda x: self.update_augmentation()).grid(row=1, column=1)
        
        tk.Label(control_frame, text="Max Offset (%):").grid(row=2, column=0, sticky="e")
        self.offset_var = tk.DoubleVar(value=20.0)
        tk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, 
                 variable=self.offset_var, resolution=1,
                 command=lambda x: self.update_augmentation()).grid(row=2, column=1)
        
        tk.Label(control_frame, text="Max Noise (%):").grid(row=3, column=0, sticky="e")
        self.noise_var = tk.DoubleVar(value=20.0)
        tk.Scale(control_frame, from_=0, to=20, orient=tk.HORIZONTAL, 
                 variable=self.noise_var, resolution=1,
                 command=lambda x: self.update_augmentation()).grid(row=3, column=1)
        
        # Keyboard shortcuts
        root.bind('<Left>', lambda event: self.navigate(-1))
        root.bind('<Right>', lambda event: self.navigate(1))
        root.bind('<space>', lambda event: self.toggle_augmentation())
        
        # Display first image
        self.update_display()
    
    def navigate(self, step):
        new_index = self.current_index + step
        if 0 <= new_index < len(self.data):
            self.current_index = new_index
            self.update_display()
    
    def toggle_augmentation(self):
        self.show_augmented = not self.show_augmented
        self.update_augmentation()
    
    def update_augmentation(self):
        if self.show_augmented:
            self.create_augmented_image()
        self.update_display()
    
    def create_augmented_image(self):
        row = self.data.iloc[self.current_index]
        pixel_data = row[1:].values
        img_size = 28
        
        # Reshape to 28x28
        img = pixel_data.reshape(img_size, img_size)
        
        # Apply augmentations based on slider values
        try:
            # Get parameter values
            max_rotation = self.rotation_var.get()  # Degrees
            max_scale_pct = self.scale_var.get() / 100.0  # Convert to proportion
            max_offset_pct = self.offset_var.get() / 100.0  # Convert to proportion
            max_noise_pct = self.noise_var.get() / 100.0  # Convert to proportion
            
            # Make a copy of the original image
            augmented = img.copy().astype(float)
            
            # Random rotation
            angle = np.random.uniform(-max_rotation, max_rotation)
            augmented = ndimage.rotate(augmented, angle, reshape=False, mode='constant', cval=0)
            
            # Random scaling
            min_scale = 1.0 - max_scale_pct
            max_scale = 1.0 + max_scale_pct
            scale = np.random.uniform(min_scale, max_scale)
            
            if scale != 1.0:
                augmented = ndimage.zoom(augmented, scale, mode='constant', cval=0)
                
                # If scaled up, crop to original size
                if scale > 1.0:
                    start = int((augmented.shape[0] - img_size) // 2)
                    augmented = augmented[start:start+img_size, start:start+img_size]
                # If scaled down, pad to original size
                else:
                    pad = int((img_size - augmented.shape[0]) // 2)
                    padded = np.zeros((img_size, img_size))
                    padded[pad:pad+augmented.shape[0], pad:pad+augmented.shape[1]] = augmented
                    augmented = padded
            
            # Apply random X and Y offsets
            max_offset = int(img_size * max_offset_pct)
            offset_x = np.random.randint(-max_offset, max_offset + 1)
            offset_y = np.random.randint(-max_offset, max_offset + 1)
            
            # Create a shifted image with zero padding
            shifted_img = np.zeros_like(augmented)
            
            # Calculate source and destination regions for the shift
            # Source region (from original image)
            src_y_start = max(0, -offset_y)
            src_y_end = min(img_size, img_size - offset_y)
            src_x_start = max(0, -offset_x)
            src_x_end = min(img_size, img_size - offset_x)
            
            # Destination region (in new image)
            dst_y_start = max(0, offset_y)
            dst_y_end = min(img_size, img_size + offset_y)
            dst_x_start = max(0, offset_x)
            dst_x_end = min(img_size, img_size + offset_x)
            
            # Copy the valid part of the image with the shift
            shifted_img[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = augmented[src_y_start:src_y_end, src_x_start:src_x_end]
            augmented = shifted_img
            
            # Add realistic noise
            # Generate a noise texture using Perlin noise or a similar technique
            noise_amplitude = max_noise_pct * 255  # Max noise intensity
            
            # Create a noise grid with similar resolution to the image
            noise = np.random.normal(0, noise_amplitude/3, size=augmented.shape)
            
            # Add salt and pepper noise
            salt_pepper_prob = max_noise_pct / 5  # Lower probability for extreme noise
            salt_mask = np.random.random(augmented.shape) < salt_pepper_prob
            pepper_mask = np.random.random(augmented.shape) < salt_pepper_prob
            
            # Salt (white dots)
            augmented[salt_mask] = np.minimum(augmented[salt_mask] + noise_amplitude, 255)
            
            # Pepper (black dots)
            augmented[pepper_mask] = np.maximum(augmented[pepper_mask] - noise_amplitude, 0)
            
            # Add gaussian noise
            augmented = np.clip(augmented + noise, 0, 255)
            
            self.augmented_image = augmented
            
        except Exception as e:
            print(f"Error in augmentation: {e}")
            self.augmented_image = img  # Use original if augmentation fails
    
    def update_display(self):
        # Get current row data
        row = self.data.iloc[self.current_index]
        label = row[0]
        image_matrix = row[1:].values.reshape(28, 28)
        
        # Create and resize original image
        orig_img = Image.fromarray(np.uint8(image_matrix))
        orig_img = orig_img.resize((280, 280), Image.NEAREST)
        orig_tk_img = ImageTk.PhotoImage(orig_img)
        
        # Update original image
        self.original_label.configure(image=orig_tk_img)
        self.original_label.image = orig_tk_img  # Keep reference
        
        # Create and update augmented image if needed
        if self.show_augmented:
            if self.augmented_image is None:
                self.create_augmented_image()
                
            aug_img = Image.fromarray(np.uint8(self.augmented_image))
            aug_img = aug_img.resize((280, 280), Image.NEAREST)
            aug_tk_img = ImageTk.PhotoImage(aug_img)
            
            self.augmented_label.configure(image=aug_tk_img)
            self.augmented_label.image = aug_tk_img  # Keep reference
        else:
            # Clear augmented image
            self.augmented_label.configure(image="")
            self.augmented_label.image = None
        
        # Update labels
        self.digit_label.configure(text=f"Label: {label}")
        self.index_label.configure(text=f"Image: {self.current_index + 1}/{len(self.data)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MnistViewer(root, "mnist_train.csv")
    root.mainloop()