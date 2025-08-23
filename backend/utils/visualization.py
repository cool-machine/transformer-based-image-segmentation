import numpy as np
import tensorflow as tf
from PIL import Image

# Define the color mapping for 8 classes
CLASS_COLOR_MAP = {
    0: (0, 0, 0),          # flat - Black
    1: (128, 0, 0),        # human - Maroon
    2: (0, 128, 0),        # vehicle - Green
    3: (128, 128, 0),      # construction - Olive
    4: (0, 0, 128),        # object - Navy
    5: (128, 0, 128),      # nature - Purple
    6: (0, 128, 128),      # sky - Teal
    7: (128, 128, 128)     # void - Gray
}

def colorize_mask(mask):
    """
    Map each class in the mask to its corresponding color.
    
    Args:
        mask: tf.Tensor or np.ndarray of shape (H, W), with values representing classes 0-7
        
    Returns:
        np.ndarray of shape (H, W, 3), with colors mapped to each class
    """
    # Convert to numpy if tensorflow tensor
    if isinstance(mask, tf.Tensor):
        mask = tf.squeeze(mask).numpy()
    else:
        mask = np.squeeze(mask)
    
    h, w = mask.shape
    colorized_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_COLOR_MAP.items():
        colorized_mask[mask == class_idx] = color
    
    return colorized_mask

def save_visualization(image_array, filename):
    """Save numpy array as image file."""
    if len(image_array.shape) == 3:
        img = Image.fromarray(image_array.astype(np.uint8))
    else:
        img = Image.fromarray(image_array.astype(np.uint8))
    img.save(filename)
    return filename

def create_comparison_grid(original, predicted, ground_truth, save_path):
    """Create a side-by-side comparison of original, predicted, and ground truth."""
    # Ensure all images have the same height
    height = min(original.shape[0], predicted.shape[0], ground_truth.shape[0])
    
    # Resize if needed
    if original.shape[0] != height:
        original = np.array(Image.fromarray(original).resize((original.shape[1], height)))
    if predicted.shape[0] != height:
        predicted = np.array(Image.fromarray(predicted).resize((predicted.shape[1], height)))
    if ground_truth.shape[0] != height:
        ground_truth = np.array(Image.fromarray(ground_truth).resize((ground_truth.shape[1], height)))
    
    # Concatenate horizontally
    comparison = np.concatenate([original, predicted, ground_truth], axis=1)
    
    # Save
    img = Image.fromarray(comparison.astype(np.uint8))
    img.save(save_path)
    return save_path