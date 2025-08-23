import numpy as np
import tensorflow as tf
from PIL import Image

# BEAUTIFUL COLOR MAP - Same as Azure Functions for consistency!
CLASS_COLOR_MAP = {
    # Main 8 classes (matching Azure Functions BEAUTIFUL_COLOR_MAP)
    0: (255, 0, 255),      # flat/road - BRIGHT MAGENTA/FUCHSIA  
    1: (255, 0, 0),        # human/person - BRIGHT RED
    2: (0, 255, 255),      # vehicle - BRIGHT CYAN
    3: (0, 255, 0),        # construction/building - BRIGHT LIME GREEN
    4: (255, 255, 0),      # object/pole/sign - BRIGHT YELLOW  
    5: (0, 128, 255),      # nature/vegetation - BRIGHT BLUE
    6: (255, 192, 203),    # sky - LIGHT PINK/ROSE
    7: (255, 165, 0),      # void/other - BRIGHT ORANGE
    
    # Full Cityscapes original class IDs with SUPER BRIGHT colors
    8: (107, 142, 35),     # vegetation - Olive Green
    17: (0, 255, 0),       # terrain - BRIGHT LIME GREEN
    20: (255, 0, 0),       # person - BRIGHT RED  
    21: (255, 105, 180),   # rider - HOT PINK
    22: (255, 69, 0),      # car - RED ORANGE
    23: (255, 20, 147),    # truck - DEEP PINK
    24: (0, 255, 255),     # bus - BRIGHT CYAN
    25: (0, 191, 255),     # train - DEEP SKY BLUE
    26: (255, 255, 0),     # motorcycle - BRIGHT YELLOW
    27: (255, 165, 0),     # bicycle - BRIGHT ORANGE
    
    # Additional possible classes
    28: (186, 85, 211),    # Additional - MEDIUM ORCHID
    29: (124, 252, 0),     # Additional - LAWN GREEN  
    30: (255, 192, 203),   # Additional - LIGHT PINK
    31: (0, 250, 154),     # Additional - MEDIUM SPRING GREEN
    32: (138, 43, 226),    # Additional - BLUE VIOLET
    33: (255, 215, 0)      # Additional - GOLD
}

def colorize_mask(mask, color_format='RGB'):
    """
    Map each class in the mask to its corresponding color with proper channel ordering.
    
    Args:
        mask: tf.Tensor or np.ndarray of shape (H, W), with values representing classes 0-7
        color_format: 'RGB' (default), 'BGR', or 'channels_first'
        
    Returns:
        np.ndarray with beautiful colors mapped to each class
        - RGB/BGR: shape (H, W, 3)
        - channels_first: shape (3, H, W)
    """
    # Convert to numpy if tensorflow tensor
    if isinstance(mask, tf.Tensor):
        mask = tf.squeeze(mask).numpy()
    else:
        mask = np.squeeze(mask)
    
    # Ensure 2D mask
    if len(mask.shape) > 2:
        mask = mask[:, :, 0] if mask.shape[-1] == 1 else mask.squeeze()
    
    h, w = mask.shape
    colorized_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply beautiful color mapping
    for class_idx, color in CLASS_COLOR_MAP.items():
        if color_format == 'BGR':
            # Convert RGB to BGR for OpenCV compatibility
            bgr_color = (color[2], color[1], color[0])  # B, G, R
            colorized_mask[mask == class_idx] = bgr_color
        else:
            # Standard RGB for PIL, matplotlib, web
            colorized_mask[mask == class_idx] = color  # R, G, B
    
    # Handle channels first format (PyTorch style)
    if color_format == 'channels_first':
        colorized_mask = np.transpose(colorized_mask, (2, 0, 1))  # (3, H, W)
    
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