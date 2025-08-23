"""
Enhanced mask visualization system that combines original images with masks.
Based on the existing matplotlib visualization function with proper normalization handling.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Azure Functions
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io
import base64
import logging

# Define the 8-class color mapping (same as existing visualization.py)
CLASS_COLOR_MAP = {
    0: [0, 0, 0],         # flat - Black
    1: [128, 0, 0],       # human - Maroon
    2: [0, 128, 0],       # vehicle - Green
    3: [128, 128, 0],     # construction - Olive
    4: [0, 0, 128],       # object - Navy
    5: [128, 0, 128],     # nature - Purple
    6: [0, 128, 128],     # sky - Teal
    7: [128, 128, 128]    # void - Gray
}

def fix_mask_normalization(mask):
    """
    Fix the normalization issue for ground truth masks.
    Ground truth masks have pixel values 0-7 which appear almost black.
    This function properly scales them for visualization.
    
    Args:
        mask: numpy array or tensor with values 0-7
        
    Returns:
        numpy array with proper scaling for visualization
    """
    if isinstance(mask, tf.Tensor):
        mask = mask.numpy()
    
    mask = np.squeeze(mask)
    
    # Check if mask has values in 0-7 range (ground truth case)
    if mask.max() <= 7:
        logging.info(f"Detected ground truth mask with values 0-7, max value: {mask.max()}")
        # Don't normalize to 0-255, keep original class indices 0-7 for color mapping
        return mask.astype(np.uint8)
    
    # If mask is already in proper range (predictions), keep as is
    logging.info(f"Mask appears to be prediction with max value: {mask.max()}")
    return mask.astype(np.uint8)

def apply_color_map(mask):
    """
    Apply 8-class color mapping to a mask.
    
    Args:
        mask: 2D numpy array with class indices 0-7
        
    Returns:
        3D numpy array (H, W, 3) with RGB colors
    """
    mask = fix_mask_normalization(mask)
    
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in CLASS_COLOR_MAP.items():
        colored_mask[mask == class_idx] = color
    
    return colored_mask

def create_overlay_visualization(original_image, mask, alpha=0.5, title="Segmentation Overlay"):
    """
    Create a matplotlib visualization overlaying a mask on the original image.
    Based on your existing plot_predictions function with proper normalization.
    
    Args:
        original_image: RGB image as numpy array
        mask: Segmentation mask with class indices 0-7
        alpha: Transparency for mask overlay
        title: Title for the visualization
        
    Returns:
        PIL Image of the visualization
    """
    try:
        # Process original image
        if isinstance(original_image, tf.Tensor):
            original_image = original_image.numpy()
        
        # Handle different image shapes and normalization
        if len(original_image.shape) == 4:
            original_image = original_image[0]  # Remove batch dimension
            
        # Handle channel-first format (C, H, W) -> (H, W, C)
        if len(original_image.shape) == 3 and original_image.shape[0] <= 3:
            original_image = np.transpose(original_image, [1, 2, 0])
        
        # Normalize image to 0-255 range if needed
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)
        else:
            original_image = np.clip(original_image, 0, 255).astype(np.uint8)
        
        # Process mask with proper normalization
        colored_mask = apply_color_map(mask)
        
        # Resize images to match if needed
        target_height, target_width = 512, 1024
        if original_image.shape[:2] != (target_height, target_width):
            original_pil = Image.fromarray(original_image)
            original_image = np.array(original_pil.resize((target_width, target_height)))
        
        if colored_mask.shape[:2] != (target_height, target_width):
            mask_pil = Image.fromarray(colored_mask)
            colored_mask = np.array(mask_pil.resize((target_width, target_height)))
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Display original image
        ax.imshow(original_image)
        
        # Overlay colored mask with transparency
        ax.imshow(colored_mask, alpha=alpha)
        
        ax.set_title(title)
        ax.axis('off')
        
        # Convert matplotlib figure to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        
        # Close matplotlib figure to free memory
        plt.close(fig)
        
        # Convert to PIL Image
        pil_image = Image.open(buf)
        return pil_image
        
    except Exception as e:
        logging.error(f"Error creating overlay visualization: {e}")
        return None

def create_three_panel_comparison(original_image, predicted_mask, ground_truth_mask):
    """
    Create a three-panel comparison: Original | Predicted | Ground Truth
    Based on your existing plot_predictions function.
    
    Args:
        original_image: RGB image as numpy array
        predicted_mask: Predicted segmentation mask
        ground_truth_mask: Ground truth segmentation mask
        
    Returns:
        PIL Image of the three-panel comparison
    """
    try:
        # Create individual visualizations
        original_overlay = create_overlay_visualization(
            original_image, predicted_mask, alpha=0.5, title="Original + Prediction"
        )
        
        ground_truth_overlay = create_overlay_visualization(
            original_image, ground_truth_mask, alpha=0.5, title="Original + Ground Truth"
        )
        
        if original_overlay is None or ground_truth_overlay is None:
            return None
        
        # Create three-panel figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Original image only
        if isinstance(original_image, tf.Tensor):
            original_display = original_image.numpy()
        else:
            original_display = original_image.copy()
            
        if len(original_display.shape) == 4:
            original_display = original_display[0]
        if len(original_display.shape) == 3 and original_display.shape[0] <= 3:
            original_display = np.transpose(original_display, [1, 2, 0])
        if original_display.max() <= 1.0:
            original_display = (original_display * 255).astype(np.uint8)
            
        axes[0].imshow(original_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Panel 2: Original + Prediction
        axes[1].imshow(original_overlay)
        axes[1].set_title('Predicted Segmentation')
        axes[1].axis('off')
        
        # Panel 3: Original + Ground Truth  
        axes[2].imshow(ground_truth_overlay)
        axes[2].set_title('Ground Truth Segmentation')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        pil_image = Image.open(buf)
        return pil_image
        
    except Exception as e:
        logging.error(f"Error creating three-panel comparison: {e}")
        return None

def convert_pil_to_base64(pil_image):
    """Convert PIL Image to base64 string for web display."""
    if pil_image is None:
        return None
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

def enhance_ground_truth_display(mask_data):
    """
    Enhance ground truth mask data for proper display.
    This addresses the "all black images" issue by proper scaling.
    
    Args:
        mask_data: Raw mask data from Azure Storage
        
    Returns:
        Enhanced image data as bytes for HTTP response
    """
    try:
        # Load mask as PIL Image
        mask_image = Image.open(io.BytesIO(mask_data))
        mask_array = np.array(mask_image)
        
        logging.info(f"Ground truth mask shape: {mask_array.shape}, min: {mask_array.min()}, max: {mask_array.max()}")
        
        # Fix normalization and apply colors
        colored_mask = apply_color_map(mask_array)
        
        # Convert back to PIL and then to bytes
        colored_pil = Image.fromarray(colored_mask)
        
        buffer = io.BytesIO()
        colored_pil.save(buffer, format='PNG')
        
        return buffer.getvalue()
        
    except Exception as e:
        logging.error(f"Error enhancing ground truth display: {e}")
        return mask_data  # Return original data as fallback