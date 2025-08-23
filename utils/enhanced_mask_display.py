"""
Enhanced mask display system using your original matplotlib functions.
Implements both overlay approach and standalone colorized masks.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io
import base64
import logging
from utils.visualization import colorize_mask, CLASS_COLOR_MAP

def plot_image_with_mask_overlay(image, mask, title='Segmentation Overlay', alpha=0.5):
    """
    Your original approach: overlay mask on image using matplotlib with 'jet' colormap.
    This creates beautiful colorful visualizations.
    
    Args:
        image: Original RGB image (numpy array or PIL Image)
        mask: Segmentation mask with class indices 0-7
        title: Title for the plot
        alpha: Transparency for mask overlay (0.5 = 50% transparent)
    
    Returns:
        PIL Image of the overlay visualization
    """
    try:
        # Process image - convert to proper format
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = np.array(image)
        
        # Ensure image is RGB uint8
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # Process mask - ensure it's 2D with class indices
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = np.array(mask)
        
        # Ensure mask is 2D
        if len(mask_array.shape) == 3:
            if mask_array.shape[2] == 1:
                mask_array = np.squeeze(mask_array, axis=2)
            else:
                mask_array = mask_array[:, :, 0]  # Take first channel
        
        # Resize to match if needed
        if image_array.shape[:2] != mask_array.shape:
            mask_pil = Image.fromarray(mask_array)
            mask_pil = mask_pil.resize((image_array.shape[1], image_array.shape[0]))
            mask_array = np.array(mask_pil)
        
        logging.info(f"Image shape: {image_array.shape}, mask shape: {mask_array.shape}")
        logging.info(f"Mask values: min={mask_array.min()}, max={mask_array.max()}")
        
        # Create matplotlib figure - YOUR EXACT APPROACH
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display original image as base
        ax.imshow(image_array)
        
        # Overlay mask with 'jet' colormap and transparency - YOUR METHOD
        ax.imshow(mask_array, cmap='jet', alpha=alpha)
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        pil_image = Image.open(buf)
        return pil_image
        
    except Exception as e:
        logging.error(f"Error in matplotlib overlay: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def create_colorized_mask_standalone(mask, title='Colorized Segmentation Mask'):
    """
    Create a standalone colorized mask without overlaying on original image.
    Uses matplotlib's 'jet' colormap to make each segment a different bright color.
    
    Args:
        mask: Segmentation mask with class indices 0-7
        title: Title for the visualization
    
    Returns:
        PIL Image of the colorized mask
    """
    try:
        # Process mask
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = np.array(mask)
        
        # Ensure mask is 2D
        if len(mask_array.shape) == 3:
            if mask_array.shape[2] == 1:
                mask_array = np.squeeze(mask_array, axis=2)
            else:
                mask_array = mask_array[:, :, 0]
        
        logging.info(f"Standalone mask shape: {mask_array.shape}")
        logging.info(f"Standalone mask values: min={mask_array.min()}, max={mask_array.max()}")
        
        # Create colorized version using matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Use 'jet' colormap to create bright colors for each class
        im = ax.imshow(mask_array, cmap='jet', vmin=0, vmax=7)
        
        # Add colorbar to show class mapping
        cbar = plt.colorbar(im, ax=ax, shrink=0.6)
        cbar.set_label('Segmentation Classes (0-7)')
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        pil_image = Image.open(buf)
        return pil_image
        
    except Exception as e:
        logging.error(f"Error in standalone colorization: {e}")
        return None

def plot_three_panel_comparison(image, predicted_mask, ground_truth_mask):
    """
    Your original 3-panel approach: Original | Ground Truth Overlay | Prediction Overlay
    Based on your plot_predictions function.
    """
    try:
        # Process inputs
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = np.array(image)
        
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
        
        # Create figure with 3 panels - YOUR LAYOUT
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Panel 1: Original image only
        axes[0].imshow(image_array)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Panel 2: Original + Ground Truth Overlay - YOUR METHOD
        axes[1].imshow(image_array)
        if ground_truth_mask is not None:
            gt_array = np.array(ground_truth_mask) if isinstance(ground_truth_mask, Image.Image) else ground_truth_mask
            if len(gt_array.shape) == 3:
                gt_array = gt_array[:, :, 0]
            axes[1].imshow(gt_array, cmap='jet', alpha=0.5)
        axes[1].set_title('Ground Truth Overlay')
        axes[1].axis('off')
        
        # Panel 3: Original + Prediction Overlay - YOUR METHOD
        axes[2].imshow(image_array)
        if predicted_mask is not None:
            pred_array = np.array(predicted_mask) if isinstance(predicted_mask, Image.Image) else predicted_mask
            if len(pred_array.shape) == 3:
                pred_array = pred_array[:, :, 0]
            axes[2].imshow(pred_array, cmap='jet', alpha=0.5)
        axes[2].set_title('Predicted Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        return Image.open(buf)
        
    except Exception as e:
        logging.error(f"Error in three-panel comparison: {e}")
        return None

def plot_beautiful_mask_overlay(image, mask, title='Beautiful Segmentation Overlay', alpha=0.6):
    """
    Create beautiful overlay using custom vibrant color palette (like internet example).
    Uses our custom CLASS_COLOR_MAP instead of jet colormap for much better colors.
    
    Args:
        image: Original RGB image (numpy array or PIL Image)
        mask: Segmentation mask with class indices 0-7
        title: Title for the plot
        alpha: Transparency for mask overlay
    
    Returns:
        PIL Image with beautiful vibrant segmentation colors
    """
    try:
        # Process image with proper channel handling
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = np.array(image)
        
        # Ensure RGB format (not BGR)
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = image_array.astype(np.uint8)
        
        # Process mask with proper channel handling
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = np.array(mask)
        
        # Ensure 2D mask
        if len(mask_array.shape) == 3:
            if mask_array.shape[2] == 1:
                mask_array = np.squeeze(mask_array, axis=2)
            else:
                mask_array = mask_array[:, :, 0]
        
        # Resize to match if needed
        if image_array.shape[:2] != mask_array.shape:
            mask_pil = Image.fromarray(mask_array)
            mask_pil = mask_pil.resize((image_array.shape[1], image_array.shape[0]))
            mask_array = np.array(mask_pil)
        
        logging.info(f"Beautiful overlay - Image: {image_array.shape}, Mask: {mask_array.shape}")
        logging.info(f"Mask values: min={mask_array.min()}, max={mask_array.max()}")
        
        # Create beautiful colorized mask using our custom colors (RGB format)
        colorized_mask_rgb = colorize_mask(mask_array, color_format='RGB')
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display original image as base
        ax.imshow(image_array)
        
        # Overlay with beautiful custom colors (ensuring RGB format)
        ax.imshow(colorized_mask_rgb, alpha=alpha)
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image (maintaining RGB)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        pil_image = Image.open(buf)
        return pil_image
        
    except Exception as e:
        logging.error(f"Error in beautiful overlay: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def create_beautiful_standalone_mask(mask, title='Beautiful Segmentation Classes'):
    """
    Create standalone colorized mask using beautiful custom colors (like internet example).
    
    Args:
        mask: Segmentation mask with class indices 0-7
        title: Title for visualization
    
    Returns:
        PIL Image with beautiful vibrant colors for each class
    """
    try:
        # Process mask
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask)
        else:
            mask_array = np.array(mask)
        
        # Ensure 2D
        if len(mask_array.shape) == 3:
            if mask_array.shape[2] == 1:
                mask_array = np.squeeze(mask_array, axis=2)
            else:
                mask_array = mask_array[:, :, 0]
        
        logging.info(f"Beautiful standalone - Mask: {mask_array.shape}")
        logging.info(f"Mask values: min={mask_array.min()}, max={mask_array.max()}")
        
        # Create beautiful colorized version
        colorized_mask_rgb = colorize_mask(mask_array, color_format='RGB')
        
        # Create matplotlib figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Display beautiful colorized mask
        ax.imshow(colorized_mask_rgb)
        
        ax.set_title(title)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
        buf.seek(0)
        plt.close(fig)
        
        pil_image = Image.open(buf)
        return pil_image
        
    except Exception as e:
        logging.error(f"Error in beautiful standalone: {e}")
        return None

def convert_to_bytes(pil_image):
    """Convert PIL Image to bytes for HTTP response."""
    if pil_image is None:
        return None
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    return buffer.getvalue()

def enhance_mask_for_display(mask_data, display_mode='overlay_ready'):
    """
    Enhance raw mask data for proper display.
    
    Args:
        mask_data: Raw mask bytes from Azure Storage
        display_mode: 'overlay_ready', 'colorized', or 'jet_colormap'
    
    Returns:
        Enhanced image bytes
    """
    try:
        # Load mask from bytes
        mask_image = Image.open(io.BytesIO(mask_data))
        mask_array = np.array(mask_image)
        
        logging.info(f"Raw mask shape: {mask_array.shape}, min: {mask_array.min()}, max: {mask_array.max()}")
        
        if display_mode == 'colorized':
            # Create standalone colorized version
            colorized_pil = create_colorized_mask_standalone(mask_array, 'Ground Truth Mask')
            if colorized_pil:
                return convert_to_bytes(colorized_pil)
        
        elif display_mode == 'jet_colormap':
            # Apply jet colormap directly
            # Normalize to 0-1 range for colormap
            normalized_mask = mask_array.astype(np.float32) / 7.0
            
            # Apply jet colormap
            import matplotlib.cm as cm
            jet_cmap = cm.get_cmap('jet')
            colored_mask = jet_cmap(normalized_mask)
            
            # Convert to 0-255 RGB
            colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)
            
            colored_pil = Image.fromarray(colored_mask_rgb)
            return convert_to_bytes(colored_pil)
        
        # Default: return original for overlay use
        return mask_data
        
    except Exception as e:
        logging.error(f"Error enhancing mask for display: {e}")
        return mask_data  # Return original as fallback