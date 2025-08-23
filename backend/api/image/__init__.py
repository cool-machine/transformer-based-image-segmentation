import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient
import numpy as np
from PIL import Image
import io

def create_test_colorized_mask():
    """Create a test colorized mask to verify the colorization logic works."""
    try:
        # Create a simple test mask with values 0-7
        test_mask = np.array([
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 2, 3],
            [4, 5, 6, 7]
        ], dtype=np.uint8)
        
        # Scale up to make it visible
        test_mask = np.repeat(np.repeat(test_mask, 256, axis=0), 512, axis=1)
        
        # Apply manual jet colormap
        h, w = test_mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Define bright colors for each class
        colors = [
            (0, 0, 255),      # Class 0: Blue
            (0, 128, 255),    # Class 1: Light Blue
            (0, 255, 255),    # Class 2: Cyan
            (0, 255, 0),      # Class 3: Green
            (255, 255, 0),    # Class 4: Yellow
            (255, 128, 0),    # Class 5: Orange
            (255, 0, 0),      # Class 6: Red
            (255, 0, 255),    # Class 7: Magenta
        ]
        
        for class_id in range(8):
            mask_pixels = test_mask == class_id
            if np.any(mask_pixels):
                colored_mask[mask_pixels] = colors[class_id]
        
        # Convert to PNG bytes
        colored_image = Image.fromarray(colored_mask)
        buffer = io.BytesIO()
        colored_image.save(buffer, format='PNG')
        return buffer.getvalue()
    
    except Exception as e:
        logging.error(f"Error creating test colorized mask: {e}")
        return None

def apply_simple_colorization(mask_data):
    """Apply simple but effective colorization to mask data."""
    try:
        logging.info("Starting colorization process")
        
        # Load mask from bytes
        mask_image = Image.open(io.BytesIO(mask_data))
        mask_array = np.array(mask_image)
        
        logging.info(f"Mask loaded: shape={mask_array.shape}, dtype={mask_array.dtype}")
        logging.info(f"Mask values: min={mask_array.min()}, max={mask_array.max()}")
        logging.info(f"Unique values: {np.unique(mask_array)[:10]}")  # Show first 10 unique values
        
        # Handle different image formats
        if len(mask_array.shape) == 3:
            logging.info("Converting from 3D to 2D mask")
            mask_array = mask_array[:, :, 0]  # Take first channel
        
        # Check if this looks like a ground truth mask (values 0-7 or 0-30)
        max_val = mask_array.max()
        unique_vals = len(np.unique(mask_array))
        
        logging.info(f"Mask analysis: max_val={max_val}, unique_vals={unique_vals}")
        
        if max_val <= 30 and unique_vals <= 31:  # Likely a ground truth mask
            logging.info("Detected ground truth mask - applying colorization")
            
            h, w = mask_array.shape
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Simple but effective color mapping
            # Use bright, distinct colors that are easy to see
            color_map = {
                0: [128, 128, 128],   # Gray for background/road
                1: [255, 0, 0],       # Red
                2: [0, 255, 0],       # Green  
                3: [0, 0, 255],       # Blue
                4: [255, 255, 0],     # Yellow
                5: [255, 0, 255],     # Magenta
                6: [0, 255, 255],     # Cyan
                7: [255, 128, 0],     # Orange
                8: [128, 255, 128],   # Light green
                9: [128, 128, 255],   # Light blue
                10: [255, 128, 128],  # Light red
                11: [255, 255, 128],  # Light yellow
            }
            
            # Apply colors
            for value in np.unique(mask_array):
                if value in color_map:
                    mask_pixels = mask_array == value
                    colored_mask[mask_pixels] = color_map[value]
                else:
                    # Default color for unmapped values
                    mask_pixels = mask_array == value
                    colored_mask[mask_pixels] = [200, 200, 200]
            
            # Convert back to PNG
            colored_image = Image.fromarray(colored_mask)
            buffer = io.BytesIO()
            colored_image.save(buffer, format='PNG')
            result = buffer.getvalue()
            
            logging.info(f"Colorization successful: {len(result)} bytes generated")
            return result
        
        else:
            logging.info(f"Not a ground truth mask (max_val={max_val}, unique_vals={unique_vals})")
            return mask_data
        
    except Exception as e:
        logging.error(f"Error in colorization: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return mask_data

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Enhanced image serving with debug logging and robust colorization."""
    
    city = req.route_params.get('city') or req.params.get('city')
    filename = req.route_params.get('filename') or req.params.get('filename')
    
    logging.info(f'🎨 ENHANCED IMAGE REQUEST: city={city}, filename={filename}')
    
    # Special test case - return a test colorized image
    if filename and "test-colorized" in filename:
        test_image = create_test_colorized_mask()
        if test_image:
            logging.info("Returning test colorized image")
            return func.HttpResponse(
                test_image,
                status_code=200,
                headers={
                    "Content-Type": "image/png",
                    "Access-Control-Allow-Origin": "*",
                }
            )
    
    try:
        # Get storage connection
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        if not storage_connection_string:
            logging.error("❌ No storage connection string")
            return func.HttpResponse("Storage not configured", status_code=500)
        
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_name = "images1"
        
        # Detect if this is a ground truth mask
        is_mask = 'gtFine_labelIds' in filename if filename else False
        
        # Try different possible blob paths
        possible_paths = []
        if filename:
            if is_mask:
                possible_paths = [
                    f"masks/{filename}",
                    f"images/{filename}",
                    f"images/masks/{filename}"
                ]
            else:
                possible_paths = [
                    f"images/{filename}",
                    f"{filename}"
                ]
        
        image_data = None
        successful_path = None
        
        for blob_path in possible_paths:
            try:
                logging.info(f"Trying blob path: {blob_path}")
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
                if blob_client.exists():
                    image_data = blob_client.download_blob().readall()
                    successful_path = blob_path
                    logging.info(f"✅ Successfully loaded from: {blob_path} ({len(image_data)} bytes)")
                    break
            except Exception as e:
                logging.warning(f"Failed to load from {blob_path}: {e}")
        
        if not image_data:
            logging.error(f"❌ Could not find image: {filename}")
            return func.HttpResponse("Image not found", status_code=404)
        
        # Apply colorization if this is a ground truth mask
        if is_mask:
            logging.info("🎨 Applying colorization to ground truth mask")
            try:
                colorized_data = apply_simple_colorization(image_data)
                if colorized_data and len(colorized_data) != len(image_data):
                    image_data = colorized_data
                    logging.info("✅ Colorization applied successfully")
                else:
                    logging.warning("⚠️ Colorization didn't change the image")
            except Exception as e:
                logging.error(f"❌ Colorization failed: {e}")
        
        return func.HttpResponse(
            image_data,
            status_code=200,
            headers={
                "Content-Type": "image/png",
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-cache",  # Disable caching for testing
                "X-Processed-Path": successful_path or "unknown",
                "X-Is-Colorized": "true" if is_mask else "false"
            }
        )
        
    except Exception as e:
        logging.error(f'❌ Error serving enhanced image: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )