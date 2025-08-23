"""
Enhanced segmentation endpoint with proper model loading and prediction.
Fixes the issues with model integration and prediction generation.
"""

import logging
import azure.functions as func
import json
import os
import sys
import tempfile
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from azure.storage.blob import BlobServiceClient

# Add the project root to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import our enhanced components
try:
    from src.mask_prediction import MaskPredictor
    from utils.enhanced_mask_display import (
        plot_image_with_mask_overlay, 
        create_colorized_mask_standalone,
        plot_three_panel_comparison,
        convert_to_bytes
    )
    ENHANCED_FEATURES = True
    logging.info("Enhanced segmentation features imported successfully")
except ImportError as e:
    ENHANCED_FEATURES = False
    logging.warning(f"Enhanced segmentation features not available: {e}")

# Global predictor instance
predictor = None

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Process image segmentation with enhanced model loading and visualization."""
    
    logging.info('Enhanced segmentation request received')
    
    try:
        # Initialize global predictor
        global predictor
        if predictor is None and ENHANCED_FEATURES:
            predictor = MaskPredictor()
            logging.info("MaskPredictor initialized")
        
        # Parse request
        if req.get_json():
            request_data = req.get_json()
            image_name = request_data.get('image_name')
            city = request_data.get('city')
        else:
            image_name = req.params.get('image_name')
            city = req.params.get('city')
        
        if not image_name:
            raise ValueError("image_name is required")
        
        # Process the segmentation
        result = process_enhanced_segmentation(image_name, city, predictor)
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logging.error(f'Enhanced segmentation failed: {str(e)}')
        import traceback
        logging.error(traceback.format_exc())
        
        error_response = {
            "error": f"Segmentation failed: {str(e)}",
            "prediction_available": False,
            "ground_truth_available": False,
            "enhanced_features": ENHANCED_FEATURES,
            "debug_info": str(e)
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )

def process_enhanced_segmentation(image_name, city, predictor):
    """Process segmentation with enhanced model loading and visualization."""
    
    logging.info(f"Processing enhanced segmentation for {image_name}")
    
    # Get storage connection
    storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
    if not storage_connection_string:
        raise ValueError("Storage connection not configured")
    
    blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
    container_name = "images1"
    
    # Load models if not already loaded
    model_loaded = False
    model_used = "None"
    
    if ENHANCED_FEATURES and predictor:
        try:
            # Try to load SegFormer first
            if not hasattr(predictor, '_segformer_loaded'):
                try:
                    predictor.load_segformer()
                    predictor._segformer_loaded = True
                    model_loaded = True
                    model_used = "SegFormer"
                    logging.info("✅ SegFormer model loaded successfully")
                except Exception as e:
                    logging.warning(f"Could not load SegFormer: {e}")
                    predictor._segformer_loaded = False
            
            # Try UNet as fallback
            if not model_loaded and not hasattr(predictor, '_unet_loaded'):
                try:
                    predictor.load_unet()
                    predictor._unet_loaded = True
                    model_loaded = True
                    model_used = "UNet"
                    logging.info("✅ UNet model loaded successfully")
                except Exception as e:
                    logging.warning(f"Could not load UNet: {e}")
                    predictor._unet_loaded = False
                    
        except Exception as e:
            logging.error(f"Error during model loading: {e}")
    
    # Find and load the original image
    image_paths = [
        f"images/{image_name}_leftImg8bit.png",
        f"images/img_{image_name}_leftImg8bit.png"
    ]
    
    original_image = None
    image_blob_path = None
    
    for path in image_paths:
        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=path)
            if blob_client.exists():
                image_data = blob_client.download_blob().readall()
                original_image = Image.open(BytesIO(image_data))
                image_blob_path = path
                logging.info(f"Loaded original image from: {path}")
                break
        except Exception as e:
            logging.warning(f"Could not load image from {path}: {e}")
    
    if original_image is None:
        raise FileNotFoundError(f"Could not find original image for {image_name}")
    
    # Convert to numpy array for processing
    image_array = np.array(original_image)
    
    # Load ground truth mask
    mask_paths = [
        f"masks/{image_name}_gtFine_labelIds.png",
        f"masks/img_{image_name}_gtFine_labelIds.png"
    ]
    
    ground_truth_mask = None
    ground_truth_available = False
    
    for path in mask_paths:
        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=path)
            if blob_client.exists():
                mask_data = blob_client.download_blob().readall()
                ground_truth_mask = Image.open(BytesIO(mask_data))
                ground_truth_available = True
                logging.info(f"Loaded ground truth mask from: {path}")
                break
        except Exception as e:
            logging.warning(f"Could not load mask from {path}: {e}")
    
    # Generate prediction
    predicted_mask = None
    prediction_available = False
    
    if model_loaded and ENHANCED_FEATURES and predictor:
        try:
            # Prepare image for prediction (use existing preprocessing)
            processed_image = preprocess_for_prediction(image_array)
            
            # Make prediction
            if hasattr(predictor, '_segformer_loaded') and predictor._segformer_loaded:
                predicted_mask = predictor.predict_segformer(processed_image)
                logging.info("✅ SegFormer prediction generated")
            elif hasattr(predictor, '_unet_loaded') and predictor._unet_loaded:
                predicted_mask = predictor.predict_unet(processed_image)
                logging.info("✅ UNet prediction generated")
            
            if predicted_mask is not None:
                prediction_available = True
                
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    # Create visualizations
    visualization_results = {}
    
    if ENHANCED_FEATURES:
        try:
            # Create ground truth overlay using YOUR matplotlib approach
            if ground_truth_available:
                gt_array = np.array(ground_truth_mask)
                gt_overlay = plot_image_with_mask_overlay(
                    image_array, gt_array, title="Ground Truth Overlay", alpha=0.5
                )
                if gt_overlay:
                    # Convert to base64 for JSON response
                    gt_bytes = convert_to_bytes(gt_overlay)
                    if gt_bytes:
                        import base64
                        visualization_results["ground_truth_overlay"] = base64.b64encode(gt_bytes).decode()
                        logging.info("✅ Ground truth overlay created using your matplotlib method")
            
            # Create prediction overlay using YOUR matplotlib approach
            if prediction_available:
                pred_overlay = plot_image_with_mask_overlay(
                    image_array, predicted_mask, title="Prediction Overlay", alpha=0.5
                )
                if pred_overlay:
                    pred_bytes = convert_to_bytes(pred_overlay)
                    if pred_bytes:
                        import base64
                        visualization_results["prediction_overlay"] = base64.b64encode(pred_bytes).decode()
                        logging.info("✅ Prediction overlay created using your matplotlib method")
            
            # Create three-panel comparison using YOUR original layout
            if ground_truth_available or prediction_available:
                three_panel = plot_three_panel_comparison(
                    image_array, 
                    predicted_mask if prediction_available else None,
                    np.array(ground_truth_mask) if ground_truth_available else None
                )
                if three_panel:
                    panel_bytes = convert_to_bytes(three_panel)
                    if panel_bytes:
                        import base64
                        visualization_results["three_panel_comparison"] = base64.b64encode(panel_bytes).decode()
                        logging.info("✅ Three-panel comparison created using your original layout")
                    
        except Exception as e:
            logging.warning(f"Visualization creation failed: {e}")
            import traceback
            logging.warning(traceback.format_exc())
    
    # Save prediction to Azure Storage (if generated)
    prediction_url = None
    if prediction_available and predicted_mask is not None:
        try:
            # Convert prediction to image and save
            pred_filename = f"{image_name}_predicted_mask.png"
            prediction_url = save_prediction_to_storage(
                predicted_mask, pred_filename, blob_service_client, container_name
            )
            logging.info(f"✅ Prediction saved to storage: {pred_filename}")
        except Exception as e:
            logging.warning(f"Could not save prediction to storage: {e}")
    
    # Prepare response
    result = {
        "image_name": image_name,
        "city": city or "unknown",
        "model_used": model_used,
        "model_loaded": model_loaded,
        "prediction_available": prediction_available,
        "ground_truth_available": ground_truth_available,
        "enhanced_features": ENHANCED_FEATURES,
        "visualization_results": visualization_results,
        "prediction_url": prediction_url,
        "original_image_url": f"https://ocp8.azurewebsites.net/api/image/{image_name}_leftImg8bit.png",
        "ground_truth_url": f"https://ocp8.azurewebsites.net/api/image/{image_name}_gtFine_labelIds.png" if ground_truth_available else None
    }
    
    return result

def preprocess_for_prediction(image_array):
    """Preprocess image array for model prediction."""
    
    # Convert to float32 and normalize to [0, 1]
    if image_array.dtype == np.uint8:
        processed = image_array.astype(np.float32) / 255.0
    else:
        processed = image_array.astype(np.float32)
    
    # Ensure we have the right shape (H, W, 3)
    if len(processed.shape) == 2:
        processed = np.stack([processed] * 3, axis=-1)
    elif processed.shape[-1] == 4:  # RGBA
        processed = processed[:, :, :3]  # Remove alpha channel
    
    # Resize to model input size (512, 1024)
    processed_pil = Image.fromarray((processed * 255).astype(np.uint8))
    processed_pil = processed_pil.resize((1024, 512))
    processed = np.array(processed_pil).astype(np.float32) / 255.0
    
    return processed

def save_prediction_to_storage(predicted_mask, filename, blob_service_client, container_name):
    """Save prediction mask to Azure Storage."""
    
    try:
        # Convert prediction to colorized image
        from utils.mask_visualization import apply_color_map
        
        # Apply color mapping
        colored_mask = apply_color_map(predicted_mask)
        
        # Convert to PIL Image
        pred_image = Image.fromarray(colored_mask)
        
        # Convert to bytes
        buffer = BytesIO()
        pred_image.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Upload to Azure Storage
        blob_path = f"predictions/{filename}"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        
        # Return URL
        return f"https://ocp8.azurewebsites.net/api/image/{filename}"
        
    except Exception as e:
        logging.error(f"Error saving prediction to storage: {e}")
        return None