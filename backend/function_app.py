"""
Step 6: Complete visualization Function App with matplotlib colorized mask generation
"""

import azure.functions as func
import logging
import json
import os
import sys
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from azure.storage.blob import BlobServiceClient
import tempfile
from pathlib import Path

# Use non-interactive backend
matplotlib.use('Agg')

# Try to import TensorFlow and Transformers (for model loading)
try:
    import tensorflow as tf
    from transformers import TFSegformerForSemanticSegmentation, SegformerConfig
    HAS_MODEL_SUPPORT = True
    logging.info("✅ Model libraries loaded successfully")
except ImportError as e:
    HAS_MODEL_SUPPORT = False
    logging.warning(f"⚠️ Model libraries not available: {e}")

app = func.FunctionApp()

# SUPER BRIGHT & DISTINCT COLORS - Every class gets maximum visibility!
BEAUTIFUL_COLOR_MAP = {
    0: (255, 0, 255),      # flat/road - BRIGHT MAGENTA/FUCHSIA  
    1: (255, 0, 0),        # human/person - BRIGHT RED
    2: (0, 255, 255),      # vehicle - BRIGHT CYAN
    3: (0, 255, 0),        # construction/building - BRIGHT LIME GREEN
    4: (255, 255, 0),      # object/pole/sign - BRIGHT YELLOW
    5: (0, 128, 255),      # nature/vegetation - BRIGHT BLUE
    6: (255, 192, 203),    # sky - LIGHT PINK/ROSE
    7: (255, 165, 0)       # void/other - BRIGHT ORANGE
}

# Cityscapes 30-class to 8-class conversion mapping
CITYSCAPES_TO_8CLASS = {
    # Original class ID -> 8-class ID
    0: 0,   # road -> flat
    1: 0,   # sidewalk -> flat  
    2: 0,   # parking -> flat
    3: 0,   # rail track -> flat
    4: 1,   # person -> human
    5: 1,   # rider -> human
    6: 2,   # car -> vehicle
    7: 2,   # truck -> vehicle
    8: 2,   # bus -> vehicle
    9: 2,   # on rails -> vehicle
    10: 2,  # motorcycle -> vehicle
    11: 2,  # bicycle -> vehicle
    12: 2,  # caravan -> vehicle
    13: 2,  # trailer -> vehicle
    14: 3,  # building -> construction
    15: 3,  # wall -> construction
    16: 3,  # fence -> construction
    17: 3,  # guard rail -> construction
    18: 3,  # bridge -> construction
    19: 3,  # tunnel -> construction
    20: 4,  # pole -> object
    21: 4,  # pole group -> object
    22: 4,  # traffic sign -> object
    23: 4,  # traffic light -> object
    24: 5,  # vegetation -> nature
    25: 5,  # terrain -> nature
    26: 6,  # sky -> sky
    27: 7,  # ground -> void
    28: 7,  # dynamic -> void
    29: 7,  # static -> void
}

def convert_cityscapes_to_8class(mask):
    """Convert Cityscapes 30-class mask to 8-class mask"""
    if len(mask.shape) > 2:
        mask = mask[:, :, 0] if mask.shape[-1] == 1 else mask.squeeze()
    
    # Create output mask
    converted_mask = np.zeros_like(mask, dtype=np.uint8)
    
    # Convert each class
    for original_id, new_id in CITYSCAPES_TO_8CLASS.items():
        converted_mask[mask == original_id] = new_id
    
    return converted_mask

def colorize_mask_beautiful(mask):
    """Apply beautiful colors to mask classes"""
    # First convert from Cityscapes 30-class to 8-class
    mask_8class = convert_cityscapes_to_8class(mask)
    
    h, w = mask_8class.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Apply colors to the 8-class mask
    for class_idx, color in BEAUTIFUL_COLOR_MAP.items():
        if class_idx < 8:  # Only use 0-7 classes
            colorized[mask_8class == class_idx] = color
    
    return colorized

# Global model storage
_segformer_model = None
_model_loaded = False

def download_model_from_azure(storage_connection_string, model_container="models"):
    """Download trained SegFormer model files from Azure Storage"""
    try:
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client(model_container)
        
        # Create temp directory for model files
        model_dir = tempfile.mkdtemp()
        logging.info(f"📁 Created temp model directory: {model_dir}")
        
        # List all blobs in model container
        blob_list = container_client.list_blobs()
        downloaded_files = []
        
        for blob in blob_list:
            blob_path = os.path.join(model_dir, blob.name)
            
            # Create subdirectories if needed
            os.makedirs(os.path.dirname(blob_path), exist_ok=True)
            
            # Download blob
            blob_client = container_client.get_blob_client(blob.name)
            with open(blob_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            downloaded_files.append(blob_path)
            logging.info(f"📥 Downloaded: {blob.name}")
        
        logging.info(f"✅ Downloaded {len(downloaded_files)} model files")
        return model_dir, downloaded_files
        
    except Exception as e:
        logging.error(f"❌ Error downloading model: {e}")
        return None, []

def load_trained_segformer_model(model_dir):
    """Load the trained SegFormer model from downloaded files"""
    global _segformer_model, _model_loaded
    
    if not HAS_MODEL_SUPPORT:
        logging.error("❌ TensorFlow/Transformers not available")
        return False
        
    try:
        # SegFormer configuration for 8-class Cityscapes
        config = SegformerConfig(
            num_labels=8,
            id2label={0: "flat", 1: "human", 2: "vehicle", 3: "construction", 
                     4: "object", 5: "nature", 6: "sky", 7: "void"},
            label2id={"flat": 0, "human": 1, "vehicle": 2, "construction": 3, 
                     "object": 4, "nature": 5, "sky": 6, "void": 7},
            image_size=(512, 1024),
        )
        
        # Load base SegFormer model
        _segformer_model = TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Look for SegFormer model files in downloaded files
        segformer_files = list(Path(model_dir).rglob("segformer*"))
        model_files = list(Path(model_dir).rglob("*.h5"))
        checkpoint_files = list(Path(model_dir).rglob("checkpoint*"))
        
        logging.info(f"🔍 Found files: segformer={len(segformer_files)}, h5={len(model_files)}, checkpoints={len(checkpoint_files)}")
        
        # Prioritize segformer-specific files
        if segformer_files:
            segformer_h5_files = [f for f in segformer_files if f.suffix in ['.h5', '.hdf5', '.keras']]
            if segformer_h5_files:
                model_file_list = segformer_h5_files
            else:
                model_file_list = model_files
        else:
            model_file_list = model_files
            
        checkpoint_file_list = checkpoint_files
        
        if model_file_list:
            # Load fine-tuned weights
            weights_file = str(model_file_list[0])
            logging.info(f"🔄 Loading fine-tuned weights from: {weights_file}")
            
            try:
                _segformer_model.load_weights(weights_file)
                logging.info("✅ Fine-tuned weights loaded successfully!")
            except Exception as e:
                logging.warning(f"⚠️ Could not load fine-tuned weights: {e}")
                logging.info("🔄 Using base pre-trained model")
        
        elif checkpoint_file_list:
            # Try to restore from checkpoint
            checkpoint_dir = str(checkpoint_file_list[0].parent)
            logging.info(f"🔄 Restoring from checkpoint: {checkpoint_dir}")
            
            try:
                checkpoint = tf.train.Checkpoint(model=_segformer_model)
                checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
                logging.info("✅ Checkpoint restored successfully!")
            except Exception as e:
                logging.warning(f"⚠️ Could not restore checkpoint: {e}")
                logging.info("🔄 Using base pre-trained model")
        else:
            logging.info("🔄 No fine-tuned weights found, using base pre-trained model")
        
        _model_loaded = True
        logging.info("✅ SegFormer model loaded and ready for inference!")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error loading SegFormer model: {e}")
        _model_loaded = False
        return False

def predict_with_segformer(image_array):
    """
    Generate prediction using the loaded SegFormer model.
    Handles proper channel order and resizing for SegFormer requirements.
    """
    global _segformer_model, _model_loaded
    
    if not _model_loaded or _segformer_model is None:
        logging.error("❌ SegFormer model not loaded")
        return None
        
    try:
        # Store original dimensions for later resizing back
        original_height, original_width = image_array.shape[:2]
        logging.info(f"🖼️ Original image shape: {image_array.shape}")
        
        # Preprocess image for SegFormer (expects channels last initially)
        if len(image_array.shape) == 3:
            # Ensure RGB format (channels last: H, W, C)
            if image_array.shape[-1] != 3:
                logging.error(f"❌ Expected 3 channels, got {image_array.shape[-1]}")
                return None
            
            # Resize to SegFormer input size (512, 1024) - keeping channels last
            image_resized = tf.image.resize(image_array, [512, 1024])
            image_normalized = tf.cast(image_resized, tf.float32) / 255.0
            
            # Add batch dimension: (1, H, W, C) - still channels last
            image_batch = tf.expand_dims(image_normalized, 0)
            
            logging.info(f"🔄 Preprocessed shape: {image_batch.shape} (channels last)")
            
        else:
            logging.error(f"❌ Invalid image format: {image_array.shape}")
            return None
        
        # SegFormer inference - try both channel orders if needed
        logging.info("🔮 Running SegFormer inference...")
        
        try:
            # First try channels last (standard TensorFlow format)
            outputs = _segformer_model(image_batch)
        except Exception as channels_error:
            logging.warning(f"⚠️ Channels last failed: {channels_error}")
            logging.info("🔄 Trying channels first format...")
            
            # Convert to channels first: (1, C, H, W)
            image_channels_first = tf.transpose(image_batch, [0, 3, 1, 2])
            logging.info(f"🔄 Channels first shape: {image_channels_first.shape}")
            
            try:
                outputs = _segformer_model(image_channels_first)
            except Exception as e:
                logging.error(f"❌ Both channel formats failed: {e}")
                return None
        
        # Extract logits and handle different output formats
        prediction_logits = outputs.logits
        logging.info(f"🧠 Model output shape: {prediction_logits.shape}")
        
        # Handle different output channel orders
        if len(prediction_logits.shape) == 4:  # [batch, height, width, classes] or [batch, classes, height, width]
            if prediction_logits.shape[1] == 8:  # Channels first: [1, 8, H, W] 
                logging.info("🔄 Converting model output from channels first to channels last")
                prediction_logits = tf.transpose(prediction_logits, [0, 2, 3, 1])  # [1, H, W, 8]
                logging.info(f"🔄 Converted output shape: {prediction_logits.shape}")
        
        # Convert logits to class predictions (argmax over channel dimension)
        predicted_mask = tf.argmax(prediction_logits, axis=-1)  # Remove class dimension
        predicted_mask = tf.squeeze(predicted_mask)  # Remove batch dimension
        
        # Convert to numpy
        predicted_mask_np = predicted_mask.numpy().astype(np.uint8)
        logging.info(f"🎯 Prediction mask shape: {predicted_mask_np.shape}")
        
        # Resize prediction back to original image size
        if predicted_mask_np.shape != (original_height, original_width):
            logging.info(f"🔄 Resizing prediction from {predicted_mask_np.shape} to ({original_height}, {original_width})")
            
            # Convert to tensor for resizing (add channel dimension temporarily)
            mask_tensor = tf.expand_dims(tf.cast(predicted_mask_np, tf.float32), -1)
            
            # Resize using nearest neighbor to preserve class labels
            mask_resized = tf.image.resize(
                mask_tensor, 
                [original_height, original_width], 
                method='nearest'
            )
            
            # Convert back to numpy and remove extra dimension
            predicted_mask_final = tf.squeeze(mask_resized).numpy().astype(np.uint8)
        else:
            predicted_mask_final = predicted_mask_np
        
        logging.info(f"✅ Final prediction shape: {predicted_mask_final.shape}")
        logging.info(f"✅ Prediction classes: {np.unique(predicted_mask_final)}")
        
        return predicted_mask_final
        
    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

# Add CORS preflight handler
@app.function_name(name="cors_preflight")
@app.route(route="{*path}", methods=["OPTIONS"], auth_level=func.AuthLevel.ANONYMOUS)
def cors_preflight(req: func.HttpRequest) -> func.HttpResponse:
    """Handle CORS preflight requests"""
    return func.HttpResponse(
        "",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

@app.function_name(name="health")
@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Enhanced health check with all visualization capabilities."""
    
    logging.info('Health check endpoint hit.')
    
    try:
        # Check matplotlib availability
        matplotlib_available = False
        matplotlib_version = None
        try:
            import matplotlib
            matplotlib_available = True
            matplotlib_version = matplotlib.__version__
        except ImportError:
            pass
        
        # Check storage configuration
        storage_configured = bool(os.getenv('IMAGES_STORAGE_CONNECTION_STRING'))
        
        result = {
            "status": "ok",
            "message": "Central US Function App with complete visualization stack",
            "version": "visualization-1.0",
            "python_version": f"{sys.version.split()[0]}",
            "storage_configured": storage_configured,
            "matplotlib_available": matplotlib_available,
            "matplotlib_version": matplotlib_version,
            "numpy_available": True,
            "numpy_version": np.__version__,
            "features": [
                "Azure Storage access",
                "PIL image processing",
                "Numpy array operations",
                "Matplotlib colorized mask generation",
                "Base64 API responses"
            ]
        }
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            mimetype="application/json",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
    except Exception as e:
        logging.error(f"Health check error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json", 
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

@app.function_name(name="images")
@app.route(route="images", auth_level=func.AuthLevel.ANONYMOUS)
def images(req: func.HttpRequest) -> func.HttpResponse:
    """List available images from storage."""
    
    logging.info('Images endpoint hit.')
    
    try:
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500
            )
        
        # Connect to Azure Storage
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client("images1")
        
        # List blobs in images folder
        image_blobs = []
        for blob in container_client.list_blobs(name_starts_with="images/"):
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None
                })
        
        result = {
            "images": image_blobs[:10],  # Limit to first 10 for testing
            "total_count": len(image_blobs),
            "container": "images1",
            "path": "images/"
        }
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            mimetype="application/json",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logging.error(f"Images endpoint error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

@app.function_name(name="masks")
@app.route(route="masks", auth_level=func.AuthLevel.ANONYMOUS)
def masks(req: func.HttpRequest) -> func.HttpResponse:
    """List available masks from storage."""
    
    logging.info('Masks endpoint hit.')
    
    try:
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500
            )
        
        # Connect to Azure Storage
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client("images1")
        
        # List blobs in masks folder
        mask_blobs = []
        for blob in container_client.list_blobs(name_starts_with="masks/"):
            if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask_blobs.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None
                })
        
        result = {
            "masks": mask_blobs[:10],  # Limit to first 10 for testing
            "total_count": len(mask_blobs),
            "container": "images1",
            "path": "masks/"
        }
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            mimetype="application/json",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logging.error(f"Masks endpoint error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

@app.function_name(name="check_model_container")
@app.route(route="check-model-container", auth_level=func.AuthLevel.ANONYMOUS)
def check_model_container(req: func.HttpRequest) -> func.HttpResponse:
    """Debug endpoint to check what's in the model container"""
    try:
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500
            )
        
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        
        # Check if model container exists
        try:
            model_container_client = blob_service_client.get_container_client("models")
            blob_list = list(model_container_client.list_blobs())
            
            model_files = []
            for blob in blob_list:
                model_files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None
                })
            
            result = {
                "status": "success",
                "model_container_exists": True,
                "files_found": len(model_files),
                "model_files": model_files,
                "has_model_support": HAS_MODEL_SUPPORT,
                "model_loaded": _model_loaded
            }
            
        except Exception as container_error:
            result = {
                "status": "error",
                "model_container_exists": False,
                "error": str(container_error),
                "has_model_support": HAS_MODEL_SUPPORT,
                "model_loaded": _model_loaded
            }
        
        return func.HttpResponse(
            json.dumps(result, indent=2),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": f"Unexpected error: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )

@app.function_name(name="test_matplotlib")
@app.route(route="test-matplotlib", auth_level=func.AuthLevel.ANONYMOUS)
def test_matplotlib(req: func.HttpRequest) -> func.HttpResponse:
    """Test matplotlib functionality."""
    
    logging.info('Test matplotlib endpoint hit.')
    
    try:
        # Create a simple test plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        
        # Create test data
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        
        ax.plot(x, y, label='sin(x)')
        ax.set_title('Matplotlib Test - Colorized Mask Generation Ready')
        ax.set_xlabel('x')
        ax.set_ylabel('sin(x)')
        ax.legend()
        ax.grid(True)
        
        # Convert to PIL Image
        buffer = BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        plt.close(fig)
        
        # Convert to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        result = {
            "status": "success",
            "message": "Matplotlib is working correctly",
            "matplotlib_version": matplotlib.__version__,
            "test_plot": f"data:image/png;base64,{img_base64}"
        }
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logging.error(f"Matplotlib test error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Matplotlib test failed: {str(e)}"}),
            mimetype="application/json",
            status_code=500
        )

def plot_beautiful_mask_overlay(image, mask, title='Beautiful Segmentation Overlay', alpha=0.6):
    """
    Your original matplotlib overlay function - the exact approach you wanted!
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Display the original image
    ax.imshow(image)
    
    # Create beautiful colorized mask and overlay it
    colorized_mask = colorize_mask_beautiful(mask)
    ax.imshow(colorized_mask, alpha=alpha)
    
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def create_three_panel_comparison(image, mask, title_prefix=''):
    """
    Create three-panel layout: Original | Mask | Overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title(f'{title_prefix}Original Image')
    axes[0].axis('off')
    
    # Colorized mask only
    im1 = axes[1].imshow(mask, cmap='jet', vmin=0, vmax=7)
    axes[1].set_title(f'{title_prefix}Colorized Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(mask, cmap='jet', alpha=0.6, vmin=0, vmax=7)
    axes[2].set_title(f'{title_prefix}Overlay')
    axes[2].axis('off')
    
    # Add colorbar
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    plt.tight_layout()
    return fig

def convert_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    plt.close(fig)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

@app.function_name(name="colorized_masks")
@app.route(route="colorized-masks", auth_level=func.AuthLevel.ANONYMOUS)
def colorized_masks(req: func.HttpRequest) -> func.HttpResponse:
    """Generate colorized mask visualizations - Your core functionality!"""
    
    logging.info('Colorized masks endpoint hit.')
    
    try:
        # Get parameters
        image_name = req.params.get('image_name')
        if not image_name:
            image_name = "lindau_000000_000019_leftImg8bit"  # Default test image
        
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500
            )
        
        # Connect to Azure Storage
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client("images1")
        
        # Load image - handle the correct naming pattern
        if not image_name.endswith("_leftImg8bit"):
            image_blob_name = f"images/{image_name}_leftImg8bit.png"
        else:
            image_blob_name = f"images/{image_name}.png"
            
        # For mask, convert from leftImg8bit to gtFine_labelIds pattern
        base_name = image_name.replace("_leftImg8bit", "")
        mask_blob_name = f"masks/{base_name}_gtFine_labelIds.png"
        
        try:
            # Download image
            image_blob = container_client.get_blob_client(image_blob_name)
            image_data = image_blob.download_blob().readall()
            image = Image.open(BytesIO(image_data))
            image_array = np.array(image)
            
            # Download mask
            mask_blob = container_client.get_blob_client(mask_blob_name)
            mask_data = mask_blob.download_blob().readall()
            mask_image = Image.open(BytesIO(mask_data))
            mask_array = np.array(mask_image)
            
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Could not load image/mask: {str(e)}"}),
                mimetype="application/json",
                status_code=404
            )
        
        # Generate individual visualizations for three panels
        
        # 1. Original Image (clean)
        orig_fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(image_array)
        ax.set_title('Original Image')
        ax.axis('off')
        original_base64 = convert_to_base64(orig_fig)
        
        # 2. Ground Truth Mask (transparent overlay on original image)
        mask_fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        # Show original image as background
        ax.imshow(image_array)
        # Overlay beautiful colorized mask with stronger transparency (alpha=0.8) for better visibility
        colorized_mask = colorize_mask_beautiful(mask_array)
        ax.imshow(colorized_mask, alpha=0.8)
        ax.set_title('Ground Truth Mask (Beautiful Colors!)')
        ax.axis('off')
        ground_truth_base64 = convert_to_base64(mask_fig)
        
        # 3. Predicted Mask - Use trained SegFormer model
        predicted_base64 = None
        
        # Try to load model if not already loaded
        if HAS_MODEL_SUPPORT and not _model_loaded:
            try:
                logging.info("🔄 Attempting to load trained SegFormer model...")
                model_dir, _ = download_model_from_azure(storage_connection_string, "models")
                if model_dir:
                    load_trained_segformer_model(model_dir)
            except Exception as e:
                logging.warning(f"⚠️ Could not load model: {e}")
        
        # Generate prediction if model is loaded
        if _model_loaded:
            try:
                logging.info("🔮 Generating AI prediction...")
                predicted_mask = predict_with_segformer(image_array)
                
                if predicted_mask is not None:
                    # Create prediction visualization
                    pred_fig, pred_ax = plt.subplots(1, 1, figsize=(12, 8))
                    
                    # Show original image as background
                    pred_ax.imshow(image_array)
                    # Overlay predicted mask with beautiful colors
                    predicted_colorized = colorize_mask_beautiful(predicted_mask)
                    pred_ax.imshow(predicted_colorized, alpha=0.8)
                    
                    pred_ax.set_title('AI Model Prediction (Fine-tuned SegFormer)')
                    pred_ax.axis('off')
                    
                    predicted_base64 = convert_to_base64(pred_fig)
                    logging.info("✅ AI prediction visualization created!")
                else:
                    logging.warning("⚠️ Prediction returned None")
            except Exception as e:
                logging.error(f"❌ Prediction failed: {e}")
                import traceback
                logging.error(traceback.format_exc())
        else:
            logging.info("ℹ️ Model not loaded, skipping prediction")
        
        result = {
            "status": "success",
            "message": "Individual visualizations generated successfully!",
            "image_name": image_name,
            "image_shape": image_array.shape,
            "mask_shape": mask_array.shape,
            "mask_unique_values": sorted(np.unique(mask_array).tolist()),
            "visualizations": {
                "original": original_base64,
                "ground_truth": ground_truth_base64,
                "predicted": predicted_base64
            },
            "generation_method": "individual panels for comparison"
        }
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logging.error(f"Colorized masks error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

@app.function_name(name="image_thumbnail")
@app.route(route="image-thumbnail", auth_level=func.AuthLevel.ANONYMOUS)
def image_thumbnail(req: func.HttpRequest) -> func.HttpResponse:
    """Serve image thumbnails for gallery display."""
    
    logging.info('Image thumbnail endpoint hit.')
    
    try:
        # Get parameters
        image_name = req.params.get('image_name')
        if not image_name:
            return func.HttpResponse(
                json.dumps({"error": "image_name parameter required"}),
                mimetype="application/json",
                status_code=400,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        
        # Connect to Azure Storage
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client("images1")
        
        # Load image - handle the correct naming pattern
        if not image_name.endswith("_leftImg8bit"):
            image_blob_name = f"images/{image_name}_leftImg8bit.png"
        else:
            image_blob_name = f"images/{image_name}.png"
            
        try:
            # Download image
            image_blob = container_client.get_blob_client(image_blob_name)
            image_data = image_blob.download_blob().readall()
            image = Image.open(BytesIO(image_data))
            
            # Create thumbnail (300x200)
            image.thumbnail((300, 200), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            result = {
                "status": "success",
                "image_name": image_name,
                "thumbnail": f"data:image/png;base64,{img_base64}"
            }
            
            return func.HttpResponse(
                json.dumps(result),
                mimetype="application/json",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
            
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Could not load image: {str(e)}"}),
                mimetype="application/json",
                status_code=404,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        
    except Exception as e:
        logging.error(f"Image thumbnail error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )

@app.function_name(name="predict_with_container")
@app.route(route="predict-with-container", auth_level=func.AuthLevel.ANONYMOUS)
def predict_with_container(req: func.HttpRequest) -> func.HttpResponse:
    """Use the containerized model API for predictions."""
    
    logging.info('Container prediction endpoint hit.')
    
    try:
        # Get parameters
        image_name = req.params.get('image_name')
        if not image_name:
            image_name = "lindau_000000_000019_leftImg8bit"  # Default test image
        
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            return func.HttpResponse(
                json.dumps({"error": "Storage connection string not configured"}),
                mimetype="application/json",
                status_code=500,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        
        # Connect to Azure Storage and get image
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client("images1")
        
        # Load image
        if not image_name.endswith("_leftImg8bit"):
            image_blob_name = f"images/{image_name}_leftImg8bit.png"
        else:
            image_blob_name = f"images/{image_name}.png"
            
        try:
            # Download image
            image_blob = container_client.get_blob_client(image_blob_name)
            image_data = image_blob.download_blob().readall()
            image = Image.open(BytesIO(image_data))
            image_array = np.array(image)
            
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": f"Could not load image: {str(e)}"}),
                mimetype="application/json",
                status_code=404,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type"
                }
            )
        
        # Convert image to base64 for API call
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Call the container API
        import requests
        container_url = "https://segformer-model-api.victoriousforest-a1dd6cc5.centralus.azurecontainerapps.io"
        
        try:
            logging.info(f"🚀 Calling container API at {container_url}/predict")
            
            response = requests.post(
                f"{container_url}/predict",
                json={"image": image_b64},
                timeout=120,  # 2 minutes timeout
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                container_result = response.json()
                
                # Generate visualization with the prediction
                if 'prediction' in container_result:
                    # Decode the prediction mask
                    prediction_b64 = container_result['prediction'].split(',')[1]
                    pred_data = base64.b64decode(prediction_b64)
                    pred_image = Image.open(BytesIO(pred_data))
                    pred_mask = np.array(pred_image)
                    
                    # Create visualization - AI predictions are already in 8-class format, no conversion needed
                    pred_fig, pred_ax = plt.subplots(1, 1, figsize=(12, 8))
                    pred_ax.imshow(image_array)
                    # Apply colors directly to AI prediction (already 8-class format)
                    pred_colorized = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                    for class_idx, color in BEAUTIFUL_COLOR_MAP.items():
                        if class_idx < 8:
                            pred_colorized[pred_mask == class_idx] = color
                    pred_ax.imshow(pred_colorized, alpha=0.8)
                    pred_ax.set_title('AI Model Prediction (Container API)')
                    pred_ax.axis('off')
                    
                    predicted_base64 = convert_to_base64(pred_fig)
                    
                    result = {
                        "status": "success",
                        "message": "Container API prediction successful!",
                        "image_name": image_name,
                        "container_api_url": container_url,
                        "model_info": {
                            "prediction_shape": container_result.get("shape"),
                            "classes_found": container_result.get("classes")
                        },
                        "prediction_visualization": predicted_base64
                    }
                else:
                    result = {
                        "status": "error",
                        "message": "Container API didn't return prediction",
                        "container_response": container_result
                    }
                
            else:
                result = {
                    "status": "error",
                    "message": f"Container API returned status {response.status_code}",
                    "response_text": response.text
                }
                
        except requests.exceptions.Timeout:
            result = {
                "status": "error",
                "message": "Container API request timed out (model may still be loading)"
            }
        except Exception as api_error:
            result = {
                "status": "error", 
                "message": f"Container API call failed: {str(api_error)}"
            }
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )
        
    except Exception as e:
        logging.error(f"Container prediction error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type"
            }
        )