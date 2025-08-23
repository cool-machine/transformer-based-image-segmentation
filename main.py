import warnings
import os
from pathlib import Path
import glob
import tempfile
import json
import numpy as np
from typing import Optional

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import tensorflow as tf
from transformers import TFSegformerForSemanticSegmentation, SegformerConfig
from azure.storage.blob import BlobServiceClient

# Import our modules
from utils.image_processing import process_image, process_mask
from utils.visualization import colorize_mask, save_visualization
from models.unet_model import create_unet_with_vgg16_encoder

app = FastAPI(title="Image Segmentation System", version="2.0.0")

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for models
unet_model = None
segformer_model = None

def download_model_from_azure():
    """Download trained SegFormer model from Azure Storage"""
    try:
        # Get storage connection from environment
        storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not storage_connection_string:
            print("⚠️ AZURE_STORAGE_CONNECTION_STRING not set, using default pre-trained model")
            return None
        
        blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
        container_client = blob_service_client.get_container_client("models")
        
        # Create temp directory for model files
        model_dir = tempfile.mkdtemp()
        print(f"📁 Created temp model directory: {model_dir}")
        
        # Download all blobs in models container
        blob_list = container_client.list_blobs()
        downloaded_files = []
        
        for blob in blob_list:
            if "segformer" in blob.name.lower():  # Only download segformer files
                blob_path = os.path.join(model_dir, blob.name)
                
                # Create subdirectories if needed
                os.makedirs(os.path.dirname(blob_path), exist_ok=True)
                
                # Download blob
                blob_client = container_client.get_blob_client(blob.name)
                with open(blob_path, "wb") as download_file:
                    download_file.write(blob_client.download_blob().readall())
                
                downloaded_files.append(blob_path)
                print(f"📥 Downloaded: {blob.name}")
        
        print(f"✅ Downloaded {len(downloaded_files)} model files")
        return model_dir
        
    except Exception as e:
        print(f"❌ Error downloading model from Azure: {e}")
        return None

def load_segformer_model(model_dir: Optional[str] = None):
    """Load SegFormer model with fine-tuned weights from Azure Storage"""
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
        model = TFSegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Load fine-tuned weights if available
        if model_dir:
            model_files = list(Path(model_dir).rglob("*.h5"))
            model_files.extend(list(Path(model_dir).rglob("*.hdf5")))
            model_files.extend(list(Path(model_dir).rglob("*.keras")))
            
            if model_files:
                weights_file = str(model_files[0])
                print(f"🔄 Loading fine-tuned weights: {weights_file}")
                try:
                    model.load_weights(weights_file)
                    print("✅ Fine-tuned weights loaded successfully!")
                except Exception as e:
                    print(f"⚠️ Could not load weights: {e}, using base model")
            else:
                print("🔄 No weight files found, using base pre-trained model")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading SegFormer model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global unet_model, segformer_model
    
    print("🚀 Starting Model Container...")
    
    # Download model from Azure Storage
    model_dir = download_model_from_azure()
    
    # Load SegFormer model
    print("🧠 Loading SegFormer model...")
    segformer_model = load_segformer_model(model_dir)
    
    if segformer_model:
        print("✅ SegFormer model loaded successfully!")
    else:
        print("❌ Failed to load SegFormer model")
    
    # Load UNet model (optional fallback)
    try:
        unet_model = create_unet_with_vgg16_encoder()
        # Try to load weights if they exist
        model_path = "models/final_model.keras"
        if os.path.exists(model_path):
            unet_model.load_weights(model_path)
            print("UNet model loaded successfully")
        else:
            print("No UNet weights found, using randomly initialized model")
    except Exception as e:
        print(f"Error loading UNet model: {e}")
        unet_model = None

def get_available_images():
    """Get all available images from the static directory."""
    images_info = []
    cities = ["frankfurt", "lindau", "munster"]
    
    for city in cities:
        image_dir = Path(f"static/images/{city}")
        if image_dir.exists():
            # Find all image files (support multiple formats)
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                image_files.extend(glob.glob(str(image_dir / ext)))
            
            for image_path in image_files:
                filename = os.path.basename(image_path)
                if filename.startswith('img'):  # Only include original images, not masks
                    display_name = filename.replace('img_', '').replace('.png', '').replace('_leftImg8bit', '')
                    images_info.append({
                        "city": city,
                        "filename": filename,
                        "display_name": display_name,
                        "full_path": f"static/images/{city}/{filename}"
                    })
    
    return images_info

def get_all_image_names():
    """Get all possible image names for dropdown (including non-displayed ones)."""
    images_info = get_available_images()
    
    # Add some additional image names that might not be displayed but could exist
    all_names = []
    for info in images_info:
        all_names.append(info['display_name'])
    
    # Add some common Cityscapes image patterns
    additional_names = [
        "aachen_000000_000019",
        "aachen_000001_000019", 
        "bochum_000000_000019",
        "bremen_000000_000019",
        "cologne_000000_000019",
        "darmstadt_000000_000019",
        "dusseldorf_000000_000019",
        "erfurt_000000_000019",
        "hamburg_000000_000019",
        "hanover_000000_000019",
        "jena_000000_000019",
        "krefeld_000000_000019",
        "monchengladbach_000000_000019",
        "strasbourg_000000_000019",
        "stuttgart_000000_000019",
        "tubingen_000000_000019",
        "ulm_000000_000019",
        "weimar_000000_000019",
        "zurich_000000_000019"
    ]
    
    all_names.extend(additional_names)
    return sorted(list(set(all_names)))

def predict_segmentation(image_array):
    """
    Generate segmentation prediction with proper channel handling
    """
    global segformer_model
    
    if segformer_model is None:
        raise HTTPException(status_code=503, detail="SegFormer model not loaded")
    
    try:
        # Store original dimensions
        original_height, original_width = image_array.shape[:2]
        print(f"🖼️ Input image shape: {image_array.shape}")
        
        # Preprocess for SegFormer (channels last: H, W, C)
        if len(image_array.shape) == 3 and image_array.shape[-1] == 3:
            # Resize to model input size
            image_resized = tf.image.resize(image_array, [512, 1024])
            image_normalized = tf.cast(image_resized, tf.float32) / 255.0
            image_batch = tf.expand_dims(image_normalized, 0)  # Add batch dimension
            
            print(f"🔄 Preprocessed shape: {image_batch.shape}")
        else:
            raise ValueError(f"Invalid image format: {image_array.shape}")
        
        # Model inference - try both channel orders
        try:
            outputs = segformer_model(image_batch)
        except Exception:
            print("🔄 Trying channels first format...")
            image_channels_first = tf.transpose(image_batch, [0, 3, 1, 2])
            outputs = segformer_model(image_channels_first)
        
        # Extract prediction logits
        prediction_logits = outputs.logits
        print(f"🧠 Model output shape: {prediction_logits.shape}")
        
        # Handle different output formats (channels first/last)
        if len(prediction_logits.shape) == 4 and prediction_logits.shape[1] == 8:
            # Convert from channels first to channels last
            prediction_logits = tf.transpose(prediction_logits, [0, 2, 3, 1])
            print(f"🔄 Converted to channels last: {prediction_logits.shape}")
        
        # Get class predictions
        predicted_mask = tf.argmax(prediction_logits, axis=-1)
        predicted_mask = tf.squeeze(predicted_mask).numpy().astype(np.uint8)
        
        # Resize back to original dimensions
        if predicted_mask.shape != (original_height, original_width):
            mask_tensor = tf.expand_dims(tf.cast(predicted_mask, tf.float32), -1)
            mask_resized = tf.image.resize(mask_tensor, [original_height, original_width], method='nearest')
            predicted_mask = tf.squeeze(mask_resized).numpy().astype(np.uint8)
        
        print(f"✅ Final prediction shape: {predicted_mask.shape}")
        print(f"✅ Unique classes: {np.unique(predicted_mask)}")
        
        return predicted_mask
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Prediction API endpoint
@app.post("/predict")
async def predict_endpoint(request: Request):
    """
    API endpoint for generating segmentation predictions
    Expected input: JSON with base64 encoded image
    """
    try:
        data = await request.json()
        
        # Decode base64 image
        import base64
        import io
        image_data = base64.b64decode(data["image"])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Generate prediction
        prediction_mask = predict_segmentation(image_array)
        
        # Convert prediction to base64 for response
        prediction_pil = Image.fromarray(prediction_mask)
        buffer = io.BytesIO()
        prediction_pil.save(buffer, format="PNG")
        prediction_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return {
            "status": "success",
            "prediction": f"data:image/png;base64,{prediction_b64}",
            "shape": prediction_mask.shape,
            "classes": np.unique(prediction_mask).tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def get_images(request: Request):
    """Main page with image selection interface."""
    images_info = get_available_images()
    all_image_names = get_all_image_names()
    
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "images_info": images_info,
        "all_image_names": all_image_names
    })

@app.post("/segment")
async def segment_image(request: Request, 
                       selected_image: str = Form(None),
                       image_dropdown: str = Form(None)):
    """Process segmentation for selected image."""
    
    # Determine which image was selected
    if selected_image:
        # Image selected from gallery
        parts = selected_image.split('/')
        city = parts[0]
        filename = parts[1]
        image_name = filename.replace('img_', '').replace('.png', '').replace('_leftImg8bit', '')
    else:
        # Image selected from dropdown
        image_name = image_dropdown
        # Try to find the image in available cities
        city = None
        filename = None
        
        for search_city in ["frankfurt", "lindau", "munster"]:
            potential_filename = f"img_{image_name}_leftImg8bit.png"
            potential_path = f"static/images/{search_city}/{potential_filename}"
            if os.path.exists(potential_path):
                city = search_city
                filename = potential_filename
                break
        
        if not city:
            # Image not found
            return templates.TemplateResponse("error.html", {
                "request": request,
                "error_message": f"Image '{image_name}' not found in any city directory."
            })
    
    try:
        # Construct paths
        image_path = f"static/images/{city}/{filename}"
        mask_filename = filename.replace("img", "mask")
        mask_path = f"static/images/{city}/{mask_filename}"
        
        # Check if files exist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Process the image
        processed_image = process_image(image_path)
        
        # Load and save original image for display
        original_image = Image.open(image_path)
        original_image.save("static/original_image.png")
        
        # Process ground truth mask if available
        ground_truth_available = os.path.exists(mask_path)
        if ground_truth_available:
            processed_mask = process_mask(mask_path)
            colorized_ground_truth = colorize_mask(processed_mask)
            save_visualization(colorized_ground_truth, "static/ground_truth_mask.png")
        
        # Make prediction using SegFormer (primary model)
        predicted_mask = None
        prediction_available = False
        
        if segformer_model:
            try:
                predicted_mask = predict_segformer(segformer_model, processed_image)
                colorized_prediction = colorize_mask(predicted_mask)
                save_visualization(colorized_prediction, "static/predicted_mask.png")
                prediction_available = True
                model_used = "SegFormer"
            except Exception as e:
                print(f"SegFormer prediction failed: {e}")
        
        # Fallback to UNet if SegFormer fails
        if not prediction_available and unet_model:
            try:
                # UNet expects different input format
                unet_input = tf.transpose(processed_image, perm=[1, 2, 0])
                unet_input = tf.expand_dims(unet_input, axis=0)
                unet_prediction = unet_model(unet_input, training=False)
                predicted_mask = tf.argmax(unet_prediction, axis=-1)
                predicted_mask = tf.squeeze(predicted_mask)
                
                colorized_prediction = colorize_mask(predicted_mask)
                save_visualization(colorized_prediction, "static/predicted_mask.png")
                prediction_available = True
                model_used = "UNet"
            except Exception as e:
                print(f"UNet prediction failed: {e}")
        
        if not prediction_available:
            raise RuntimeError("No segmentation models available or all models failed")
        
        return templates.TemplateResponse("result.html", {
            "request": request,
            "image_name": image_name,
            "city": city,
            "model_used": model_used,
            "ground_truth_available": ground_truth_available,
            "prediction_available": prediction_available
        })
        
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": str(e)
        })

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_status = {
        "segformer_loaded": segformer_model is not None,
        "unet_loaded": unet_model is not None
    }
    
    return {
        "status": "healthy",
        "models": models_status,
        "available_images": len(get_available_images())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)