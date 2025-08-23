import logging
import azure.functions as func
import json
import os
from azure.storage.blob import BlobServiceClient
import re

def get_available_images():
    """Get all available images from Azure Storage."""
    images_info = []
    
    # Get storage connection string
    storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
    
    if storage_connection_string:
        # Azure Storage - read from blob storage
        try:
            blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
            container_name = "images1"
            
            # List all blobs to understand the structure
            container_client = blob_service_client.get_container_client(container_name)
            all_blobs = list(container_client.list_blobs())
            
            logging.info(f"Found {len(all_blobs)} total blobs in container")
            for blob in all_blobs[:10]:  # Log first 10 for debugging
                logging.info(f"Blob: {blob.name}")
            
            # List blobs that could be images (in any directory structure)
            blobs = [blob for blob in all_blobs if blob.name.endswith('_leftImg8bit.png')]
            
            for blob in blobs:
                blob_name = blob.name
                logging.info(f"Processing image blob: {blob_name}")
                
                # Extract the actual filename (last part of path)
                filename = blob_name.split('/')[-1]
                
                # Extract display name: lindau_000000_000019
                display_name = filename.replace('_leftImg8bit.png', '')
                
                # Extract city from filename
                city_match = re.match(r'^([a-z]+)_', display_name)
                city = city_match.group(1) if city_match else 'unknown'
                
                images_info.append({
                    "city": city,
                    "filename": filename,
                    "display_name": display_name,
                    "blob_path": blob_name,
                    "full_path": f"api/image/blob/{blob_name}"
                })
                    
            logging.info(f"Found {len(images_info)} images in Azure Storage")
            
        except Exception as e:
            logging.error(f"Error reading from Azure Storage: {e}")
            # Fallback to sample data
            images_info = get_sample_images()
    else:
        # No storage connection - use sample data
        logging.warning("No storage connection string found, using sample data")
        images_info = get_sample_images()
    
    return images_info

def get_sample_images():
    """Get sample images when Azure Storage is not available."""
    samples = [
        {
            "city": "lindau",
            "filename": "lindau_000000_000019_leftImg8bit.png",
            "display_name": "lindau_000000_000019",
            "blob_path": "images/lindau_000000_000019_leftImg8bit.png",
            "full_path": "api/image/blob/images/lindau_000000_000019_leftImg8bit.png"
        },
        {
            "city": "lindau", 
            "filename": "lindau_000001_000019_leftImg8bit.png",
            "display_name": "lindau_000001_000019",
            "blob_path": "images/lindau_000001_000019_leftImg8bit.png", 
            "full_path": "api/image/blob/images/lindau_000001_000019_leftImg8bit.png"
        }
    ]
    
    return samples

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Get all available image names for dropdown."""
    
    logging.info('Image names requested')
    
    try:
        # Get available images using the same logic as images API
        images_info = get_available_images()
        
        # Extract display names
        image_names = [info['display_name'] for info in images_info]
        
        # Add common Cityscapes image patterns that might exist
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
        
        # Combine and sort unique names
        all_names = sorted(list(set(image_names + additional_names)))
        
        return func.HttpResponse(
            json.dumps(all_names, indent=2),
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logging.error(f'Error getting image names: {str(e)}')
        
        # Return empty list on error
        return func.HttpResponse(
            json.dumps([]),
            status_code=500,
            headers={
                "Content-Type": "application/json", 
                "Access-Control-Allow-Origin": "*"
            }
        )