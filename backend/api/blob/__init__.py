import logging
import azure.functions as func
import os
from azure.storage.blob import BlobServiceClient

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Serve image files from Azure Blob Storage using blob paths."""
    
    blob_path = req.route_params.get('blob_path')
    
    logging.info(f'Blob requested: {blob_path}')
    
    try:
        # Get storage connection string
        storage_connection_string = os.getenv('IMAGES_STORAGE_CONNECTION_STRING')
        
        if not storage_connection_string:
            logging.error("No storage connection string found")
            return func.HttpResponse(
                "Storage not configured",
                status_code=500,
                headers={
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
        # Azure Storage - read from blob storage
        try:
            blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)
            container_name = "images1"
            
            # Get blob client and download
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
            
            if not blob_client.exists():
                return func.HttpResponse(
                    "Blob not found",
                    status_code=404,
                    headers={
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            
            # Download blob data
            blob_data = blob_client.download_blob().readall()
            
            # Determine content type based on file extension
            content_type = "image/png"
            if blob_path.lower().endswith(('.jpg', '.jpeg')):
                content_type = "image/jpeg"
            elif blob_path.lower().endswith('.gif'):
                content_type = "image/gif"
            elif blob_path.lower().endswith('.webp'):
                content_type = "image/webp"
            
            return func.HttpResponse(
                blob_data,
                status_code=200,
                headers={
                    "Content-Type": content_type,
                    "Access-Control-Allow-Origin": "*",
                    "Cache-Control": "public, max-age=3600"
                }
            )
            
        except Exception as e:
            logging.error(f"Error reading from Azure Storage: {e}")
            return func.HttpResponse(
                "Error loading blob from storage",
                status_code=500,
                headers={
                    "Access-Control-Allow-Origin": "*"
                }
            )
        
    except Exception as e:
        logging.error(f'Error serving blob {blob_path}: {str(e)}')
        
        return func.HttpResponse(
            "Error loading blob",
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*"
            }
        )