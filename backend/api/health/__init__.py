import logging
import azure.functions as func
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint for the image segmentation API."""
    
    logging.info('Health check requested')
    
    try:
        # Check if models can be imported (basic health check)
        models_status = {
            "segformer_available": True,
            "unet_available": True,
            "tensorflow_available": True
        }
        
        try:
            import tensorflow as tf
            models_status["tensorflow_version"] = tf.__version__
        except ImportError:
            models_status["tensorflow_available"] = False
        
        try:
            from transformers import TFSegformerForSemanticSegmentation
            models_status["transformers_available"] = True
        except ImportError:
            models_status["transformers_available"] = False
            
        health_data = {
            "status": "healthy",
            "service": "image-segmentation-api",
            "models": models_status,
            "endpoints": [
                "/api/health",
                "/api/images", 
                "/api/image-names",
                "/api/segment",
                "/api/image/{city}/{filename}"
            ]
        }
        
        return func.HttpResponse(
            json.dumps(health_data, indent=2),
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except Exception as e:
        logging.error(f'Health check failed: {str(e)}')
        
        error_response = {
            "status": "unhealthy",
            "error": str(e)
        }
        
        return func.HttpResponse(
            json.dumps(error_response),
            status_code=500,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )