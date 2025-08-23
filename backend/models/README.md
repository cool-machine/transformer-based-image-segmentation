# Model Files

This directory should contain the trained model weights. Due to GitHub file size limitations, model files are not stored in this repository.

## Required Model Files

### SegFormer Model
- The application will automatically download the SegFormer model from HuggingFace
- Model: `nvidia/segformer-b0-finetuned-cityscapes-512-1024`
- No manual download required

### UNet Model (Optional)
Place your trained UNet model files here:
- `final_model.keras` - Main UNet model weights
- `my_model.hdf5` - Alternative model format

### Checkpoint Directory
Create subdirectories for model checkpoints:
- `checkpoints/` - Current model checkpoints
- `old_checkpoints/` - Previous model versions

## Model Storage Options

For production deployment, consider:
1. **Azure Blob Storage** - Store models in Azure and download during startup
2. **Git LFS** - Use Git Large File Storage for version control
3. **Docker Image** - Include models in Docker build process
4. **Model Registry** - Use MLflow or similar for model management

## Usage in Application

The application will:
1. First try to load SegFormer from HuggingFace (requires internet)
2. Fall back to local UNet model if available
3. Display appropriate error messages if no models are found