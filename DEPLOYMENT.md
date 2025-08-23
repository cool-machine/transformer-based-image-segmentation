# 🚀 Deployment Guide

## Quick Start

### Frontend Deployment (GitHub Pages)
1. Fork this repository
2. Enable GitHub Pages in repository settings
3. The site will be automatically deployed to `https://yourusername.github.io/image-segmentation/`

### Backend Deployment (Azure Functions)

#### Prerequisites
- Azure CLI installed and logged in
- Azure Functions Core Tools v4+
- Python 3.12+

#### Steps
1. **Create Azure Resources**
   ```bash
   # Create resource group
   az group create --name image-segmentation-rg --location "Central US"
   
   # Create storage account
   az storage account create --name yourstorageaccount --resource-group image-segmentation-rg --location "Central US" --sku Standard_LRS
   
   # Create function app
   az functionapp create --resource-group image-segmentation-rg --consumption-plan-location "Central US" --runtime python --runtime-version 3.12 --functions-version 4 --name your-function-app --storage-account yourstorageaccount
   ```

2. **Configure Local Settings**
   ```bash
   cd backend
   cp local.settings.json.template local.settings.json
   # Edit local.settings.json with your Azure connection strings
   ```

3. **Deploy Functions**
   ```bash
   func azure functionapp publish your-function-app
   ```

## Environment Variables

### Frontend
No environment variables required - uses GitHub Pages static hosting.

### Backend (Azure Functions)
- `AZURE_STORAGE_CONNECTION_STRING`: Azure Storage connection string for image data
- `CONTAINER_ENDPOINT`: ML model container endpoint (optional)

## Architecture

- **Frontend**: Static web app hosted on GitHub Pages
- **Backend**: Azure Functions (Python) for API endpoints
- **Storage**: Azure Blob Storage for images and masks
- **ML Models**: Containerized SegFormer and UNet models

## Development

### Local Development
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run locally: `func start` (in backend directory)
4. Open `index.html` in browser

### Testing
- Frontend health check: Visit your deployed site
- Backend health check: `GET /api/health`
- Full pipeline test: Upload an image and generate segmentation

## Monitoring

- Azure Application Insights integration
- GitHub Actions for automated deployment
- Health check endpoints for system monitoring