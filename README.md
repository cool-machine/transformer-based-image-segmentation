# 🔍 Image Segmentation System v2.0

[![Deploy Functions](https://github.com/cool-machine/image-segmentation/actions/workflows/deploy-functions.yml/badge.svg)](https://github.com/cool-machine/image-segmentation/actions/workflows/deploy-functions.yml)
[![Deploy to GitHub Pages](https://github.com/cool-machine/image-segmentation/actions/workflows/deploy.yml/badge.svg)](https://github.com/cool-machine/image-segmentation/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌐 Live Demo

**GitHub Pages**: [https://cool-machine.github.io/image-segmentation/](https://cool-machine.github.io/image-segmentation/)

### Available Pages:
- **Main Application**: `/` - Full-featured image segmentation interface
- **Standalone Test**: `/standalone.html` - Self-contained diagnostic page  
- **Simple Test**: `/test.html` - Minimal API functionality test

## 🎯 **What is this?**

A **production-ready image segmentation system** that performs semantic segmentation on urban scenes using state-of-the-art deep learning models. Built with modern **Static Web App + Azure Functions** architecture for optimal cost and performance.

### ✨ **Key Features**

🤖 **Dual Model Support**: SegFormer B0 (primary) + UNet with VGG16 (fallback)  
🖼️ **8-Class Cityscapes Segmentation**: Roads, vehicles, buildings, sky, etc.  
🌐 **Modern Web Interface**: Responsive design with dropdown + gallery selection  
☁️ **Cloud-Native Architecture**: Azure Static Web Apps + Functions  
💰 **Cost-Optimized**: ~$5-15/month vs $20-40 for container solutions  
🚀 **Auto-Deployment**: GitHub Actions CI/CD pipeline  

---

## 🏗️ **Architecture**

```
┌─────────────────────┐    ┌─────────────────────┐
│  Static Web App     │    │  Azure Functions    │
│  (Frontend)         │───▶│  (API Backend)      │
├─────────────────────┤    ├─────────────────────┤
│ • HTML/CSS/JS       │    │ • /api/health       │
│ • Image Gallery     │    │ • /api/images       │
│ • Dropdown Menu     │    │ • /api/segment      │
│ • Results Display   │    │ • /api/image-names  │
└─────────────────────┘    └─────────────────────┘
         │                            │
         └────────────────────────────┘
              HTTPS + CORS Enabled
```

📊 **Detailed Architecture Diagrams**: 
- [Complete System Architecture](docs/architecture-diagram.md) - Technical flow diagrams with Mermaid
- [Visual Flow Diagrams](docs/visual-flow-diagram.md) - User journey and data processing flows

### **Technology Stack**
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **Backend**: Python 3.11, Azure Functions
- **ML Models**: TensorFlow, HuggingFace Transformers
- **Infrastructure**: Azure Static Web Apps, Azure Functions
- **CI/CD**: GitHub Actions

---

## 🚀 **Quick Start**

### **Live Demo**
- **Frontend**: https://kind-beach-0371f0d10.2.azurestaticapps.net
- **API**: https://ocp8.azurewebsites.net/api/health

### **Local Development**

#### **Frontend (Static Web App)**
```bash
# Serve frontend locally
cd frontend/public
python -m http.server 8080
# Access: http://localhost:8080
```

#### **Backend (Azure Functions)**
```bash
# Install Azure Functions Core Tools
npm install -g azure-functions-core-tools@4

# Install dependencies
cd backend
pip install -r requirements.txt

# Run Functions locally
func start --port 7071
# Access: http://localhost:7071/api/health
```

---

## 🖼️ **Usage**

### **Web Interface**

1. **Select Image**: Choose from dropdown menu or click gallery preview
2. **Process**: System automatically segments the image
3. **View Results**: See original, predicted mask, and ground truth side-by-side

### **Segmentation Classes**

| Class | Description | Color |
|-------|-------------|-------|
| **Flat** | Roads, sidewalks, parking | Black |
| **Human** | People, riders | Maroon |
| **Vehicle** | Cars, trucks, buses, bikes | Green |
| **Construction** | Buildings, walls, fences | Olive |
| **Object** | Poles, signs, traffic lights | Navy |
| **Nature** | Vegetation, terrain | Purple |
| **Sky** | Sky regions | Teal |
| **Void** | Ground, dynamic objects | Gray |

### **API Endpoints**

```bash
# Health check
GET https://ocp8.azurewebsites.net/api/health

# Get available images
GET https://ocp8.azurewebsites.net/api/images

# Get image names for dropdown
GET https://ocp8.azurewebsites.net/api/image-names

# Segment image
POST https://ocp8.azurewebsites.net/api/segment
Content-Type: application/json
{
  "image_name": "frankfurt_000000_000019",
  "city": "frankfurt"
}

# Get image file
GET https://ocp8.azurewebsites.net/api/image/frankfurt/img_frankfurt_000000_000019_leftImg8bit.png
```

---

## 🛠️ **Development**

### **Project Structure**

```
image-segmentation/
├── 📁 frontend/                    # Static Web App
│   ├── 📁 public/
│   │   └── 📄 index.html          # Main web interface
│   ├── 📁 src/
│   │   └── 📄 app.js              # Frontend JavaScript logic
│   └── 📄 staticwebapp.config.json # Static Web App configuration
├── 📁 backend/                     # Azure Functions API
│   ├── 📁 api/
│   │   ├── 📁 health/             # Health check endpoint
│   │   ├── 📁 images/             # Images list endpoint
│   │   ├── 📁 image-names/        # Dropdown options endpoint
│   │   ├── 📁 segment/            # Image processing endpoint
│   │   └── 📁 image/              # Image serving endpoint
│   ├── 📁 models/                 # ML model definitions
│   │   ├── 📄 segformer_model.py  # SegFormer B0 implementation
│   │   └── 📄 unet_model.py       # UNet + VGG16 implementation
│   ├── 📁 utils/                  # Processing utilities
│   │   ├── 📄 image_processing.py # Image preprocessing
│   │   └── 📄 visualization.py    # Result visualization
│   ├── 📁 src/                    # Core processing logic
│   │   ├── 📄 image_preprocessing.py
│   │   ├── 📄 mask_prediction.py
│   │   └── 📄 predict.py
│   ├── 📄 host.json               # Azure Functions configuration
│   └── 📄 requirements.txt        # Python dependencies
├── 📁 .github/workflows/           # CI/CD pipelines
│   ├── 📄 deploy-functions.yml     # Backend deployment
│   └── 📄 deploy-static-web-app.yml # Frontend deployment
├── 📄 README.md                   # This file
└── 📄 .gitignore                  # Git ignore rules
```

### **Adding New Models**

1. **Create Model File**: Add to `backend/models/your_model.py`
2. **Implement Interface**:
   ```python
   def create_your_model(num_classes=8):
       # Model creation logic
       return model
   
   def predict_your_model(model, processed_image):
       # Prediction logic
       return predicted_mask
   ```
3. **Update Segment Function**: Modify `backend/api/segment/__init__.py`
4. **Test Locally**: Use Azure Functions Core Tools
5. **Deploy**: Push to GitHub (auto-deployment via Actions)

### **Adding New Images**

1. **Local Development**: Place in `static/images/{city}/`
2. **Azure Deployment**: Upload to Azure Blob Storage
3. **Update API**: Modify `backend/api/images/__init__.py` if needed

---

## ☁️ **Deployment**

### **Automated Deployment (Recommended)**

Push to GitHub `master` branch to trigger automatic deployments:

```bash
git add .
git commit -m "Your changes"
git push origin master
```

### **Manual Deployment**

#### **Frontend (Static Web App)**
```bash
# Using Azure CLI
az staticwebapp create \
  --name "image-segmentation-frontend" \
  --resource-group "your-rg" \
  --source "https://github.com/your-username/image-segmentation" \
  --branch "master" \
  --app-location "/frontend/public"
```

#### **Backend (Azure Functions)**
```bash
# Using Azure Functions Core Tools
cd backend
func azure functionapp publish your-function-app-name
```

### **Environment Variables**

Set these in Azure portal or local.settings.json:

```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "DefaultEndpointsProtocol=https;...",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "FUNCTIONS_EXTENSION_VERSION": "~4",
    "STATIC_FILES_PATH": "./static/images",
    "AZURE_STORAGE_CONNECTION_STRING": "your-storage-connection"
  }
}
```

---

## 🧪 **Testing**

### **API Testing**

```bash
# Health check
curl https://ocp8.azurewebsites.net/api/health

# Segment image
curl -X POST https://ocp8.azurewebsites.net/api/segment \
  -H "Content-Type: application/json" \
  -d '{"image_name": "frankfurt_000000_000019"}'
```

### **Frontend Testing**

1. Open browser to Static Web App URL
2. Select image from dropdown or gallery
3. Verify results display correctly
4. Check browser console for errors

### **Load Testing**

```bash
# Install artillery
npm install -g artillery

# Create test config
echo 'config:
  target: "https://ocp8.azurewebsites.net"
scenarios:
  - name: "Health check"
    requests:
      - get:
          url: "/api/health"' > loadtest.yml

# Run test
artillery run loadtest.yml
```

---

## 📊 **Performance & Monitoring**

### **Expected Performance**
- **Cold Start**: 10-30 seconds (first request)
- **Warm Requests**: 2-5 seconds per image
- **Throughput**: ~10-20 concurrent requests
- **Model Loading**: ~5-15 seconds on startup

### **Monitoring**

1. **Azure Application Insights**: Automatic performance monitoring
2. **Function Logs**: View in Azure portal Functions > Monitor
3. **Static Web App Analytics**: Built-in usage statistics

### **Cost Monitoring**

- **Static Web App**: FREE tier (100GB bandwidth)
- **Azure Functions**: ~$0.20 per million executions
- **Typical Monthly Cost**: $5-15 for moderate usage

---

## 🔧 **Troubleshooting**

### **Common Issues**

#### **Frontend Not Loading**
```bash
# Check if Static Web App is deployed
az staticwebapp show --name "image-segmentation-frontend"

# Verify CORS settings in Functions
curl -I https://ocp8.azurewebsites.net/api/health
```

#### **API Errors**
```bash
# Check Function logs
az functionapp logs tail --name "ocp8" --resource-group "ocp8"

# Test individual endpoints
curl https://ocp8.azurewebsites.net/api/health
```

#### **Model Loading Failures**
1. **Check Dependencies**: Verify `requirements.txt` includes all ML libraries
2. **Memory Limits**: Upgrade to Premium Functions plan if needed
3. **Timeout Issues**: Increase `functionTimeout` in `host.json`

#### **Image Not Found Errors**
1. **Local Development**: Ensure images are in `static/images/{city}/`
2. **Azure Deployment**: Upload images to Azure Blob Storage
3. **File Naming**: Follow Cityscapes convention: `img_{name}_leftImg8bit.png`

### **Debug Mode**

Enable detailed logging in `backend/host.json`:

```json
{
  "logging": {
    "logLevel": {
      "default": "Information"
    }
  }
}
```

---

## 🤝 **Contributing**

### **Development Workflow**

1. **Fork** the repository
2. **Create** feature branch: `git checkout -b feature/amazing-feature`
3. **Make** changes and test locally
4. **Commit** changes: `git commit -m "Add amazing feature"`
5. **Push** to branch: `git push origin feature/amazing-feature`
6. **Create** Pull Request

### **Code Style**

- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use ES6+ features, semicolons optional
- **HTML/CSS**: Semantic HTML5, mobile-first responsive design

### **Testing Requirements**

- Test all API endpoints locally before submitting
- Verify frontend works across browsers (Chrome, Firefox, Safari)
- Include error handling for edge cases

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Cityscapes Dataset**: Urban scene segmentation benchmark
- **HuggingFace**: Pre-trained SegFormer models
- **TensorFlow**: Deep learning framework
- **Azure**: Cloud infrastructure and deployment platform

---

## 📞 **Support**

### **Getting Help**

- 🐛 **Issues**: [GitHub Issues](https://github.com/cool-machine/image-segmentation/issues)
- 📖 **Documentation**: [Project Wiki](https://github.com/cool-machine/image-segmentation/wiki)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/cool-machine/image-segmentation/discussions)

### **Contact**

For questions or support, please create an issue on GitHub with:
- System details (OS, browser, etc.)
- Error messages or logs
- Steps to reproduce the issue
- Expected vs actual behavior

---

## 🔄 **Version History**

### **v2.0** - Current
- ✅ Static Web App + Azure Functions architecture
- ✅ Dual model support (SegFormer + UNet)
- ✅ Modern responsive web interface
- ✅ Automated CI/CD deployment
- ✅ Cost-optimized cloud infrastructure

### **v1.0** - Legacy
- FastAPI monolithic application
- Single model support
- Manual deployment process

---

## 🎯 **Roadmap**

### **Short Term**
- [ ] Batch processing API endpoint
- [ ] Model performance benchmarks
- [ ] Extended image format support (WEBP, BMP)
- [ ] Real-time processing metrics

### **Long Term**
- [ ] Custom model training interface
- [ ] Multi-dataset support (ADE20K, PASCAL VOC)
- [ ] Video segmentation capabilities
- [ ] Mobile app version

---

**Happy Segmenting! 🖼️✨**