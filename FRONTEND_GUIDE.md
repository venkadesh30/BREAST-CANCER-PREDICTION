# 🎗️ Breast Cancer Detection System - Frontend User Guide

## Overview
This project includes **two modern frontend options** for breast cancer histopathology image analysis:

1. **Streamlit App** - Interactive web interface with tabs and advanced features
2. **Flask Web App** - Full-featured professional web application with REST API

Choose the one that best fits your needs!

---

## Prerequisites

Make sure you have:
- ✅ Trained model at `models/best_model.pth`
- ✅ All dependencies installed: `pip install -r requirements.txt`
- ✅ Python 3.7+ installed

---

## Option 1: Streamlit Application 🎯

### Why Streamlit?
- **Fastest to run** - No additional setup needed
- **Interactive** - Real-time updates and responsive UI
- **Feature-rich** - Multiple tabs, batch processing, history tracking
- **Professional** - Medical-grade disclaimers and guidance

### Running the Streamlit App

```bash
# From the project root directory
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Features

#### 📋 Menu Navigation
- **Diagnosis** - Main analysis interface
- **Model Info** - Technical specifications and architecture
- **About** - Usage guidelines and disclaimer

#### 🔬 Diagnosis Page

**Single Image Analysis:**
1. Navigate to "📤 Single Image" tab
2. Upload a histopathology image (PNG, JPG, JPEG)
3. Click "Analyze Image" button
4. View results with:
   - Color-coded prediction (Benign/Malignant)
   - Confidence percentage
   - Risk level assessment
   - Probability distribution chart
   - Option to download analysis report

**Batch Analysis:**
1. Navigate to "📁 Batch Analysis" tab
2. Upload multiple images at once
3. View summary statistics
4. See individual results in table format
5. Download batch report

#### ⚙️ Settings
- Adjust confidence threshold for alerts
- Visual warning if confidence falls below threshold

#### 📊 Model Information
- Architecture details (ResNet50)
- Performance specifications
- Training methodology explanation
- Transfer Learning benefits

#### 📚 About Section
- How the system works
- Important medical disclaimers
- Best practices and recommendations
- Technical background
- Prediction history tracking

### Example Usage

```python
# The app handles everything automatically:
# 1. Load and preprocess the image
# 2. Run inference on GPU/CPU
# 3. Display results with confidence
# 4. Provide risk assessment
# 5. Generate downloadable reports
```

---

## Option 2: Flask Web Application 🌐

### Why Flask?
- **Professional** - Full REST API backend
- **Scalable** - Can be deployed to production servers
- **Customizable** - Easily extend with additional features
- **Responsive** - Modern HTML/CSS/JavaScript frontend

### Running the Flask App

```bash
# From the project root directory
python web_app.py
```

The app will start at `http://localhost:5000`

### Features

#### 🎨 Modern UI
- Responsive design (works on desktop/mobile)
- Beautiful gradient styling
- Smooth animations
- Professional color scheme

#### 📤 Single Image Analysis
1. Upload an image via drag-and-drop or file browser
2. View live preview
3. See analysis results with:
   - Classification (Benign/Malignant)
   - Confidence score with visual bar
   - Probability distribution
   - Risk assessment

#### 📁 Batch Analysis
1. Upload multiple images
2. Process them automatically
3. View summary statistics
4. See detailed results for each image

#### 📊 Statistics Dashboard
- Total predictions count
- Malignant vs Benign breakdown
- Percentage distribution
- Recent predictions history
- One-click history clearing

#### 🔧 REST API Endpoints

If you want to integrate this with other applications:

```
POST /api/predict
- Upload a single image
- Returns: Prediction, confidence, probabilities, risk level

POST /api/batch-predict
- Upload multiple images
- Returns: Array of predictions with statistics

GET /api/history
- Retrieve prediction history
- Returns: Last 20 predictions

GET /api /stats
- Get overall statistics
- Returns: Total count, malignant count, benign count

POST /api/clear-history
- Clear prediction history

GET /api/model-info
- Get model specifications
- Returns: Model architecture details
```

### Example API Usage

```bash
# Using curl to analyze a single image
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict

# Response:
{
  "filename": "image.jpg",
  "prediction": "Malignant",
  "confidence": "92.45%",
  "benign_prob": "7.55%",
  "malignant_prob": "92.45%",
  "risk_level": "🔴 HIGH RISK",
  "timestamp": "2024-02-10T15:30:45.123456"
}
```

---

## Quick Comparison

| Feature | Streamlit | Flask |
|---------|-----------|-------|
| **Setup Time** | Instant | ~1 min |
| **Performance** | Fast | Very Fast |
| **Scalability** | Limited | Excellent |
| **Customization** | Limited | Extensive |
| **Mobile Support** | Partial | Full Responsive |
| **API Integration** | No | Yes (REST) |
| **Learning Curve** | Easiest | Moderate |
| **Production Ready** | Limited | Yes |

---

## Understanding Results

### Prediction Output

**BENIGN (Class 0)**
- No cancer cells detected
- Green color indicator
- Low risk classification

**MALIGNANT (Class 1)**
- Cancer cells detected
- Red color indicator  
- High risk classification

### Confidence Levels

| Confidence | Interpretation |
|-----------|-----------------|
| > 80% | High confidence - Clear classification |
| 60-80% | Moderate confidence - Generally reliable |
| < 60% | Low confidence - Consider expert review |

### Risk Assessment Categories

- 🟢 **LOW RISK**: Benign with >80% confidence
- 🟡 **MODERATE CONFIDENCE**: 60-80% confidence
- 🟠 **MEDIUM RISK**: Malignant with moderate confidence
- 🔴 **HIGH RISK**: Malignant with >80% confidence

---

## Image Requirements

### Optimal Images for Best Results
- ✅ **Format**: PNG, JPG, or JPEG
- ✅ **Size**: 224x224 pixels (auto-resized)
- ✅ **Quality**: High resolution histopathology slides
- ✅ **Staining**: Standard H&E (Hematoxylin & Eosin) stained slides

### Tips for Better Accuracy
1. **Ensure good lighting** in the slide image
2. **Use clear, focused** histopathology images
3. **Avoid artifacts** or slide damage
4. **Check for proper staining** (standard H&E preferred)
5. **Use appropriate magnification** for the tissue type

---

## Reports and Export

### Streamlit App Reports
- Download as `.txt` file
- Includes timestamp, filename, prediction, confidence
- Contains risk assessment and clinical notes
- Available after each analysis

### Flask App (Via API)
- JSON response format
- Structured data for integration
- Batch analysis summary available

---

## Important Medical Disclaimers ⚠️

### NOT FOR CLINICAL USE
This system is designed for:
- ✅ Educational purposes
- ✅ Research demonstrations
- ✅ AI reference implementation
- ❌ Direct clinical diagnosis
- ❌ Treatment decisions
- ❌ Standalone diagnosis

### Critical Points
1. **Always consult qualified pathologists** for clinical diagnosis
2. **Never replace professional judgment** with AI results
3. **Consider confidence scores** - low confidence needs expert review
4. **Maintain patient privacy** when uploading real patient data
5. **Keep records** of all analyses performed
6. **Understand limitations** - AI can make errors

---

## Troubleshooting

### Model Loading Error
**Problem**: "Model file not found"
```
Solution: Train the model first using src/train.py
python src/train.py
```

### Out of Memory Error
**Problem**: CUDA out of memory (GPU)
```
Solution: Use CPU instead
Edit config.py and change: DEVICE = torch.device('cpu')
```

### Port Already in Use
**Problem**: "Address already in use"
```
# Stream lit (change port)
streamlit run app.py --server.port=8502

# Flask (change port)
python web_app.py --port=5001
```

### Slow Performance
**Problem**: Analysis is taking too long
```
Solutions:
1. Use a GPU (CUDA-enabled graphics card)
2. Use smaller batch sizes
3. Close other applications
4. Check image resolution isn't too large
```

---

## Advanced Usage

### Custom Model Integration
To use with a different trained model:

**Streamlit:**
```python
# Edit app.py, line ~50
model_path = 'path/to/your/model.pth'
```

**Flask:**
```python
# Edit web_app.py, line ~60
model_path = 'path/to/your/model.pth'
```

### Batch Processing Script
```python
# For automated batch processing without GUI:
from inference import load_model, predict
import os

model = load_model()
image_dir = 'path/to/images'

for image_file in os.listdir(image_dir):
    prediction, confidence = predict(model, f'{image_dir}/{image_file}')
    print(f"{image_file}: {prediction} ({confidence:.2%})")
```

---

## Performance Metrics

### Typical Analysis Times
- **Single Image**: 0.5-2 seconds (GPU), 3-5 seconds (CPU)
- **Batch (10 images)**: 2-5 seconds (GPU), 15-30 seconds (CPU)
- **Model Loading**: ~2 seconds (both)

### Hardware Requirements
| Hardware | Speed | Memory |
|----------|-------|--------|
| GPU (NVIDIA) | ⚡⚡⚡ Fast | ~2GB VRAM |
| CPU (Modern) | ⚡ Moderate | ~1GB RAM |
| CPU (Older) | 🐌 Slow | ~500MB RAM |

---

## Support & Contribution

### Need Help?
1. Check the troubleshooting section above
2. Review the project README.md
3. Check the TROUBLESHOOTING.md file
4. Review source code comments in `src/` folder

### Want to Improve?
1. The frontend is fully modular and extensible
2. Both Streamlit and Flask apps use the same model backend
3. Easy to add new features or modify existing ones

---

## Next Steps

1. **Try the Streamlit app** - Best for quick testing:
   ```bash
   streamlit run app.py
   ```

2. **Or run the Flask app** - Best for production:
   ```bash
   python web_app.py
   ```

3. **Upload test images** from the `data/` folder

4. **Review results** and understand the confidence scores

5. **Explore model information** to understand how transfer learning works

---

## Version Information
- **Framework**: PyTorch
- **Web Frameworks**: Streamlit & Flask
- **Model**: ResNet50 with Transfer Learning
- **Python**: 3.7+
- **Last Updated**: 2024

---

**Happy Analyzing! 🎉**

*Remember: This tool is for educational and research purposes only. Always consult qualified medical professionals for actual diagnosis.*
