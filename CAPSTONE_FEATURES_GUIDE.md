# Breast Cancer Detection System - Capstone Features Guide

## 🎯 Overview

This enhanced breast cancer detection system now includes:
- **Explainable AI (XAI)**: Grad-CAM, SHAP, and Integrated Gradients visualizations
- **Clinical Decision Support**: Evidence-based medical recommendations
- **Automated Reporting**: Professional PDF report generation
- **Enhanced Inference Pipeline**: Comprehensive analysis workflow

---

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**New dependencies added:**
- `pytorch-grad-cam` - For Grad-CAM heatmap generation
- `shap` - For SHAP explanations
- `captum` - For Integrated Gradients
- `reportlab` - For PDF report generation
- `opencv-python` - For image processing

### 2. Verify Installation

```bash
python -c "import torch, shap, captum, reportlab; print('All dependencies installed!')"
```

---

## 🚀 Usage

### Option 1: Command-Line Inference

#### Standard Mode (Basic Prediction)
```bash
python inference.py data/0/sample_image.png
```

**Output:**
- Prediction (Benign/Malignant)
- Confidence score
- Basic interpretation

#### Comprehensive Mode (Full Analysis)
```bash
python inference.py data/0/sample_image.png --comprehensive
```

**Output:**
- Prediction with probabilities
- Grad-CAM, SHAP, and Integrated Gradients visualizations
- Clinical risk assessment
- Evidence-based recommendations
- Automated PDF report

#### With Patient ID
```bash
python inference.py data/0/sample_image.png --comprehensive --patient-id PATIENT001
```

#### Skip PDF Report
```bash
python inference.py data/0/sample_image.png --comprehensive --no-report
```

---

### Option 2: Streamlit Web Interface

#### Launch the App
```bash
streamlit run app.py
```

#### Features Available:
1. **Upload histopathology images** (PNG, JPG, JPEG)
2. **View prediction results** with confidence scores
3. **Explore Grad-CAM heatmaps** showing influential tissue regions
4. **Read clinical recommendations** with color-coded risk levels
5. **Generate and download PDF reports**
6. **Batch analysis** for multiple images
7. **Prediction history** tracking

---

## 📊 Understanding the Outputs

### 1. Explainability Visualizations

#### Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Location**: `outputs/explanations/gradcam/`
- **Purpose**: Shows which tissue regions influenced the AI's decision
- **Interpretation**:
  - 🔴 Red/Yellow areas = High importance
  - 🔵 Blue areas = Low importance
- **Medical Use**: Helps pathologists validate if the AI is focusing on clinically relevant features

#### SHAP (SHapley Additive exPlanations)
- **Location**: `outputs/explanations/shap/`
- **Purpose**: Pixel-level attribution showing positive/negative contributions
- **Interpretation**: Quantifies each pixel's contribution to the prediction
- **Medical Use**: Provides detailed feature importance analysis

#### Integrated Gradients
- **Location**: `outputs/explanations/integrated_gradients/`
- **Purpose**: Attribution method with theoretical guarantees
- **Interpretation**: Shows gradual attribution from baseline to input
- **Medical Use**: Offers mathematically rigorous explanation

---

### 2. Clinical Decision Support

#### Risk Levels
- **LOW**: High confidence benign classification → Routine monitoring
- **MEDIUM**: Moderate confidence or borderline cases → Closer monitoring
- **HIGH**: Moderate confidence malignant → Specialist review required
- **CRITICAL**: High confidence malignant → Immediate oncologist consultation

#### Recommendations
The system provides evidence-based recommendations based on:
- Prediction (Benign vs Malignant)
- Confidence score
- Medical best practices

**Example Recommendations:**
- **Benign (High Confidence)**: "Routine screening after 12 months"
- **Malignant (High Confidence)**: "⚠️ IMMEDIATE ONCOLOGIST CONSULTATION REQUIRED"
- **Malignant (Low Confidence)**: "⚠️ INCONCLUSIVE - Additional biopsy required"

---

### 3. PDF Medical Reports

#### Report Contents
1. **Header**: Report ID, timestamp, patient ID
2. **Diagnostic Results**: Prediction, confidence, probability distribution
3. **Image Analysis**: Original image and Grad-CAM heatmap
4. **Clinical Recommendations**: Risk level, follow-up timeline, specialist referral
5. **Disclaimer**: Important medical and regulatory notices
6. **Technical Details**: Model architecture, processing information

#### Report Location
- **Directory**: `outputs/reports/`
- **Naming**: `report_YYYYMMDD_HHMMSS_PATIENTID.pdf`

---

## 🧪 Testing the New Features

### Test Explainability Module
```bash
cd src
python explainability.py
```

**Expected Output:**
- Generates Grad-CAM, SHAP, and IG visualizations for a sample image
- Saves files to `outputs/explanations/`

### Test Decision Support System
```bash
cd src
python decision_support.py
```

**Expected Output:**
- Displays recommendations for various scenarios (benign/malignant, different confidence levels)

### Test Report Generator
```bash
cd src
python report_generator.py
```

**Expected Output:**
- Generates sample PDF reports
- Saves to `outputs/reports/`

### Test End-to-End Pipeline
```bash
# Find a sample image in your data directory
python inference.py data/0/sample_image.png --comprehensive --patient-id TEST001
```

**Expected Output:**
- Complete analysis with all features
- PDF report generated

---

## 📁 Project Structure

```
CNN/
├── src/
│   ├── model.py                    # ResNet50 model definition
│   ├── train.py                    # Training script
│   ├── data_loader.py              # Data loading utilities
│   ├── preprocessing.py            # Image preprocessing
│   ├── config.py                   # Configuration settings
│   ├── explainability.py           # ✨ NEW: XAI visualizations
│   ├── decision_support.py         # ✨ NEW: Clinical recommendations
│   └── report_generator.py         # ✨ NEW: PDF report generation
├── outputs/
│   ├── explanations/               # ✨ NEW: Explanation visualizations
│   │   ├── gradcam/
│   │   ├── shap/
│   │   └── integrated_gradients/
│   └── reports/                    # ✨ NEW: Generated PDF reports
├── models/
│   └── best_model.pth              # Trained model weights
├── data/                           # Training/test data
├── inference.py                    # ✨ ENHANCED: Comprehensive inference
├── app.py                          # ✨ ENHANCED: Streamlit web app
├── requirements.txt                # ✨ UPDATED: New dependencies
└── README.md
```

---

## ⚠️ Important Notes

### Medical Disclaimer
- This system is for **research and educational purposes only**
- **NOT approved for clinical diagnostic use**
- All findings must be validated by qualified medical professionals
- Do NOT make treatment decisions based solely on AI predictions

### Performance Considerations
- **Explainability generation** may take 30-60 seconds per image
- **SHAP** is the most computationally intensive (requires background samples)
- **PDF generation** is fast (~2-3 seconds)
- For batch processing, consider disabling explanations for speed

### Troubleshooting

#### "Module not found" errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

#### CUDA/GPU issues
```bash
# Force CPU mode if GPU causes issues
export CUDA_VISIBLE_DEVICES=""
python inference.py ...
```

#### Memory errors with SHAP
- Reduce `num_background` parameter in `explainability.py` (default: 50)
- Use CPU instead of GPU for SHAP

---

## 🔬 Advanced Usage

### Customize Recommendations
Edit `src/decision_support.py` to modify:
- Confidence thresholds
- Recommendation text
- Follow-up timelines
- Additional tests

### Customize PDF Reports
Edit `src/report_generator.py` to modify:
- Report layout and styling
- Included sections
- Color schemes
- Logo/branding

### Integrate with Existing Systems
The modules are designed to be modular:
```python
from explainability import generate_all_explanations
from decision_support import generate_comprehensive_report
from report_generator import generate_pdf_report

# Use in your own pipeline
explanations = generate_all_explanations(model, image_path, prediction)
report_data = generate_comprehensive_report(prediction, confidence)
pdf_path = generate_pdf_report(...)
```

---

## 📚 References

### Explainable AI Methods
- **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
- **SHAP**: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions"
- **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks"

### Medical Guidelines
- Recommendations based on standard breast cancer screening and diagnostic protocols
- Conservative approach prioritizing patient safety
- Emphasizes multidisciplinary review for all cases

---

## 🎓 Educational Use Cases

1. **Medical AI Education**: Demonstrate XAI techniques in healthcare
2. **Research**: Study model behavior and decision-making patterns
3. **Clinical Validation**: Compare AI predictions with expert pathologist assessments
4. **Workflow Integration**: Prototype AI-assisted diagnostic workflows

---

## 🤝 Contributing

To extend this system:
1. Add new explainability methods in `src/explainability.py`
2. Enhance decision logic in `src/decision_support.py`
3. Customize report templates in `src/report_generator.py`
4. Integrate additional medical knowledge bases

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the implementation plan: `implementation_plan.md`
3. Test individual modules separately to isolate issues

---

**Version**: 1.0  
**Last Updated**: February 2026  
**Status**: Capstone-Level AI Medical System ✅
