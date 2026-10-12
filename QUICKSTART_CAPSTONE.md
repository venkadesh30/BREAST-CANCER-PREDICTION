# Breast Cancer Detection System - Quick Start

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Usage

### Web Interface (Recommended)
```bash
streamlit run app.py
```
Then upload an image and explore all features!

### Command Line

**Basic prediction:**
```bash
python inference.py data/0/sample_image.png
```

**Full analysis with explanations and PDF report:**
```bash
python inference.py data/0/sample_image.png --comprehensive --patient-id PATIENT001
```

## New Features

✅ **Grad-CAM Heatmaps** - See which tissue regions influenced the AI  
✅ **Clinical Recommendations** - Evidence-based medical guidance  
✅ **PDF Reports** - Professional diagnostic reports  
✅ **Risk Stratification** - LOW/MEDIUM/HIGH/CRITICAL risk levels  

## Outputs

- **Explanations**: `outputs/explanations/`
- **PDF Reports**: `outputs/reports/`

## Important

⚠️ **For research/education only - NOT for clinical diagnosis**  
⚠️ **Always consult qualified medical professionals**

See `CAPSTONE_FEATURES_GUIDE.md` for detailed documentation.
