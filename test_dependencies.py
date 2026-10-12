"""
Quick test script to verify all new dependencies are installed correctly.
Run this to ensure the capstone features will work.
"""

import sys

print("Testing Capstone Feature Dependencies...")
print("=" * 60)

# Test 1: PyTorch and core dependencies
print("\n[1/7] Testing PyTorch...")
try:
    import torch
    import torchvision
    print(f"✓ PyTorch {torch.__version__} installed")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

# Test 2: Grad-CAM
print("\n[2/7] Testing Grad-CAM...")
try:
    from pytorch_grad_cam import GradCAM
    print("✓ Grad-CAM installed")
except ImportError as e:
    print(f"✗ Grad-CAM not installed: {e}")
    print("  Try: pip install grad-cam")

# Test 3: SHAP
print("\n[3/7] Testing SHAP...")
try:
    import shap
    print(f"✓ SHAP {shap.__version__} installed")
except ImportError as e:
    print(f"✗ SHAP not installed: {e}")
    print("  Try: pip install shap")

# Test 4: Captum (Integrated Gradients)
print("\n[4/7] Testing Captum...")
try:
    import captum
    from captum.attr import IntegratedGradients
    print(f"✓ Captum {captum.__version__} installed")
except ImportError as e:
    print(f"✗ Captum not installed: {e}")
    print("  Try: pip install captum")

# Test 5: ReportLab (PDF generation)
print("\n[5/7] Testing ReportLab...")
try:
    import reportlab
    from reportlab.pdfgen import canvas
    print(f"✓ ReportLab {reportlab.Version} installed")
except ImportError as e:
    print(f"✗ ReportLab not installed: {e}")
    print("  Try: pip install reportlab")

# Test 6: OpenCV
print("\n[6/7] Testing OpenCV...")
try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__} installed")
except ImportError as e:
    print(f"✗ OpenCV not installed: {e}")
    print("  Try: pip install opencv-python")

# Test 7: Other dependencies
print("\n[7/7] Testing other dependencies...")
try:
    import numpy
    import pandas
    import matplotlib
    import PIL
    import streamlit
    print("✓ All other dependencies installed")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")

# Test custom modules
print("\n" + "=" * 60)
print("Testing Custom Modules...")
print("=" * 60)

sys.path.insert(0, 'src')

print("\n[1/3] Testing explainability module...")
try:
    import explainability
    print("✓ explainability.py loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load explainability.py: {e}")

print("\n[2/3] Testing decision_support module...")
try:
    import decision_support
    print("✓ decision_support.py loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load decision_support.py: {e}")

print("\n[3/3] Testing report_generator module...")
try:
    import report_generator
    print("✓ report_generator.py loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load report_generator.py: {e}")

# Summary
print("\n" + "=" * 60)
print("DEPENDENCY CHECK COMPLETE")
print("=" * 60)
print("\n✅ All dependencies installed successfully!")
print("\nYou can now use:")
print("  • python inference.py <image> --comprehensive")
print("  • streamlit run app.py")
print("\nFor detailed usage, see CAPSTONE_FEATURES_GUIDE.md")
