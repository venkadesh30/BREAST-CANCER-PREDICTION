"""
Explainability Module for Breast Cancer Detection System

This module provides multiple explainable AI (XAI) techniques to interpret
model predictions on histopathology images:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- SHAP: SHapley Additive exPlanations
- Integrated Gradients: Attribution method with theoretical guarantees

Medical Relevance:
- Helps pathologists understand which tissue regions influenced the AI decision
- Provides visual evidence for model predictions
- Enables validation of AI reasoning against medical knowledge
- Supports clinical decision-making with transparent AI
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Grad-CAM imports
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not installed. Grad-CAM will not be available.")

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not installed. SHAP explanations will not be available.")

# Integrated Gradients imports
try:
    from captum.attr import IntegratedGradients
    from captum.attr import visualization as viz
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Warning: captum not installed. Integrated Gradients will not be available.")

import config

# Create output directories
EXPLANATIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs', 'explanations')
GRADCAM_DIR = os.path.join(EXPLANATIONS_DIR, 'gradcam')
SHAP_DIR = os.path.join(EXPLANATIONS_DIR, 'shap')
IG_DIR = os.path.join(EXPLANATIONS_DIR, 'integrated_gradients')

for dir_path in [GRADCAM_DIR, SHAP_DIR, IG_DIR]:
    os.makedirs(dir_path, exist_ok=True)


def load_and_preprocess_image(image_path):
    """
    Load and preprocess image for model input.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (preprocessed_tensor, original_image_rgb, original_image_pil)
    """
    from torchvision import transforms
    
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Preprocessing transform (same as inference)
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform for visualization (no normalization)
    transform_viz = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(original_image).unsqueeze(0)
    viz_tensor = transform_viz(original_image)
    viz_image = viz_tensor.permute(1, 2, 0).numpy()
    
    return input_tensor, viz_image, original_image


def generate_gradcam(model, image_path, target_class=None, save_path=None):
    """
    Generate Grad-CAM heatmap for the given image.
    
    Grad-CAM highlights the regions in the image that are most important
    for the model's prediction. For histopathology images, this shows
    which tissue structures influenced the benign/malignant classification.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the input image
        target_class: Target class for visualization (0=Benign, 1=Malignant)
                     If None, uses the predicted class
        save_path: Optional custom save path
        
    Returns:
        str: Path to saved Grad-CAM visualization
    """
    if not GRADCAM_AVAILABLE:
        print("Error: pytorch-grad-cam not installed. Install with: pip install grad-cam")
        return None
    
    try:
        # Load and preprocess image
        input_tensor, viz_image, _ = load_and_preprocess_image(image_path)
        input_tensor = input_tensor.to(config.DEVICE)
        
        # Set model to eval mode
        model.eval()
        
        # Get prediction if target_class not specified
        if target_class is None:
            with torch.no_grad():
                outputs = model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        # Determine target layer based on model architecture
        # For ResNet50, use the last convolutional layer (layer4)
        if hasattr(model, 'backbone'):
            if hasattr(model.backbone, 'layer4'):
                target_layers = [model.backbone.layer4[-1]]
            elif hasattr(model.backbone, 'features'):
                # For VGG16
                target_layers = [model.backbone.features[-1]]
            else:
                raise ValueError("Could not identify target layer for Grad-CAM")
        else:
            raise ValueError("Model structure not recognized")
        
        # Initialize Grad-CAM
        cam = GradCAM(model=model, target_layers=target_layers)
        
        # Generate CAM with targets
        targets = [ClassifierOutputTarget(target_class)]
        
        # Important: Detach and clone to avoid in-place modification
        input_for_cam = input_tensor.detach().clone()
        input_for_cam.requires_grad = True
        
        grayscale_cam = cam(input_tensor=input_for_cam, targets=targets, eigen_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        
        # Create visualization
        visualization = show_cam_on_image(viz_image, grayscale_cam, use_rgb=True)
        
        # Save visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"gradcam_{timestamp}_class{target_class}.png"
            save_path = os.path.join(GRADCAM_DIR, filename)
        
        # Create figure with original and Grad-CAM side by side
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(viz_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(visualization)
        class_name = 'Malignant' if target_class == 1 else 'Benign'
        axes[1].set_title(f'Grad-CAM Heatmap\nPrediction: {class_name}', 
                         fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Grad-CAM saved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_shap(model, image_path, target_class=None, save_path=None, num_background=50):
    """
    Generate SHAP explanation for the given image.
    
    SHAP provides pixel-level importance scores showing which pixels
    contributed positively or negatively to the prediction.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the input image
        target_class: Target class for explanation (0=Benign, 1=Malignant)
        save_path: Optional custom save path
        num_background: Number of background samples for SHAP
        
    Returns:
        str: Path to saved SHAP visualization
    """
    if not SHAP_AVAILABLE:
        print("Error: shap not installed. Install with: pip install shap")
        return None
    
    try:
        # Load and preprocess image
        input_tensor, viz_image, _ = load_and_preprocess_image(image_path)
        input_tensor = input_tensor.to(config.DEVICE)
        
        # Get prediction if target_class not specified
        if target_class is None:
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        # Create background dataset (random noise or black images)
        # For efficiency, we use a small background set
        background = torch.zeros((num_background, 3, config.IMG_SIZE, config.IMG_SIZE)).to(config.DEVICE)
        
        # Initialize SHAP DeepExplainer
        model.eval()
        explainer = shap.DeepExplainer(model, background)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(input_tensor)
        
        # SHAP values shape: [num_classes, batch_size, channels, height, width]
        # We want the SHAP values for the target class
        if isinstance(shap_values, list):
            shap_values_target = shap_values[target_class][0]
        else:
            shap_values_target = shap_values[0, target_class]
        
        # Aggregate across color channels (take mean)
        shap_values_aggregated = np.mean(np.abs(shap_values_target), axis=0)
        
        # Normalize for visualization
        shap_values_normalized = (shap_values_aggregated - shap_values_aggregated.min()) / \
                                 (shap_values_aggregated.max() - shap_values_aggregated.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * shap_values_normalized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(
            (viz_image * 255).astype(np.uint8), 0.6,
            heatmap, 0.4, 0
        )
        
        # Save visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"shap_{timestamp}_class{target_class}.png"
            save_path = os.path.join(SHAP_DIR, filename)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(viz_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(shap_values_normalized, cmap='hot')
        axes[1].set_title('SHAP Attribution Heatmap', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        class_name = 'Malignant' if target_class == 1 else 'Benign'
        axes[2].set_title(f'SHAP Overlay\nPrediction: {class_name}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ SHAP explanation saved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error generating SHAP explanation: {str(e)}")
        return None


def generate_integrated_gradients(model, image_path, target_class=None, save_path=None):
    """
    Generate Integrated Gradients attribution for the given image.
    
    Integrated Gradients computes the attribution of each pixel by integrating
    gradients along a path from a baseline (black image) to the input image.
    This method has strong theoretical guarantees (completeness and sensitivity).
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the input image
        target_class: Target class for attribution (0=Benign, 1=Malignant)
        save_path: Optional custom save path
        
    Returns:
        str: Path to saved IG visualization
    """
    if not CAPTUM_AVAILABLE:
        print("Error: captum not installed. Install with: pip install captum")
        return None
    
    try:
        # Load and preprocess image
        input_tensor, viz_image, _ = load_and_preprocess_image(image_path)
        input_tensor = input_tensor.to(config.DEVICE)
        input_tensor.requires_grad = True
        
        # Get prediction if target_class not specified
        if target_class is None:
            model.eval()
            with torch.no_grad():
                outputs = model(input_tensor)
                target_class = torch.argmax(outputs, dim=1).item()
        
        # Initialize Integrated Gradients
        ig = IntegratedGradients(model)
        
        # Baseline: black image (all zeros after normalization)
        baseline = torch.zeros_like(input_tensor).to(config.DEVICE)
        
        # Generate attributions
        attributions = ig.attribute(input_tensor, baseline, target=target_class, n_steps=50)
        
        # Convert to numpy and aggregate across channels
        attributions_np = attributions.squeeze().cpu().detach().numpy()
        attributions_aggregated = np.mean(np.abs(attributions_np), axis=0)
        
        # Normalize for visualization
        attributions_normalized = (attributions_aggregated - attributions_aggregated.min()) / \
                                  (attributions_aggregated.max() - attributions_aggregated.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attributions_normalized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay on original image
        overlay = cv2.addWeighted(
            (viz_image * 255).astype(np.uint8), 0.6,
            heatmap, 0.4, 0
        )
        
        # Save visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ig_{timestamp}_class{target_class}.png"
            save_path = os.path.join(IG_DIR, filename)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(viz_image)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(attributions_normalized, cmap='hot')
        axes[1].set_title('Integrated Gradients Attribution', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        class_name = 'Malignant' if target_class == 1 else 'Benign'
        axes[2].set_title(f'IG Overlay\nPrediction: {class_name}', 
                         fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Integrated Gradients saved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error generating Integrated Gradients: {str(e)}")
        return None


def generate_all_explanations(model, image_path, prediction=None):
    """
    Generate all available explanations for the given image.
    
    This is the main interface for the explainability module.
    It generates Grad-CAM, SHAP, and Integrated Gradients explanations.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to the input image
        prediction: Predicted class (0=Benign, 1=Malignant)
                   If None, will be computed from the model
        
    Returns:
        dict: Dictionary with paths to all generated explanations
              {
                  'gradcam': path or None,
                  'shap': path or None,
                  'integrated_gradients': path or None,
                  'prediction': predicted class
              }
    """
    print("\n" + "="*60)
    print("Generating Explainability Visualizations")
    print("="*60)
    
    # Get prediction if not provided
    if prediction is None:
        input_tensor, _, _ = load_and_preprocess_image(image_path)
        input_tensor = input_tensor.to(config.DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.argmax(outputs, dim=1).item()
    
    class_name = 'Malignant' if prediction == 1 else 'Benign'
    print(f"Target Class: {class_name} (class {prediction})")
    print("-"*60)
    
    results = {
        'prediction': prediction,
        'gradcam': None,
        'shap': None,
        'integrated_gradients': None
    }
    
    # Generate Grad-CAM
    print("\n[1/3] Generating Grad-CAM...")
    results['gradcam'] = generate_gradcam(model, image_path, target_class=prediction)
    
    # Generate SHAP
    print("\n[2/3] Generating SHAP explanation...")
    results['shap'] = generate_shap(model, image_path, target_class=prediction)
    
    # Generate Integrated Gradients
    print("\n[3/3] Generating Integrated Gradients...")
    results['integrated_gradients'] = generate_integrated_gradients(model, image_path, target_class=prediction)
    
    print("\n" + "="*60)
    print("Explainability Generation Complete")
    print("="*60)
    
    # Summary
    successful = sum(1 for v in results.values() if v is not None and v != prediction)
    print(f"\nSuccessfully generated {successful}/3 explanations")
    
    return results


if __name__ == "__main__":
    """
    Test the explainability module with a sample image.
    Usage: python explainability.py
    """
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from model import BreastCancerClassifier
    
    print("Testing Explainability Module")
    print("="*60)
    
    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'models', 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python src/train.py")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model = BreastCancerClassifier(model_name='resnet50', num_classes=2, freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Find a sample image
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    sample_image = None
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_image = os.path.join(root, file)
                break
        if sample_image:
            break
    
    if sample_image is None:
        print(f"Error: No sample images found in {data_dir}")
        sys.exit(1)
    
    print(f"\nUsing sample image: {sample_image}")
    
    # Generate all explanations
    results = generate_all_explanations(model, sample_image)
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nGenerated files:")
    for key, path in results.items():
        if key != 'prediction' and path:
            print(f"  {key}: {path}")
