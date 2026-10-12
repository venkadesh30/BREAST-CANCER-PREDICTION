import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import BreastCancerClassifier
import config

# Import new capstone modules
try:
    from explainability import generate_all_explanations
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False
    print("Warning: Explainability module not available")

try:
    from decision_support import (
        get_medical_recommendation, 
        get_risk_level,
        generate_comprehensive_report
    )
    DECISION_SUPPORT_AVAILABLE = True
except ImportError:
    DECISION_SUPPORT_AVAILABLE = False
    print("Warning: Decision support module not available")

try:
    from report_generator import generate_pdf_report
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False
    print("Warning: Report generator module not available")

def load_model(model_path='models/best_model.pth'):
    """Load the trained model from checkpoint."""
    model = BreastCancerClassifier(model_name='resnet50', num_classes=2, freeze_backbone=True)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess a single image for inference."""
    # Same normalization as training (ImageNet stats)
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def predict(model, image_path):
    """
    Predict the class of a single image.
    
    Args:
        model: Trained model
        image_path: Path to the image file
    
    Returns:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Probability of the predicted class
    """
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()

def predict_comprehensive(model, image_path, patient_id=None, generate_report=True):
    """
    Perform comprehensive prediction with explainability, recommendations, and reporting.
    
    This is the enhanced prediction pipeline that integrates:
    1. Standard prediction (benign/malignant classification)
    2. Explainability visualizations (Grad-CAM, SHAP, Integrated Gradients)
    3. Clinical decision support (risk assessment and recommendations)
    4. Automated PDF report generation
    
    Args:
        model: Trained model
        image_path: Path to the image file
        patient_id: Optional patient identifier for the report
        generate_report: Whether to generate PDF report (default: True)
    
    Returns:
        dict: Comprehensive results containing:
            - prediction: 0 (Benign) or 1 (Malignant)
            - confidence: Probability of the predicted class
            - class_name: 'Benign' or 'Malignant'
            - benign_prob: Probability of benign
            - malignant_prob: Probability of malignant
            - risk_level: Risk stratification (LOW/MEDIUM/HIGH/CRITICAL)
            - recommendation: Clinical recommendation text
            - explanations: Dictionary with paths to explanation visualizations
            - report_path: Path to generated PDF report (if generated)
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE BREAST CANCER DETECTION ANALYSIS")
    print("="*70)
    print(f"Image: {image_path}")
    if patient_id:
        print(f"Patient ID: {patient_id}")
    print("-"*70)
    
    # Step 1: Standard prediction
    print("\n[1/4] Performing prediction...")
    prediction, confidence = predict(model, image_path)
    
    # Get probabilities
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        benign_prob = probabilities[0][0].item()
        malignant_prob = probabilities[0][1].item()
    
    class_name = 'Malignant' if prediction == 1 else 'Benign'
    print(f"✓ Prediction: {class_name}")
    print(f"  Confidence: {confidence:.2%}")
    print(f"  Benign probability: {benign_prob:.2%}")
    print(f"  Malignant probability: {malignant_prob:.2%}")
    
    # Step 2: Generate explanations
    explanations = None
    if EXPLAINABILITY_AVAILABLE:
        print("\n[2/4] Generating explainability visualizations...")
        try:
            explanations = generate_all_explanations(model, image_path, prediction)
        except Exception as e:
            print(f"Warning: Error generating explanations: {str(e)}")
            explanations = {'gradcam': None, 'shap': None, 'integrated_gradients': None}
    else:
        print("\n[2/4] Skipping explanations (module not available)")
        explanations = {'gradcam': None, 'shap': None, 'integrated_gradients': None}
    
    # Step 3: Get clinical recommendations
    recommendation = None
    risk_level = None
    if DECISION_SUPPORT_AVAILABLE:
        print("\n[3/4] Generating clinical recommendations...")
        try:
            recommendation = get_medical_recommendation(prediction, confidence)
            risk_level = get_risk_level(prediction, confidence)
            print(f"✓ Risk Level: {risk_level}")
            print(f"✓ Recommendation: {recommendation[:100]}...")
        except Exception as e:
            print(f"Warning: Error generating recommendations: {str(e)}")
            recommendation = "Unable to generate recommendation"
            risk_level = "UNKNOWN"
    else:
        print("\n[3/4] Skipping recommendations (module not available)")
        recommendation = "Decision support module not available"
        risk_level = "UNKNOWN"
    
    # Step 4: Generate PDF report
    report_path = None
    if generate_report and REPORT_GENERATOR_AVAILABLE:
        print("\n[4/4] Generating PDF report...")
        try:
            report_path = generate_pdf_report(
                image_path=image_path,
                prediction=prediction,
                confidence=confidence,
                recommendation=recommendation,
                explanation_summary=explanations,
                patient_id=patient_id,
                benign_prob=benign_prob,
                malignant_prob=malignant_prob
            )
        except Exception as e:
            print(f"Warning: Error generating report: {str(e)}")
            report_path = None
    else:
        print("\n[4/4] Skipping PDF report generation")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    # Return comprehensive results
    return {
        'prediction': prediction,
        'confidence': confidence,
        'class_name': class_name,
        'benign_prob': benign_prob,
        'malignant_prob': malignant_prob,
        'risk_level': risk_level,
        'recommendation': recommendation,
        'explanations': explanations,
        'report_path': report_path
    }

def main():
    """
    Main function for inference script.
    
    Usage:
        Standard mode:
            python inference.py <path_to_image>
        
        Comprehensive mode (with all features):
            python inference.py <path_to_image> --comprehensive
            python inference.py <path_to_image> --comprehensive --patient-id PATIENT123
            python inference.py <path_to_image> --comprehensive --no-report
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Breast Cancer Detection Inference')
    parser.add_argument('image_path', type=str, help='Path to the histopathology image')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive analysis with explanations and report')
    parser.add_argument('--patient-id', type=str, default=None,
                       help='Patient identifier for the report')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip PDF report generation (only with --comprehensive)')
    
    # Parse arguments (handle both old and new style)
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image> [--comprehensive] [--patient-id ID] [--no-report]")
        print("\nExamples:")
        print("  Standard mode:       python inference.py data/0/sample_image.png")
        print("  Comprehensive mode:  python inference.py data/0/sample_image.png --comprehensive")
        print("  With patient ID:     python inference.py data/0/sample_image.png --comprehensive --patient-id PATIENT001")
        return
    
    args = parser.parse_args()
    image_path = args.image_path
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    print("Loading model...")
    model = load_model()
    
    if args.comprehensive:
        # Comprehensive mode with all features
        print("\n🔬 Running COMPREHENSIVE analysis mode...")
        results = predict_comprehensive(
            model, 
            image_path, 
            patient_id=args.patient_id,
            generate_report=not args.no_report
        )
        
        # Display summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Prediction:     {results['class_name']}")
        print(f"Confidence:     {results['confidence']:.2%}")
        print(f"Risk Level:     {results['risk_level']}")
        print(f"\nBenign Prob:    {results['benign_prob']:.2%}")
        print(f"Malignant Prob: {results['malignant_prob']:.2%}")
        print(f"\nRecommendation:\n{results['recommendation']}")
        
        if results['explanations']:
            print("\nExplanation Files:")
            for key, path in results['explanations'].items():
                if key != 'prediction' and path:
                    print(f"  {key}: {path}")
        
        if results['report_path']:
            print(f"\n📄 PDF Report: {results['report_path']}")
        
        print("="*70)
        
    else:
        # Standard mode (backward compatible)
        print(f"\nProcessing image: {image_path}")
        prediction, confidence = predict(model, image_path)
        
        class_names = {0: "Benign", 1: "Malignant"}
        print(f"\nPrediction: {class_names[prediction]}")
        print(f"Confidence: {confidence:.2%}")
        
        # Additional interpretation
        if prediction == 1 and confidence > 0.8:
            print("\n⚠️  High confidence malignant prediction - recommend immediate review")
        elif prediction == 1 and confidence < 0.6:
            print("\n⚠️  Low confidence malignant prediction - recommend additional testing")
        elif prediction == 0 and confidence > 0.8:
            print("\n✓ High confidence benign prediction")
        else:
            print("\n⚠️  Borderline prediction - recommend expert review")
        
        print("\n💡 Tip: Use --comprehensive flag for detailed analysis with explanations and PDF report")

if __name__ == "__main__":
    main()
