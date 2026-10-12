"""
Flask-based Web Application for Breast Cancer Detection
Modern HTML/CSS/JavaScript frontend with RESTful API backend
"""

from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import json
from datetime import datetime
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model import BreastCancerClassifier
import config

# Flask app configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model cache
MODEL = None
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
PREDICTION_HISTORY = []

def load_model():
    """Load and cache the model."""
    global MODEL
    if MODEL is None:
        try:
            MODEL = BreastCancerClassifier(model_name='resnet50', num_classes=2, freeze_backbone=True)
            model_path = 'models/best_model.pth'
            if not os.path.exists(model_path):
                return None
            MODEL.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            MODEL.to(config.DEVICE)
            MODEL.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return MODEL

def allowed_file(filename):
    """Check if file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model."""
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def get_prediction(model, image):
    """Get prediction from model."""
    input_tensor = preprocess_image(image).to(config.DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return {
            'prediction': predicted.item(),
            'confidence': confidence.item(),
            'benign_prob': probabilities[0][0].item(),
            'malignant_prob': probabilities[0][1].item()
        }

def get_risk_level(prediction, confidence):
    """Determine risk level."""
    if prediction == 1:  # Malignant
        if confidence > 0.8:
            return "🔴 HIGH RISK"
        elif confidence > 0.6:
            return "🟠 MEDIUM RISK"
        else:
            return "🟡 LOW CONFIDENCE"
    else:  # Benign
        if confidence > 0.8:
            return "🟢 LOW RISK"
        elif confidence > 0.6:
            return "🟡 MODERATE CONFIDENCE"
        else:
            return "🟠 LOW CONFIDENCE"

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for prediction."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format'}), 400
        
        # Load model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded. Train the model first.'}), 500
        
        # Process image
        image = Image.open(file.stream)
        
        # Get prediction
        result = get_prediction(model, image)
        
        # Prepare response
        response = {
            'filename': secure_filename(file.filename),
            'prediction': 'Malignant' if result['prediction'] == 1 else 'Benign',
            'confidence': f"{result['confidence']:.2%}",
            'benign_prob': f"{result['benign_prob']:.2%}",
            'malignant_prob': f"{result['malignant_prob']:.2%}",
            'risk_level': get_risk_level(result['prediction'], result['confidence']),
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        PREDICTION_HISTORY.append(response)
        
        # Convert image to base64 for display
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        response['image'] = f"data:image/png;base64,{img_base64}"
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch prediction."""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files selected'}), 400
        
        # Load model
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        results = []
        errors = []
        
        for file in files:
            try:
                if not allowed_file(file.filename):
                    errors.append(f"{file.filename}: Invalid format")
                    continue
                
                image = Image.open(file.stream)
                result = get_prediction(model, image)
                
                result_item = {
                    'filename': secure_filename(file.filename),
                    'prediction': 'Malignant' if result['prediction'] == 1 else 'Benign',
                    'confidence': f"{result['confidence']:.2%}",
                    'risk_level': get_risk_level(result['prediction'], result['confidence'])
                }
                results.append(result_item)
                PREDICTION_HISTORY.append(result_item)
            
            except Exception as e:
                errors.append(f"{file.filename}: {str(e)}")
        
        # Calculate statistics
        malignant_count = sum(1 for r in results if r['prediction'] == 'Malignant')
        benign_count = len(results) - malignant_count
        
        return jsonify({
            'total': len(results),
            'malignant': malignant_count,
            'benign': benign_count,
            'results': results,
            'errors': errors
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history."""
    return jsonify({
        'total': len(PREDICTION_HISTORY),
        'history': PREDICTION_HISTORY[-20:]  # Last 20
    }), 200

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    """Clear prediction history."""
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    return jsonify({'message': 'History cleared'}), 200

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics."""
    if not PREDICTION_HISTORY:
        return jsonify({
            'total_predictions': 0,
            'malignant': 0,
            'benign': 0,
            'average_confidence': 'N/A'
        }), 200
    
    malignant_count = sum(1 for p in PREDICTION_HISTORY if p['prediction'] == 'Malignant')
    return jsonify({
        'total_predictions': len(PREDICTION_HISTORY),
        'malignant': malignant_count,
        'benign': len(PREDICTION_HISTORY) - malignant_count
    }), 200

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information."""
    return jsonify({
        'model_type': 'ResNet50',
        'training_method': 'Transfer Learning',
        'base_model': 'Pre-trained on ImageNet',
        'input_size': '224 × 224 pixels',
        'output_classes': 2,
        'device': str(config.DEVICE),
        'framework': 'PyTorch'
    }), 200

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("Starting Flask server...")
    app.run(debug=True, port=5000)
