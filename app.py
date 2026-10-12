import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys
import os
import io
from datetime import datetime
import base64

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import model definition from our project
from model import BreastCancerClassifier
import config

# Import new capstone modules
try:
    from explainability import generate_all_explanations
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

try:
    from decision_support import get_medical_recommendation, get_risk_level
    DECISION_SUPPORT_AVAILABLE = True
except ImportError:
    DECISION_SUPPORT_AVAILABLE = False

try:
    from report_generator import generate_pdf_report
    REPORT_GENERATOR_AVAILABLE = True
except ImportError:
    REPORT_GENERATOR_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Detection System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    :root {
        --primary-color: #e94b3c;
        --secondary-color: #4CAF50;
        --warning-color: #ff9800;
        --safe-color: #2196F3;
    }
    
    .main {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .header-container {
        background: linear-gradient(135deg, #e94b3c 0%, #c62828 100%);
        padding: 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-box {
        padding: 25px;
        border-radius: 10px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    
    .malignant-box {
        background-color: #ffebee;
        border-left-color: #c62828;
    }
    
    .benign-box {
        background-color: #e8f5e9;
        border-left-color: #2e7d32;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #4CAF50 0%, #FFC107 50%, #FF5722 100%);
        height: 30px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        margin: 15px 0;
    }
    
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .model-info {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 10px 0;
    }
    
    .disclaimer {
        background: #fff3e0;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #f57c00;
        margin: 15px 0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #e94b3c 0%, #c62828 100%);
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

@st.cache_resource
def load_model():
    """Load the trained model and cache it for performance."""
    try:
        model = BreastCancerClassifier(model_name='resnet50', num_classes=2, freeze_backbone=True)
        model_path = 'models/best_model.pth'
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at {model_path}. Please train the model first.")
            return None
            
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.to(config.DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for the model."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = image.convert('RGB')
    return transform(image).unsqueeze(0)

def perform_prediction(model, image):
    """Perform prediction on the image."""
    input_tensor = preprocess_image(image).to(config.DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        prediction_idx = predicted.item()
        confidence_score = confidence.item()
        benign_prob = probabilities[0][0].item()
        malignant_prob = probabilities[0][1].item()
    
    return {
        'prediction': prediction_idx,
        'confidence': confidence_score,
        'benign_prob': benign_prob,
        'malignant_prob': malignant_prob
    }

def get_risk_category(confidence, prediction):
    """Determine risk category based on prediction and confidence."""
    if prediction == 1:  # Malignant
        if confidence > 0.8:
            return "🔴 HIGH RISK", "High confidence malignant detection"
        elif confidence > 0.6:
            return "🟠 MEDIUM RISK", "Moderate confidence malignant detection"
        else:
            return "🟡 LOW CONFIDENCE", "Low confidence malignant detection - Additional testing recommended"
    else:  # Benign
        if confidence > 0.8:
            return "🟢 LOW RISK", "High confidence benign classification"
        elif confidence > 0.6:
            return "🟡 MODERATE CONFIDENCE", "Moderate confidence benign classification"
        else:
            return "🟠 LOW CONFIDENCE", "Low confidence benign classification - Additional testing recommended"

def generate_report(image_array, prediction_result, filename):
    """Generate a text report of the analysis."""
    report = f"""
BREAST CANCER DETECTION ANALYSIS REPORT
=====================================
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Filename: {filename}

PREDICTION RESULTS:
------------------
Classification: {'MALIGNANT' if prediction_result['prediction'] == 1 else 'BENIGN'}
Confidence: {prediction_result['confidence']:.2%}
Benign Probability: {prediction_result['benign_prob']:.2%}
Malignant Probability: {prediction_result['malignant_prob']:.2%}

RISK ASSESSMENT:
----------------
{get_risk_category(prediction_result['confidence'], prediction_result['prediction'])[0]}
{get_risk_category(prediction_result['confidence'], prediction_result['prediction'])[1]}

RECOMMENDATIONS:
---------------
- This analysis is for informational purposes only
- Always consult with a qualified pathologist or oncologist for diagnosis
- This AI-assisted analysis should complement, not replace, professional medical judgment
- Consider additional diagnostic tests if results are borderline (40-60% confidence)

Model Information:
- Architecture: ResNet50 with Transfer Learning
- Training Dataset: Histopathology images
- Device: {config.DEVICE}
"""
    return report

def main():
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>🎗️ Breast Cancer Detection System</h1>
        <p style="font-size: 18px; margin-top: 10px;">Automated Histopathology Analysis using Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Navigation and Settings
    with st.sidebar:
        st.title("📋 Menu")
        page = st.radio("Select Page:", ["🏥 Diagnosis", "📊 Model Info", "📚 About"])
        
        st.markdown("---")
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold for Alert",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Show warning if confidence falls below this threshold"
        )
        
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
            <strong>⚠️ IMPORTANT DISCLAIMER</strong><br/>
            This tool is for educational and research purposes only. 
            It should NOT be used for actual medical diagnosis or treatment decisions. 
            Always consult qualified medical professionals.
        </div>
        """, unsafe_allow_html=True)
    
    # Page routing
    if page == "🏥 Diagnosis":
        show_diagnosis_page(confidence_threshold)
    elif page == "📊 Model Info":
        show_model_info_page()
    elif page == "📚 About":
        show_about_page()

def show_diagnosis_page(confidence_threshold):
    """Main diagnosis/prediction page."""
    st.subheader("🔬 Image Analysis")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📤 Single Image", "📁 Batch Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Upload a histopathology image for analysis**")
            uploaded_file = st.file_uploader(
                "Choose an image (PNG, JPG, JPEG):",
                type=["jpg", "jpeg", "png"],
                key="single_upload"
            )
        
        with col2:
            st.info("💡 **Supported Formats:** PNG, JPG, JPEG")
            st.info("📐 **Recommended:** Images from histopathology datasets")
        
        if uploaded_file is not None:
            analyze_single_image(uploaded_file, confidence_threshold)
    
    with tab2:
        st.write("**Upload multiple images for batch analysis**")
        uploaded_files = st.file_uploader(
            "Choose images:",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="batch_upload"
        )
        
        if uploaded_files:
            analyze_batch_images(uploaded_files, confidence_threshold)

def analyze_single_image(uploaded_file, confidence_threshold):
    """Analyze a single uploaded image."""
    try:
        image = Image.open(uploaded_file)
        
        # Create layout
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Image info
            st.markdown("""
            <div class="info-card">
                <strong>📸 Image Information</strong><br/>
                Size: {0} × {1} pixels
            </div>
            """.format(image.width, image.height), unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔍 Analysis Results")
            
            # Load model and predict
            model = load_model()
            
            if model:
                with st.spinner("🔄 Analyzing image..."):
                    prediction_result = perform_prediction(model, image)
                
                # Display prediction
                prediction_idx = prediction_result['prediction']
                confidence_score = prediction_result['confidence']
                benign_prob = prediction_result['benign_prob']
                malignant_prob = prediction_result['malignant_prob']
                
                # Risk category
                risk_level, risk_desc = get_risk_category(confidence_score, prediction_idx)
                
                # Color-coded prediction box
                if prediction_idx == 1:
                    st.markdown(f"""
                    <div class="prediction-box malignant-box">
                        <h2 style="color: #c62828;">⚠️ MALIGNANT</h2>
                        <p style="font-size: 18px;">Confidence: <b>{confidence_score:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box benign-box">
                        <h2 style="color: #2e7d32;">✅ BENIGN</h2>
                        <p style="font-size: 18px;">Confidence: <b>{confidence_score:.2%}</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Risk assessment
                st.markdown(f"### {risk_level}")
                st.write(f"_{risk_desc}_")
                
                # Confidence warning if below threshold
                if confidence_score < confidence_threshold:
                    st.warning(
                        f"⚠️ **Low Confidence Alert**: Confidence ({confidence_score:.2%}) "
                        f"is below your threshold ({confidence_threshold:.2%}). "
                        f"Consider additional professional review.",
                        icon="⚠️"
                    )
                
                # Probability distribution
                st.markdown("### 📊 Probability Distribution")
                col_b, col_m = st.columns(2)
                with col_b:
                    st.metric("Benign Probability", f"{benign_prob:.2%}")
                with col_m:
                    st.metric("Malignant Probability", f"{malignant_prob:.2%}")
                
                # Probability visualization
                prob_data = {
                    'Benign': benign_prob,
                    'Malignant': malignant_prob
                }
                st.bar_chart(prob_data)
                
                # ==================== EXPLAINABILITY SECTION ====================
                if EXPLAINABILITY_AVAILABLE:
                    with st.expander("🔍 Model Explainability (Grad-CAM Heatmap)", expanded=True):
                        st.markdown("""
                        The heatmap below shows which regions of the tissue influenced the AI's decision.
                        **Red/yellow areas** indicate high importance, **blue areas** indicate low importance.
                        """)
                        
                        with st.spinner("Generating Grad-CAM heatmap..."):
                            try:
                                # Save uploaded image temporarily
                                import tempfile
                                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                    image.save(tmp_file.name)
                                    temp_image_path = tmp_file.name
                                
                                # Import the specific function we need
                                from explainability import generate_gradcam
                                
                                # Generate only Grad-CAM (fastest method)
                                gradcam_path = generate_gradcam(model, temp_image_path, target_class=prediction_idx)
                                
                                # Display Grad-CAM if available
                                if gradcam_path and os.path.exists(gradcam_path):
                                    st.image(gradcam_path, 
                                            caption='Grad-CAM Heatmap Visualization',
                                            use_column_width=True)
                                    st.success("✅ Grad-CAM heatmap generated successfully!")
                                else:
                                    st.warning("⚠️ Could not generate Grad-CAM heatmap. The model may need retraining or the grad-cam package may not be installed correctly.")
                                
                                # Clean up temp file
                                try:
                                    os.unlink(temp_image_path)
                                except:
                                    pass
                                    
                            except Exception as e:
                                st.error(f"❌ Error generating Grad-CAM: {str(e)}")
                                st.info("💡 Tip: Make sure 'grad-cam' package is installed: `pip install grad-cam`")
                else:
                    st.info("ℹ️ Explainability features not available. Install required packages: `pip install grad-cam`")
                
                
                # ==================== CLINICAL RECOMMENDATIONS ====================
                if DECISION_SUPPORT_AVAILABLE:
                    st.markdown("### 🏥 Clinical Recommendations")
                    
                    try:
                        recommendation = get_medical_recommendation(prediction_idx, confidence_score)
                        clinical_risk = get_risk_level(prediction_idx, confidence_score)
                        
                        # Color-coded recommendation box
                        risk_colors = {
                            'LOW': '#c8e6c9',
                            'MEDIUM': '#fff9c4',
                            'HIGH': '#ffccbc',
                            'CRITICAL': '#ffcdd2'
                        }
                        risk_color = risk_colors.get(clinical_risk, '#f5f5f5')
                        
                        st.markdown(f"""
                        <div style="background-color: {risk_color}; padding: 15px; border-radius: 8px; 
                                    border-left: 5px solid #424242; margin: 10px 0;">
                            <strong>Risk Level:</strong> {clinical_risk}<br/>
                            <strong>Recommendation:</strong><br/>
                            {recommendation}
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not generate clinical recommendations: {str(e)}")
                
                # ==================== REPORT GENERATION ====================
                st.markdown("### 📄 Generate Medical Report")
                
                # Text report (existing)
                report_text = generate_report(
                    np.array(image),
                    prediction_result,
                    uploaded_file.name
                )
                
                col_download1, col_download2, col_download3 = st.columns(3)
                
                with col_download1:
                    st.download_button(
                        label="📥 Download Report (TXT)",
                        data=report_text,
                        file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                
                # PDF report generation
                with col_download2:
                    if REPORT_GENERATOR_AVAILABLE:
                        if st.button("📄 Generate PDF Report"):
                            with st.spinner("Generating comprehensive PDF report..."):
                                try:
                                    # Save uploaded image temporarily
                                    import tempfile
                                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                                        image.save(tmp_file.name)
                                        temp_image_path = tmp_file.name
                                    
                                    # Get recommendation if available
                                    if DECISION_SUPPORT_AVAILABLE:
                                        recommendation = get_medical_recommendation(prediction_idx, confidence_score)
                                    else:
                                        recommendation = "Clinical decision support not available"
                                    
                                    # Generate explanations for report
                                    if EXPLAINABILITY_AVAILABLE:
                                        explanations = generate_all_explanations(model, temp_image_path, prediction_idx)
                                    else:
                                        explanations = {'gradcam': None, 'shap': None, 'integrated_gradients': None}
                                    
                                    # Generate PDF
                                    pdf_path = generate_pdf_report(
                                        image_path=temp_image_path,
                                        prediction=prediction_idx,
                                        confidence=confidence_score,
                                        recommendation=recommendation,
                                        explanation_summary=explanations,
                                        patient_id=None,
                                        benign_prob=benign_prob,
                                        malignant_prob=malignant_prob
                                    )
                                    
                                    if pdf_path and os.path.exists(pdf_path):
                                        with open(pdf_path, 'rb') as pdf_file:
                                            st.download_button(
                                                label="📥 Download PDF Report",
                                                data=pdf_file.read(),
                                                file_name=os.path.basename(pdf_path),
                                                mime="application/pdf"
                                            )
                                        st.success("✅ PDF report generated successfully!")
                                    
                                    # Clean up temp file
                                    try:
                                        os.unlink(temp_image_path)
                                    except:
                                        pass
                                        
                                except Exception as e:
                                    st.error(f"Error generating PDF report: {str(e)}")
                
                with col_download3:
                    if st.button("💾 Save to History"):
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'filename': uploaded_file.name,
                            'prediction': 'Malignant' if prediction_idx == 1 else 'Benign',
                            'confidence': confidence_score,
                            'risk_level': risk_level
                        })
                        st.success("✅ Added to prediction history!")
                
                # Detailed clinical notes
                with st.expander("📋 Clinical Interpretation Guide"):
                    st.markdown("""
                    **Understanding the Results:**
                    
                    - **High Confidence (>80%)**: Model is very certain about the classification
                    - **Medium Confidence (60-80%)**: Model is moderately confident
                    - **Low Confidence (<60%)**: Results are borderline; professional review strongly recommended
                    
                    **Important Notes:**
                    - This AI analysis is an assistive tool only
                    - Cross-validation with pathological examination is essential
                    - False positives and negatives can occur
                    - Clinical judgment should always take precedence
                    """)
    
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")

def analyze_batch_images(uploaded_files, confidence_threshold):
    """Analyze multiple images in batch mode."""
    st.info(f"📁 Processing {len(uploaded_files)} images...")
    
    model = load_model()
    if not model:
        return
    
    results = []
    progress_bar = st.progress(0)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            image = Image.open(uploaded_file)
            prediction_result = perform_prediction(model, image)
            
            results.append({
                'filename': uploaded_file.name,
                'prediction': 'Malignant' if prediction_result['prediction'] == 1 else 'Benign',
                'confidence': prediction_result['confidence'],
                'benign_prob': prediction_result['benign_prob'],
                'malignant_prob': prediction_result['malignant_prob']
            })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        except Exception as e:
            st.warning(f"⚠️ Error processing {uploaded_file.name}: {str(e)}")
    
    # Display results table
    st.markdown("### 📊 Batch Analysis Results")
    
    # Create results dataframe
    import pandas as pd
    df_results = pd.DataFrame(results)
    df_results['confidence'] = df_results['confidence'].apply(lambda x: f"{x:.2%}")
    df_results['benign_prob'] = df_results['benign_prob'].apply(lambda x: f"{x:.2%}")
    df_results['malignant_prob'] = df_results['malignant_prob'].apply(lambda x: f"{x:.2%}")
    
    st.dataframe(df_results, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        malignant_count = sum(1 for r in results if r['prediction'] == 'Malignant')
        st.metric("Malignant Detections", malignant_count)
    with col2:
        benign_count = len(results) - malignant_count
        st.metric("Benign Classifications", benign_count)
    with col3:
        avg_confidence = np.mean([r['confidence'] for r in results])
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    # Download batch report
    batch_report = "BATCH ANALYSIS REPORT\n" + "="*50 + "\n"
    batch_report += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    batch_report += f"Total Images: {len(results)}\n"
    batch_report += f"Malignant: {malignant_count}\n"
    batch_report += f"Benign: {benign_count}\n\n"
    batch_report += "DETAILED RESULTS:\n" + "-"*50 + "\n"
    for r in results:
        batch_report += f"\n{r['filename']}:\n"
        batch_report += f"  Prediction: {r['prediction']}\n"
        batch_report += f"  Confidence: {r['confidence']}\n"
    
    st.download_button(
        label="📥 Download Batch Report",
        data=batch_report,
        file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def show_model_info_page():
    """Display model information and specifications."""
    st.header("📊 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="model-info">
            <h3>🔧 Architecture Details</h3>
            <ul>
                <li><strong>Model Type:</strong> ResNet50</li>
                <li><strong>Training Method:</strong> Transfer Learning</li>
                <li><strong>Base Model:</strong> Pre-trained on ImageNet</li>
                <li><strong>Input Size:</strong> 224 × 224 pixels</li>
                <li><strong>Output Classes:</strong> 2 (Benign, Malignant)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="model-info">
            <h3>📈 Performance Metrics</h3>
            <ul>
                <li><strong>Device:</strong> {0}</li>
                <li><strong>Batch Size:</strong> 32</li>
                <li><strong>Image Normalization:</strong> ImageNet Stats</li>
                <li><strong>Threshold:</strong> 0.5</li>
                <li><strong>Framework:</strong> PyTorch</li>
            </ul>
        </div>
        """.format(config.DEVICE), unsafe_allow_html=True)
    
    st.markdown("### What is Transfer Learning?")
    st.markdown("""
    Transfer Learning leverages pre-trained models that have learned features from large datasets 
    (like ImageNet). By fine-tuning these models on our specific medical imaging task, we achieve:
    
    ✅ **Better Performance** - Leverages pre-learned features from millions of images  
    ✅ **Faster Training** - Requires less data and training time  
    ✅ **Reduced Overfitting** - Pre-trained features are more generalizable  
    ✅ **Medical Applicability** - Works well with limited medical datasets  
    
    ### Model Advantages for Medical Imaging
    - **ResNet50**: 50-layer deep network, balances complexity and efficiency
    - **Feature Extraction**: Captures both low-level (edges) and high-level (structures) features
    - **Backbone Freezing**: Preserves learned ImageNet features while adapting to histopathology
    """)

def show_about_page():
    """Display about and usage information."""
    st.header("📚 About This System")
    
    st.markdown("""
    ### 🎯 Purpose
    This application demonstrates the application of **Transfer Learning** and **Deep Learning** 
    to breast cancer histopathology image analysis. It's designed as an educational and 
    demonstration tool for AI-assisted medical imaging.
    
    ### ⚕️ Medical Disclaimer
    **⚠️ CRITICAL:** This tool is for **research and educational purposes only**. 
    
    - **NOT FOR CLINICAL USE** without proper validation and regulatory approval
    - **NOT A SUBSTITUTE** for professional medical diagnosis
    - **ALWAYS CONSULT** qualified pathologists and oncologists
    - **RESULTS MAY CONTAIN ERRORS** - Never rely solely on AI predictions
    
    ### 🔬 How It Works
    1. Upload a histopathology image
    2. Image is preprocessed to 224×224 pixels
    3. ResNet50 model extracts features
    4. Classification head predicts: Benign or Malignant
    5. Confidence score indicates prediction certainty
    
    ### 📊 Understanding Results
    
    **Benign (Class 0)** - No cancer cells detected
    - High confidence (>80%): Clear benign characteristic
    - Low confidence (<60%): Ambiguous features - expert review needed
    
    **Malignant (Class 1)** - Cancer cells detected
    - High confidence (>80%): Clear malignant characteristic  
    - Low confidence (<60%): Ambiguous features - expert review needed
    
    ### 💡 Best Practices
    - Use high-quality histopathology images
    - Verify results with qualified medical professionals
    - Consider the confidence score - low confidence results are less reliable
    - Keep records of all analyses
    - Never adjust doses or treatment based on AI results alone
    
    ### 👥 User Recommendations
    - **Best For:** Educational demonstrations, medical AI research
    - **Not Ideal For:** Direct clinical patient diagnosis without validation
    - **Optimal Scenario:** Supporting pathologists in routine screening and triage
    
    ### 📖 Technical Background
    """)
    
    st.info("""
    **Transfer Learning Workflow:**
    1. Start with ResNet50 pre-trained on ImageNet (1.2M images, 1000 classes)
    2. Remove final classification layer
    3. Add new layers for binary classification (2 classes)
    4. Fine-tune on histopathology dataset
    5. Deploy for inference
    
    This approach significantly outperforms training from scratch on limited medical data.
    """)
    
    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("### 📋 Recent Predictions")
        history_data = []
        for pred in st.session_state.prediction_history[-10:]:  # Last 10
            history_data.append({
                'Time': pred['timestamp'].strftime('%H:%M:%S'),
                'File': pred['filename'],
                'Prediction': pred['prediction'],
                'Confidence': f"{pred['confidence']:.2%}",
                'Risk': pred['risk_level']
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.rerun()

if __name__ == "__main__":
    main()
