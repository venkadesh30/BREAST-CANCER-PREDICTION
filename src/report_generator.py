"""
Medical Report Generator for Breast Cancer Detection System

This module generates professional PDF diagnostic reports containing:
- Patient information and report metadata
- Original histopathology image
- AI prediction results with confidence scores
- Explainability visualizations (Grad-CAM heatmaps)
- Clinical recommendations and risk assessment
- Technical details and disclaimers

Reports are suitable for clinical documentation and archival purposes.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import numpy as np

# Create output directory
REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           'outputs', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for adding page numbers and headers/footers."""
    
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.grey)
        self.drawRightString(
            letter[0] - 0.5 * inch,
            0.5 * inch,
            f"Page {self._pageNumber} of {page_count}"
        )
        self.drawString(
            0.5 * inch,
            0.5 * inch,
            "AI-Assisted Breast Cancer Detection Report"
        )


def create_probability_chart(benign_prob, malignant_prob, save_path):
    """
    Create a bar chart showing probability distribution.
    
    Args:
        benign_prob: Probability of benign classification
        malignant_prob: Probability of malignant classification
        save_path: Path to save the chart image
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    
    categories = ['Benign', 'Malignant']
    probabilities = [benign_prob, malignant_prob]
    colors_list = ['#4CAF50', '#f44336']
    
    bars = ax.bar(categories, probabilities, color=colors_list, alpha=0.7, edgecolor='black')
    
    # Add percentage labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Probability', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.set_title('Classification Probability Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def generate_pdf_report(image_path, prediction, confidence, recommendation, 
                       explanation_summary, patient_id=None, 
                       benign_prob=None, malignant_prob=None):
    """
    Generate a comprehensive PDF diagnostic report.
    
    Args:
        image_path: Path to the original histopathology image
        prediction: Predicted class (0=Benign, 1=Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        recommendation: Clinical recommendation text
        explanation_summary: Dictionary with paths to explanation images
                           {'gradcam': path, 'shap': path, 'integrated_gradients': path}
        patient_id: Optional patient identifier
        benign_prob: Probability of benign (if None, computed from prediction/confidence)
        malignant_prob: Probability of malignant (if None, computed from prediction/confidence)
        
    Returns:
        str: Path to the generated PDF report
    """
    # Generate report filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    patient_suffix = f"_{patient_id}" if patient_id else ""
    report_filename = f"report_{timestamp}{patient_suffix}.pdf"
    report_path = os.path.join(REPORTS_DIR, report_filename)
    
    # Compute probabilities if not provided
    if benign_prob is None or malignant_prob is None:
        if prediction == 1:  # Malignant
            malignant_prob = confidence
            benign_prob = 1 - confidence
        else:  # Benign
            benign_prob = confidence
            malignant_prob = 1 - confidence
    
    # Create probability chart
    chart_path = os.path.join(REPORTS_DIR, f"chart_{timestamp}.png")
    create_probability_chart(benign_prob, malignant_prob, chart_path)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        report_path,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1976d2'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#424242'),
        spaceAfter=10,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#616161'),
        spaceAfter=8,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )
    
    # ==================== HEADER ====================
    elements.append(Paragraph("BREAST CANCER DETECTION", title_style))
    elements.append(Paragraph("AI-Assisted Histopathology Analysis Report", styles['Heading3']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata table
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_id = f"BCR-{timestamp}"
    
    metadata = [
        ['Report ID:', report_id],
        ['Generated:', report_time],
        ['Patient ID:', patient_id if patient_id else 'N/A'],
        ['Image File:', os.path.basename(image_path)]
    ]
    
    metadata_table = Table(metadata, colWidths=[1.5*inch, 4*inch])
    metadata_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#424242')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5'))
    ]))
    
    elements.append(metadata_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # ==================== DIAGNOSTIC RESULTS ====================
    elements.append(Paragraph("DIAGNOSTIC RESULTS", heading_style))
    
    # Prediction result box
    prediction_text = "MALIGNANT" if prediction == 1 else "BENIGN"
    prediction_color = colors.HexColor('#d32f2f') if prediction == 1 else colors.HexColor('#388e3c')
    
    result_data = [
        ['Classification:', prediction_text],
        ['Confidence Score:', f'{confidence:.1%}'],
        ['Benign Probability:', f'{benign_prob:.1%}'],
        ['Malignant Probability:', f'{malignant_prob:.1%}']
    ]
    
    result_table = Table(result_data, colWidths=[2*inch, 3.5*inch])
    result_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (1, 0), (1, 0), 14),
        ('TEXTCOLOR', (1, 0), (1, 0), prediction_color),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('BACKGROUND', (1, 0), (1, 0), colors.HexColor('#fff9c4') if prediction == 0 else colors.HexColor('#ffebee'))
    ]))
    
    elements.append(result_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Probability chart
    if os.path.exists(chart_path):
        elements.append(Paragraph("Probability Distribution", subheading_style))
        chart_img = Image(chart_path, width=4.5*inch, height=2.25*inch)
        elements.append(chart_img)
        elements.append(Spacer(1, 0.2*inch))
    
    # ==================== IMAGE ANALYSIS ====================
    elements.append(Paragraph("HISTOPATHOLOGY IMAGE ANALYSIS", heading_style))
    
    # Original image
    if os.path.exists(image_path):
        elements.append(Paragraph("Original Histopathology Image", subheading_style))
        try:
            img = Image(image_path, width=3.5*inch, height=3.5*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            elements.append(Paragraph(f"Error loading image: {str(e)}", normal_style))
    
    # Grad-CAM heatmap
    if explanation_summary and explanation_summary.get('gradcam') and os.path.exists(explanation_summary['gradcam']):
        elements.append(Paragraph("Explainability: Grad-CAM Heatmap", subheading_style))
        elements.append(Paragraph(
            "The heatmap below highlights the tissue regions that most influenced the AI's decision. "
            "Red/yellow areas indicate high importance, while blue areas indicate low importance.",
            normal_style
        ))
        try:
            gradcam_img = Image(explanation_summary['gradcam'], width=5*inch, height=2.5*inch)
            elements.append(gradcam_img)
            elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            elements.append(Paragraph(f"Error loading Grad-CAM: {str(e)}", normal_style))
    
    # ==================== CLINICAL RECOMMENDATIONS ====================
    elements.append(PageBreak())
    elements.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    
    # Recommendation text
    rec_para = Paragraph(recommendation, normal_style)
    elements.append(rec_para)
    elements.append(Spacer(1, 0.2*inch))
    
    # Risk assessment
    from decision_support import get_risk_level, get_followup_timeline, get_specialist_referral
    
    risk_level = get_risk_level(prediction, confidence)
    followup = get_followup_timeline(prediction, confidence)
    specialist = get_specialist_referral(prediction, confidence)
    
    risk_data = [
        ['Risk Level:', risk_level],
        ['Follow-up Timeline:', followup],
        ['Specialist Referral:', specialist]
    ]
    
    risk_table = Table(risk_data, colWidths=[2*inch, 3.5*inch])
    risk_color = {
        'LOW': colors.HexColor('#c8e6c9'),
        'MEDIUM': colors.HexColor('#fff9c4'),
        'HIGH': colors.HexColor('#ffccbc'),
        'CRITICAL': colors.HexColor('#ffcdd2')
    }.get(risk_level, colors.white)
    
    risk_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
        ('BACKGROUND', (1, 0), (1, 0), risk_color)
    ]))
    
    elements.append(risk_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # ==================== DISCLAIMER ====================
    elements.append(Paragraph("IMPORTANT MEDICAL DISCLAIMER", heading_style))
    
    disclaimer_text = """
    <b>This report is generated by an AI-assisted diagnostic system and is intended for 
    decision support purposes only.</b><br/><br/>
    
    <b>Key Points:</b><br/>
    • This analysis must be validated by qualified medical professionals<br/>
    • Do NOT make treatment decisions based solely on this AI analysis<br/>
    • Always consult with board-certified pathologists and oncologists<br/>
    • This system is not a substitute for professional medical judgment<br/>
    • False positives and false negatives can occur<br/>
    • Clinical correlation and additional diagnostic tests are essential<br/><br/>
    
    <b>Regulatory Status:</b><br/>
    This AI system is for research and educational purposes. It has not been approved 
    by regulatory agencies (FDA, CE, etc.) for clinical diagnostic use.<br/><br/>
    
    <b>Professional Review Required:</b><br/>
    All findings must be confirmed through standard pathological examination and 
    clinical evaluation by licensed healthcare providers.
    """
    
    disclaimer_para = Paragraph(disclaimer_text, normal_style)
    
    disclaimer_box = Table([[disclaimer_para]], colWidths=[6*inch])
    disclaimer_box.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor('#ff9800')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fff3e0')),
        ('PADDING', (0, 0), (-1, -1), 12)
    ]))
    
    elements.append(disclaimer_box)
    elements.append(Spacer(1, 0.3*inch))
    
    # ==================== TECHNICAL DETAILS ====================
    elements.append(Paragraph("TECHNICAL DETAILS", heading_style))
    
    tech_details = f"""
    <b>Model Architecture:</b> ResNet50 with Transfer Learning<br/>
    <b>Training Dataset:</b> Histopathology images (Benign/Malignant classification)<br/>
    <b>Input Image Size:</b> 224 × 224 pixels<br/>
    <b>Preprocessing:</b> ImageNet normalization<br/>
    <b>Explainability Methods:</b> Grad-CAM, SHAP, Integrated Gradients<br/>
    <b>Software Version:</b> Breast Cancer Detection System v1.0<br/>
    <b>Processing Timestamp:</b> {report_time}
    """
    
    elements.append(Paragraph(tech_details, normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # ==================== FOOTER ====================
    footer_text = """
    <i>This report was automatically generated by the AI-Assisted Breast Cancer Detection System. 
    For questions or concerns, please consult with your healthcare provider.</i>
    """
    elements.append(Paragraph(footer_text, styles['Italic']))
    
    # Build PDF
    try:
        doc.build(elements, canvasmaker=NumberedCanvas)
        print(f"✓ PDF report generated: {report_path}")
        
        # Clean up temporary chart
        if os.path.exists(chart_path):
            os.remove(chart_path)
        
        return report_path
    
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        return None


if __name__ == "__main__":
    """
    Test the report generator with sample data.
    """
    print("Testing PDF Report Generator")
    print("=" * 70)
    
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
        exit(1)
    
    print(f"Using sample image: {sample_image}")
    
    # Test case 1: Malignant with high confidence
    print("\nTest Case 1: Malignant - High Confidence")
    report_path = generate_pdf_report(
        image_path=sample_image,
        prediction=1,
        confidence=0.92,
        recommendation="⚠️ IMMEDIATE ONCOLOGIST CONSULTATION REQUIRED. High confidence malignant detection.",
        explanation_summary={'gradcam': None, 'shap': None, 'integrated_gradients': None},
        patient_id="TEST001",
        benign_prob=0.08,
        malignant_prob=0.92
    )
    
    if report_path:
        print(f"✓ Report generated: {report_path}")
    
    # Test case 2: Benign with high confidence
    print("\nTest Case 2: Benign - High Confidence")
    report_path = generate_pdf_report(
        image_path=sample_image,
        prediction=0,
        confidence=0.88,
        recommendation="Routine screening after 12 months. Continue regular breast health monitoring.",
        explanation_summary={'gradcam': None, 'shap': None, 'integrated_gradients': None},
        patient_id="TEST002",
        benign_prob=0.88,
        malignant_prob=0.12
    )
    
    if report_path:
        print(f"✓ Report generated: {report_path}")
    
    print("\n" + "=" * 70)
    print("PDF Report Generator test completed!")
    print(f"Reports saved in: {REPORTS_DIR}")
