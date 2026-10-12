"""
Clinical Decision Support System for Breast Cancer Detection

This module provides evidence-based clinical recommendations based on
AI predictions and confidence scores. It implements a rule-based system
that stratifies risk and suggests appropriate clinical actions.

Medical Rationale:
- High confidence predictions (≥85%) warrant immediate action
- Medium confidence (60-85%) requires specialist review
- Low confidence (<60%) necessitates additional testing
- Conservative approach prioritizes patient safety
- All recommendations emphasize human expert oversight

IMPORTANT DISCLAIMER:
This system provides decision support only. All recommendations must be
validated by qualified medical professionals. This is NOT a substitute
for clinical judgment or pathological examination.
"""

from typing import Dict, Tuple


def get_risk_level(prediction: int, confidence: float) -> str:
    """
    Determine risk level based on prediction and confidence.
    
    Risk stratification helps prioritize cases for clinical review
    and determines urgency of follow-up actions.
    
    Args:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        
    Returns:
        str: Risk level - "LOW", "MEDIUM", "HIGH", or "CRITICAL"
    """
    if prediction == 1:  # Malignant
        if confidence >= 0.85:
            return "CRITICAL"
        elif confidence >= 0.60:
            return "HIGH"
        else:
            return "MEDIUM"  # Low confidence malignant still warrants caution
    else:  # Benign
        if confidence >= 0.85:
            return "LOW"
        elif confidence >= 0.60:
            return "MEDIUM"
        else:
            return "MEDIUM"  # Low confidence benign needs verification


def get_medical_recommendation(prediction: int, confidence: float) -> str:
    """
    Generate evidence-based clinical recommendation.
    
    Recommendations follow conservative medical guidelines:
    - Benign cases: Regular monitoring with appropriate intervals
    - Malignant cases: Immediate specialist consultation
    - Borderline cases: Additional testing and expert review
    
    Args:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        
    Returns:
        str: Clinical recommendation text
    """
    if prediction == 0:  # Benign
        if confidence >= 0.85:
            return (
                "Routine screening after 12 months. "
                "Continue regular breast health monitoring. "
                "Maintain healthy lifestyle and attend scheduled screenings."
            )
        elif confidence >= 0.60:
            return (
                "Follow-up screening in 6 months recommended. "
                "While the analysis suggests benign characteristics, "
                "moderate confidence warrants closer monitoring. "
                "Consult with your healthcare provider for personalized guidance."
            )
        else:
            return (
                "Additional diagnostic testing recommended. "
                "Low confidence in benign classification requires further investigation. "
                "Schedule appointment with specialist for comprehensive evaluation "
                "including additional imaging or biopsy if indicated."
            )
    
    else:  # Malignant (prediction == 1)
        if confidence >= 0.85:
            return (
                "⚠️ IMMEDIATE ONCOLOGIST CONSULTATION REQUIRED. "
                "High confidence malignant detection indicates urgent need for "
                "specialist evaluation. Schedule appointment with oncologist immediately. "
                "Additional diagnostic procedures (biopsy, staging) will be necessary "
                "to confirm diagnosis and determine treatment plan."
            )
        elif confidence >= 0.60:
            return (
                "⚠️ RETEST AND SPECIALIST REVIEW RECOMMENDED. "
                "Moderate confidence malignant detection requires verification. "
                "Schedule immediate appointment with specialist for comprehensive evaluation. "
                "Additional imaging, biopsy, or pathological examination strongly advised "
                "to confirm findings and rule out false positive."
            )
        else:
            return (
                "⚠️ INCONCLUSIVE - ADDITIONAL BIOPSY AND EXPERT PATHOLOGIST REVIEW REQUIRED. "
                "Low confidence malignant prediction indicates ambiguous features. "
                "Immediate specialist consultation essential. "
                "Recommend tissue biopsy with expert pathological examination, "
                "additional imaging studies, and multidisciplinary review "
                "to establish accurate diagnosis."
            )


def get_followup_timeline(prediction: int, confidence: float) -> str:
    """
    Determine recommended follow-up timeline.
    
    Args:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        
    Returns:
        str: Follow-up timeline (e.g., "Immediate", "1 week", "6 months")
    """
    if prediction == 1:  # Malignant
        if confidence >= 0.60:
            return "Immediate (within 24-48 hours)"
        else:
            return "Urgent (within 1 week)"
    else:  # Benign
        if confidence >= 0.85:
            return "12 months (routine screening)"
        elif confidence >= 0.60:
            return "6 months (precautionary)"
        else:
            return "1-3 months (additional testing)"


def get_specialist_referral(prediction: int, confidence: float) -> str:
    """
    Determine appropriate specialist referral.
    
    Args:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        
    Returns:
        str: Specialist referral recommendation
    """
    if prediction == 1:  # Malignant
        return "Oncologist and Pathologist"
    else:  # Benign
        if confidence < 0.60:
            return "Radiologist or Pathologist for second opinion"
        else:
            return "Primary Care Physician (routine follow-up)"


def get_additional_tests(prediction: int, confidence: float) -> list:
    """
    Recommend additional diagnostic tests based on prediction and confidence.
    
    Args:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        
    Returns:
        list: Recommended additional tests
    """
    tests = []
    
    if prediction == 1:  # Malignant
        tests.append("Core needle biopsy for histopathological confirmation")
        tests.append("Immunohistochemistry (IHC) panel")
        
        if confidence >= 0.85:
            tests.append("Staging imaging (CT/MRI/PET scan)")
            tests.append("Tumor marker blood tests")
            tests.append("Genetic testing (BRCA1/BRCA2 if indicated)")
        else:
            tests.append("Repeat imaging with different modality")
            tests.append("Second pathologist review")
    
    else:  # Benign
        if confidence < 0.60:
            tests.append("Repeat biopsy or excisional biopsy")
            tests.append("Additional mammography or ultrasound")
            tests.append("Second pathologist review")
        elif confidence < 0.85:
            tests.append("Follow-up imaging in 6 months")
    
    return tests


def generate_comprehensive_report(prediction: int, confidence: float, 
                                  benign_prob: float = None, 
                                  malignant_prob: float = None) -> Dict[str, any]:
    """
    Generate comprehensive clinical decision support report.
    
    This function consolidates all decision support information into
    a structured format suitable for clinical documentation.
    
    Args:
        prediction: 0 (Benign) or 1 (Malignant)
        confidence: Model confidence score (0.0 to 1.0)
        benign_prob: Probability of benign classification (optional)
        malignant_prob: Probability of malignant classification (optional)
        
    Returns:
        dict: Comprehensive clinical decision support information
    """
    return {
        'prediction': 'Malignant' if prediction == 1 else 'Benign',
        'prediction_code': prediction,
        'confidence': confidence,
        'benign_probability': benign_prob if benign_prob is not None else (1 - confidence if prediction == 1 else confidence),
        'malignant_probability': malignant_prob if malignant_prob is not None else (confidence if prediction == 1 else 1 - confidence),
        'risk_level': get_risk_level(prediction, confidence),
        'recommendation': get_medical_recommendation(prediction, confidence),
        'followup_timeline': get_followup_timeline(prediction, confidence),
        'specialist_referral': get_specialist_referral(prediction, confidence),
        'additional_tests': get_additional_tests(prediction, confidence),
        'disclaimer': (
            "This AI-assisted analysis is for decision support only and must be "
            "validated by qualified medical professionals. Do not make treatment "
            "decisions based solely on this analysis. Always consult with board-certified "
            "pathologists and oncologists for definitive diagnosis and treatment planning."
        )
    }


def format_recommendation_for_display(report: Dict[str, any]) -> str:
    """
    Format the comprehensive report for human-readable display.
    
    Args:
        report: Dictionary from generate_comprehensive_report()
        
    Returns:
        str: Formatted text for display or printing
    """
    output = []
    output.append("=" * 70)
    output.append("CLINICAL DECISION SUPPORT REPORT")
    output.append("=" * 70)
    output.append("")
    
    output.append(f"PREDICTION: {report['prediction']}")
    output.append(f"CONFIDENCE: {report['confidence']:.1%}")
    output.append(f"RISK LEVEL: {report['risk_level']}")
    output.append("")
    
    output.append("PROBABILITY DISTRIBUTION:")
    output.append(f"  Benign:    {report['benign_probability']:.1%}")
    output.append(f"  Malignant: {report['malignant_probability']:.1%}")
    output.append("")
    
    output.append("CLINICAL RECOMMENDATION:")
    output.append("-" * 70)
    output.append(report['recommendation'])
    output.append("")
    
    output.append(f"FOLLOW-UP TIMELINE: {report['followup_timeline']}")
    output.append(f"SPECIALIST REFERRAL: {report['specialist_referral']}")
    output.append("")
    
    if report['additional_tests']:
        output.append("RECOMMENDED ADDITIONAL TESTS:")
        for i, test in enumerate(report['additional_tests'], 1):
            output.append(f"  {i}. {test}")
        output.append("")
    
    output.append("IMPORTANT DISCLAIMER:")
    output.append("-" * 70)
    output.append(report['disclaimer'])
    output.append("")
    output.append("=" * 70)
    
    return "\n".join(output)


if __name__ == "__main__":
    """
    Test the clinical decision support system with various scenarios.
    """
    print("Testing Clinical Decision Support System")
    print("=" * 70)
    print()
    
    # Test scenarios
    test_cases = [
        (0, 0.95, "Benign - High Confidence"),
        (0, 0.75, "Benign - Medium Confidence"),
        (0, 0.50, "Benign - Low Confidence"),
        (1, 0.92, "Malignant - High Confidence"),
        (1, 0.70, "Malignant - Medium Confidence"),
        (1, 0.55, "Malignant - Low Confidence"),
    ]
    
    for prediction, confidence, description in test_cases:
        print(f"\nSCENARIO: {description}")
        print("-" * 70)
        
        report = generate_comprehensive_report(prediction, confidence)
        formatted = format_recommendation_for_display(report)
        print(formatted)
        print("\n" + "=" * 70)
        print()
    
    print("\nAll test scenarios completed successfully!")
