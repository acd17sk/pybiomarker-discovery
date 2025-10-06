import torch
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class ClinicalReportGenerator:
    """Generate interpretable clinical reports from biomarker predictions"""
    
    def __init__(self, disease_names: List[str], biomarker_names: Dict[str, List[str]]):
        self.disease_names = disease_names
        self.biomarker_names = biomarker_names
        
    def generate_report(self,
                       predictions: Dict[str, torch.Tensor],
                       patient_info: Dict[str, Any],
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive clinical report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'patient_info': patient_info,
            'risk_assessment': self._assess_risk(predictions),
            'biomarker_analysis': self._analyze_biomarkers(predictions),
            'recommendations': self._generate_recommendations(predictions),
            'confidence_analysis': self._analyze_confidence(predictions)
        }
        
        # Generate visualizations
        if save_path:
            self._create_visualizations(predictions, report, save_path)
        
        return report
    
    def _assess_risk(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Assess disease risk from predictions"""
        probs = predictions['probabilities'].cpu().numpy()
        
        risk_scores = {}
        for i, disease in enumerate(self.disease_names):
            risk_scores[disease] = {
                'probability': float(probs[0, i]),
                'risk_level': self._categorize_risk(probs[0, i]),
                'percentile': self._calculate_percentile(probs[0, i])
            }
        
        # Top risks
        top_risks_idx = np.argsort(probs[0])[-3:][::-1]
        top_risks = [
            {
                'disease': self.disease_names[idx],
                'probability': float(probs[0, idx])
            }
            for idx in top_risks_idx if probs[0, idx] > 0.1
        ]
        
        return {
            'risk_scores': risk_scores,
            'top_risks': top_risks,
            'overall_health_score': float(1.0 - np.max(probs[0, 1:]))  # Exclude healthy
        }
    
    def _analyze_biomarkers(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Analyze individual biomarkers"""
        biomarkers = predictions.get('biomarkers', {})
        
        analysis = {}
        for modality, markers in biomarkers.items():
            if isinstance(markers, dict):
                analysis[modality] = {}
                for marker_name, values in markers.items():
                    values_np = values.cpu().numpy()
                    analysis[modality][marker_name] = {
                        'value': float(np.mean(values_np)),
                        'std': float(np.std(values_np)),
                        'interpretation': self._interpret_biomarker(
                            marker_name, np.mean(values_np)
                        )
                    }
        
        return analysis
    
    def _generate_recommendations(self, predictions: Dict[str, torch.Tensor]) -> List[str]:
        """Generate clinical recommendations based on predictions"""
        recommendations = []
        probs = predictions['probabilities'].cpu().numpy()
        
        # High-risk conditions
        for i, disease in enumerate(self.disease_names):
            if probs[0, i] > 0.7 and disease != 'healthy':
                recommendations.append(
                    f"High risk for {disease}. Recommend specialist consultation."
                )
            elif probs[0, i] > 0.4 and disease != 'healthy':
                recommendations.append(
                    f"Moderate risk for {disease}. Consider follow-up screening."
                )
        
        # Biomarker-specific recommendations
        if 'biomarkers' in predictions:
            biomarkers = predictions['biomarkers']
            
            # Voice biomarkers
            if 'voice' in biomarkers and 'tremor_score' in biomarkers['voice']:
                tremor = biomarkers['voice']['tremor_score'].item()
                if tremor > 0.7:
                    recommendations.append(
                        "Significant voice tremor detected. Neurological evaluation recommended."
                    )
            
            # Movement biomarkers
            if 'movement' in biomarkers and 'bradykinesia' in biomarkers['movement']:
                brady = biomarkers['movement']['bradykinesia'].mean().item()
                if brady > 0.6:
                    recommendations.append(
                        "Bradykinesia detected. Movement disorder specialist consultation advised."
                    )
        
        if not recommendations:
            recommendations.append("Continue regular health monitoring.")
        
        return recommendations
    
    def _analyze_confidence(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze prediction confidence and uncertainty"""
        confidence_analysis = {}
        
        if 'confidence' in predictions:
            confidence = predictions['confidence'].cpu().numpy()
            confidence_analysis['mean_confidence'] = float(np.mean(confidence))
            confidence_analysis['min_confidence'] = float(np.min(confidence))
        
        if 'uncertainty' in predictions:
            uncertainty = predictions['uncertainty'].cpu().numpy()
            confidence_analysis['mean_uncertainty'] = float(np.mean(uncertainty))
            confidence_analysis['max_uncertainty'] = float(np.max(uncertainty))
        
        # Entropy of predictions
        probs = predictions['probabilities'].cpu().numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)
        confidence_analysis['prediction_entropy'] = float(entropy[0])
        
        return confidence_analysis
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk level based on probability"""
        if probability < 0.2:
            return "Low"
        elif probability < 0.5:
            return "Moderate"
        elif probability < 0.8:
            return "High"
        else:
            return "Very High"
    
    def _calculate_percentile(self, probability: float) -> float:
        """Calculate risk percentile (placeholder)"""
        # In practice, compare with population statistics
        return min(99.0, probability * 100)
    
    def _interpret_biomarker(self, marker_name: str, value: float) -> str:
        """Interpret biomarker value"""
        interpretations = {
            'tremor_score': {
                0.0: "No tremor detected",
                0.3: "Mild tremor",
                0.6: "Moderate tremor",
                0.8: "Severe tremor"
            },
            'cognitive_load': {
                0.0: "Low cognitive load",
                0.3: "Normal cognitive load",
                0.6: "High cognitive load",
                0.8: "Very high cognitive load"
            }
        }
        
        if marker_name in interpretations:
            thresholds = interpretations[marker_name]
            for threshold in sorted(thresholds.keys(), reverse=True):
                if value >= threshold:
                    return thresholds[threshold]
        
        return f"Value: {value:.3f}"
    
    def _create_visualizations(self,
                              predictions: Dict[str, torch.Tensor],
                              report: Dict[str, Any],
                              save_path: str):
        """Create visualization plots for the report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Disease risk bar chart
        ax = axes[0, 0]
        diseases = list(report['risk_assessment']['risk_scores'].keys())
        risks = [report['risk_assessment']['risk_scores'][d]['probability'] 
                for d in diseases]
        ax.bar(diseases, risks)
        ax.set_title('Disease Risk Scores')
        ax.set_ylabel('Probability')
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        
        # Biomarker heatmap
        ax = axes[0, 1]
        if 'biomarker_analysis' in report:
            biomarker_data = []
            biomarker_labels = []
            for modality, markers in report['biomarker_analysis'].items():
                for marker, data in markers.items():
                    biomarker_data.append(data['value'])
                    biomarker_labels.append(f"{modality}_{marker}")
            
            if biomarker_data:
                sns.heatmap(
                    np.array(biomarker_data).reshape(-1, 1),
                    ax=ax,
                    yticklabels=biomarker_labels,
                    cmap='RdYlBu_r',
                    center=0.5,
                    vmin=0,
                    vmax=1
                )
                ax.set_title('Biomarker Values')
        
        # Confidence distribution
        ax = axes[1, 0]
        if 'confidence' in predictions:
            confidence = predictions['confidence'].cpu().numpy().flatten()
            ax.hist(confidence, bins=20, alpha=0.7)
            ax.set_title('Prediction Confidence Distribution')
            ax.set_xlabel('Confidence')
            ax.set_ylabel('Frequency')
        
        # Feature importance (if available)
        ax = axes[1, 1]
        if 'attention_scores' in predictions:
            attention = predictions['attention_scores'].cpu().numpy()[0]
            ax.bar(range(len(attention)), attention)
            ax.set_title('Disease-Specific Attention Scores')
            ax.set_xlabel('Disease Class')
            ax.set_ylabel('Attention Weight')
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/clinical_report_visualizations.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_to_json(self, report: Dict[str, Any], filepath: str):
        """Export report to JSON format"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def export_to_html(self, report: Dict[str, Any], filepath: str):
        """Export report to HTML format"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Biomarker Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1, h2 { color: #333; }
                .risk-high { color: #d9534f; font-weight: bold; }
                .risk-moderate { color: #f0ad4e; }
                .risk-low { color: #5cb85c; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .recommendation { 
                    background-color: #e7f3ff; 
                    padding: 10px; 
                    margin: 10px 0;
                    border-left: 4px solid #2196F3;
                }
            </style>
        </head>
        <body>
            <h1>Clinical Biomarker Analysis Report</h1>
            <p><strong>Generated:</strong> {timestamp}</p>
            
            <h2>Patient Information</h2>
            <p><strong>ID:</strong> {patient_id}</p>
            <p><strong>Age:</strong> {age}</p>
            <p><strong>Gender:</strong> {gender}</p>
            
            <h2>Risk Assessment</h2>
            <table>
                <tr>
                    <th>Disease</th>
                    <th>Risk Level</th>
                    <th>Probability</th>
                </tr>
                {risk_rows}
            </table>
            
            <h2>Clinical Recommendations</h2>
            {recommendations}
            
            <h2>Confidence Analysis</h2>
            <p><strong>Mean Confidence:</strong> {mean_confidence:.2%}</p>
            <p><strong>Prediction Entropy:</strong> {entropy:.3f}</p>
        </body>
        </html>
        """
        
        # Format risk rows
        risk_rows = ""
        for disease, data in report['risk_assessment']['risk_scores'].items():
            risk_class = f"risk-{data['risk_level'].lower()}"
            risk_rows += f"""
                <tr>
                    <td>{disease}</td>
                    <td class="{risk_class}">{data['risk_level']}</td>
                    <td>{data['probability']:.2%}</td>
                </tr>
            """
        
        # Format recommendations
        recommendations_html = ""
        for rec in report['recommendations']:
            recommendations_html += f'<div class="recommendation">{rec}</div>'
        
        # Fill template
        html_content = html_template.format(
            timestamp=report['timestamp'],
            patient_id=report['patient_info'].get('patient_id', 'N/A'),
            age=report['patient_info'].get('age', 'N/A'),
            gender=report['patient_info'].get('gender', 'N/A'),
            risk_rows=risk_rows,
            recommendations=recommendations_html,
            mean_confidence=report['confidence_analysis'].get('mean_confidence', 0),
            entropy=report['confidence_analysis'].get('prediction_entropy', 0)
        )
        
        with open(filepath, 'w') as f:
            f.write(html_content)