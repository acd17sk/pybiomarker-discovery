"""Custom metrics for biomarker evaluation"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    roc_auc_score, confusion_matrix, roc_curve,
    cohen_kappa_score, matthews_corrcoef
)
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ClinicalMetrics:
    """Clinical evaluation metrics for biomarker models"""
    
    def __init__(self, 
                 disease_names: Optional[List[str]] = None,
                 threshold: float = 0.5):
        self.disease_names = disease_names
        self.threshold = threshold
        
    def calculate_all(self,
                     y_true: Union[np.ndarray, torch.Tensor],
                     y_pred: Union[np.ndarray, torch.Tensor],
                     y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Dict[str, float]:
        """Calculate all clinical metrics"""
        # Convert to numpy
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = self.accuracy(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Calculate for each disease
        num_classes = len(np.unique(y_true))
        for i in range(num_classes):
            disease = self.disease_names[i] if self.disease_names else f"class_{i}"
            
            # Binary metrics for this class
            binary_true = (y_true == i).astype(int)
            binary_pred = (y_pred == i).astype(int)
            
            metrics[f'{disease}_precision'] = precision[i]
            metrics[f'{disease}_recall'] = recall[i]
            metrics[f'{disease}_f1'] = f1[i]
            metrics[f'{disease}_support'] = support[i]
            
            # Sensitivity and Specificity
            tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred).ravel()
            
            metrics[f'{disease}_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics[f'{disease}_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics[f'{disease}_ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            metrics[f'{disease}_npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            # Diagnostic Odds Ratio
            dor = self.diagnostic_odds_ratio(tp, tn, fp, fn)
            metrics[f'{disease}_dor'] = dor
            
            # Youden's Index
            metrics[f'{disease}_youden'] = metrics[f'{disease}_sensitivity'] + metrics[f'{disease}_specificity'] - 1
            
            # AUC if probabilities available
            if y_prob is not None and len(y_prob.shape) > 1:
                try:
                    auc = roc_auc_score(binary_true, y_prob[:, i])
                    metrics[f'{disease}_auc'] = auc
                except:
                    metrics[f'{disease}_auc'] = 0.5
        
        # Overall metrics
        metrics['macro_f1'] = np.mean(f1)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Multi-class AUC if probabilities available
        if y_prob is not None:
            try:
                metrics['macro_auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
            except:
                metrics['macro_auc'] = 0.5
        
        return metrics
    
    def sensitivity_at_specificity(self,
                                  y_true: np.ndarray,
                                  y_score: np.ndarray,
                                  target_specificity: float = 0.95) -> float:
        """Calculate sensitivity at a given specificity"""
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        specificity = 1 - fpr
        
        # Find sensitivity at target specificity
        idx = np.argmin(np.abs(specificity - target_specificity))
        return tpr[idx]
    
    def diagnostic_odds_ratio(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """Calculate Diagnostic Odds Ratio (DOR)"""
        if fp == 0 or fn == 0:
            # Add small epsilon to avoid division by zero
            fp = max(fp, 0.5)
            fn = max(fn, 0.5)
        
        if tp == 0 or tn == 0:
            return 0.0
        
        dor = (tp * tn) / (fp * fn)
        return dor
    
    def number_needed_to_diagnose(self,
                                  sensitivity: float,
                                  specificity: float) -> float:
        """Calculate Number Needed to Diagnose (NND)"""
        if sensitivity + specificity <= 1:
            return float('inf')
        
        nnd = 1 / (sensitivity - (1 - specificity))
        return abs(nnd)
    
    def likelihood_ratios(self,
                         sensitivity: float,
                         specificity: float) -> Tuple[float, float]:
        """Calculate positive and negative likelihood ratios"""
        # Positive likelihood ratio
        if specificity == 1:
            lr_pos = float('inf')
        else:
            lr_pos = sensitivity / (1 - specificity)
        
        # Negative likelihood ratio
        if specificity == 0:
            lr_neg = float('inf')
        else:
            lr_neg = (1 - sensitivity) / specificity
        
        return lr_pos, lr_neg
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy"""
        return accuracy_score(y_true, y_pred)
    
    def _to_numpy(self, tensor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert tensor to numpy array"""
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor
    
    def clinical_agreement(self,
                          rater1: np.ndarray,
                          rater2: np.ndarray) -> Dict[str, float]:
        """Calculate inter-rater agreement metrics"""
        # Cohen's Kappa
        kappa = cohen_kappa_score(rater1, rater2)
        
        # Percentage agreement
        agreement = np.mean(rater1 == rater2)
        
        # Gwet's AC1 (less affected by prevalence)
        po = agreement  # Observed agreement
        
        # Calculate expected agreement for AC1
        n_categories = len(np.unique(np.concatenate([rater1, rater2])))
        pe_ac1 = (1 / n_categories) * (1 - 1 / n_categories)
        
        ac1 = (po - pe_ac1) / (1 - pe_ac1) if pe_ac1 < 1 else 0
        
        return {
            'cohen_kappa': kappa,
            'percentage_agreement': agreement,
            'gwet_ac1': ac1
        }


class BiomarkerMetrics:
    """Metrics specific to biomarker reliability and validity"""
    
    def __init__(self):
        pass
    
    def test_retest_reliability(self,
                               measurements1: np.ndarray,
                               measurements2: np.ndarray) -> Dict[str, float]:
        """Calculate test-retest reliability metrics"""
        # Intraclass Correlation Coefficient (ICC)
        icc = self._calculate_icc(measurements1, measurements2)
        
        # Pearson correlation
        correlation, p_value = stats.pearsonr(measurements1.flatten(), 
                                             measurements2.flatten())
        
        # Coefficient of Variation
        cv = self._coefficient_of_variation(measurements1, measurements2)
        
        # Bland-Altman analysis
        mean_diff, std_diff, limits_of_agreement = self._bland_altman(
            measurements1, measurements2
        )
        
        return {
            'icc': icc,
            'pearson_r': correlation,
            'pearson_p': p_value,
            'coefficient_of_variation': cv,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'lower_loa': limits_of_agreement[0],
            'upper_loa': limits_of_agreement[1]
        }
    
    def _calculate_icc(self,
                      measurements1: np.ndarray,
                      measurements2: np.ndarray,
                      icc_type: str = 'icc_2_1') -> float:
        """
        Calculate Intraclass Correlation Coefficient
        
        icc_type options:
        - 'icc_1_1': One-way random effects, single measurement
        - 'icc_2_1': Two-way random effects, single measurement
        - 'icc_3_1': Two-way mixed effects, single measurement
        """
        # Reshape data for ICC calculation
        n = len(measurements1)
        data = np.array([measurements1, measurements2]).T
        
        # Calculate mean squares
        grand_mean = np.mean(data)
        
        # Between-subject mean square
        subject_means = np.mean(data, axis=1)
        ms_between = np.sum((subject_means - grand_mean) ** 2) * 2 / (n - 1)
        
        # Within-subject mean square
        ms_within = np.sum((data - subject_means.reshape(-1, 1)) ** 2) / n
        
        # Calculate ICC based on type
        if icc_type == 'icc_1_1':
            icc = (ms_between - ms_within) / (ms_between + ms_within)
        elif icc_type == 'icc_2_1':
            icc = (ms_between - ms_within) / (ms_between + ms_within)
        elif icc_type == 'icc_3_1':
            icc = (ms_between - ms_within) / ms_between
        else:
            icc = 0.0
        
        return max(0, min(1, icc))  # Bound between 0 and 1
    
    def _coefficient_of_variation(self,
                                 measurements1: np.ndarray,
                                 measurements2: np.ndarray) -> float:
        """Calculate coefficient of variation for repeated measurements"""
        # Pool measurements
        all_measurements = np.concatenate([measurements1.flatten(), 
                                          measurements2.flatten()])
        
        # Calculate CV
        mean = np.mean(all_measurements)
        std = np.std(all_measurements)
        
        cv = (std / mean) * 100 if mean != 0 else 0
        return cv
    
    def _bland_altman(self,
                     measurements1: np.ndarray,
                     measurements2: np.ndarray) -> Tuple[float, float, Tuple[float, float]]:
        """Perform Bland-Altman analysis"""
        # Calculate differences and means
        differences = measurements1 - measurements2
        means = (measurements1 + measurements2) / 2
        
        # Calculate statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        # 95% limits of agreement
        lower_loa = mean_diff - 1.96 * std_diff
        upper_loa = mean_diff + 1.96 * std_diff
        
        return mean_diff, std_diff, (lower_loa, upper_loa)
    
    def minimal_detectable_change(self,
                                 test_std: float,
                                 reliability: float,
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate Minimal Detectable Change (MDC)
        
        Args:
            test_std: Standard deviation of the test
            reliability: Test-retest reliability coefficient (e.g., ICC)
            confidence_level: Confidence level for MDC
        """
        # Standard error of measurement
        sem = test_std * np.sqrt(1 - reliability)
        
        # Z-score for confidence level
        if confidence_level == 0.95:
            z = 1.96
        elif confidence_level == 0.90:
            z = 1.645
        else:
            z = stats.norm.ppf((1 + confidence_level) / 2)
        
        # MDC
        mdc = sem * z * np.sqrt(2)
        return mdc
    
    def responsiveness(self,
                      pre_scores: np.ndarray,
                      post_scores: np.ndarray,
                      external_criterion: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate responsiveness metrics for detecting change
        
        Args:
            pre_scores: Scores before intervention
            post_scores: Scores after intervention
            external_criterion: External measure of change (optional)
        """
        # Effect size (Cohen's d)
        change = post_scores - pre_scores
        effect_size = np.mean(change) / np.std(pre_scores)
        
        # Standardized Response Mean (SRM)
        srm = np.mean(change) / np.std(change)
        
        # Guyatt's Responsiveness Index
        if external_criterion is not None:
            # Split by external criterion (e.g., improved vs not improved)
            improved = external_criterion > 0
            
            if np.any(improved) and np.any(~improved):
                change_improved = change[improved]
                change_stable = change[~improved]
                
                gri = np.mean(change_improved) / np.std(change_stable)
            else:
                gri = 0.0
        else:
            gri = None
        
        results = {
            'effect_size': effect_size,
            'standardized_response_mean': srm,
            'mean_change': np.mean(change),
            'std_change': np.std(change)
        }
        
        if gri is not None:
            results['guyatt_responsiveness'] = gri
        
        return results
    
    def construct_validity(self,
                          biomarker_scores: np.ndarray,
                          criterion_scores: np.ndarray,
                          hypothesis_direction: str = 'positive') -> Dict[str, float]:
        """
        Assess construct validity against criterion measure
        
        Args:
            biomarker_scores: Biomarker measurements
            criterion_scores: Gold standard or criterion measurements
            hypothesis_direction: Expected direction of correlation
        """
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(biomarker_scores, criterion_scores)
        
        # Spearman correlation (for non-linear relationships)
        r_spearman, p_spearman = stats.spearmanr(biomarker_scores, criterion_scores)
        
        # Check if correlation matches hypothesis
        if hypothesis_direction == 'positive':
            hypothesis_confirmed = r_pearson > 0 and p_pearson < 0.05
        elif hypothesis_direction == 'negative':
            hypothesis_confirmed = r_pearson < 0 and p_pearson < 0.05
        else:
            hypothesis_confirmed = p_pearson < 0.05
        
        return {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'hypothesis_confirmed': hypothesis_confirmed
        }
    
    def biomarker_stability(self,
                           measurements: np.ndarray,
                           time_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Assess biomarker stability over time
        
        Args:
            measurements: Array of measurements (subjects x time_points)
            time_points: Optional array of time values
        """
        if len(measurements.shape) == 1:
            measurements = measurements.reshape(-1, 1)
        
        n_subjects, n_timepoints = measurements.shape
        
        # Within-subject variability
        within_subject_cv = []
        for i in range(n_subjects):
            subject_data = measurements[i, :]
            if np.mean(subject_data) != 0:
                cv = (np.std(subject_data) / np.mean(subject_data)) * 100
                within_subject_cv.append(cv)
        
        mean_within_cv = np.mean(within_subject_cv) if within_subject_cv else 0
        
        # Between-subject variability
        between_subject_cv = []
        for t in range(n_timepoints):
            timepoint_data = measurements[:, t]
            if np.mean(timepoint_data) != 0:
                cv = (np.std(timepoint_data) / np.mean(timepoint_data)) * 100
                between_subject_cv.append(cv)
        
        mean_between_cv = np.mean(between_subject_cv) if between_subject_cv else 0
        
        # Trend analysis if time points provided
        if time_points is not None and len(time_points) == n_timepoints:
            trends = []
            for i in range(n_subjects):
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    time_points, measurements[i, :]
                )
                trends.append(slope)
            
            mean_trend = np.mean(trends)
            trend_p_value = stats.ttest_1samp(trends, 0).pvalue
        else:
            mean_trend = 0
            trend_p_value = 1.0
        
        return {
            'mean_within_subject_cv': mean_within_cv,
            'mean_between_subject_cv': mean_between_cv,
            'stability_ratio': mean_within_cv / mean_between_cv if mean_between_cv > 0 else float('inf'),
            'mean_trend': mean_trend,
            'trend_p_value': trend_p_value
        }


def calculate_clinical_metrics(y_true: Union[np.ndarray, torch.Tensor],
                              y_pred: Union[np.ndarray, torch.Tensor],
                              y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None,
                              disease_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calculate comprehensive clinical metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        disease_names: Names of diseases/classes (optional)
    """
    metrics_calculator = ClinicalMetrics(disease_names=disease_names)
    return metrics_calculator.calculate_all(y_true, y_pred, y_prob)


def calculate_biomarker_reliability(test_measurements: np.ndarray,
                                   retest_measurements: np.ndarray) -> Dict[str, float]:
    """
    Calculate biomarker reliability metrics
    
    Args:
        test_measurements: First set of measurements
        retest_measurements: Second set of measurements
    """
    metrics_calculator = BiomarkerMetrics()
    return metrics_calculator.test_retest_reliability(test_measurements, retest_measurements)


def calculate_roc_metrics(y_true: np.ndarray,
                         y_score: np.ndarray,
                         pos_label: int = 1) -> Dict[str, Any]:
    """
    Calculate ROC curve metrics
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities
        pos_label: Positive class label
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=pos_label)
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_score)
    
    # Find optimal threshold (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    
    # Calculate metrics at optimal threshold
    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
    
    return {
        'auc': auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold,
        'optimal_sensitivity': optimal_tpr,
        'optimal_specificity': 1 - optimal_fpr,
        'optimal_accuracy': (tp + tn) / (tp + tn + fp + fn)
    }


def calculate_calibration_metrics(y_true: np.ndarray,
                                 y_prob: np.ndarray,
                                 n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate calibration metrics for probability predictions
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration plot
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate calibration data
    bin_centers = []
    bin_true_probs = []
    bin_predicted_probs = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_true_probs.append(np.mean(y_true[in_bin]))
            bin_predicted_probs.append(np.mean(y_prob[in_bin]))
            bin_counts.append(np.sum(in_bin))
    
    # Calculate Expected Calibration Error (ECE)
    total_samples = len(y_true)
    ece = 0
    for i in range(len(bin_counts)):
        weight = bin_counts[i] / total_samples
        ece += weight * abs(bin_true_probs[i] - bin_predicted_probs[i])
    
    # Calculate Maximum Calibration Error (MCE)
    if len(bin_true_probs) > 0:
        mce = max([abs(true_p - pred_p) 
                  for true_p, pred_p in zip(bin_true_probs, bin_predicted_probs)])
    else:
        mce = 0
    
    # Hosmer-Lemeshow test for calibration
    try:
        # Group predictions into deciles
        deciles = np.percentile(y_prob, np.arange(10, 100, 10))
        groups = np.digitize(y_prob, deciles)
        
        observed = []
        expected = []
        
        for g in range(max(groups) + 1):
            group_mask = groups == g
            if np.sum(group_mask) > 0:
                observed.append(np.sum(y_true[group_mask]))
                expected.append(np.sum(y_prob[group_mask]))
        
        # Chi-square test
        if len(observed) > 1:
            chi2, p_value = stats.chisquare(observed, expected)
        else:
            chi2, p_value = 0, 1.0
    except:
        chi2, p_value = 0, 1.0
    
    return {
        'ece': ece,
        'mce': mce,
        'hosmer_lemeshow_chi2': chi2,
        'hosmer_lemeshow_p': p_value,
        'bin_centers': bin_centers,
        'bin_true_probs': bin_true_probs,
        'bin_predicted_probs': bin_predicted_probs,
        'bin_counts': bin_counts
    }


class MetricTracker:
    """Track metrics over time during training"""
    
    def __init__(self, metrics_to_track: List[str]):
        self.metrics_to_track = metrics_to_track
        self.history = {metric: [] for metric in metrics_to_track}
        self.best_values = {metric: None for metric in metrics_to_track}
        self.best_epochs = {metric: 0 for metric in metrics_to_track}
        
    def update(self, metrics: Dict[str, float], epoch: int):
        """Update tracked metrics"""
        for metric in self.metrics_to_track:
            if metric in metrics:
                value = metrics[metric]
                self.history[metric].append(value)
                
                # Update best value
                if self.best_values[metric] is None or self._is_better(metric, value):
                    self.best_values[metric] = value
                    self.best_epochs[metric] = epoch
    
    def _is_better(self, metric: str, value: float) -> bool:
        """Check if new value is better than current best"""
        # Metrics where higher is better
        higher_better = ['accuracy', 'precision', 'recall', 'f1', 'auc', 
                        'sensitivity', 'specificity', 'ppv', 'npv']
        
        # Metrics where lower is better
        lower_better = ['loss', 'error', 'ece', 'mce']
        
        if any(m in metric.lower() for m in higher_better):
            return value > self.best_values[metric]
        elif any(m in metric.lower() for m in lower_better):
            return value < self.best_values[metric]
        else:
            # Default: higher is better
            return value > self.best_values[metric]
    
    def get_best(self, metric: str) -> Tuple[float, int]:
        """Get best value and epoch for a metric"""
        return self.best_values.get(metric), self.best_epochs.get(metric)
    
    def get_history(self, metric: str) -> List[float]:
        """Get history for a metric"""
        return self.history.get(metric, [])
    
    def summary(self) -> str:
        """Generate summary of tracked metrics"""
        summary = "Metric Tracking Summary:\n"
        summary += "-" * 50 + "\n"
        
        for metric in self.metrics_to_track:
            if self.best_values[metric] is not None:
                summary += f"{metric}:\n"
                summary += f"  Best Value: {self.best_values[metric]:.4f}\n"
                summary += f"  Best Epoch: {self.best_epochs[metric]}\n"
                summary += f"  Current: {self.history[metric][-1]:.4f}\n"
                summary += "\n"
        
        return summary