import pytest
import numpy as np
from biomarkers.core.metrics import (
    ClinicalMetrics,
    BiomarkerMetrics,
    MetricTracker,
    calculate_clinical_metrics,
    calculate_biomarker_reliability,
    calculate_roc_metrics,
    calculate_calibration_metrics,
)


@pytest.fixture
def sample_data():
    """Fixture for sample prediction data."""
    return {
        "y_true": np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
        "y_pred": np.array([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]),
        "y_prob": np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.8, 0.2],
                [0.1, 0.9],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.2, 0.8],
            ]
        ),
        "disease_names": ["healthy", "diseased"],
    }


@pytest.fixture
def reliability_data():
    """Fixture for sample reliability data."""
    return {
        "measurements1": np.array([1.2, 1.5, 1.8, 2.1, 2.4]),
        "measurements2": np.array([1.3, 1.6, 1.9, 2.2, 2.5]),
    }


class TestClinicalMetrics:
    """Tests for the ClinicalMetrics class."""

    def test_calculate_all(self, sample_data):
        """Test the calculate_all method."""
        metrics_calculator = ClinicalMetrics(disease_names=sample_data["disease_names"])
        metrics = metrics_calculator.calculate_all(
            y_true=sample_data["y_true"],
            y_pred=sample_data["y_pred"],
            y_prob=sample_data["y_prob"],
        )

        assert "accuracy" in metrics
        assert "healthy_precision" in metrics
        assert "diseased_recall" in metrics
        assert "macro_f1" in metrics
        assert "healthy_auc" in metrics
        assert "macro_auc" in metrics
        assert metrics["accuracy"] == 0.8

    def test_sensitivity_at_specificity(self, sample_data):
        """Test sensitivity calculation at a given specificity."""
        metrics_calculator = ClinicalMetrics()
        sensitivity = metrics_calculator.sensitivity_at_specificity(
            y_true=sample_data["y_true"], y_score=sample_data["y_prob"][:, 1]
        )
        assert 0.0 <= sensitivity <= 1.0

    def test_likelihood_ratios(self):
        """Test likelihood ratio calculations."""
        metrics_calculator = ClinicalMetrics()
        lr_pos, lr_neg = metrics_calculator.likelihood_ratios(
            sensitivity=0.9, specificity=0.8
        )
        assert np.isclose(lr_pos, 4.5)
        assert np.isclose(lr_neg, 0.125)


class TestBiomarkerMetrics:
    """Tests for the BiomarkerMetrics class."""

    def test_test_retest_reliability(self, reliability_data):
        """Test test-retest reliability metrics."""
        metrics_calculator = BiomarkerMetrics()
        metrics = metrics_calculator.test_retest_reliability(
            measurements1=reliability_data["measurements1"],
            measurements2=reliability_data["measurements2"],
        )
        assert "icc" in metrics
        assert "pearson_r" in metrics
        assert "coefficient_of_variation" in metrics
        assert "mean_difference" in metrics
        assert 0.0 <= metrics["icc"] <= 1.0

    def test_minimal_detectable_change(self):
        """Test minimal detectable change calculation."""
        metrics_calculator = BiomarkerMetrics()
        mdc = metrics_calculator.minimal_detectable_change(
            test_std=0.5, reliability=0.9
        )
        assert mdc > 0


def test_calculate_clinical_metrics(sample_data):
    """Test the calculate_clinical_metrics function."""
    metrics = calculate_clinical_metrics(**sample_data)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics


def test_calculate_biomarker_reliability(reliability_data):
    """Test the calculate_biomarker_reliability function."""
    metrics = calculate_biomarker_reliability(
        test_measurements=reliability_data["measurements1"],
        retest_measurements=reliability_data["measurements2"],
    )
    assert "icc" in metrics
    assert "pearson_r" in metrics


def test_calculate_roc_metrics(sample_data):
    """Test the calculate_roc_metrics function."""
    metrics = calculate_roc_metrics(
        y_true=sample_data["y_true"], y_score=sample_data["y_prob"][:, 1]
    )
    assert "auc" in metrics
    assert "optimal_threshold" in metrics
    assert "optimal_sensitivity" in metrics
    assert "optimal_specificity" in metrics


def test_calculate_calibration_metrics(sample_data):
    """Test the calculate_calibration_metrics function."""
    metrics = calculate_calibration_metrics(
        y_true=sample_data["y_true"], y_prob=sample_data["y_prob"][:, 1]
    )
    assert "ece" in metrics
    assert "mce" in metrics
    assert "hosmer_lemeshow_p" in metrics


class TestMetricTracker:
    """Tests for the MetricTracker class."""

    def test_tracker_update_and_best(self):
        """Test updating the tracker and getting the best values."""
        tracker = MetricTracker(metrics_to_track=["loss", "accuracy"])
        tracker.update({"loss": 0.5, "accuracy": 0.8}, epoch=1)
        tracker.update({"loss": 0.4, "accuracy": 0.85}, epoch=2)
        tracker.update({"loss": 0.6, "accuracy": 0.82}, epoch=3)

        best_loss, best_loss_epoch = tracker.get_best("loss")
        best_acc, best_acc_epoch = tracker.get_best("accuracy")

        assert best_loss == 0.4
        assert best_loss_epoch == 2
        assert best_acc == 0.85
        assert best_acc_epoch == 2
        assert len(tracker.get_history("loss")) == 3