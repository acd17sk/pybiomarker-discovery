import pytest
import torch
import torch.nn as nn
from pathlib import Path
from biomarkers.core.base import (
    BiomarkerConfig,
    BiomarkerModel,
    BiomarkerPredictor,
)


# Dummy model for testing
class DummyModel(BiomarkerModel):
    def _build_model(self):
        self.fc = nn.Linear(self.config.input_dim, self.config.output_dim)

    def extract_features(self, x):
        return self.fc(x)

    def forward(self, x, **kwargs):
        features = self.extract_features(x)
        return {
            "logits": features,
            "probabilities": torch.softmax(features, dim=-1),
            "features": features,
        }


@pytest.fixture
def dummy_config():
    """Fixture for a dummy BiomarkerConfig."""
    return BiomarkerConfig(
        modality="test",
        model_type="dummy",
        input_dim=10,
        output_dim=2,
    )


class TestBiomarkerConfig:
    def test_config_serialization(self, dummy_config, tmp_path):
        """Test saving and loading a config."""
        config_path = tmp_path / "config.json"
        dummy_config.save(config_path)
        loaded_config = BiomarkerConfig.load(config_path)

        assert loaded_config.modality == "test"
        assert loaded_config.input_dim == 10


class TestBiomarkerModel:
    def test_model_creation(self, dummy_config):
        """Test the creation of a BiomarkerModel."""
        model = DummyModel(dummy_config)
        assert model.num_parameters > 0
        assert model.modality == "test"

    def test_model_save_and_load(self, dummy_config, tmp_path):
        """Test saving and loading a model."""
        model = DummyModel(dummy_config)
        model_path = tmp_path / "model.pth"
        model.save(model_path)

        loaded_model = DummyModel.load(model_path)
        assert isinstance(loaded_model, DummyModel)
        assert loaded_model.config.input_dim == 10
        assert torch.allclose(
            model.fc.weight, loaded_model.fc.weight
        )


class TestBiomarkerPredictor:
    def test_predictor(self, dummy_config):
        """Test the BiomarkerPredictor."""
        model = DummyModel(dummy_config)
        predictor = BiomarkerPredictor(model)
        input_data = torch.randn(1, 10)
        result = predictor.predict_single(input_data)

        assert "logits" in result
        assert "probabilities" in result
        assert "features" in result
        assert result["logits"].shape == (1, 2)