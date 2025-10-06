import pytest
from pathlib import Path
from biomarkers.core.registry import ModelRegistry, register_model, get_model, create_model
from biomarkers.core.base import BiomarkerModel, BiomarkerConfig
import torch.nn as nn
import torch


# Dummy model for testing
class DummyModel(BiomarkerModel):
    def _build_model(self):
        self.fc = nn.Linear(10, 2)

    def extract_features(self, x):
        return x

    def forward(self, x, **kwargs):
        logits = self.fc(x)
        return {
            "logits": logits,
            "probabilities": torch.softmax(logits, dim=-1),
            "features": x,
        }


@pytest.fixture
def registry():
    """Fixture for a clean ModelRegistry instance."""
    return ModelRegistry()


class TestModelRegistry:
    def test_register_and_get(self, registry):
        """Test registering and retrieving a model."""
        config = BiomarkerConfig(modality="test", model_type="dummy")
        registry.register("dummy", DummyModel, config)
        model_class = registry.get("dummy")
        assert model_class == DummyModel

    def test_create_model(self, registry):
        """Test creating a model instance from the registry."""
        config = BiomarkerConfig(modality="test", model_type="dummy", input_dim=10)
        registry.register("dummy", DummyModel, config)
        model = registry.create("dummy")
        assert isinstance(model, DummyModel)
        assert model.config.modality == "test"

    def test_list_models(self, registry):
        """Test listing registered models."""
        registry.register("dummy1", DummyModel)
        registry.register("dummy2", DummyModel)
        assert set(registry.list_models()) == {"dummy1", "dummy2"}

    def test_save_and_load_registry(self, registry, tmp_path):
        """Test saving and loading the registry."""
        config = BiomarkerConfig(modality="test", model_type="dummy", input_dim=10)
        registry.register("dummy", DummyModel, config)
        registry_path = tmp_path / "registry.json"
        registry.save_registry(registry_path)

        new_registry = ModelRegistry()
        new_registry.load_registry(registry_path)
        assert "dummy" in new_registry.list_models()
        model = new_registry.create("dummy")
        assert isinstance(model, DummyModel)

    def test_search(self, registry):
        """Test searching for models in the registry."""
        config1 = BiomarkerConfig(modality="ecg", model_type="dummy1")
        config2 = BiomarkerConfig(modality="eeg", model_type="dummy2")
        registry.register("ecg_model", DummyModel, config1)
        registry.register("eeg_model", DummyModel, config2)

        ecg_models = registry.search(modality="ecg")
        assert ecg_models == ["ecg_model"]

        model_search = registry.search(pattern="model")
        assert set(model_search) == {"ecg_model", "eeg_model"}


# Test global registry functions
def test_global_registration():
    """Test the global registration functions."""
    register_model("global_dummy", DummyModel)
    model_class = get_model("global_dummy")
    assert model_class == DummyModel

    model = create_model("global_dummy", config={"modality": "test", "model_type": "dummy", "input_dim": 10})

    assert isinstance(model, DummyModel)