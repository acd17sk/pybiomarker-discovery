"""
Multi-Modal Biomarker Discovery Platform

A PyTorch-based framework for discovering and analyzing digital biomarkers
from multiple data modalities for early disease detection and monitoring.
"""

__version__ = "0.1.0"
__author__ = "Biomarker Discovery Team"

from biomarkers.core.registry import ModelRegistry, register_model, get_model
from biomarkers.core.base import (
    BiomarkerModel,
    BiomarkerDataset,
    BiomarkerPredictor
)
from biomarkers.training.trainer import BiomarkerTrainer
from biomarkers.utils.clinical_report import ClinicalReportGenerator

# Initialize global model registry
model_registry = ModelRegistry()

# Register built-in models
def _register_builtin_models():
    """Register all built-in biomarker models"""
    from biomarkers.models.voice.speech_biomarker import VoiceBiomarkerModel
    from biomarkers.models.movement.movement_biomarker import MovementBiomarkerModel
    from biomarkers.models.fusion.multimodal_fusion import MultiModalBiomarkerFusion
    
    model_registry.register("voice_biomarker", VoiceBiomarkerModel)
    model_registry.register("movement_biomarker", MovementBiomarkerModel)
    model_registry.register("multimodal_fusion", MultiModalBiomarkerFusion)

_register_builtin_models()

__all__ = [
    "ModelRegistry",
    "register_model",
    "get_model",
    "BiomarkerModel",
    "BiomarkerDataset",
    "BiomarkerPredictor",
    "BiomarkerTrainer",
    "ClinicalReportGenerator",
    "model_registry"
]