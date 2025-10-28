"""Base classes for biomarker models and components"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import json
import numpy as np
from dataclasses import dataclass, field, fields


logger = logging.getLogger(__name__)

@dataclass
class BiomarkerConfig:
    """Configuration for biomarker models"""
    modality: str
    model_type: str
    input_dim: Optional[int] = None
    hidden_dim: int = 256
    output_dim: Optional[int] = None
    dropout: float = 0.3
    num_diseases: int = 10
    use_uncertainty: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        # --- MODIFIED ---
        # Also include metadata keys at the top level for easier serialization
        d = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'metadata'
        }
        d.update(self.metadata)
        return d
        # --------------
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BiomarkerConfig':
        """Create config from dictionary"""
        
        # --- MODIFIED ---
        """Create config from dictionary, handling extra keys."""
        config_keys = {f.name for f in fields(cls)}
        init_args = {k: v for k, v in config_dict.items() if k in config_keys}
        metadata = init_args.get('metadata', {})
        
        # Put extra keys into metadata
        for k, v in config_dict.items():
            if k not in config_keys:
                metadata[k] = v
        
        init_args['metadata'] = metadata
        return cls(**init_args)
        # --------------
            
    def save(self, path: Union[str, Path]):
        """Save configuration to file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BiomarkerConfig':
        """Load configuration from file"""
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


class BiomarkerModel(nn.Module, ABC):
    """Base class for all biomarker models"""
    
    def __init__(self, config: Union[Dict[str, Any], BiomarkerConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = BiomarkerConfig.from_dict(config)
        self.config = config
        self.device = torch.device(config.device)
        self._model_built = False
        self._build_model()
        self._model_built = True

        self.to(self.device)
        
    @abstractmethod
    def _build_model(self):
        """Build model architecture - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract biomarker features from input"""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning dict with at minimum:
        - 'logits': raw predictions
        - 'probabilities': softmax probabilities
        - 'features': extracted features
        """
        pass
    
    def get_biomarkers(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract interpretable biomarkers from input"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_biomarkers=True)
            if 'biomarkers' in output:
                return output['biomarkers']
            else:
                # Fallback to raw features
                features = self.extract_features(x)
                return self._interpret_features(features)
    
    def _interpret_features(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert features to interpretable biomarkers"""
        return {"raw_features": features}
    
    def predict(self, x: torch.Tensor, 
                return_confidence: bool = True) -> Dict[str, torch.Tensor]:
        """Make predictions with optional confidence scores"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x, return_uncertainty=return_confidence)
            
            # Add prediction labels
            output['predictions'] = output['logits'].argmax(dim=-1)
            
            return output
    
    def save(self, path: Union[str, Path], 
             save_config: bool = True,
             save_optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'config': self.config.to_dict() if save_config else None
        }
        
        if save_optimizer is not None:
            checkpoint['optimizer_state_dict'] = save_optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path], 
             map_location: Optional[str] = None) -> 'BiomarkerModel':
        """Load model from checkpoint"""
        path = Path(path)
        checkpoint = torch.load(path, map_location=map_location)
        
        # Create model instance
        config = BiomarkerConfig.from_dict(checkpoint['config'])
        model = cls(config)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
        
        return model
    
    @property
    def num_parameters(self) -> int:
        """Count number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def modality(self) -> str:
        """Get model modality"""
        return self.config.modality
    
    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        # Override in subclasses to freeze specific layers
        pass
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.parameters():
            param.requires_grad = True


class BiomarkerDataset(torch.utils.data.Dataset, ABC):
    """Base class for biomarker datasets"""
    
    def __init__(self, 
                 data_dir: Union[str, Path],
                 transform: Optional[Any] = None,
                 target_transform: Optional[Any] = None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._load_samples()
        
    @abstractmethod
    def _load_samples(self) -> List[Any]:
        """Load sample list - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Get item by index - must be implemented by subclasses"""
        pass
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.samples)
    
    def get_labels(self) -> np.ndarray:
        """Get all labels in dataset"""
        labels = []
        for i in range(len(self)):
            _, label = self[i]
            labels.append(label)
        return np.array(labels)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes"""
        labels = self.get_labels()
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


class BiomarkerPredictor:
    """High-level predictor for biomarker models"""
    
    def __init__(self, 
                 model: BiomarkerModel,
                 preprocessing: Optional[Any] = None,
                 postprocessing: Optional[Any] = None):
        self.model = model
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.device = model.device
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def predict_single(self, 
                      input_data: Any,
                      return_biomarkers: bool = True,
                      return_confidence: bool = True) -> Dict[str, Any]:
        """Predict on single input"""
        # Preprocess if needed
        if self.preprocessing is not None:
            input_data = self.preprocessing(input_data)
        
        # Convert to tensor if needed
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(input_data.shape) == len(self.model.config.metadata.get('input_shape', [0, 0, 0])):
            input_data = input_data.unsqueeze(0)
        
        # Move to device
        input_data = input_data.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output = self.model.forward(
                input_data,
                return_biomarkers=return_biomarkers,
                return_uncertainty=return_confidence
            )
        
        # Postprocess if needed
        if self.postprocessing is not None:
            output = self.postprocessing(output)
        
        # Convert to CPU and numpy for output
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                output[key] = value.cpu().numpy()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor):
                        output[key][k] = v.cpu().numpy()
        
        return output
    
    def predict_batch(self,
                     input_batch: List[Any],
                     batch_size: int = 32,
                     return_biomarkers: bool = True,
                     return_confidence: bool = True) -> List[Dict[str, Any]]:
        """Predict on batch of inputs"""
        results = []
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]
            
            # Process each item in batch
            batch_results = []
            for item in batch:
                result = self.predict_single(
                    item,
                    return_biomarkers=return_biomarkers,
                    return_confidence=return_confidence
                )
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def stream_predict(self, 
                      input_stream,
                      return_biomarkers: bool = True,
                      return_confidence: bool = True):
        """Generator for streaming predictions"""
        for input_data in input_stream:
            yield self.predict_single(
                input_data,
                return_biomarkers=return_biomarkers,
                return_confidence=return_confidence
            )


class BiomarkerTransform(ABC):
    """Base class for biomarker data transforms"""
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply transform to data"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ComposeTransform(BiomarkerTransform):
    """Compose multiple transforms"""
    
    def __init__(self, transforms: List[BiomarkerTransform]):
        self.transforms = transforms
    
    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n    ' + repr(t)
        format_string += '\n)'
        return format_string