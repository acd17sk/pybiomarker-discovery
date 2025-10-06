"""Model registry system for managing biomarker models"""

import logging
from typing import Dict, Type, Optional, List, Any, Union
from pathlib import Path
import json
import importlib
import inspect

from biomarkers.core.base import BiomarkerModel, BiomarkerConfig

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for biomarker models"""
    
    def __init__(self):
        self._models: Dict[str, Type[BiomarkerModel]] = {}
        self._configs: Dict[str, BiomarkerConfig] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        
    def register(self, 
                name: str,
                model_class: Type[BiomarkerModel],
                config: Optional[BiomarkerConfig] = None,
                metadata: Optional[Dict[str, Any]] = None,
                override: bool = False):
        """
        Register a model in the registry
        
        Args:
            name: Name to register model under
            model_class: Model class (must inherit from BiomarkerModel)
            config: Default configuration for model
            metadata: Additional metadata about the model
            override: Whether to override existing registration
        """
        if not issubclass(model_class, BiomarkerModel):
            raise ValueError(f"{model_class} must inherit from BiomarkerModel")
        
        if name in self._models and not override:
            raise ValueError(f"Model '{name}' already registered. Use override=True to replace.")
        
        self._models[name] = model_class
        
        if config is not None:
            self._configs[name] = config
        
        if metadata is not None:
            self._metadata[name] = metadata
        else:
            # Extract metadata from class
            self._metadata[name] = self._extract_metadata(model_class)
        
        logger.info(f"Registered model '{name}' ({model_class.__name__})")
    
    def get(self, name: str) -> Type[BiomarkerModel]:
        """Get model class by name"""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found in registry. "
                          f"Available models: {self.list_models()}")
        return self._models[name]
    
    def create(self, 
              name: str,
              config: Optional[Union[Dict[str, Any], BiomarkerConfig]] = None,
              **kwargs) -> BiomarkerModel:
        """
        Create model instance from registry
        
        Args:
            name: Name of registered model
            config: Configuration for model (overrides default)
            **kwargs: Additional arguments to pass to model constructor
        """
        model_class = self.get(name)
        
        # Use default config if available
        if config is None and name in self._configs:
            config = self._configs[name]
        elif config is None:
            config = BiomarkerConfig(
                modality="unknown",
                model_type=name,
                **kwargs
            )
        elif isinstance(config, dict):
            # Merge with default config if available
            if name in self._configs:
                default_config = self._configs[name].to_dict()
                default_config.update(config)
                config = BiomarkerConfig.from_dict(default_config)
            else:
                config = BiomarkerConfig.from_dict(config)
        
        # Create model instance
        model = model_class(config)
        logger.info(f"Created model instance of '{name}' with {model.num_parameters:,} parameters")
        
        return model
    
    def list_models(self) -> List[str]:
        """List all registered models"""
        return list(self._models.keys())
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a model"""
        if name not in self._metadata:
            return {}
        return self._metadata[name]
    
    def describe(self, name: str) -> str:
        """Get description of a model"""
        if name not in self._models:
            return f"Model '{name}' not found"
        
        model_class = self._models[name]
        metadata = self._metadata.get(name, {})
        
        description = f"Model: {name}\n"
        description += f"Class: {model_class.__name__}\n"
        description += f"Module: {model_class.__module__}\n"
        
        if metadata:
            description += "Metadata:\n"
            for key, value in metadata.items():
                description += f"  {key}: {value}\n"
        
        if name in self._configs:
            description += f"Default Config: {self._configs[name].to_dict()}\n"
        
        if model_class.__doc__:
            description += f"Documentation:\n{model_class.__doc__}\n"
        
        return description
    
    def save_registry(self, path: Union[str, Path]):
        """Save registry to file"""
        path = Path(path)
        
        registry_data = {
            'models': {
                name: {
                    'module': cls.__module__,
                    'class': cls.__name__,
                    'metadata': self._metadata.get(name, {}),
                    'config': self._configs[name].to_dict() if name in self._configs else None
                }
                for name, cls in self._models.items()
            }
        }
        
        with open(path, 'w') as f:
            json.dump(registry_data, f, indent=2)
        
        logger.info(f"Registry saved to {path}")
    
    def load_registry(self, path: Union[str, Path], override: bool = False):
        """Load registry from file"""
        path = Path(path)
        
        with open(path, 'r') as f:
            registry_data = json.load(f)
        
        for name, model_info in registry_data['models'].items():
            # Import module and get class
            module = importlib.import_module(model_info['module'])
            model_class = getattr(module, model_info['class'])
            
            # Create config if available
            config = None
            if model_info['config']:
                config = BiomarkerConfig.from_dict(model_info['config'])
            
            # Register model
            self.register(
                name=name,
                model_class=model_class,
                config=config,
                metadata=model_info.get('metadata'),
                override=override
            )
        
        logger.info(f"Registry loaded from {path}")
    
    def _extract_metadata(self, model_class: Type[BiomarkerModel]) -> Dict[str, Any]:
        """Extract metadata from model class"""
        metadata = {
            'class_name': model_class.__name__,
            'module': model_class.__module__,
            'has_docstring': model_class.__doc__ is not None
        }
        
        # Check for specific methods
        methods = ['extract_features', 'get_biomarkers', 'predict']
        for method in methods:
            metadata[f'has_{method}'] = hasattr(model_class, method)
        
        # Get init signature
        try:
            sig = inspect.signature(model_class.__init__)
            metadata['init_params'] = list(sig.parameters.keys())
        except:
            metadata['init_params'] = []
        
        return metadata
    
    def search(self, 
              modality: Optional[str] = None,
              pattern: Optional[str] = None) -> List[str]:
        """
        Search for models in registry
        
        Args:
            modality: Filter by modality (if in metadata)
            pattern: String pattern to match in model name
        """
        results = []
        
        for name in self._models:
            # Check pattern match
            if pattern and pattern.lower() not in name.lower():
                continue
            
            # Check modality match
            if modality:
                metadata = self._metadata.get(name, {})
                model_modality = metadata.get('modality', '')
                if modality.lower() != model_modality.lower():
                    # Also check in config
                    if name in self._configs:
                        if modality.lower() != self._configs[name].modality.lower():
                            continue
                    else:
                        continue
            
            results.append(name)
        
        return results
    
    def __contains__(self, name: str) -> bool:
        """Check if model is in registry"""
        return name in self._models
    
    def __len__(self) -> int:
        """Get number of registered models"""
        return len(self._models)
    
    def __repr__(self) -> str:
        return f"ModelRegistry({len(self._models)} models registered)"


# Global registry instance
_global_registry = ModelRegistry()


def register_model(name: str,
                  model_class: Type[BiomarkerModel] = None,
                  config: Optional[BiomarkerConfig] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  override: bool = False):
    """
    Register a model in the global registry
    
    Can be used as a decorator:
    @register_model("my_model")
    class MyModel(BiomarkerModel):
        ...
    
    Or as a function:
    register_model("my_model", MyModel)
    """
    def decorator(cls):
        _global_registry.register(name, cls, config, metadata, override)
        return cls
    
    if model_class is None:
        # Used as decorator
        return decorator
    else:
        # Used as function
        _global_registry.register(name, model_class, config, metadata, override)


def get_model(name: str) -> Type[BiomarkerModel]:
    """Get model class from global registry"""
    return _global_registry.get(name)


def create_model(name: str,
                config: Optional[Union[Dict[str, Any], BiomarkerConfig]] = None,
                **kwargs) -> BiomarkerModel:
    """Create model instance from global registry"""
    return _global_registry.create(name, config, **kwargs)


def list_models() -> List[str]:
    """List all models in global registry"""
    return _global_registry.list_models()


def describe_model(name: str) -> str:
    """Get description of model in global registry"""
    return _global_registry.describe(name)


def search_models(modality: Optional[str] = None,
                 pattern: Optional[str] = None) -> List[str]:
    """Search for models in global registry"""
    return _global_registry.search(modality, pattern)


def save_registry(path: Union[str, Path]):
    """Save global registry to file"""
    _global_registry.save_registry(path)


def load_registry(path: Union[str, Path], override: bool = False):
    """Load global registry from file"""
    _global_registry.load_registry(path, override)  