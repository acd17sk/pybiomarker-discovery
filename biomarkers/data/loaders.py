import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import librosa
import cv2
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class MultiModalBiomarkerDataset(Dataset):
    """Dataset for multi-modal biomarker data"""
    
    def __init__(self,
                 data_dir: str,
                 modalities: List[str],
                 transform: Optional[Dict[str, Any]] = None,
                 target_diseases: List[str] = None,
                 cache_data: bool = False,
                 quality_threshold: float = 0.8):
        self.data_dir = Path(data_dir)
        self.modalities = modalities
        self.transform = transform or {}
        self.target_diseases = target_diseases
        self.cache_data = cache_data
        self.quality_threshold = quality_threshold
        self._cache = {} if cache_data else None
        
        # Load metadata
        self.metadata = pd.read_csv(self.data_dir / 'metadata.csv')
        self.samples = self._prepare_samples()
        
        # Calculate class weights for balanced sampling
        self._calculate_class_weights()
        
    def _prepare_samples(self) -> List[Dict]:
        """Prepare sample list with paths"""
        samples = []
        for idx, row in self.metadata.iterrows():
            # Check data quality if threshold is set
            if 'quality_score' in row and row['quality_score'] < self.quality_threshold:
                logger.warning(f"Skipping sample {row['patient_id']} due to low quality score")
                continue
                
            sample = {
                'id': row['patient_id'],
                'disease': row['disease'],
                'severity': row.get('severity', 0),
                'age': row.get('age', 0),
                'gender': row.get('gender', 'unknown'),
                'quality_score': row.get('quality_score', 1.0)
            }
            
            # Add paths for each modality
            for modality in self.modalities:
                if modality == 'voice':
                    sample['voice_path'] = self.data_dir / 'voice' / f"{row['patient_id']}.wav"
                elif modality == 'movement':
                    sample['movement_path'] = self.data_dir / 'movement' / f"{row['patient_id']}.csv"
                elif modality == 'vision':
                    sample['vision_path'] = self.data_dir / 'vision' / f"{row['patient_id']}.mp4"
                elif modality == 'text':
                    sample['text_path'] = self.data_dir / 'text' / f"{row['patient_id']}.txt"
                elif modality == 'ecg':
                    sample['ecg_path'] = self.data_dir / 'ecg' / f"{row['patient_id']}.npy"
                elif modality == 'eeg':
                    sample['eeg_path'] = self.data_dir / 'eeg' / f"{row['patient_id']}.edf"
            
            samples.append(sample)
        
        logger.info(f"Prepared {len(samples)} samples from {len(self.metadata)} total")
        return samples
    
    def _calculate_class_weights(self):
        """Calculate class weights for balanced sampling"""
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample['disease']] += 1
        
        total_samples = len(self.samples)
        num_classes = len(class_counts)
        
        self.class_weights = {}
        for disease, count in class_counts.items():
            self.class_weights[disease] = total_samples / (num_classes * count)
        
        # Create sample weights
        self.sample_weights = torch.tensor([
            self.class_weights[sample['disease']] for sample in self.samples
        ])
    
    def _load_voice(self, path: Path) -> torch.Tensor:
        """Load and preprocess voice data"""
        if self.cache_data and str(path) in self._cache:
            return self._cache[str(path)]
            
        # Load audio
        audio, sr = librosa.load(path, sr=16000)
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=80, n_fft=512, hop_length=160
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Apply transforms
        if 'voice' in self.transform:
            mel_spec_db = self.transform['voice'](mel_spec_db)
        
        result = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        
        if self.cache_data:
            self._cache[str(path)] = result
            
        return result
    
    def _load_movement(self, path: Path) -> torch.Tensor:
        """Load and preprocess movement data"""
        if self.cache_data and str(path) in self._cache:
            return self._cache[str(path)]
            
        # Load sensor data
        data = pd.read_csv(path)
        
        # Expected columns: timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        sensor_data = data[sensor_cols].values
        
        # Normalize
        sensor_data = (sensor_data - sensor_data.mean(axis=0)) / (sensor_data.std(axis=0) + 1e-8)
        
        # Apply transforms
        if 'movement' in self.transform:
            sensor_data = self.transform['movement'](sensor_data)
        
        result = torch.FloatTensor(sensor_data).transpose(0, 1)
        
        if self.cache_data:
            self._cache[str(path)] = result
            
        return result
    
    def _load_vision(self, path: Path) -> torch.Tensor:
        """Load and preprocess vision data (facial features)"""
        if self.cache_data and str(path) in self._cache:
            return self._cache[str(path)]
            
        # For simplicity, extract key frames
        cap = cv2.VideoCapture(str(path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = np.linspace(0, total_frames-1, 10, dtype=int)
        
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Resize and normalize
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        
        frames = np.stack(frames)
        
        # Apply transforms
        if 'vision' in self.transform:
            frames = self.transform['vision'](frames)
        
        result = torch.FloatTensor(frames).permute(0, 3, 1, 2)
        
        if self.cache_data:
            self._cache[str(path)] = result
            
        return result
    
    def _load_text(self, path: Path) -> torch.Tensor:
        """Load and preprocess text data"""
        if self.cache_data and str(path) in self._cache:
            return self._cache[str(path)]
            
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Apply transforms (should include proper tokenization)
        if 'text' in self.transform:
            result = self.transform['text'](text)
        else:
            # Fallback tokenization
            tokens = text.split()[:512]
            token_ids = [hash(token) % 10000 for token in tokens]
            if len(token_ids) < 512:
                token_ids += [0] * (512 - len(token_ids))
            result = torch.LongTensor(token_ids)
        
        if self.cache_data:
            self._cache[str(path)] = result
            
        return result
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        # Load data for each modality
        data = {}
        
        for modality in self.modalities:
            path_key = f'{modality}_path'
            if path_key in sample and sample[path_key].exists():
                try:
                    if modality == 'voice':
                        data[modality] = self._load_voice(sample[path_key])
                    elif modality == 'movement':
                        data[modality] = self._load_movement(sample[path_key])
                    elif modality == 'vision':
                        data[modality] = self._load_vision(sample[path_key])
                    elif modality == 'text':
                        data[modality] = self._load_text(sample[path_key])
                except Exception as e:
                    logger.error(f"Error loading {modality} for sample {sample['id']}: {e}")
                    # Return zero tensor as fallback
                    data[modality] = self._get_zero_tensor(modality)
        
        # Disease label
        if self.target_diseases:
            target = self.target_diseases.index(sample['disease'])
        else:
            target = 0  # Binary classification
        
        return {
            'input': data,
            'target': torch.tensor(target, dtype=torch.long),
            'metadata': {
                'patient_id': sample['id'],
                'age': sample['age'],
                'gender': sample['gender'],
                'severity': sample['severity'],
                'quality_score': sample.get('quality_score', 1.0)
            }
        }
    
    def _get_zero_tensor(self, modality: str) -> torch.Tensor:
        """Get zero tensor for fallback"""
        shapes = {
            'voice': (1, 80, 100),
            'movement': (6, 1000),
            'vision': (10, 3, 224, 224),
            'text': (512,)
        }
        shape = shapes.get(modality, (1, 100))
        return torch.zeros(shape)
    
    def get_sampler(self, balanced: bool = True) -> Optional[WeightedRandomSampler]:
        """Get sampler for balanced training"""
        if balanced:
            return WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.sample_weights),
                replacement=True
            )
        return None


class BiomarkerDataLoader(DataLoader):
    """Custom DataLoader with additional features for biomarker data"""
    
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 balanced_sampling: bool = True,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 persistent_workers: bool = True,
                 **kwargs):
        
        # Get sampler if balanced sampling requested
        sampler = None
        if balanced_sampling and hasattr(dataset, 'get_sampler'):
            sampler = dataset.get_sampler(balanced=True)
            shuffle = False  # Cannot use shuffle with sampler
        
        # ✅ FIX: Only add prefetch_factor if multiprocessing is enabled
        if num_workers > 0:
            kwargs['prefetch_factor'] = prefetch_factor

        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            persistent_workers=persistent_workers and num_workers > 0,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders"""
    
    # Create datasets
    train_dataset = MultiModalBiomarkerDataset(
        data_dir=config['data_dir'] + '/train',
        modalities=config['modalities'],
        transform=config.get('transforms'),
        target_diseases=config.get('target_diseases'),
        cache_data=config.get('cache_data', False),
        quality_threshold=config.get('quality_threshold', 0.8)
    )
    
    val_dataset = MultiModalBiomarkerDataset(
        data_dir=config['data_dir'] + '/val',
        modalities=config['modalities'],
        transform=config.get('transforms'),
        target_diseases=config.get('target_diseases'),
        cache_data=config.get('cache_data', True),  # Cache validation data
        quality_threshold=config.get('quality_threshold', 0.8)
    )
    
    test_dataset = MultiModalBiomarkerDataset(
        data_dir=config['data_dir'] + '/test',
        modalities=config['modalities'],
        transform=config.get('transforms'),
        target_diseases=config.get('target_diseases'),
        cache_data=config.get('cache_data', True),  # Cache test data
        quality_threshold=config.get('quality_threshold', 0.8)
    )
    
    # Create dataloaders
    train_loader = BiomarkerDataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        balanced_sampling=config.get('balanced_sampling', True),
        num_workers=config.get('num_workers', 4),
        persistent_workers=config.get('persistent_workers', True)
    )
    
    val_loader = BiomarkerDataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        balanced_sampling=False,
        num_workers=config.get('num_workers', 4),
        persistent_workers=config.get('persistent_workers', True)
    )
    
    test_loader = BiomarkerDataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        balanced_sampling=False,
        num_workers=config.get('num_workers', 4),
        persistent_workers=config.get('persistent_workers', True)
    )
    
    # Log dataset statistics
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader