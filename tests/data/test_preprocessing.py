import pytest
import numpy as np
import soundfile as sf
from biomarkers.data.preprocessing import (
    SignalQualityChecker,
    VoicePreprocessor,
    DataCleaner,
    FeatureExtractor
)

def test_signal_quality_checker_voice():
    """Test the voice quality checker."""
    checker = SignalQualityChecker(modality='voice', quality_threshold=0.7)
    
    # Test a good signal
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    good_signal = np.sin(2 * np.pi * 440 * t) * 0.9


    quality = checker.check_quality(good_signal)
    print("Quality result:", quality)

    assert quality['passes'] == True
    
    # Test a clipped signal
    clipped_signal = np.ones(16000)
    quality = checker.check_quality(clipped_signal)
    assert quality['passes'] == False
    assert "clipping" in quality['issues'][0]
    
    # Test a silent signal
    silent_signal = np.zeros(16000)
    quality = checker.check_quality(silent_signal)
    assert quality['passes'] == False
    assert "weak" in quality['issues'][0]

def test_voice_preprocessor(tmp_path):
    """Test the voice preprocessing pipeline."""
    # Create dummy audio file with silence
    signal = np.concatenate([np.zeros(8000), np.random.randn(16000), np.zeros(8000)])
    path = tmp_path / "test.wav"
    sf.write(path, signal, 16000)
    
    preprocessor = VoicePreprocessor(trim_silence=True, extract_features=['mel_spectrogram', 'mfcc'])
    features = preprocessor.process(path)
    
    assert 'waveform' in features
    assert 'mel_spectrogram' in features
    assert 'mfcc' in features
    
    # Check that silence was trimmed
    assert len(features['waveform']) < len(signal)
    assert len(features['waveform']) > 15000

def test_data_cleaner():
    """Test the DataCleaner for outlier and NaN handling."""
    cleaner = DataCleaner(remove_outliers=True, interpolate_missing=True)
    # Create data with NaNs and a large outlier
    data = np.array([1.0, 1.1, 1.2, np.nan, 0.9, 10.0, 1.3])
    
    cleaned_data = cleaner.clean(data)
    
    assert not np.any(np.isnan(cleaned_data)) # NaNs should be gone
    assert np.max(cleaned_data) < 5.0 # Outlier should be replaced
    assert np.isclose(cleaned_data[3], 1.05) # Check interpolated value

def test_feature_extractor():
    """Test the feature extractor."""
    extractor = FeatureExtractor(modality='movement', feature_set='comprehensive')
    data = np.random.randn(500, 6) # 5 seconds of 100Hz data
    
    features = extractor.extract(data)
    
    assert 'statistical' in features
    assert 'frequency' in features
    assert 'gait' in features
    assert len(features['statistical']) > 5
    assert len(features['frequency']) > 5