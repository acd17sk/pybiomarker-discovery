import pandas as pd
import numpy as np
import torch
from biomarkers.data.datasets import MovementBiomarkerDataset

def test_movement_dataset_windowing(tmp_path):
    """Test that the movement dataset correctly creates windows."""
    # Create dummy data directory
    sensor_dir = tmp_path / "sensors"
    sensor_dir.mkdir()
    
    # Create a long sensor file
    mov_data = pd.DataFrame(np.random.randn(5000, 6), columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
    mov_data.to_csv(sensor_dir / "mov_01.csv", index=False)
    
    # Create metadata
    metadata = pd.DataFrame([
        {"filename": "mov_01.csv", "patient_id": "p01", "disease": "A"}
    ])
    metadata.to_csv(tmp_path / "metadata.csv", index=False)

    dataset = MovementBiomarkerDataset(
        data_dir=str(tmp_path),
        window_size=1000,
        overlap=0.5
    )
    
    # Calculation: (5000 - 1000) / (1000 * (1-0.5)) + 1 = 4000 / 500 + 1 = 9 windows
    assert len(dataset) == 9

    # Check one sample
    data, target = dataset[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (6, 1000) # [channels, time]
    assert target['disease'] == 'A'
    assert 'features' in target
    assert 'mean' in target['features']