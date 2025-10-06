import pytest
import pandas as pd
import numpy as np
import soundfile as sf
import cv2


@pytest.fixture(scope="session")
def dummy_data_dir(tmp_path_factory):
    """
    Creates a temporary directory with a full data structure for testing
    loaders and datasets.
    """
    base_dir = tmp_path_factory.mktemp("biomarker_data")

    # Define patients and diseases
    patients = [
        {"patient_id": "p001", "disease": "A", "quality_score": 0.9, "age": 55},
        {"patient_id": "p002", "disease": "B", "quality_score": 0.95, "age": 62},
        {"patient_id": "p003", "disease": "A", "quality_score": 0.75, "age": 71},
        {"patient_id": "p004", "disease": "C", "quality_score": 0.6, "age": 48}, # Low quality
    ]

    def create_split(split_name, patient_list):
        split_dir = base_dir / split_name
        
        # Create modality subdirectories
        (split_dir / "voice").mkdir(parents=True, exist_ok=True)
        (split_dir / "movement").mkdir(parents=True, exist_ok=True)
        (split_dir / "vision").mkdir(parents=True, exist_ok=True)
        (split_dir / "text").mkdir(parents=True, exist_ok=True)

        # Create metadata
        metadata = pd.DataFrame(patient_list)
        metadata.to_csv(split_dir / "metadata.csv", index=False)

        # Create dummy data files
        for p in patient_list:
            pid = p["patient_id"]
            
            # Voice (.wav)
            sf.write(split_dir / "voice" / f"{pid}.wav", np.random.randn(16000), 16000)
            
            # Movement (.csv)
            mov_data = pd.DataFrame(np.random.randn(100, 6), columns=['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'])
            mov_data.to_csv(split_dir / "movement" / f"{pid}.csv", index=False)

            # Vision (.mp4)
            video_path = str(split_dir / "vision" / f"{pid}.mp4")
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (64, 64))
            for _ in range(30):
                writer.write(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
            writer.release()
            
            # Text (.txt)
            with open(split_dir / "text" / f"{pid}.txt", "w") as f:
                f.write(f"This is a test transcript for patient {pid}.")

    # Create train, val, test splits
    create_split("train", patients)
    create_split("val", patients[:2])
    create_split("test", patients[:2])
    
    return base_dir