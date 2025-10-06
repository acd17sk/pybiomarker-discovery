#!/usr/bin/env python3
import argparse
import yaml
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from biomarkers.models.fusion.multimodal_fusion import MultiModalBiomarkerFusion
from biomarkers.training.trainer import BiomarkerTrainer
from biomarkers.data.loaders import create_dataloaders

def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config['data'])
    
    # Create model
    model = MultiModalBiomarkerFusion(config['model'])
    
    # Create trainer
    trainer = BiomarkerTrainer(
        model=model,
        config=config['training'],
        use_wandb=not args.no_wandb
    )
    
    # Train
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir=args.save_dir
    )
    
    # Final evaluation
    test_metrics = trainer.evaluate(test_loader, clinical_metrics=True)
    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    args = parser.parse_args()
    
    main(args)