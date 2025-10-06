import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Callable
import wandb
from tqdm import tqdm
import numpy as np
from pathlib import Path

class BiomarkerTrainer:
    """Trainer for biomarker models with privacy and clinical metrics"""
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any],
                 use_wandb: bool = True):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project="biomarker-discovery", config=config)
            
        self._setup_training()
        
    def _setup_training(self):
        """Setup optimizer, scheduler, and loss"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss functions
        self.criterion = nn.CrossEntropyLoss()
        self.auxiliary_losses = {}
        
    def train_epoch(self, 
                   dataloader: DataLoader,
                   epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        metrics = {'loss': 0, 'accuracy': 0}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss = self.criterion(outputs['logits'], targets)
            
            # Add auxiliary losses (e.g., contrastive, reconstruction)
            for name, loss_fn in self.auxiliary_losses.items():
                aux_loss = loss_fn(outputs, batch)
                loss += self.config.get(f'{name}_weight', 0.1) * aux_loss
                metrics[f'{name}_loss'] = aux_loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Update metrics
            metrics['loss'] += loss.item()
            predictions = outputs['logits'].argmax(dim=1)
            metrics['accuracy'] += (predictions == targets).float().mean().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': metrics['accuracy'] / (batch_idx + 1)
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Average metrics
        num_batches = len(dataloader)
        for key in metrics:
            metrics[key] /= num_batches
            
        self.scheduler.step()
        
        return metrics
    
    def evaluate(self,
                dataloader: DataLoader,
                clinical_metrics: bool = True) -> Dict[str, float]:
        """Evaluate model with clinical metrics"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs, return_uncertainty=True)
                
                all_predictions.append(outputs['logits'].argmax(dim=1).cpu())
                all_targets.append(targets.cpu())
                all_probabilities.append(outputs['probabilities'].cpu())
                
                if 'uncertainty' in outputs:
                    all_uncertainties.append(outputs['uncertainty'].cpu())
        
        # Concatenate results
        predictions = torch.cat(all_predictions)
        targets = torch.cat(all_targets)
        probabilities = torch.cat(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, targets, probabilities)
        
        if clinical_metrics:
            clinical = self._calculate_clinical_metrics(
                predictions, targets, probabilities
            )
            metrics.update(clinical)
        
        if all_uncertainties:
            uncertainties = torch.cat(all_uncertainties)
            metrics['mean_uncertainty'] = uncertainties.mean().item()
            
        return metrics
    
    def _calculate_metrics(self, predictions, targets, probabilities):
        """Calculate standard metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        
        accuracy = accuracy_score(targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average='macro'
        )
        
        # AUC for multi-class
        try:
            auc = roc_auc_score(
                targets, probabilities, multi_class='ovr', average='macro'
            )
        except:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
    
    def _calculate_clinical_metrics(self, predictions, targets, probabilities):
        """Calculate clinical-specific metrics"""
        # Sensitivity and Specificity per disease
        metrics = {}
        num_classes = probabilities.shape[1]
        
        for disease_idx in range(num_classes):
            # Binary classification for this disease
            binary_targets = (targets == disease_idx).float()
            binary_predictions = (predictions == disease_idx).float()
            
            tp = ((binary_predictions == 1) & (binary_targets == 1)).sum().item()
            tn = ((binary_predictions == 0) & (binary_targets == 0)).sum().item()
            fp = ((binary_predictions == 1) & (binary_targets == 0)).sum().item()
            fn = ((binary_predictions == 0) & (binary_targets == 1)).sum().item()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            metrics[f'disease_{disease_idx}_sensitivity'] = sensitivity
            metrics[f'disease_{disease_idx}_specificity'] = specificity
            metrics[f'disease_{disease_idx}_ppv'] = ppv
            metrics[f'disease_{disease_idx}_npv'] = npv
        
        return metrics
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            save_dir: str = './checkpoints'):
        """Full training loop"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_metric = 0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            # Log metrics
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['f1_score']:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/f1_score': val_metrics['f1_score'],
                    'val/auc': val_metrics['auc']
                })
            
            # Save best model
            if val_metrics['f1_score'] > best_val_metric:
                best_val_metric = val_metrics['f1_score']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics,
                    'config': self.config
                }, save_path / 'best_model.pt')
                
                print(f"Saved best model with F1 score: {best_val_metric:.4f}")