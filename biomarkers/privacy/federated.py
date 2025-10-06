import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import copy
import numpy as np
from collections import OrderedDict

class FederatedLearning:
    """Federated learning for privacy-preserving biomarker discovery"""
    
    def __init__(self, 
                 model: nn.Module,
                 num_clients: int,
                 config: Dict[str, Any]):
        self.global_model = model
        self.num_clients = num_clients
        self.config = config
        self.client_models = [copy.deepcopy(model) for _ in range(num_clients)]
        
    def client_update(self,
                     client_id: int,
                     client_loader,
                     epochs: int = 5) -> Dict[str, Any]:
        """Update a single client's model"""
        model = self.client_models[client_id]
        model.train()
        
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.config.get('client_lr', 0.01),
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        total_samples = 0
        
        for _ in range(epochs):
            for batch in client_loader:
                inputs = batch['input']
                targets = batch['target']
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs['logits'], targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(targets)
                total_samples += len(targets)
        
        # Return model updates (difference from global model)
        client_update = {}
        for name, param in model.state_dict().items():
            client_update[name] = param - self.global_model.state_dict()[name]
        
        return {
            'updates': client_update,
            'num_samples': total_samples,
            'loss': total_loss / total_samples
        }
    
    def aggregate_updates(self,
                         client_updates: List[Dict[str, Any]]) -> None:
        """Aggregate client updates using FedAvg"""
        total_samples = sum([c['num_samples'] for c in client_updates])
        
        # Weighted average of client updates
        aggregated_update = {}
        for name in self.global_model.state_dict().keys():
            aggregated_update[name] = torch.zeros_like(
                self.global_model.state_dict()[name]
            )
            
            for client in client_updates:
                weight = client['num_samples'] / total_samples
                aggregated_update[name] += weight * client['updates'][name]
        
        # Update global model
        global_state = self.global_model.state_dict()
        for name in global_state.keys():
            global_state[name] += aggregated_update[name]
        
        self.global_model.load_state_dict(global_state)
        
        # Sync client models
        for client_model in self.client_models:
            client_model.load_state_dict(self.global_model.state_dict())
    
    def secure_aggregation(self,
                          client_updates: List[Dict[str, torch.Tensor]],
                          use_encryption: bool = True) -> Dict[str, torch.Tensor]:
        """Secure aggregation with optional homomorphic encryption"""
        if use_encryption:
            # Placeholder for homomorphic encryption
            # In practice, use libraries like PySyft or TenSEAL
            pass
        
        # Add noise for additional privacy (simple differential privacy)
        noise_scale = self.config.get('noise_scale', 0.01)
        
        aggregated = {}
        for name in client_updates[0]['updates'].keys():
            # Average updates
            stacked = torch.stack([c['updates'][name] for c in client_updates])
            aggregated[name] = stacked.mean(dim=0)
            
            # Add Gaussian noise
            if noise_scale > 0:
                noise = torch.randn_like(aggregated[name]) * noise_scale
                aggregated[name] += noise
        
        return aggregated
    
    def train_round(self,
                   client_loaders: List,
                   selected_clients: Optional[List[int]] = None) -> Dict[str, float]:
        """Execute one round of federated training"""
        if selected_clients is None:
            # Random client selection
            num_selected = max(1, int(self.num_clients * self.config.get('client_fraction', 0.1)))
            selected_clients = np.random.choice(
                self.num_clients, num_selected, replace=False
            ).tolist()
        
        # Collect updates from selected clients
        client_updates = []
        for client_id in selected_clients:
            update = self.client_update(
                client_id,
                client_loaders[client_id],
                epochs=self.config.get('client_epochs', 5)
            )
            client_updates.append(update)
        
        # Aggregate updates
        self.aggregate_updates(client_updates)
        
        # Calculate average metrics
        avg_loss = np.mean([c['loss'] for c in client_updates])
        
        return {
            'avg_loss': avg_loss,
            'num_clients': len(selected_clients)
        }