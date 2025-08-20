import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import json
import os
from datetime import datetime
from typing import Dict, Tuple, Optional
from tqdm.auto import tqdm
from .regression_utils import denormalize_material_params
#import wandb  # Optional: for experiment tracking


    
class PulseParameterNet(nn.Module):
    """CNN architecture for predicting material parameters from time domain pulses."""
    
    def __init__(self, input_length: int = 1024, num_params: int = 9):
        super().__init__()
        
        # 1D CNN backbone - designed for temporal features
        self.backbone = nn.Sequential(
            # Layer 1: Capture broad pulse features
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            
            # Layer 2: Medium-scale delay patterns
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(64), 
            nn.ReLU(inplace=True),
            
            # Layer 3: Fine pulse shape features
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Layer 4: Local amplitude variations
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2), 
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_params)  # Output raw parameters
        )
        
    def forward(self, x):
        # x: (batch, 1, 1024)
        features = self.backbone(x)  # (batch, 256, 1)
        features = features.squeeze(-1)  # (batch, 256)
        params = self.head(features)  # (batch, 9)
        return params


class ParameterTrainer:
    """Training and evaluation framework for parameter estimation models."""
    
    def __init__(self, model: nn.Module, param_ranges: Dict, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.param_ranges = param_ranges
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
        # Target accuracies for evaluation
        self.target_accuracies = {
            'n': 0.1,
            'k': 0.001, 
            'd': 50e-6
        }
        
    def train_epoch(self, dataloader: DataLoader, optimizer, criterion):
        """Train for one epoch with progress bar."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc='Training', leave=False)
        for pulses, params in pbar:
            pulses = pulses.to(self.device)
            params = params.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(pulses)
            loss = criterion(predictions, params)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
        return total_loss / num_batches
    
    def evaluate(self, dataloader: DataLoader, criterion) -> Dict:
        """Evaluate model and return comprehensive metrics."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(dataloader, desc='Evaluating', leave=False)
        with torch.no_grad():
            for pulses, params in pbar:
                pulses = pulses.to(self.device)
                params = params.to(self.device)
                
                predictions = self.model(pulses)
                loss = criterion(predictions, params)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(params.cpu())
                
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Denormalize for evaluation
        dataset = dataloader.dataset
        pred_denorm = denormalize_material_params(predictions)
        target_denorm = denormalize_material_params(targets)
        
        # Calculate metrics
        metrics = self._calculate_metrics(pred_denorm, target_denorm)
        metrics['loss'] = total_loss / len(dataloader)
        
        return metrics
    
    def _calculate_metrics(self, predictions: torch.Tensor, 
                          targets: torch.Tensor) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        param_names = ['n1', 'k1', 'd1', 'n2', 'k2', 'd2', 'n3', 'k3', 'd3']
        param_types = ['n', 'k', 'd'] * 3
        
        # Overall metrics
        mae = F.l1_loss(predictions, targets).item()
        rmse = torch.sqrt(F.mse_loss(predictions, targets)).item()
        
        metrics['overall'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2_score(targets.numpy(), predictions.numpy())
        }
        
        # Per-parameter metrics
        metrics['parameters'] = {}
        within_tolerance_count = 0
        
        for i, (param_name, param_type) in enumerate(zip(param_names, param_types)):
            pred_param = predictions[:, i].numpy()
            target_param = targets[:, i].numpy()
            
            param_mae = np.mean(np.abs(pred_param - target_param))
            param_rmse = np.sqrt(np.mean((pred_param - target_param) ** 2))
            param_r2 = r2_score(target_param, pred_param)
            
            # Check accuracy requirements
            tolerance = self.target_accuracies[param_type]
            within_tolerance = np.mean(np.abs(pred_param - target_param) <= tolerance)
            
            metrics['parameters'][param_name] = {
                'mae': param_mae,
                'rmse': param_rmse, 
                'r2': param_r2,
                'within_tolerance': within_tolerance,
                'tolerance': tolerance
            }
            
            within_tolerance_count += within_tolerance
        
        # Average tolerance satisfaction
        metrics['avg_tolerance_satisfaction'] = within_tolerance_count / 9
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, lr: float = 1e-3, 
              save_dir: str = 'regression_experiments', experiment_name: str = None):
        """Full training loop with validation and model saving."""
        
        if experiment_name is None:
            experiment_name = f"pulse_param_net_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        print(f"Starting training: {experiment_name}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")
        
        # Main training loop with progress bar
        epoch_pbar = tqdm(range(num_epochs), desc='Training Progress')
        
        for epoch in epoch_pbar:
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.train_losses.append(train_loss)
            
            # Validation
            val_metrics = self.evaluate(val_loader, criterion)
            val_loss = val_metrics['loss']
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.6f}',
                'Val Loss': f'{val_loss:.6f}', 
                'Val R²': f'{val_metrics["overall"]["r2"]:.4f}',
                'Tolerance': f'{val_metrics["avg_tolerance_satisfaction"]:.3f}'
            })
            
            # Detailed progress every 10 epochs
            if epoch % 10 == 0:
                tqdm.write(f"Epoch {epoch:3d}: Train Loss={train_loss:.6f}, "
                        f"Val Loss={val_loss:.6f}, Val R²={val_metrics['overall']['r2']:.4f}, "
                        f"Avg Tolerance={val_metrics['avg_tolerance_satisfaction']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model checkpoint
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'param_ranges': self.param_ranges,
                    'val_metrics': val_metrics,
                    'epoch': epoch
                }, os.path.join(save_dir, f"{experiment_name}_best.pth"))
                
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation
        print("\nRunning final evaluation...")
        final_metrics = self.evaluate(val_loader, criterion)
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'final_metrics': final_metrics,
            'param_ranges': self.param_ranges
        }
        
        with open(os.path.join(save_dir, f"{experiment_name}_history.json"), 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        return final_metrics
    
    def plot_training_history(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.7)
        plt.plot(self.val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses, label='Val Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.title('Validation Loss')
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()


# --- Vectorize raw target from dataset ---
def convert_target_to_vector(material_params_entry):
    """
    Convert [(n_complex, d), ...] to 9D vector: [n1, k1, d1, n2, k2, d2, ...]
    """
    out = []
    for n_complex, d in material_params_entry:
        n = n_complex.real
        k = n_complex.imag  # extinction coefficient (-ve means loss)
        out.extend([n, k, d])
    return torch.tensor(out, dtype=torch.float32)


# Example usage and testing
def load_dataset_from_pt(file_path: str, param_ranges: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load dataset from .pt file with the specified format."""
    if param_ranges is None:
        param_ranges = {
            'n': (1.0, 8.0),
            'k': (-0.1, 0.001),
            'd': (0.05e-3, 0.5e-3)
        }
    
    print(f"Loading dataset from {file_path}...")
    
    # Load the dataset
    data = torch.load(file_path, weights_only=False)
    
    # Extract components
    synthetic_data = data["synthetic_data"]  # Should be (N, pulse_length) 
    material_params = data["material_params"]  # Should be (N, 9) for 3 layers
    material_params = convert_target_to_vector(material_params)
    num_layers = data["num_layers"]
    
    print(f"Loaded {len(synthetic_data)} samples")
    print(f"Pulse shape: {synthetic_data.shape}")
    print(f"Parameters shape: {material_params.shape}")
    print(f"Number of layers: {num_layers}")
    
    # Convert to numpy if needed
    if isinstance(synthetic_data, torch.Tensor):
        pulses = synthetic_data.numpy()
    else:
        pulses = np.array(synthetic_data)
        
    if isinstance(material_params, torch.Tensor):
        parameters = material_params.numpy()
    else:
        parameters = np.array(material_params)
    
    # Verify parameter ranges
    print("\nParameter statistics:")
    param_names = ['n1', 'k1', 'd1', 'n2', 'k2', 'd2', 'n3', 'k3', 'd3']
    for i, name in enumerate(param_names):
        if i < parameters.shape[1]:
            print(f"{name}: min={parameters[:, i].min():.6f}, max={parameters[:, i].max():.6f}")
    
    return pulses, parameters
    

'''
def main_notebook(data_file_path: str, test_size: float = 0.2, batch_size: int = 32, 
                 num_epochs: int = 100, lr: float = 1e-3):
    """Main function optimized for notebook usage."""
    
    # Configuration
    param_ranges = {
        'n': (1.0, 8.0),
        'k': (-0.1, 0.001),
        'd': (0.05e-3, 0.5e-3)
    }
    
    # Load your real data
    pulses, parameters = load_dataset_from_pt(data_file_path, param_ranges)
    
    # Train/test split
    n_samples = len(pulses)
    split_idx = int((1 - test_size) * n_samples)
    
    # Random shuffle for better split
    indices = np.random.permutation(n_samples)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_pulses, test_pulses = pulses[train_indices], pulses[test_indices]
    train_params, test_params = parameters[train_indices], parameters[test_indices]
    
    print(f"\nData split:")
    print(f"Training samples: {len(train_pulses)}")
    print(f"Test samples: {len(test_pulses)}")
    
    # Create datasets
    train_dataset = PulseDataset(train_pulses, train_params, param_ranges, augment=True)
    test_dataset = PulseDataset(test_pulses, test_params, param_ranges, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=0, pin_memory=True)  # num_workers=0 for notebooks
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # Initialize model and trainer
    pulse_length = pulses.shape[1]
    model = PulseParameterNet(input_length=pulse_length)
    trainer = ParameterTrainer(model, param_ranges)
    
    # Display model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel initialized:")
    print(f"Total parameters: {total_params:,}")
    print(f"Input pulse length: {pulse_length}")
    print(f"Device: {trainer.device}")
    
    # Train model
    print(f"\nStarting training for {num_epochs} epochs...")
    final_metrics = trainer.train(train_loader, test_loader, 
                                 num_epochs=num_epochs, lr=lr,
                                 experiment_name="notebook_experiment")
    
    # Print final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Overall R²: {final_metrics['overall']['r2']:.4f}")
    print(f"Overall RMSE: {final_metrics['overall']['rmse']:.6f}")
    print(f"Overall MAE: {final_metrics['overall']['mae']:.6f}")
    print(f"Average Tolerance Satisfaction: {final_metrics['avg_tolerance_satisfaction']:.4f}")
    
    print("\nPer-parameter results:")
    print(f"{'Parameter':<12} {'R²':<8} {'RMSE':<12} {'MAE':<12} {'Within Tol':<12} {'Target Tol':<12}")
    print("-" * 80)
    
    for param, metrics in final_metrics['parameters'].items():
        print(f"{param:<12} {metrics['r2']:<8.4f} {metrics['rmse']:<12.6f} "
              f"{metrics['mae']:<12.6f} {metrics['within_tolerance']:<12.3f} "
              f"{metrics['tolerance']:<12.6f}")
    
    # Plot training history
    trainer.plot_training_history()
    
    return trainer, final_metrics

'''