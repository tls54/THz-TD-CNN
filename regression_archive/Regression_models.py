import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


class ResidualBlock1D(nn.Module):
    """1D Residual block for better gradient flow"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm1 = nn.GroupNorm(min(32, channels//4), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.norm2 = nn.GroupNorm(min(32, channels//4), channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        return self.relu(out)


class MultiScaleBlock(nn.Module):
    """Multi-scale feature extraction with different kernel sizes"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Different kernel sizes to capture different temporal patterns
        self.conv_small = nn.Conv1d(in_channels, out_channels//3, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(in_channels, out_channels//3, kernel_size=7, padding=3)
        self.conv_large = nn.Conv1d(in_channels, out_channels//3, kernel_size=15, padding=7)
        
        self.norm = nn.GroupNorm(min(32, out_channels//4), out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        small = self.conv_small(x)
        medium = self.conv_medium(x)
        large = self.conv_large(x)
        
        out = torch.cat([small, medium, large], dim=1)
        return self.relu(self.norm(out))


class AdvancedTHzRegressor(nn.Module):
    """Advanced CNN architecture specifically designed for high-precision THz regression"""
    def __init__(self, input_channels=1, output_dim=9):
        super().__init__()
        
        # Initial feature extraction with larger kernels for THz signals
        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=15, padding=7)
        self.norm1 = nn.GroupNorm(16, 128)
        self.pool1 = nn.MaxPool1d(2)
        
        # Multi-scale feature extraction
        self.multiscale1 = MultiScaleBlock(128, 256)
        self.pool2 = nn.MaxPool1d(2)
        
        # Residual blocks for deeper learning
        self.res1 = ResidualBlock1D(256, kernel_size=5)
        self.res2 = ResidualBlock1D(256, kernel_size=5)
        
        # More feature extraction
        self.conv2 = nn.Conv1d(256, 512, kernel_size=7, padding=3)
        self.norm2 = nn.GroupNorm(32, 512)
        self.pool3 = nn.MaxPool1d(2)
        
        # Additional residual blocks
        self.res3 = ResidualBlock1D(512, kernel_size=3)
        self.res4 = ResidualBlock1D(512, kernel_size=3)
        
        # Final conv layer
        self.conv3 = nn.Conv1d(512, 512, kernel_size=5, padding=2)
        self.norm3 = nn.GroupNorm(32, 512)
        self.pool4 = nn.MaxPool1d(2)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Large MLP head for precision
        self.fc1 = nn.Linear(512, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Feature extraction
        x = self.pool1(self.relu(self.norm1(self.conv1(x))))
        x = self.pool2(self.multiscale1(x))
        
        # Residual processing
        x = self.res1(x)
        x = self.res2(x)
        
        # More feature extraction
        x = self.pool3(self.relu(self.norm2(self.conv2(x))))
        
        # More residual processing
        x = self.res3(x)
        x = self.res4(x)
        
        # Final processing
        x = self.pool4(self.relu(self.norm3(self.conv3(x))))
        x = self.global_pool(x).squeeze(-1)
        
        # MLP head
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


class UltraHighPrecisionTHzRegressor(nn.Module):
    """Ultra-large architecture for extreme precision requirements"""
    def __init__(self, input_channels=1, output_dim=9):
        super().__init__()
        
        # Much larger initial feature extraction
        self.conv1 = nn.Conv1d(input_channels, 256, kernel_size=31, padding=15)
        self.norm1 = nn.GroupNorm(32, 256)
        
        # Sequential deeper architecture instead of parallel paths
        self.feature_extractor = nn.Sequential(
            # Block 1
            nn.Conv1d(256, 512, kernel_size=15, padding=7),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock1D(512, kernel_size=7),
            ResidualBlock1D(512, kernel_size=7),
            
            # Block 2
            nn.Conv1d(512, 768, kernel_size=11, padding=5),
            nn.GroupNorm(64, 768),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock1D(768, kernel_size=5),
            ResidualBlock1D(768, kernel_size=5),
            
            # Block 3
            nn.Conv1d(768, 1024, kernel_size=7, padding=3),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock1D(1024, kernel_size=5),
            ResidualBlock1D(1024, kernel_size=5),
            
            # Block 4
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ResidualBlock1D(1024, kernel_size=3),
            ResidualBlock1D(1024, kernel_size=3),
            ResidualBlock1D(1024, kernel_size=3),
        )
        
        # Self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=16, batch_first=True)
        self.attention_norm = nn.LayerNorm(1024)
        
        # Final processing
        self.final_conv = nn.Sequential(
            nn.Conv1d(1024, 1024, kernel_size=5, padding=2),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            ResidualBlock1D(1024, kernel_size=3),
            ResidualBlock1D(1024, kernel_size=3),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Ultra-large MLP with multiple stages for extreme precision
        self.precision_head = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim)
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Initial feature extraction
        x = self.relu(self.norm1(self.conv1(x)))
        
        # Deep feature extraction
        x = self.feature_extractor(x)
        
        # Self-attention for long-range dependencies
        # Reshape for attention: [B, L, C]
        B, C, L = x.shape
        attn_input = x.permute(0, 2, 1)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = self.attention_norm(attn_output + attn_input)
        
        # Back to conv format: [B, C, L]
        x = attn_output.permute(0, 2, 1)
        
        # Final processing
        features = self.final_conv(x).squeeze(-1)
        
        # Ultra-precision prediction head
        output = self.precision_head(features)
        
        return output


class MegaPrecisionTHzRegressor(nn.Module):
    """Even larger model without attention - simpler but more parameters"""
    def __init__(self, input_channels=1, output_dim=9):
        super().__init__()
        
        # Progressive channel expansion
        self.layers = nn.Sequential(
            # Stage 1: 1024 -> 512 length
            nn.Conv1d(input_channels, 128, kernel_size=31, padding=15),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(128, kernel_size=15),
            ResidualBlock1D(128, kernel_size=15),
            
            # Stage 2: 512 -> 256 length  
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.GroupNorm(32, 256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(256, kernel_size=11),
            ResidualBlock1D(256, kernel_size=11),
            ResidualBlock1D(256, kernel_size=7),
            
            # Stage 3: 256 -> 128 length
            nn.Conv1d(256, 512, kernel_size=11, padding=5),
            nn.GroupNorm(64, 512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(512, kernel_size=7),
            ResidualBlock1D(512, kernel_size=7),
            ResidualBlock1D(512, kernel_size=5),
            
            # Stage 4: 128 -> 64 length
            nn.Conv1d(512, 1024, kernel_size=7, padding=3),
            nn.GroupNorm(64, 1024),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(1024, kernel_size=5),
            ResidualBlock1D(1024, kernel_size=5),
            ResidualBlock1D(1024, kernel_size=3),
            ResidualBlock1D(1024, kernel_size=3),
            
            # Stage 5: 64 -> 32 length
            nn.Conv1d(1024, 1536, kernel_size=5, padding=2),
            nn.GroupNorm(64, 1536),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(1536, kernel_size=3),
            ResidualBlock1D(1536, kernel_size=3),
            ResidualBlock1D(1536, kernel_size=3),
            
            # Final processing
            nn.Conv1d(1536, 2048, kernel_size=3, padding=1),
            nn.GroupNorm(64, 2048),
            nn.ReLU(),
            
            ResidualBlock1D(2048, kernel_size=3),
            ResidualBlock1D(2048, kernel_size=3),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Massive MLP head
        self.head = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(4096, 4096),
            nn.ReLU(), 
            nn.Dropout(0.3),
            
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        x = self.layers(x).squeeze(-1)
        return self.head(x)


class MultiHeadTHzRegressor(nn.Module):
    """Multi-head architecture with separate heads for different parameter types"""
    def __init__(self, input_channels=1):
        super().__init__()
        
        # Shared backbone - same as above but without final FC layers
        self.backbone = self._build_backbone(input_channels)
        
        # Separate heads for different parameter types
        self.n_head = nn.Sequential(  # Refractive indices (n1, n2, n3)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.k_head = nn.Sequential(  # Extinction coefficients (k1, k2, k3)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.d_head = nn.Sequential(  # Thicknesses (d1, d2, d3)
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def _build_backbone(self, input_channels):
        return nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=15, padding=7),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            MultiScaleBlock(128, 256),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(256, kernel_size=5),
            ResidualBlock1D(256, kernel_size=5),
            
            nn.Conv1d(256, 512, kernel_size=7, padding=3),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            ResidualBlock1D(512, kernel_size=3),
            ResidualBlock1D(512, kernel_size=3),
            
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.GroupNorm(32, 512),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        
        n_pred = self.n_head(features)
        k_pred = self.k_head(features)
        d_pred = self.d_head(features)
        
        # Concatenate in the expected order: [n1,k1,d1,n2,k2,d2,n3,k3,d3]
        output = torch.empty(x.size(0), 9, device=x.device, dtype=x.dtype)
        output[:, [0, 3, 6]] = n_pred  # n1, n2, n3
        output[:, [1, 4, 7]] = k_pred  # k1, k2, k3
        output[:, [2, 5, 8]] = d_pred  # d1, d2, d3
        
        return output


class WeightedParameterLoss(nn.Module):
    """Weighted loss function emphasizing parameters that need higher precision"""
    def __init__(self, loss_type='smooth_l1'):
        super().__init__()
        # Weights based on target accuracy requirements
        # Higher weights for parameters needing higher precision
        self.register_buffer('weights', torch.tensor([
            10.0,  # n1 (need 0.01 accuracy)
            1000.0, # k1 (need 1e-5 accuracy) 
            100.0,  # d1 (need 1e-6 accuracy)
            10.0,   # n2
            1000.0, # k2
            100.0,  # d2
            10.0,   # n3
            1000.0, # k3
            100.0   # d3
        ], dtype=torch.float32))
        
        self.loss_type = loss_type
        
    def forward(self, pred, target):
        if self.loss_type == 'smooth_l1':
            loss = nn.functional.smooth_l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'mse':
            loss = nn.functional.mse_loss(pred, target, reduction='none')
        else:  # mae
            loss = nn.functional.l1_loss(pred, target, reduction='none')
            
        weighted_loss = loss * self.weights.unsqueeze(0)
        return weighted_loss.mean()


class MultiTaskLoss(nn.Module):
    """Separate losses for different parameter types"""
    def __init__(self):
        super().__init__()
        self.n_loss = nn.SmoothL1Loss()  # For refractive indices
        self.k_loss = nn.SmoothL1Loss()  # For extinction coefficients  
        self.d_loss = nn.SmoothL1Loss()  # For thicknesses
        
    def forward(self, pred, target):
        # Split predictions and targets by parameter type
        n_pred = pred[:, [0, 3, 6]]  # n1, n2, n3
        k_pred = pred[:, [1, 4, 7]]  # k1, k2, k3
        d_pred = pred[:, [2, 5, 8]]  # d1, d2, d3
        
        n_target = target[:, [0, 3, 6]]
        k_target = target[:, [1, 4, 7]]
        d_target = target[:, [2, 5, 8]]
        
        # Weight k and d losses more heavily due to precision requirements
        n_loss = self.n_loss(n_pred, n_target)
        k_loss = self.k_loss(k_pred, k_target) * 10.0  # Higher weight
        d_loss = self.d_loss(d_pred, d_target) * 5.0   # Higher weight
        
        return n_loss + k_loss + d_loss


def evaluate_model(model, dataloader, denormalize_fn, device):
    """Comprehensive evaluation including parameter-wise metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Denormalize for real-world metrics
    pred_denorm = denormalize_fn(all_preds)
    target_denorm = denormalize_fn(all_targets)
    
    # Calculate metrics for each parameter
    param_names = ['n1', 'k1', 'd1', 'n2', 'k2', 'd2', 'n3', 'k3', 'd3']
    metrics = {}
    
    for i, param in enumerate(param_names):
        pred_param = pred_denorm[:, i].numpy()
        target_param = target_denorm[:, i].numpy()
        
        rmse = np.sqrt(np.mean((pred_param - target_param) ** 2))
        mae = np.mean(np.abs(pred_param - target_param))
        r2 = r2_score(target_param, pred_param)
        max_err = np.max(np.abs(pred_param - target_param))
        
        metrics[param] = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'max_error': max_err
        }
    
    return metrics, pred_denorm, target_denorm


def advanced_train_model(
    model,
    train_loader,
    val_loader=None,
    num_epochs=500,
    initial_lr=1e-3,
    loss_fn=None,
    denormalize_fn=None,
    device=None,
    patience=50,
    min_delta=1e-6,
    save_best=True,
    model_save_path="best_model.pt"
):
    """Advanced training loop with learning rate scheduling, early stopping, and comprehensive monitoring"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Loss function
    if loss_fn is None:
        loss_fn = WeightedParameterLoss(loss_type='smooth_l1')
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    # Progress bar
    pbar = tqdm(range(num_epochs), desc='Training Progress')
    
    for epoch in pbar:
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'batch_loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Early stopping and best model saving
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                if save_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                    }, model_save_path)
            else:
                epochs_without_improvement += 1
            
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'patience': f'{epochs_without_improvement}/{patience}'
            })
            
            # Early stopping
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        else:
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Update learning rate
        scheduler.step()
        
        # Periodic detailed evaluation
        if (epoch + 1) % 50 == 0 and val_loader is not None and denormalize_fn is not None:
            print(f"\n--- Detailed Evaluation at Epoch {epoch+1} ---")
            metrics, _, _ = evaluate_model(model, val_loader, denormalize_fn, device)
            
            print("Parameter-wise RMSE in original units:")
            for param, metric in metrics.items():
                print(f"  {param}: RMSE = {metric['rmse']:.2e}, R² = {metric['r2']:.4f}")
    
    # Load best model if saved
    if save_best and val_loader is not None:
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nBest model loaded from epoch {checkpoint['epoch']+1}")
    
    return model, history


# Convenience function to get the right model
def get_thz_model(model_type='advanced', input_channels=1, output_dim=9):
    """Factory function to get the appropriate model"""
    if model_type == 'advanced':
        return AdvancedTHzRegressor(input_channels, output_dim)
    elif model_type == 'multihead':
        return MultiHeadTHzRegressor(input_channels)
    elif model_type == 'ultra':
        return UltraHighPrecisionTHzRegressor(input_channels, output_dim)
    elif model_type == 'mega':
        return MegaPrecisionTHzRegressor(input_channels, output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        

class AdaptivePrecisionLoss(nn.Module):
    """Adaptive loss that increases weights during training for harder parameters"""
    def __init__(self, initial_weights=None):
        super().__init__()
        if initial_weights is None:
            initial_weights = torch.tensor([
                50.0,   # n1 
                5000.0, # k1 (even higher weight)
                500.0,  # d1 
                50.0,   # n2
                5000.0, # k2 
                500.0,  # d2
                50.0,   # n3
                5000.0, # k3 
                500.0   # d3
            ], dtype=torch.float32)
        
        self.register_buffer('weights', initial_weights)
        self.epoch = 0
        
    def forward(self, pred, target):
        # Adaptive weighting - increase focus on harder parameters over time
        adaptive_weights = self.weights * (1 + 0.01 * self.epoch)  # Gradually increase
        
        # Use smooth L1 loss with adaptive weights
        loss = nn.functional.smooth_l1_loss(pred, target, reduction='none')
        weighted_loss = loss * adaptive_weights.unsqueeze(0)
        
        return weighted_loss.mean()
    
    def step_epoch(self):
        self.epoch += 1


# Convenience function to get the right loss
def get_loss_function(loss_type='weighted'):
    """Factory function to get the appropriate loss function"""
    if loss_type == 'weighted':
        return WeightedParameterLoss(loss_type='smooth_l1')
    elif loss_type == 'multitask':
        return MultiTaskLoss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'mse':
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")