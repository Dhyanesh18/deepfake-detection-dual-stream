"""
Ablation Study Script
Systematically evaluates each component's contribution to the model
Uses pre-extracted face images (no video processing or MTCNN needed)
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.fftpack import dct
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
from datetime import datetime

# Import models
from models import XceptionBaseline
from models.backbones import XceptionWithCBAM, DCTStream, DualStreamModel


# ==================== DATASET CLASS (No MTCNN - Pre-extracted faces) ====================

def dct2d(block):
    """Apply 2D DCT on an image channel"""
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def rgb_to_dct(image):
    """Convert RGB image to DCT domain"""
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    dct_channels = []
    for c in range(3):
        dct_ch = dct2d(image[c].astype(np.float32))
        dct_channels.append(dct_ch)
    
    dct_image = np.stack(dct_channels, axis=0)
    return torch.from_numpy(dct_image).float()


def normalize_dct(dct_tensor):
    """Normalize DCT coefficients"""
    mean = dct_tensor.mean(dim=(1, 2), keepdim=True)
    std = dct_tensor.std(dim=(1, 2), keepdim=True)
    return (dct_tensor - mean) / (std + 1e-8)


class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection using pre-extracted face images
    No MTCNN needed - images are already cropped faces
    """
    def __init__(self, real_dir, fake_dir, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        
        # Get all image paths
        real_images = sorted(glob(os.path.join(real_dir, '*.png')) + 
                           glob(os.path.join(real_dir, '*.jpg')))
        fake_images = sorted(glob(os.path.join(fake_dir, '*.png')) + 
                           glob(os.path.join(fake_dir, '*.jpg')))
        
        # Create dataset: (path, label)
        self.samples = [(img, 0) for img in real_images] + [(img, 1) for img in fake_images]
        
        print(f"{mode.upper()} - Real: {len(real_images)}, Fake: {len(fake_images)}, Total: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            # Load pre-extracted face image
            image = Image.open(img_path).convert('RGB')
            
            # Resize to 160x160 (standard face size)
            image = transforms.Resize((160, 160))(image)
            
            # Apply augmentations (only during training, before tensor conversion)
            if self.transform and self.mode == 'train':
                image = self.transform(image)
            
            # Convert to tensor [0, 1]
            face_tensor = transforms.ToTensor()(image)
            
            # Normalize RGB for spatial stream (ImageNet stats)
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
            rgb_tensor = normalize(face_tensor)
            
            # Convert to DCT domain for frequency stream
            # Scale to [0, 255] for DCT conversion
            dct_tensor = rgb_to_dct(face_tensor * 255.0)
            dct_tensor = normalize_dct(dct_tensor)
            
            return rgb_tensor, dct_tensor, label
            
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random valid sample
            return self.__getitem__(np.random.randint(0, len(self)))


# ==================== TRAINING & EVALUATION FUNCTIONS ====================

def train_epoch(model, dataloader, criterion, optimizer, device, model_type='dual'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        rgb, dct, labels = batch
        rgb = rgb.to(device)
        dct = dct.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'baseline':
            outputs = model(rgb)
        else:  # dual stream
            outputs, _ = model(rgb, dct)
        
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        running_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device, model_type='dual'):
    """Evaluate model on validation/test set"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
        for batch in progress_bar:
            rgb, dct, labels = batch
            rgb = rgb.to(device)
            dct = dct.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if model_type == 'baseline':
                outputs = model(rgb)
            else:  # dual stream
                outputs, _ = model(rgb, dct)
            
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
            
            running_loss += loss.item()
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Debug: Check unique values
    unique_labels = np.unique(all_labels)
    unique_preds = np.unique(all_preds)
    
    # Calculate metrics
    loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Check if this is truly binary classification
    if len(unique_labels) > 2 or len(unique_preds) > 2:
        print(f"\nWarning: Detected more than 2 classes!")
        print(f"Unique labels: {unique_labels}")
        print(f"Unique predictions: {unique_preds}")
        # Use weighted average for multiclass
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        auc = 0.0  # AUC not well-defined for multiclass
    elif len(unique_preds) < 2:
        # Model is predicting only one class
        print(f"\nWarning: Model predicting only class {unique_preds[0]}")
        precision = 0.0
        recall = 0.0
        f1 = 0.0
        auc = 0.0
    else:
        # Binary classification - use binary metrics
        # Make sure labels are actually 0 and 1
        assert set(unique_labels).issubset({0, 1}), f"Labels must be 0 or 1, got {unique_labels}"
        assert set(unique_preds).issubset({0, 1}), f"Predictions must be 0 or 1, got {unique_preds}"
        
        precision = precision_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        recall = recall_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='binary', pos_label=1, zero_division=0)
        
        # Handle AUC calculation
        try:
            if len(unique_labels) >= 2:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                auc = 0.0
        except ValueError as e:
            print(f"\nWarning: Could not calculate AUC - {e}")
            auc = 0.0
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


# ==================== ABLATION STUDY CONFIGURATIONS ====================

def get_ablation_configs():
    """Define all ablation study configurations"""
    configs = {
        'xception_cbam': {
            'name': 'Xception + CBAM',
            'model_type': 'xception_cbam',
            'description': 'Xception with CBAM attention modules'
        },
        'dual_full': {
            'name': 'Full Model (All Components)',
            'model_type': 'dual',
            'description': 'Complete model with all proposed components'
        }
    }
    return configs


def create_model(config_name, num_classes=2, device='cuda'):
    """Create model based on configuration"""
    
    if config_name == 'baseline':
        model = XceptionBaseline(num_classes=num_classes, pretrained=True)
        model_type = 'baseline'
    
    elif config_name == 'xception_cbam':
        # Create Xception with CBAM but no DCT stream
        base_model = XceptionWithCBAM(pretrained=True)
        
        # Get the feature dimension
        if hasattr(base_model, 'feature_dim'):
            feature_dim = base_model.feature_dim
        else:
            # Try to infer from a dummy forward pass
            dummy_input = torch.randn(1, 3, 160, 160).to(device)
            with torch.no_grad():
                dummy_output = base_model(dummy_input)
                if isinstance(dummy_output, tuple):
                    dummy_output = dummy_output[0]
                feature_dim = dummy_output.shape[1] if len(dummy_output.shape) > 1 else dummy_output.shape[0]
        
        # Create a wrapper model with classifier
        class XceptionCBAMClassifier(nn.Module):
            def __init__(self, backbone, feature_dim, num_classes):
                super().__init__()
                self.backbone = backbone
                self.avgpool = nn.AdaptiveAvgPool2d(1)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(feature_dim, num_classes)
            
            def forward(self, x):
                x = self.backbone(x)
                if len(x.shape) == 4:  # (B, C, H, W)
                    x = self.avgpool(x)
                x = self.flatten(x)
                x = self.fc(x)
                return x
        
        model = XceptionCBAMClassifier(base_model, feature_dim, num_classes)
        model_type = 'baseline'
    
    elif config_name == 'dual_simple':
        # Dual stream with simple concatenation
        model = DualStreamModel(num_classes=num_classes, pretrained=True)
        model_type = 'dual'
    
    elif config_name == 'dual_learnable':
        # Import learnable fusion model if available
        try:
            from models.fusion import LearnableFusion
            model = DualStreamModel(num_classes=num_classes, pretrained=True)
            model_type = 'dual'
        except ImportError:
            print("LearnableFusion not available, using DualStreamModel")
            model = DualStreamModel(num_classes=num_classes, pretrained=True)
            model_type = 'dual'
    
    elif config_name == 'dual_full':
        # Full model with all components
        model = DualStreamModel(num_classes=num_classes, pretrained=True)
        model_type = 'dual'
    
    else:
        raise ValueError(f"Unknown config: {config_name}")
    
    model = model.to(device)
    
    # Verify output shape
    dummy_input = torch.randn(2, 3, 160, 160).to(device)
    with torch.no_grad():
        if model_type == 'baseline':
            dummy_output = model(dummy_input)
        else:
            dummy_dct = torch.randn(2, 3, 160, 160).to(device)
            dummy_output, _ = model(dummy_input, dummy_dct)
    
    assert dummy_output.shape[1] == num_classes, \
        f"Model output shape {dummy_output.shape} doesn't match num_classes={num_classes}"
    
    print(f"Model output shape verified: {dummy_output.shape} (batch_size, {num_classes})")
    
    return model, model_type


# ==================== MAIN ABLATION STUDY ====================

def run_ablation_study(
    dataset_root='data_dfdc',
    output_dir='ablation_results',
    batch_size=16,
    epochs=10,
    learning_rate=1e-4
):
    """Run complete ablation study"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f'ablation_results_{timestamp}.json')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY - Pre-extracted Face Images")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Dataset: {dataset_root}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*70}\n")
    
    # Check dataset
    if not os.path.exists(dataset_root):
        raise ValueError(f"Dataset not found at {dataset_root}")
    
    # Dataset paths
    train_real = os.path.join(dataset_root, 'train', 'real')
    train_fake = os.path.join(dataset_root, 'train', 'fake')
    val_real = os.path.join(dataset_root, 'val', 'real')
    val_fake = os.path.join(dataset_root, 'val', 'fake')
    test_real = os.path.join(dataset_root, 'test', 'real')
    test_fake = os.path.join(dataset_root, 'test', 'fake')
    
    # Augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DeepfakeDataset(train_real, train_fake, transform=train_transform, mode='train')
    val_dataset = DeepfakeDataset(val_real, val_fake, mode='val')
    test_dataset = DeepfakeDataset(test_real, test_fake, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=8, pin_memory=True)
    
    print(f"Datasets loaded successfully\n")
    
    # Get ablation configurations
    configs = get_ablation_configs()
    
    # Store results
    all_results = {}
    
    # Run each configuration
    for config_key, config_info in configs.items():
        print(f"\n{'='*70}")
        print(f"Running: {config_info['name']}")
        print(f"Description: {config_info['description']}")
        print(f"{'='*70}\n")
        
        # Create model
        model, model_type = create_model(config_key, num_classes=2, device=device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}\n")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                         patience=2, factor=0.5, verbose=True)
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_metrics': []
        }
        
        best_val_acc = 0.0
        best_model_path = os.path.join(output_dir, f'{config_key}_best.pth')
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 70)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, model_type
            )
            
            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device, model_type)
            
            # Update scheduler
            scheduler.step(val_metrics['accuracy'])
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_metrics'].append(val_metrics)
            
            # Print epoch results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
            print(f"        Prec: {val_metrics['precision']:.4f}, Rec: {val_metrics['recall']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            print(f"        Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), best_model_path)
                print(f"✓ Saved best model (Val Acc: {best_val_acc:.4f})")
        
        # Load best model for testing
        print(f"\nLoading best model for testing...")
        model.load_state_dict(torch.load(best_model_path))
        
        # Test evaluation
        print("Testing...")
        test_metrics = evaluate(model, test_loader, criterion, device, model_type)
        
        print(f"\n{'='*70}")
        print(f"FINAL TEST RESULTS - {config_info['name']}")
        print(f"{'='*70}")
        print(f"Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall:    {test_metrics['recall']:.4f}")
        print(f"F1-Score:  {test_metrics['f1']:.4f}")
        print(f"AUC-ROC:   {test_metrics['auc']:.4f}")
        print(f"{'='*70}\n")
        
        # Store results
        all_results[config_key] = {
            'config': config_info,
            'model_params': {
                'total': int(total_params),
                'trainable': int(trainable_params)
            },
            'training_history': history,
            'best_val_accuracy': float(best_val_acc),
            'test_metrics': {k: float(v) for k, v in test_metrics.items()}
        }
        
        # Save intermediate results
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Intermediate results saved to: {results_file}\n")
    
    # Print summary comparison
    print(f"\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}\n")
    print(f"{'Configuration':<35} {'Test Acc':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 70)
    
    for config_key, results in all_results.items():
        config_name = results['config']['name']
        test_acc = results['test_metrics']['accuracy']
        test_f1 = results['test_metrics']['f1']
        test_auc = results['test_metrics']['auc']
        print(f"{config_name:<35} {test_acc:<12.4f} {test_f1:<12.4f} {test_auc:<12.4f}")
    
    print(f"\n{'='*70}")
    print(f"Final results saved to: {results_file}")
    print(f"Model checkpoints saved to: {output_dir}/")
    print(f"{'='*70}\n")
    
    return all_results


if __name__ == '__main__':
    # Configuration
    DATASET_ROOT = 'data_dfdc'  # Pre-extracted face images
    OUTPUT_DIR = 'ablation_results'
    BATCH_SIZE = 48
    EPOCHS = 10  # Set to 10-15 for proper ablation study
    LEARNING_RATE = 1e-4
    
    # Run ablation study
    results = run_ablation_study(
        dataset_root=DATASET_ROOT,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )