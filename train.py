import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.fftpack import dct
import os
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from facenet_pytorch import MTCNN
import json
from datetime import datetime

# Import your models
from models import XceptionBaseline, DualStreamModel


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


# DATASET CLASS
class DeepfakeDataset(Dataset):
    """
    Dataset for deepfake detection with face detection and DCT preprocessing
    """
    def __init__(self, real_dir, fake_dir, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        
        # Shared MTCNN instance for face detection
        self.mtcnn = MTCNN(
            keep_all=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            post_process=False  # We'll handle normalization ourselves
        )
        
        # Load image paths
        self.real_images = glob(os.path.join(real_dir, '*.png')) + \
                            glob(os.path.join(real_dir, '*.jpg'))
        self.fake_images = glob(os.path.join(fake_dir, '*.png')) + \
                            glob(os.path.join(fake_dir, '*.jpg'))
        
        # Create labels (0 = real, 1 = fake)
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        print(f"{mode.capitalize()} dataset: {len(self.real_images)} real, "
                f"{len(self.fake_images)} fake (Total: {len(self.image_paths)})")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Detect and crop face using MTCNN
        try:
            face_tensor = self.mtcnn(image)
            if face_tensor is None:
                # If face not detected, use original image
                face = transforms.Resize((160, 160))(image)
            else:
                # Convert tensor back to PIL for consistency
                face = transforms.ToPILImage()(face_tensor)
                face = transforms.Resize((160, 160))(face)
        except Exception:
            # Fallback if MTCNN fails
            face = transforms.Resize((160, 160))(image)
        
        # Apply augmentations (only during training, before tensor conversion)
        if self.transform and self.mode == 'train':
            face = self.transform(face)
        
        # Convert to tensor [0, 1]
        face_tensor = transforms.ToTensor()(face)
        
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


# TRAINING FUNCTIONS
def train_epoch(model, dataloader, criterion, optimizer, device, model_type='dual'):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        rgb, dct, labels = batch
        rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'dual':
            outputs, _ = model(rgb, dct)
        else:  # baseline
            outputs = model(rgb)
        
        # Backward pass
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Metrics
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
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
            rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
            
            # Forward pass
            if model_type == 'dual':
                outputs, _ = model(rgb, dct)
            else:  # baseline
                outputs = model(rgb)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions and probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of fake class
    
    # Calculate metrics
    loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Handle AUC calculation safely
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0  # If only one class present in batch
    
    return loss, accuracy, precision, recall, f1, auc


def main():
    """Main training script"""
    
    # CONFIGURATION 
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 1
    LEARNING_RATE = 1e-4
    
    # Model selection: 'dual' or 'baseline'
    MODEL_TYPE = 'dual'  # Change to 'baseline' for XceptionNet baseline
    
    # Dataset paths - UPDATE THESE TO MATCH YOUR DATASET
    DATASET_ROOT = 'data_dfdc'  # Change if using different dataset
    train_real_dir = f'{DATASET_ROOT}/train/real'
    train_fake_dir = f'{DATASET_ROOT}/train/fake'
    val_real_dir = f'{DATASET_ROOT}/val/real'
    val_fake_dir = f'{DATASET_ROOT}/val/fake'
    test_real_dir = f'{DATASET_ROOT}/test/real'
    test_fake_dir = f'{DATASET_ROOT}/test/fake'
    
    # Check if dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"Error: Dataset not found at '{DATASET_ROOT}'")
        print("Please run create_dataset.py first or update DATASET_ROOT path")
        return
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("DEEPFAKE DETECTION - TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model Type: {MODEL_TYPE.upper()}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Dataset: {DATASET_ROOT}")
    print("="*70 + "\n")
    
    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    
    # Augmentation for training (applied to PIL images before tensor conversion)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DeepfakeDataset(
        train_real_dir, train_fake_dir, 
        transform=train_transform, mode='train'
    )
    val_dataset = DeepfakeDataset(val_real_dir, val_fake_dir, mode='val')
    test_dataset = DeepfakeDataset(test_real_dir, test_fake_dir, mode='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"Datasets loaded successfully\n")
    

    # MODEL INITIALIZATION
    
    print(f"Initializing {MODEL_TYPE.upper()} model...")
    if MODEL_TYPE == 'dual':
        model = DualStreamModel(num_classes=2, pretrained=True).to(device)
    else:
        model = XceptionBaseline(num_classes=2, pretrained=True).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model initialized\n")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    # TRAINING LOOP
    
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    print("Starting training...")
    print("="*70 + "\n")
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-"*70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, MODEL_TYPE
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device, MODEL_TYPE
        )
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"        Prec: {val_prec:.4f}, Rec: {val_rec:.4f}, "
                f"F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"        Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f'best_model_{MODEL_TYPE}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (acc: {best_val_acc:.4f})")
        
        print()
    
    # FINAL TEST EVALUATION
    
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    # Load best model
    model.load_state_dict(torch.load(f'best_model_{MODEL_TYPE}.pth'))
    
    # Evaluate on test set
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device, MODEL_TYPE
    )
    
    print(f"\nTest Results ({MODEL_TYPE.upper()} model):")
    print("-"*70)
    print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_auc:.4f}")
    print("="*70)
    
    
    summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': MODEL_TYPE,
        'hyperparameters': {
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LEARNING_RATE,
            'total_params': total_params,
            'trainable_params': trainable_params
        },
        'best_val_accuracy': float(best_val_acc),
        'test_results': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1_score': float(test_f1),
            'roc_auc': float(test_auc)
        },
        'training_history': training_history
    }
    
    summary_path = f'training_summary_{MODEL_TYPE}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nTraining summary saved to: {summary_path}")
    print(f"Best model saved to: best_model_{MODEL_TYPE}.pth")
    print("\nTraining complete!")


if __name__ == '__main__':
    main()