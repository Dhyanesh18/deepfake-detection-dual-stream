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

# Dataset Class
class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        
        self.mtcnn = MTCNN(
            keep_all=False, 
            device='cuda' if torch.cuda.is_available() else 'cpu',
            post_process=False
        )
        
        self.real_images = glob(os.path.join(real_dir, '*.png')) + \
                            glob(os.path.join(real_dir, '*.jpg'))
        self.fake_images = glob(os.path.join(fake_dir, '*.png')) + \
                            glob(os.path.join(fake_dir, '*.jpg'))
        
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        print(f"{mode} dataset: {len(self.real_images)} real, {len(self.fake_images)} fake")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        try:
            face_tensor = self.mtcnn(image)
            if face_tensor is None:
                face = transforms.Resize((299, 299))(image)
            else:
                face = transforms.ToPILImage()(face_tensor)
                face = transforms.Resize((299, 299))(face)
        except:
            face = transforms.Resize((299, 299))(image)
        
        if self.transform and self.mode == 'train':
            face = self.transform(face)
        
        face_tensor = transforms.ToTensor()(face)
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        rgb_tensor = normalize(face_tensor)
        
        dct_tensor = rgb_to_dct(face_tensor * 255.0)
        dct_tensor = normalize_dct(dct_tensor)
        
        return rgb_tensor, dct_tensor, label

# ============================================================================
# Training & Evaluation Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, model_type='dual'):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc='Training', leave=False):
        rgb, dct, labels = batch
        rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'dual':
            outputs, _ = model(rgb, dct)
        else:
            outputs = model(rgb)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device, model_type='dual'):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            rgb, dct, labels = batch
            rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
            
            if model_type == 'dual':
                outputs, _ = model(rgb, dct)
            else:
                outputs = model(rgb)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    loss = running_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0
    
    return loss, accuracy, precision, recall, f1, auc

# ============================================================================
# Main Ablation Study Function
# ============================================================================

def run_ablation_study(
    train_real_dir, train_fake_dir,
    val_real_dir, val_fake_dir,
    test_real_dir, test_fake_dir,
    batch_size=16,
    epochs=15,
    learning_rate=1e-4
):
    """
    Run comprehensive ablation study
    
    Configurations tested:
    1. Baseline: XceptionNet only (spatial stream)
    2. Dual-Stream (Full): Both streams + CBAM + Learnable Fusion
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = DeepfakeDataset(train_real_dir, train_fake_dir, 
                                   transform=train_transform, mode='train')
    val_dataset = DeepfakeDataset(val_real_dir, val_fake_dir, mode='val')
    test_dataset = DeepfakeDataset(test_real_dir, test_fake_dir, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    criterion = nn.CrossEntropyLoss()
    
    # Store results
    results = {
        'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'hyperparameters': {
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate
        },
        'configurations': {}
    }
    
    # Configuration 1: Baseline (Spatial Only - XceptionNet)
    print("\n" + "="*70)
    print("CONFIGURATION 1: Baseline (XceptionNet - Spatial Stream Only)")
    print("="*70)
    
    model = XceptionBaseline(num_classes=2, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, 'baseline')
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device, 'baseline'
        )
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val: Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, "
                f"F1={val_f1:.4f}, AUC={val_auc:.4f}")
        
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ablation_baseline.pth')
            print(f"Saved best baseline model (acc: {best_val_acc:.4f})")
    
    # Test baseline
    model.load_state_dict(torch.load('ablation_baseline.pth'))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device, 'baseline'
    )
    
    results['configurations']['baseline'] = {
        'description': 'XceptionNet spatial stream only',
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc)
    }
    
    print(f"\nBaseline Test Results:")
    print(f"  Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    print(f"  F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    

    # Configuration 2: Dual-Stream (Full Model)
    print("\n" + "="*70)
    print("CONFIGURATION 2: Dual-Stream (Full Model with CBAM + Learnable Fusion)")
    print("="*70)
    
    model = DualStreamModel(num_classes=2, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    best_val_acc = 0.0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device, 'dual')
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device, 'dual'
        )
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"Val: Acc={val_acc:.4f}, Prec={val_prec:.4f}, Rec={val_rec:.4f}, "
                f"F1={val_f1:.4f}, AUC={val_auc:.4f}")
        
        scheduler.step(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'ablation_dual_full.pth')
            print(f"Saved best dual-stream model (acc: {best_val_acc:.4f})")
    
    # Test full model
    model.load_state_dict(torch.load('ablation_dual_full.pth'))
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device, 'dual'
    )
    
    results['configurations']['dual_full'] = {
        'description': 'Dual-stream with CBAM and learnable fusion',
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1),
        'test_auc': float(test_auc)
    }
    
    print(f"\nDual-Stream Full Model Test Results:")
    print(f"  Accuracy: {test_acc:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
    print(f"  F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
    
    # Save Results and Print Summary
    
    with open('ablation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print comparison
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("="*70)
    
    baseline_acc = results['configurations']['baseline']['test_accuracy']
    dual_acc = results['configurations']['dual_full']['test_accuracy']
    improvement = dual_acc - baseline_acc
    improvement_pct = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
    
    print(f"\n1. Baseline (Spatial Only)")
    print(f"   Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)")
    
    print(f"\n2. Dual-Stream (Full)")
    print(f"   Accuracy: {dual_acc:.4f} ({dual_acc*100:.2f}%)")
    print(f"   Improvement: +{improvement:.4f} ({improvement_pct:.2f}% relative improvement)")
    
    print("\nDetailed Metrics Comparison:")
    print("-" * 70)
    print(f"{'Configuration':<30} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'AUC':<8}")
    print("-" * 70)
    
    for config_name, config_data in results['configurations'].items():
        desc = config_data['description'][:28]
        print(f"{desc:<30} "
                f"{config_data['test_accuracy']:<8.4f} "
                f"{config_data['test_precision']:<8.4f} "
                f"{config_data['test_recall']:<8.4f} "
                f"{config_data['test_f1']:<8.4f} "
                f"{config_data['test_auc']:<8.4f}")
    
    print("-" * 70)
    print(f"\nResults saved to: ablation_results.json")
    print("="*70)
    
    return results

# Main Execution

def main():
    """Run ablation study with your dataset"""
    
    # Configuration - UPDATE THESE PATHS!
    train_real_dir = 'data_small/train/real'
    train_fake_dir = 'data_small/train/fake'
    val_real_dir = 'data_small/val/real'
    val_fake_dir = 'data_small/val/fake'
    test_real_dir = 'data_small/test/real'
    test_fake_dir = 'data_small/test/fake'
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    
    print("="*70)
    print("ABLATION STUDY - Deepfake Detection")
    print("="*70)
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*70)
    
    # Run ablation study
    results = run_ablation_study(
        train_real_dir, train_fake_dir,
        val_real_dir, val_fake_dir,
        test_real_dir, test_fake_dir,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    print("\nAblation study complete!")

if __name__ == '__main__':
    main()