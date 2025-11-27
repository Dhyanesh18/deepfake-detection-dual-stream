import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.backbone_resnet import DCTStream
from train_resnet import DeepfakeDataset  # Reuse your dataset


# ============================================================================
# STAGE 1: Pre-train Frequency Stream Alone
# ============================================================================

def pretrain_frequency_stream(train_loader, val_loader, device, epochs=10):
    """Train frequency stream independently as a classifier"""
    
    print("\n" + "="*70)
    print("STAGE 1: PRE-TRAINING FREQUENCY STREAM")
    print("="*70)
    
    # Frequency stream + classifier
    freq_model = nn.Sequential(
        DCTStream(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(freq_model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Train
        freq_model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for rgb, dct, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            dct, labels = dct.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = freq_model(dct)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        
        # Validate
        freq_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for rgb, dct, labels in val_loader:
                dct, labels = dct.to(device), labels.to(device)
                outputs = freq_model(dct)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(freq_model[0].state_dict(), 'pretrained_frequency_stream.pth')
            print(f"✓ Saved best frequency stream (Val Acc: {best_val_acc:.4f})")
        
        scheduler.step(val_acc)
    
    print(f"\n✓ Frequency stream pre-training complete. Best Val Acc: {best_val_acc:.4f}")
    return 'pretrained_frequency_stream.pth'


# ============================================================================
# STAGE 2: Train Full Dual-Stream Model
# ============================================================================

def train_dual_stream_with_pretrained(train_loader, val_loader, device, 
                                     pretrained_freq_path, epochs=15):
    """Train full dual-stream model with pre-trained frequency stream"""
    
    print("\n" + "="*70)
    print("STAGE 2: TRAINING FULL DUAL-STREAM MODEL")
    print("="*70)
    
    from models.backbone_resnet import DualStreamModelResNet
    
    model = DualStreamModelResNet(num_classes=2, pretrained=True).to(device)
    
    # Load pre-trained frequency stream weights
    print(f"Loading pre-trained frequency stream from {pretrained_freq_path}")
    model.frequency_stream.load_state_dict(torch.load(pretrained_freq_path))
    
    # Freeze frequency stream initially (optional - helps stability)
    for param in model.frequency_stream.parameters():
        param.requires_grad = False
    
    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates
    optimizer = optim.Adam([
        {'params': model.spatial_stream.parameters(), 'lr': 1e-5},
        {'params': model.fusion.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Unfreeze frequency stream after 3 epochs
        if epoch == 3:
            print("\n→ Unfreezing frequency stream for fine-tuning")
            for param in model.frequency_stream.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': model.frequency_stream.parameters(), 'lr': 1e-4})
        
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for rgb, dct, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs, attn = model(rgb, dct)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = correct / total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for rgb, dct, labels in val_loader:
                rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
                outputs, attn = model(rgb, dct)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model_dual_pretrained.pth')
            print(f"✓ Saved best model (Val Acc: {best_val_acc:.4f})")
        
        scheduler.step(val_acc)
    
    print(f"\n✓ Full training complete. Best Val Acc: {best_val_acc:.4f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    from torchvision import transforms
    
    BATCH_SIZE = 32
    DATASET_ROOT = 'data_dfdc'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
    
    # Load datasets
    train_dataset = DeepfakeDataset(
        f'{DATASET_ROOT}/train/real',
        f'{DATASET_ROOT}/train/fake',
        transform=train_transform,
        mode='train'
    )
    val_dataset = DeepfakeDataset(
        f'{DATASET_ROOT}/val/real',
        f'{DATASET_ROOT}/val/fake',
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Stage 1: Pre-train frequency stream
    pretrained_path = pretrain_frequency_stream(
        train_loader, val_loader, device, epochs=10
    )
    
    # Stage 2: Train full dual-stream model
    train_dual_stream_with_pretrained(
        train_loader, val_loader, device, pretrained_path, epochs=15
    )


if __name__ == '__main__':
    main()