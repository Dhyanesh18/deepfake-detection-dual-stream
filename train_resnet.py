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
import json
from datetime import datetime

from models import XceptionBaseline
from models.backbone_resnet import DualStreamModelResNet 

# ============================================================================
# SHARED UTILS (Dataset & DCT)
# ============================================================================
def dct2d(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def rgb_to_dct(image):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    dct_channels = []
    for c in range(3):
        dct_ch = dct2d(image[c].astype(np.float32))
        dct_channels.append(dct_ch)
    dct_image = np.stack(dct_channels, axis=0)
    return torch.from_numpy(dct_image).float()

def normalize_dct(dct_tensor):
    mean = dct_tensor.mean(dim=(1, 2), keepdim=True)
    std = dct_tensor.std(dim=(1, 2), keepdim=True)
    return (dct_tensor - mean) / (std + 1e-8)

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.real_images = glob(os.path.join(real_dir, '*.png')) + glob(os.path.join(real_dir, '*.jpg'))
        self.fake_images = glob(os.path.join(fake_dir, '*.png')) + glob(os.path.join(fake_dir, '*.jpg'))
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        print(f"{mode.upper()} - Real: {len(self.real_images)}, Fake: {len(self.fake_images)}, Total: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            face = transforms.Resize((160, 160))(image)
            if self.transform and self.mode == 'train':
                face = self.transform(face)
            face_tensor = transforms.ToTensor()(face)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            rgb_tensor = normalize(face_tensor)
            dct_tensor = rgb_to_dct(face_tensor * 255.0)
            dct_tensor = normalize_dct(dct_tensor)
            return rgb_tensor, dct_tensor, label
        except Exception as e:
            return self.__getitem__(np.random.randint(0, len(self)))

# ============================================================================
# TRAINING HELPERS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    for batch in progress_bar:
        rgb, dct, labels = batch
        rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(rgb, dct)
            
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            rgb, dct, labels = batch
            rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
            
            outputs, _ = model(rgb, dct)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try: 
        auc = roc_auc_score(all_labels, all_probs)
    except: 
        auc = 0.0
    return loss, acc, prec, rec, f1, auc

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        if self.best_score is None:
            self.best_score = val_metric
        elif val_metric < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_metric
            self.counter = 0

# ============================================================================
# MAIN
# ============================================================================
def main():
    # CONFIG
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 7
    DATASET_ROOT = 'data_dfdc'
    SAVE_DIR = 'trained'
    ARCH = 'resnet'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {ARCH.upper()} Dual Stream on {device}")
    print(f"Saving to {SAVE_DIR}/\n")

    # --- DATA PREPARATION ---
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    
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
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}\n")

    # Model Init
    model = DualStreamModelResNet(num_classes=2, pretrained=True).to(device)
    save_name = 'best_model_resnet_dual.pth'
    
    print(f"Model initialized: ResNet50 Dual-Stream")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Optimizer
    num_real = len(train_dataset.real_images)
    num_fake = len(train_dataset.fake_images)
    weights = torch.tensor([(num_real+num_fake)/num_real, (num_real+num_fake)/num_fake]).to(device)
    
    print(f"Class weights: Real={weights[0]:.3f}, Fake={weights[1]:.3f}\n")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # Training Loop
    best_val_acc = 0.0
    
    print("=" * 70)
    print("STARTING TRAINING")
    print("Training on train set only")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Prec: {val_prec:.4f} | Val   Rec: {val_rec:.4f}")
        print(f"Val   F1:   {val_f1:.4f} | Val   AUC: {val_auc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, save_name))
            print(f"✓ Saved {save_name} (Val Acc: {best_val_acc:.4f})")
            
        scheduler.step(val_acc)
        early_stopping(val_acc)
        
        if early_stopping.early_stop:
            print("\n⚠️  Early stopping triggered")
            break

    # Save Summary
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary = {
        'model': ARCH,
        'timestamp': timestamp,
        'training_strategy': 'Train set only',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'best_val_acc': best_val_acc,
        'epochs': epoch + 1
    }
    
    with open(os.path.join(SAVE_DIR, f'summary_{ARCH}_{timestamp}.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()