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
import torch.multiprocessing as mp

# Import DCT stream
from models.backbones import DCTStream

# ============================================================================
# DCT UTILITIES
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

# ============================================================================
# DCT-ONLY DATASET
# ============================================================================
class DCTOnlyDataset(Dataset):
    """Dataset that returns ONLY DCT features (no RGB)"""
    def __init__(self, real_dir, fake_dir, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.real_images = glob(os.path.join(real_dir, '*.png')) + glob(os.path.join(real_dir, '*.jpg'))
        self.fake_images = glob(os.path.join(fake_dir, '*.png')) + glob(os.path.join(fake_dir, '*.jpg'))
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        print(f"{mode.upper()} - Real: {len(self.real_images)}, Fake: {len(self.fake_images)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            face = transforms.Resize((160, 160))(image)
            
            # Apply augmentation if training
            if self.transform and self.mode == 'train':
                face = self.transform(face)
            
            face_tensor = transforms.ToTensor()(face)
            
            # Convert to DCT and normalize
            dct_tensor = rgb_to_dct(face_tensor * 255.0)
            dct_tensor = normalize_dct(dct_tensor)
            
            return dct_tensor, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))

# ============================================================================
# DCT-ONLY MODEL
# ============================================================================
class DCTOnlyModel(nn.Module):
    """Standalone DCT stream model with classifier"""
    def __init__(self, num_classes=2):
        super(DCTOnlyModel, self).__init__()
        self.dct_stream = DCTStream()
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, dct_input):
        features = self.dct_stream(dct_input)
        logits = self.classifier(features)
        return logits

# ============================================================================
# TRAINING HELPERS
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc='Training DCT-Only', leave=False)
    for dct, labels in progress_bar:
        dct, labels = dct.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(dct)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for dct, labels in tqdm(dataloader, desc='Validating', leave=False):
            dct, labels = dct.to(device), labels.to(device)
            
            outputs = model(dct)
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

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# ============================================================================
# MAIN
# ============================================================================
def main():
    # CONFIG
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 7
    DATASET_ROOT = 'data_dfdc'
    SAVE_DIR = 'trained_dct_only'
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training DCT-Only Model on {device}")
    print(f"Save directory: {SAVE_DIR}/\n")

    # DATA
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    
    train_dataset = DCTOnlyDataset(
        f'{DATASET_ROOT}/train/real', 
        f'{DATASET_ROOT}/train/fake', 
        transform=train_transform, 
        mode='train'
    )
    val_dataset = DCTOnlyDataset(
        f'{DATASET_ROOT}/val/real', 
        f'{DATASET_ROOT}/val/fake', 
        mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # MODEL
    model = DCTOnlyModel(num_classes=2).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # OPTIMIZER
    num_real = len(train_dataset.real_images)
    num_fake = len(train_dataset.fake_images)
    class_weights = torch.tensor([
        (num_real + num_fake) / num_real, 
        (num_real + num_fake) / num_fake
    ]).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

    # TRAINING LOOP
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'val_auc': []
    }
    
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(model, val_loader, criterion, device)
        
        # Store history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_f1'].append(val_f1)
        training_history['val_auc'].append(val_auc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"Val   Prec: {val_prec:.4f} | Val   Rec: {val_rec:.4f}")
        print(f"Val   F1:   {val_f1:.4f} | Val   AUC: {val_auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVE_DIR, 'best_dct_only.pth')
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved best model (Acc: {val_acc:.4f}) to {save_path}")
        
        scheduler.step(val_acc)
        early_stopping(val_acc)
        
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            break

    # Save training summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary = {
        'best_val_acc': best_val_acc,
        'total_epochs': epoch + 1,
        'history': training_history,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'model': 'DCT-Only',
            'timestamp': timestamp
        }
    }
    
    summary_path = os.path.join(SAVE_DIR, f'training_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"✓ Summary saved to: {summary_path}")

# Add this BEFORE main()
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()