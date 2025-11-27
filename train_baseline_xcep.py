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
from sklearn.metrics import accuracy_score

# IMPORT PLAIN XCEPTION
from models import XceptionBaseline 

# ============================================================================
# DATASET (Standard)
# ============================================================================
def dct2d(block):
    return dct(dct(block.T, norm="ortho").T, norm="ortho")

def rgb_to_dct(image):
    # We still keep this to maintain Dataset compatibility, 
    # but the model won't use it.
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    dct_channels = []
    for c in range(3):
        dct_ch = dct2d(image[c].astype(np.float32))
        dct_channels.append(dct_ch)
    dct_image = np.stack(dct_channels, axis=0)
    return torch.from_numpy(dct_image).float()

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None, mode='train'):
        self.transform = transform
        self.real_images = glob(os.path.join(real_dir, '*.png')) + glob(os.path.join(real_dir, '*.jpg'))
        self.fake_images = glob(os.path.join(fake_dir, '*.png')) + glob(os.path.join(fake_dir, '*.jpg'))
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        print(f"[{mode.upper()}] Baseline Data - Real: {len(self.real_images)}, Fake: {len(self.fake_images)}")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            face = transforms.Resize((160, 160))(image)
            if self.transform:
                face = self.transform(face)
            face_tensor = transforms.ToTensor()(face)
            # Xception standard normalization
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            rgb_tensor = normalize(face_tensor)
            
            # We still compute DCT to keep the class consistent, but ignore it in the loop
            dct_tensor = rgb_to_dct(face_tensor * 255.0) 
            
            return rgb_tensor, dct_tensor, label
        except Exception:
            return self.__getitem__(np.random.randint(0, len(self)))

# ============================================================================
# TRAINING LOOP (RGB ONLY)
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for rgb, _, labels in tqdm(dataloader, desc='Train Baseline Xception', leave=False):
        # NOTE: We use '_' to ignore the DCT tensor
        rgb, labels = rgb.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(rgb) # Single stream input
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for rgb, _, labels in dataloader:
            rgb, labels = rgb.to(device), labels.to(device)
            outputs = model(rgb)
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

# ============================================================================
# MAIN
# ============================================================================
def main():
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 1e-4
    DATASET_ROOT = 'data_dfdc'
    SAVE_DIR = 'trained_baseline_xception'
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2),
    ])
    train_ds = DeepfakeDataset(f'{DATASET_ROOT}/train/real', f'{DATASET_ROOT}/train/fake', transform=train_transform)
    val_ds = DeepfakeDataset(f'{DATASET_ROOT}/val/real', f'{DATASET_ROOT}/val/fake', mode='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # MODEL
    print("Initializing Baseline Xception...")
    model = XceptionBaseline(num_classes=2, pretrained=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{SAVE_DIR}/baseline_xception.pth')
            print("✓ Saved Best Model")

if __name__ == '__main__':
    main()