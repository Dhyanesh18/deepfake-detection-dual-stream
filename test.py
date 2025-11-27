import torch
import torch.nn as nn
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

# --- IMPORTS ---
from models import DualStreamModel
from models.backbone_resnet import DualStreamModelResNet
from models.baseline import XceptionBaseline
from models.baseline_resnet import ResNetBaseline

# ============================================================================
# MODEL CONFIGURATIONS - Add/remove models here
# ============================================================================
MODEL_CONFIGS = [
    {
        'name': 'baseline_xception',
        'file': 'trained_baseline_xception/baseline_xception.pth',
        'class': XceptionBaseline,
        'dual_stream': False,
    },
    {
        'name': 'baseline_resnet',
        'file': 'trained_baseline_resnet/baseline_resnet.pth',
        'class': ResNetBaseline,
        'dual_stream': False,
    },
    {
        'name': 'custom_xception',
        'file': 'trained/best_model_dual.pth',
        'class': DualStreamModel,
        'dual_stream': True,
    },
    {
        'name': 'custom_resnet',
        'file': 'trained/best_model_resnet_dual.pth',
        'class': DualStreamModelResNet,
        'dual_stream': True,
    },
]

# ============================================================================
# SHARED UTILS
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
    return torch.from_numpy(np.stack(dct_channels, axis=0)).float()

def normalize_dct(dct_tensor):
    mean = dct_tensor.mean(dim=(1, 2), keepdim=True)
    std = dct_tensor.std(dim=(1, 2), keepdim=True)
    return (dct_tensor - mean) / (std + 1e-8)

class DeepfakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir):
        self.real_images = glob(os.path.join(real_dir, '*.png')) + glob(os.path.join(real_dir, '*.jpg'))
        self.fake_images = glob(os.path.join(fake_dir, '*.png')) + glob(os.path.join(fake_dir, '*.jpg'))
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            face = transforms.Resize((160, 160))(image)
            face_tensor = transforms.ToTensor()(face)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            rgb_tensor = normalize(face_tensor)
            dct_tensor = normalize_dct(rgb_to_dct(face_tensor * 255.0))
            return rgb_tensor, dct_tensor, label
        except:
            return self.__getitem__(np.random.randint(0, len(self)))

# ============================================================================
# EVALUATE
# ============================================================================
def evaluate(model, dataloader, device, is_dual_stream):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for rgb, dct, labels in tqdm(dataloader, desc='  Testing', leave=False):
            rgb, dct, labels = rgb.to(device), dct.to(device), labels.to(device)
            
            if is_dual_stream:
                outputs, _ = model(rgb, dct)
            else:
                outputs = model(rgb)
            
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1, 'auc': auc}

# ============================================================================
# MAIN
# ============================================================================
def main():
    BATCH_SIZE = 32
    DATASET_ROOT = 'data_dfdc'
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load dataset once
    test_dataset = DeepfakeDataset(f'{DATASET_ROOT}/test/real', f'{DATASET_ROOT}/test/fake')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    print(f"Test Data - Real: {len(test_dataset.real_images)}, Fake: {len(test_dataset.fake_images)}\n")
    
    # Store all results for comparison
    all_results = []
    
    print("=" * 70)
    print("TESTING ALL MODELS")
    print("=" * 70)
    
    for config in MODEL_CONFIGS:
        name = config['name']
        filepath = config['file']
        model_class = config['class']
        is_dual = config['dual_stream']
        
        print(f"\n[{name.upper()}]")
        print(f"  File: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"  ✗ SKIPPED - File not found")
            continue
        
        # Initialize model
        try:
            model = model_class(num_classes=2, pretrained=False).to(device)
        except Exception as e:
            print(f"  ✗ SKIPPED - Model init error: {e}")
            continue
        
        # Load weights
        try:
            model.load_state_dict(torch.load(filepath, map_location=device))
            print(f"  ✓ Weights loaded")
        except RuntimeError as e:
            print(f"  ✗ SKIPPED - Weight loading error")
            print(f"    {str(e)[:100]}...")
            continue
        
        # Evaluate
        metrics = evaluate(model, test_loader, device, is_dual)
        
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        
        # Store results
        all_results.append({
            'model': name,
            'file': filepath,
            'metrics': {k: float(v) for k, v in metrics.items()}
        })
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 70)
    
    for r in all_results:
        m = r['metrics']
        print(f"{r['model']:<20} {m['accuracy']:>8.4f} {m['precision']:>8.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['auc']:>8.4f}")
    
    print("=" * 70)
    
    # ========================================================================
    # SAVE ALL RESULTS TO JSON
    # ========================================================================
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(RESULTS_DIR, f'all_models_comparison_{timestamp}.json')
    
    with open(save_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'device': str(device),
            'results': all_results
        }, f, indent=4)
    
    print(f"\n✓ All results saved to: {save_path}")

if __name__ == '__main__':
    main()