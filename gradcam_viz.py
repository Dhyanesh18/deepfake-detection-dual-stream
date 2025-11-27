import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from scipy.fftpack import dct

import timm
from models.fusion import LearnableFusion
from models.attention import CBAM


class XceptionWithCBAM(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        full_model = timm.create_model('xception', pretrained=pretrained)
        
        self.block1 = full_model.conv1
        self.bn1 = full_model.bn1
        self.act1 = full_model.act1
        
        self.block2 = full_model.conv2
        self.bn2 = full_model.bn2
        self.act2 = full_model.act2
        
        self.block3 = nn.Sequential(full_model.block1, full_model.block2)
        
        self.block4 = nn.Sequential(
            full_model.block3, full_model.block4, full_model.block5,
            full_model.block6, full_model.block7, full_model.block8,
            full_model.block9, full_model.block10, full_model.block11
        )
        self.cbam_stage3 = CBAM(728)
        
        self.block5 = full_model.block12
        self.cbam_stage4 = CBAM(1024)
        
        self.conv3 = full_model.conv3
        self.bn3 = full_model.bn3
        self.act3 = full_model.act3
        
        self.conv4 = full_model.conv4
        self.bn4 = full_model.bn4
        self.act4 = full_model.act4
        
        self.feature_dim = 2048
    
    def forward(self, x):
        x = self.act1(self.bn1(self.block1(x)))
        x = self.act2(self.bn2(self.block2(x)))
        x = self.block3(x)
        x = self.block4(x)
        x = self.cbam_stage3(x)
        x = self.block5(x)
        x = self.cbam_stage4(x)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        return x


class DCTStream(nn.Module):
    def __init__(self):
        super(DCTStream, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        )
        self.cbam1 = CBAM(128)  
        self.cbam2 = CBAM(256)  
        self.cbam3 = CBAM(512)  
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.cbam1(x)
        x = self.conv3(x)
        x = self.cbam2(x)
        x = self.conv4(x)
        x = self.cbam3(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x


class DualStreamModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DualStreamModel, self).__init__()
        self.spatial_stream = XceptionWithCBAM(pretrained=pretrained)
        self.frequency_stream = DCTStream()
        self.fusion = LearnableFusion(
            spatial_dim=2048,
            frequency_dim=512,
            fusion_dim=512
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, rgb_input, dct_input):
        spatial_features = self.spatial_stream(rgb_input)
        frequency_features = self.frequency_stream(dct_input)
        fused_features, attention_weights = self.fusion(spatial_features, frequency_features)
        logits = self.classifier(fused_features)
        return logits, attention_weights


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


class GradCAM:
    """GradCAM implementation for dual-stream model"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        self.model.eval()
        output = self.model(*input_image)
        
        if target_class is None:
            target_class = output[0].argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output[0])
        one_hot[0][target_class] = 1
        output[0].backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def apply_colormap_on_image(org_img, cam, alpha=0.5):
    """Overlay heatmap on original image"""
    height, width = org_img.shape[:2]
    cam = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    cam_img = heatmap * alpha + org_img * (1 - alpha)
    cam_img = cam_img / cam_img.max()
    return np.uint8(255 * cam_img)


def visualize_dual_stream(model, rgb_path, save_path=None, device='cuda'):
    """
    Generate GradCAM visualizations for both spatial and frequency streams
    
    Args:
        model: Trained DualStreamModel
        rgb_path: Path to RGB image
        save_path: Path to save visualization
        device: 'cuda' or 'cpu'
    """
    model = model.to(device)
    model.eval()
    
    # Load RGB image
    rgb_img = Image.open(rgb_path).convert('RGB')
    rgb_img_resized = rgb_img.resize((160, 160))
    rgb_np = np.array(rgb_img_resized)
    
    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])
    
    rgb_tensor = transform(rgb_img)  # [3, 160, 160]
    
    # Generate DCT from RGB
    dct_tensor = rgb_to_dct(rgb_tensor)
    dct_tensor = normalize_dct(dct_tensor)
    
    # Normalize RGB for model
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
    rgb_tensor = normalize(rgb_tensor).unsqueeze(0).to(device)
    dct_tensor = dct_tensor.unsqueeze(0).to(device)
    
    # DCT visualization (log scale for better visibility)
    dct_vis = np.log(np.abs(dct_tensor[0].cpu().numpy()) + 1)
    dct_vis = (dct_vis - dct_vis.min()) / (dct_vis.max() - dct_vis.min())
    dct_vis = np.transpose(dct_vis, (1, 2, 0))
    dct_vis = (dct_vis * 255).astype(np.uint8)
    
    # Get prediction
    with torch.no_grad():
        logits, attn_weights = model(rgb_tensor, dct_tensor)
        pred_class = logits.argmax(dim=1).item()
        confidence = F.softmax(logits, dim=1)[0][pred_class].item()
    
    # GradCAM for spatial stream
    spatial_gradcam = GradCAM(model, model.spatial_stream.cbam_stage4)
    spatial_cam = spatial_gradcam.generate_cam((rgb_tensor, dct_tensor), target_class=pred_class)
    spatial_vis = apply_colormap_on_image(rgb_np, spatial_cam)
    
    # GradCAM for frequency stream
    freq_gradcam = GradCAM(model, model.frequency_stream.cbam3)
    freq_cam = freq_gradcam.generate_cam((rgb_tensor, dct_tensor), target_class=pred_class)
    freq_vis = apply_colormap_on_image(dct_vis, freq_cam)
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Spatial stream
    axes[0, 0].imshow(rgb_np)
    axes[0, 0].set_title('RGB Input', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(spatial_cam, cmap='jet')
    axes[0, 1].set_title('Spatial GradCAM', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(spatial_vis)
    axes[0, 2].set_title('Spatial Overlay', fontsize=12)
    axes[0, 2].axis('off')
    
    # Frequency stream
    axes[1, 0].imshow(dct_vis)
    axes[1, 0].set_title('DCT Input (log scale)', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(freq_cam, cmap='jet')
    axes[1, 1].set_title('Frequency GradCAM', fontsize=12)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(freq_vis)
    axes[1, 2].set_title('Frequency Overlay', fontsize=12)
    axes[1, 2].axis('off')
    
    # Add prediction info
    class_name = 'FAKE' if pred_class == 1 else 'REAL'
    attn_spatial, attn_freq = attn_weights[0].cpu().numpy()
    
    fig.suptitle(
        f'Prediction: {class_name} (Confidence: {confidence:.3f})\n'
        f'Attention Weights - Spatial: {attn_spatial:.3f}, Frequency: {attn_freq:.3f}',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualStreamModel(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load('ablation_results/dual_full_best.pth', map_location=device),strict=False)
    model = model.to(device)
    
    # Visualize sample
    rgb_path = r'data_dfdc\test\fake\aagfhgtpmv_57.png'
    
    visualize_dual_stream(
        model=model,
        rgb_path=rgb_path,
        save_path='gradcam_visualization.png',
        device=device
    )