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

from models.backbone_resnet import DualStreamModelResNet


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
    
    # GradCAM for spatial stream (ResNet18 - last stage)
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
    model = DualStreamModelResNet(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load('best_model_dual_pretrained.pth', map_location=device))
    model = model.to(device)
    
    # Visualize sample
    rgb_path = r'data_dfdc\test\fake\aagfhgtpmv_57.png'
    
    visualize_dual_stream(
        model=model,
        rgb_path=rgb_path,
        save_path='gradcam_visualization.png',
        device=device
    )