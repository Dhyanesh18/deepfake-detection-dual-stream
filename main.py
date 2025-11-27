import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from scipy.fftpack import dct
import cv2
from typing import Dict, Any
import io
from facenet_pytorch import MTCNN

# Import model architecture
from models.backbones import DualStreamModel

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Deepfake Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .real-prediction {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fake-prediction {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
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

def preprocess_image(image: Image.Image):
    """Preprocess image for model input"""
    # FIXED: Convert to RGB to handle RGBA, grayscale, etc.
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((160, 160))
    
    # Convert to tensor
    transform = transforms.ToTensor()
    rgb_tensor = transform(image)
    
    # VALIDATION: Ensure 3 channels
    if rgb_tensor.shape[0] != 3:
        raise ValueError(f"Expected 3 channels, got {rgb_tensor.shape[0]}")
    
    # Generate DCT
    dct_tensor = rgb_to_dct(rgb_tensor * 255.0)
    dct_tensor = normalize_dct(dct_tensor)
    
    # Normalize RGB
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    rgb_tensor = normalize(rgb_tensor)
    
    return rgb_tensor.unsqueeze(0), dct_tensor.unsqueeze(0)

# ============================================================================
# FACE EXTRACTION WITH MTCNN
# ============================================================================
@st.cache_resource
def load_mtcnn():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Try with post_process=False and manual normalization
    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        min_face_size=40,
        thresholds=[0.5, 0.6, 0.7],  # Lower thresholds
        factor=0.709,
        post_process=False,  # Try False
        device=device,
        keep_all=False
    )
    return mtcnn

def extract_face_with_mtcnn(image: Image.Image, mtcnn):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        face_tensor = mtcnn(image)
        
        if face_tensor is None:
            return None, False
        
        # Manual denormalization (if post_process=False)
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Denormalize from standardized values
        face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min() + 1e-8)
        face_np = (face_np * 255).astype(np.uint8)
        
        if np.mean(face_np) < 5:
            return None, False
        
        return Image.fromarray(face_np), True
        
    except Exception as e:
        st.error(f"MTCNN Error: {str(e)}")
        return None, False

# ============================================================================
# GRADCAM IMPLEMENTATION
# ============================================================================
class GradCAM:
    """GradCAM for visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(*input_image)
        
        if target_class is None:
            target_class = output[0].argmax(dim=1)
        
        self.model.zero_grad()
        one_hot = torch.zeros_like(output[0])
        one_hot[0][target_class] = 1
        output[0].backward(gradient=one_hot, retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
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

def generate_gradcam(model, rgb_tensor, dct_tensor, original_image, pred_class, device):
    """Generate GradCAM visualizations"""
    # Spatial GradCAM
    spatial_gradcam = GradCAM(model, model.spatial_stream.cbam_stage4)
    spatial_cam = spatial_gradcam.generate_cam((rgb_tensor, dct_tensor), target_class=pred_class)
    
    # Frequency GradCAM
    freq_gradcam = GradCAM(model, model.frequency_stream.cbam3)
    freq_cam = freq_gradcam.generate_cam((rgb_tensor, dct_tensor), target_class=pred_class)
    
    # Original image as numpy
    original_np = np.array(original_image.resize((160, 160)))
    
    # Apply colormaps
    spatial_overlay = apply_colormap_on_image(original_np, spatial_cam)
    freq_overlay = apply_colormap_on_image(original_np, freq_cam)
    
    return {
        'spatial_cam': spatial_cam,
        'spatial_overlay': spatial_overlay,
        'freq_cam': freq_cam,
        'freq_overlay': freq_overlay,
        'original': original_np
    }

# ============================================================================
# MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model():
    """Load model (cached)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'trained/best_model_dual.pth'
    
    try:
        model = DualStreamModel(num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model = model.to(device)
        model.eval()
        return model, device, True, None
    except Exception as e:
        return None, device, False, str(e)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_image(model, device, image: Image.Image, use_mtcnn=False, mtcnn=None):
    """Run prediction on image"""
    try:
        # FIXED: Convert to RGB first
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract face if requested
        if use_mtcnn and mtcnn is not None:
            face_img, success = extract_face_with_mtcnn(image, mtcnn)
            if not success:
                return {"error": "No face detected in the image. Please try another image or disable face detection."}
            processed_image = face_img
        else:
            processed_image = image
        
        # Preprocess
        rgb_tensor, dct_tensor = preprocess_image(processed_image)
        rgb_tensor = rgb_tensor.to(device)
        dct_tensor = dct_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            logits, attn_weights = model(rgb_tensor, dct_tensor)
            probs = F.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()
            confidence = probs[0][pred_class].item()
        
        # Generate GradCAM
        gradcam_results = generate_gradcam(
            model, rgb_tensor, dct_tensor, processed_image, pred_class, device
        )
        
        # Prepare results
        prediction = "FAKE" if pred_class == 1 else "REAL"
        spatial_weight, freq_weight = attn_weights[0].cpu().numpy()
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "class_id": pred_class,
            "probabilities": {
                "real": probs[0][0].item(),
                "fake": probs[0][1].item()
            },
            "fusion_weights": {
                "spatial": spatial_weight,
                "frequency": freq_weight
            },
            "gradcam": gradcam_results,
            "processed_image": processed_image
        }
    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}

# ============================================================================
# VIDEO PROCESSING
# ============================================================================
def extract_frames_from_video(video_file, max_frames=10):
    """Extract frames from video file"""
    # Save uploaded video temporarily
    video_bytes = video_file.read()
    temp_path = "temp_video.mp4"
    
    with open(temp_path, "wb") as f:
        f.write(video_bytes)
    
    # Extract frames
    cap = cv2.VideoCapture(temp_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly
    interval = max(1, frame_count // max_frames)
    
    for i in range(0, frame_count, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            if len(frames) >= max_frames:
                break
    
    cap.release()
    
    # Clean up
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return frames

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Header
    st.markdown('<h1 class="main-header">Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Dual-Stream Xception with CBAM Attention")
    
    # Load model
    model, device, success, error = load_model()
    
    if not success:
        st.error(f"Failed to load model: {error}")
        st.stop()
    
    st.success(f"Model loaded successfully on {device}")
    
    # Load MTCNN
    mtcnn = load_mtcnn()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Face extraction option
        st.markdown("### Face Detection")
        use_face_detection = st.radio(
            "Extract face before analysis?",
            options=["No - Use full image", "Yes - Extract face with MTCNN"],
            index=0,
            help="Enable to automatically detect and extract faces from images/videos"
        )
        use_mtcnn = (use_face_detection == "Yes - Extract face with MTCNN")
        
        st.markdown("---")
        
        mode = st.radio("Select Mode:", ["Single Image", "Video Analysis"])
        
        st.markdown("---")
        st.markdown("### Model Info")
        st.info("""
        **Architecture**: Dual-Stream Xception
        - **Spatial Stream**: RGB features
        - **Frequency Stream**: DCT features
        - **Attention**: CBAM modules
        - **Fusion**: Learnable attention weights
        """)
        
        if use_mtcnn:
            st.markdown("---")
            st.markdown("### MTCNN Active")
            st.warning("Face detection is enabled. Only detected faces will be analyzed.")
        
        st.markdown("---")
        st.markdown("### How to Use")
        st.markdown("""
        1. Choose face detection mode
        2. Upload an image or video
        3. Wait for processing
        4. View prediction & GradCAM
        5. Analyze attention weights
        """)
    
    # Main content
    if mode == "Single Image":
        st.header("Single Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=['png', 'jpg', 'jpeg', 'webp', 'bmp'],
            help="Upload a face image to check if it's real or deepfake"
        )
        
        if uploaded_file is not None:
            # Display original image
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Image")
                st.image(image, use_container_width=True)
                st.caption(f"Mode: {image.mode} | Size: {image.size}")
            
            with col2:
                st.subheader("Processing...")
                
                with st.spinner("Analyzing image..."):
                    result = predict_image(model, device, image, use_mtcnn, mtcnn)
                
                if "error" in result:
                    st.error(f"{result['error']}")
                else:
                    # Show extracted face if MTCNN was used
                    if use_mtcnn:
                        st.info("Face detected and extracted")
                        st.image(result['processed_image'], caption="Extracted Face", use_container_width=True)
                    
                    # Prediction
                    prediction = result['prediction']
                    confidence = result['confidence']
                    
                    if prediction == "REAL":
                        st.markdown(
                            f'<div class="prediction-box real-prediction">REAL<br/>{confidence*100:.2f}% Confidence</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box fake-prediction">FAKE<br/>{confidence*100:.2f}% Confidence</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Metrics
                    st.markdown("### Detailed Metrics")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Real Probability", f"{result['probabilities']['real']*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Spatial Weight", f"{result['fusion_weights']['spatial']:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Fake Probability", f"{result['probabilities']['fake']*100:.2f}%")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Frequency Weight", f"{result['fusion_weights']['frequency']:.3f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # GradCAM Visualizations
                    st.markdown("---")
                    st.header("GradCAM Visualizations")
                    st.markdown("*Heatmaps show where the model focuses its attention*")
                    
                    viz_col1, viz_col2, viz_col3 = st.columns(3)
                    
                    with viz_col1:
                        st.subheader("Original Image")
                        st.image(result['gradcam']['original'], use_container_width=True)
                    
                    with viz_col2:
                        st.subheader("Spatial Stream Focus")
                        st.image(result['gradcam']['spatial_overlay'], use_container_width=True)
                        st.caption("RGB feature attention")
                    
                    with viz_col3:
                        st.subheader("Frequency Stream Focus")
                        st.image(result['gradcam']['freq_overlay'], use_container_width=True)
                        st.caption("DCT feature attention")
    
    else:  # Video Analysis
        st.header("Video Analysis")
        st.info("Upload a video to analyze frame-by-frame")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video to analyze multiple frames"
        )
        
        if uploaded_video is not None:
            max_frames = st.slider("Number of frames to analyze:", 5, 20, 10)
            
            if st.button("Start Analysis"):
                with st.spinner("Extracting frames..."):
                    frames = extract_frames_from_video(uploaded_video, max_frames)
                
                st.success(f"Extracted {len(frames)} frames")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                skipped = 0
                
                for i, frame in enumerate(frames):
                    if use_mtcnn:
                        status_text.text(f"Detecting face in frame {i+1}/{len(frames)}...")
                    else:
                        status_text.text(f"Analyzing frame {i+1}/{len(frames)}...")
                    
                    result = predict_image(model, device, frame, use_mtcnn, mtcnn)
                    
                    if "error" not in result:
                        results.append({
                            'frame': result.get('processed_image', frame),
                            'original_frame': frame,
                            'prediction': result['prediction'],
                            'confidence': result['confidence'],
                            'face_detected': use_mtcnn
                        })
                    else:
                        skipped += 1
                    
                    progress_bar.progress((i + 1) / len(frames))
                
                status_text.text("Analysis complete!")
                
                if skipped > 0:
                    st.warning(f"Skipped {skipped} frames (no face detected)")
                
                if len(results) == 0:
                    st.error("No faces detected in any frames. Try disabling face detection.")
                    st.stop()
                
                # Summary
                st.markdown("---")
                st.header("Video Analysis Summary")
                
                fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
                real_count = len(results) - fake_count
                
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                
                with sum_col1:
                    st.metric("Total Frames Analyzed", len(results))
                
                with sum_col2:
                    st.metric("Real Frames", real_count)
                
                with sum_col3:
                    st.metric("Fake Frames", fake_count)
                
                # Overall verdict
                overall = "FAKE" if fake_count > real_count else "REAL"
                confidence = max(fake_count, real_count) / len(results) * 100
                
                if overall == "REAL":
                    st.markdown(
                        f'<div class="prediction-box real-prediction">Overall Verdict: REAL<br/>{confidence:.1f}% of frames</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box fake-prediction">Overall Verdict: FAKE<br/>{confidence:.1f}% of frames</div>',
                        unsafe_allow_html=True
                    )
                
                # Frame-by-frame results
                st.markdown("---")
                st.subheader("Frame-by-Frame Results")
                
                cols_per_row = 4
                for i in range(0, len(results), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(results):
                            r = results[i + j]
                            with cols[j]:
                                display_img = r['frame'] if use_mtcnn else r['original_frame']
                                st.image(display_img, use_container_width=True)
                                st.caption(f"{r['prediction']} ({r['confidence']*100:.1f}%)")

if __name__ == "__main__":
    main()