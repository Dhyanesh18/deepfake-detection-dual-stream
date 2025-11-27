"""
Celeb-DF Dataset Preprocessing Pipeline
Extracts faces from videos using MTCNN and creates train/val/test splits
"""

import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
import json
import random
from PIL import Image

class CelebDFPreprocessor:
    def __init__(self, 
                 celebdf_root='archive',
                 output_root='data_celebdf',
                 frames_per_video=10,
                 face_size=160,
                 device=None):
        """
        Args:
            celebdf_root: Path to archive/ folder with Celeb-real/, Celeb-synthesis/, YouTube-real/
            output_root: Where to save extracted faces
            frames_per_video: Number of frames to extract per video
            face_size: Output face image size (160x160 for FaceNet/Xception)
            device: 'cuda' or 'cpu'
        """
        self.celebdf_root = Path(celebdf_root)
        self.output_root = Path(output_root)
        self.frames_per_video = frames_per_video
        self.face_size = face_size
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize MTCNN for face detection
        print(f"Initializing MTCNN on {self.device}...")
        # ✅ FIXED: Use post_process=False with manual denormalization
        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=20,
            min_face_size=40,
            thresholds=[0.5, 0.6, 0.7],  # Lower thresholds for better detection
            factor=0.709,
            post_process=False,  # ✅ FIXED: Use False and manual denormalization
            device=self.device,
            keep_all=False,
            selection_method='largest'
        )
        
        # Load official test split
        self.test_videos = self._load_test_split()
        
        # ✅ ADD: Statistics tracking
        self.stats_tracker = {
            'total_frames_processed': 0,
            'faces_detected': 0,
            'faces_saved': 0,
            'detection_failures': 0
        }
        
        print(f"CelebDF Preprocessor initialized")
        print(f"Device: {self.device}")
        print(f"Frames per video: {frames_per_video}")
        print(f"Face size: {face_size}x{face_size}")
    
    def _load_test_split(self):
        """Load official test video list"""
        possible_names = [
            'List_of_testing_videos.txt',
            'Listing_of_testing_videos.txt',
            'test_list.txt'
        ]
        
        for name in possible_names:
            test_file = self.celebdf_root / name
            if test_file.exists():
                with open(test_file, 'r') as f:
                    test_videos = set()
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            test_videos.add(line)
                            test_videos.add(Path(line).name)
                            test_videos.add(Path(line).stem)
                
                print(f"✅ Loaded {len(test_videos)} official test video identifiers from {name}")
                return test_videos
        
        print("⚠️  Warning: Test video list not found, will create random split")
        return set()
    
    def extract_frames_from_video(self, video_path, max_frames=None):
        """Extract uniformly sampled frames from video"""
        if max_frames is None:
            max_frames = self.frames_per_video
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return []
        
        # Sample frame indices uniformly
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        cap.release()
        return frames
    
    def detect_and_save_faces(self, frames, output_dir, video_name):
        """Detect faces in frames and save them"""
        saved_count = 0
        
        for frame_idx, frame in enumerate(frames):
            self.stats_tracker['total_frames_processed'] += 1
            
            try:
                # ✅ VALIDATE: Check frame is not empty
                if frame is None or frame.size == 0:
                    self.stats_tracker['detection_failures'] += 1
                    continue
                
                # Convert to PIL Image
                pil_frame = Image.fromarray(frame)
                
                # ✅ VALIDATE: Ensure RGB mode
                if pil_frame.mode != 'RGB':
                    pil_frame = pil_frame.convert('RGB')
                
                # Detect face using MTCNN
                face_tensor = self.mtcnn(pil_frame)
                
                # ✅ CRITICAL CHECK: Verify face was detected
                if face_tensor is None:
                    self.stats_tracker['detection_failures'] += 1
                    continue
                
                self.stats_tracker['faces_detected'] += 1
                
                # ✅ VALIDATE: Check tensor dimensions
                if face_tensor.dim() != 3 or face_tensor.shape[0] != 3:
                    self.stats_tracker['detection_failures'] += 1
                    continue
                
                # ✅ FIXED: Manual denormalization (since post_process=False)
                face_np = face_tensor.permute(1, 2, 0).cpu().numpy()
                
                # Denormalize from standardized values to [0, 255]
                face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min() + 1e-8)
                face_np = (face_np * 255).astype(np.uint8)
                
                # ✅ VALIDATE: Check for valid contrast
                if face_np.max() - face_np.min() < 10:  # Too low contrast
                    self.stats_tracker['detection_failures'] += 1
                    continue
                
                # ✅ VALIDATE: Check not all black
                if np.mean(face_np) < 5:
                    self.stats_tracker['detection_failures'] += 1
                    continue
                
                # Save face
                filename = f"{video_name}_frame{frame_idx:03d}.jpg"
                save_path = output_dir / filename
                
                # Save as PIL Image
                face_img = Image.fromarray(face_np)
                face_img.save(save_path, quality=95)
                
                self.stats_tracker['faces_saved'] += 1
                saved_count += 1
            
            except Exception as e:
                self.stats_tracker['detection_failures'] += 1
                # Optionally log errors for debugging
                # print(f"Error processing {video_name} frame {frame_idx}: {e}")
                continue
        
        return saved_count
    
    def process_video_folder(self, folder_path, label, split_dict):
        """Process all videos in a folder"""
        video_files = list(folder_path.glob('*.mp4')) + list(folder_path.glob('*.avi'))
        
        print(f"\nProcessing {folder_path.name} ({label}):")
        print(f"Found {len(video_files)} videos")
        
        stats = {'train': 0, 'val': 0, 'test': 0}
        
        for video_path in tqdm(video_files, desc=f"Processing {folder_path.name}"):
            video_name = video_path.stem
            video_filename = video_path.name
            
            # Check if test video
            is_test = False
            if (video_name in self.test_videos or 
                video_filename in self.test_videos or
                str(video_path.relative_to(self.celebdf_root)) in self.test_videos):
                is_test = True
            
            # Determine split
            if is_test:
                split = 'test'
            else:
                split = split_dict.get(video_name, 'train')
            
            # Create output directory
            output_dir = self.output_root / split / label
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            frames = self.extract_frames_from_video(video_path)
            
            if len(frames) == 0:
                continue
            
            # Detect and save faces
            saved = self.detect_and_save_faces(frames, output_dir, video_name)
            stats[split] += saved
        
        return stats
    
    def prepare_dataset(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        """Main processing pipeline"""
        random.seed(seed)
        
        print("="*70)
        print("CELEB-DF DATASET PREPROCESSING")
        print("="*70)
        print(f"Source: {self.celebdf_root}")
        print(f"Output: {self.output_root}")
        print("="*70)
        
        # Create output directories
        for split in ['train', 'val', 'test']:
            for label in ['real', 'fake']:
                (self.output_root / split / label).mkdir(parents=True, exist_ok=True)
        
        # Get all videos for splitting (exclude test videos)
        real_videos = []
        for folder in ['Celeb-real', 'YouTube-real']:
            folder_path = self.celebdf_root / folder
            if folder_path.exists():
                for video_path in folder_path.glob('*.mp4'):
                    video_name = video_path.stem
                    video_filename = video_path.name
                    
                    is_test = (video_name in self.test_videos or 
                              video_filename in self.test_videos or
                              str(video_path.relative_to(self.celebdf_root)) in self.test_videos)
                    
                    if not is_test:
                        real_videos.append(video_name)
        
        fake_videos = []
        folder_path = self.celebdf_root / 'Celeb-synthesis'
        if folder_path.exists():
            for video_path in folder_path.glob('*.mp4'):
                video_name = video_path.stem
                video_filename = video_path.name
                
                is_test = (video_name in self.test_videos or 
                          video_filename in self.test_videos or
                          str(video_path.relative_to(self.celebdf_root)) in self.test_videos)
                
                if not is_test:
                    fake_videos.append(video_name)
        
        # Create train/val split
        split_dict = {}
        
        for videos in [real_videos, fake_videos]:
            if len(videos) == 0:
                continue
            
            random.shuffle(videos)
            n_total = len(videos)
            n_train = int(n_total * train_ratio / (train_ratio + val_ratio))
            
            train_vids = videos[:n_train]
            val_vids = videos[n_train:]
            
            for v in train_vids:
                split_dict[v] = 'train'
            for v in val_vids:
                split_dict[v] = 'val'
        
        print(f"\nSplit assignment:")
        print(f"Train videos: {sum(1 for v in split_dict.values() if v == 'train')}")
        print(f"Val videos: {sum(1 for v in split_dict.values() if v == 'val')}")
        print(f"Test videos (from official list): {len(self.test_videos) // 3}")
        
        # Process real videos
        total_stats = {'train': 0, 'val': 0, 'test': 0}
        
        for folder in ['Celeb-real', 'YouTube-real']:
            folder_path = self.celebdf_root / folder
            if folder_path.exists():
                stats = self.process_video_folder(folder_path, 'real', split_dict)
                for k in stats:
                    total_stats[k] += stats[k]
        
        # Process fake videos
        folder_path = self.celebdf_root / 'Celeb-synthesis'
        if folder_path.exists():
            fake_stats = self.process_video_folder(folder_path, 'fake', split_dict)
            for k in fake_stats:
                total_stats[k] += fake_stats[k]
        
        # Print final statistics
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        
        for split in ['train', 'val', 'test']:
            real_dir = self.output_root / split / 'real'
            fake_dir = self.output_root / split / 'fake'
            
            n_real = len(list(real_dir.glob('*.jpg'))) if real_dir.exists() else 0
            n_fake = len(list(fake_dir.glob('*.jpg'))) if fake_dir.exists() else 0
            
            print(f"{split.upper():5} - Real: {n_real:5}, Fake: {n_fake:5}, Total: {n_real+n_fake:5}")
        
        # ✅ PRINT DETECTION STATISTICS
        print("\n" + "="*70)
        print("FACE DETECTION STATISTICS")
        print("="*70)
        print(f"Total frames processed:  {self.stats_tracker['total_frames_processed']}")
        print(f"Faces detected:          {self.stats_tracker['faces_detected']}")
        print(f"Faces saved:             {self.stats_tracker['faces_saved']}")
        print(f"Detection failures:      {self.stats_tracker['detection_failures']}")
        
        if self.stats_tracker['total_frames_processed'] > 0:
            success_rate = (self.stats_tracker['faces_saved'] / self.stats_tracker['total_frames_processed']) * 100
            print(f"Success rate:            {success_rate:.2f}%")
        
        print("="*70)
        
        if self.stats_tracker['faces_saved'] == 0:
            print("\n⚠️  WARNING: NO FACES WERE SAVED!")
            print("Possible issues:")
            print("  1. Videos do not contain clear faces")
            print("  2. Video quality is too poor")
            print("  3. MTCNN thresholds need adjustment")
            print("  4. Try adjusting thresholds to [0.4, 0.5, 0.6] for more lenient detection")
        elif self.stats_tracker['faces_saved'] < self.stats_tracker['total_frames_processed'] * 0.5:
            print(f"\n⚠️  WARNING: Low detection rate ({success_rate:.1f}%)")
            print("Consider adjusting MTCNN thresholds if videos contain clear faces")
        
        print(f"\nDataset saved to: {self.output_root}")
        print("You can now run training with DATASET_ROOT='data_celebdf'")


if __name__ == '__main__':
    # Configuration
    CELEBDF_ROOT = 'archive'  # Your Celeb-DF folder
    OUTPUT_ROOT = 'data_celebdf'
    FRAMES_PER_VIDEO = 10
    
    # Initialize preprocessor
    preprocessor = CelebDFPreprocessor(
        celebdf_root=CELEBDF_ROOT,
        output_root=OUTPUT_ROOT,
        frames_per_video=FRAMES_PER_VIDEO,
        face_size=160
    )
    
    # Process dataset
    preprocessor.prepare_dataset(
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )