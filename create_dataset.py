"""
Create a small subset of Celeb-DF dataset for quick testing
Extracts frames from a limited number of videos
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm
import random

def extract_frames_from_video(video_path, output_dir, num_frames=10):
    """
    Extract evenly-spaced frames from a video
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        num_frames: Number of frames to extract
    
    Returns:
        Number of frames successfully extracted
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"⚠ Warning: Could not read {video_path}")
        return 0
    
    # Calculate frame indices (evenly spaced)
    frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    
    extracted = 0
    video_name = Path(video_path).stem
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            frame_path = os.path.join(output_dir, f"{video_name}_frame_{idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted += 1
    
    cap.release()
    return extracted

def create_small_dataset(
    celeb_df_root,
    output_root,
    num_real_videos=25,
    num_fake_videos=25,
    frames_per_video=10,
    train_split=0.7,
    val_split=0.15,
    seed=42
):
    """
    Create a small subset dataset from Celeb-DF
    
    Args:
        celeb_df_root: Path to Celeb-DF-v2 folder
        output_root: Where to save the small dataset
        num_real_videos: Number of real videos to use (default: 25)
        num_fake_videos: Number of fake videos to use (default: 25)
        frames_per_video: Frames to extract per video (default: 10)
        train_split: Fraction for training (default: 0.7)
        val_split: Fraction for validation (default: 0.15)
        seed: Random seed for reproducibility
    """
    
    random.seed(seed)
    
    print("="*70)
    print(f"Creating Small Celeb-DF Dataset")
    print("="*70)
    print(f"Real videos: {num_real_videos}")
    print(f"Fake videos: {num_fake_videos}")
    print(f"Frames per video: {frames_per_video}")
    print(f"Total frames: {(num_real_videos + num_fake_videos) * frames_per_video}")
    print(f"Split: {train_split*100:.0f}% train, {val_split*100:.0f}% val, "
            f"{(1-train_split-val_split)*100:.0f}% test")
    print("="*70 + "\n")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(output_root, split, label), exist_ok=True)
    

    # Process REAL videos
    real_video_dir = os.path.join(celeb_df_root, 'Celeb-real')
    all_real_videos = list(Path(real_video_dir).glob('*.mp4'))
    
    if len(all_real_videos) < num_real_videos:
        print(f"Warning: Found only {len(all_real_videos)} real videos, "
                f"requested {num_real_videos}")
        num_real_videos = len(all_real_videos)
    
    # Randomly select subset
    selected_real_videos = random.sample(all_real_videos, num_real_videos)
    
    print(f"Selected {len(selected_real_videos)} REAL videos from {len(all_real_videos)} available")
    
    # Split into train/val/test
    n_train = int(len(selected_real_videos) * train_split)
    n_val = int(len(selected_real_videos) * val_split)
    
    train_real = selected_real_videos[:n_train]
    val_real = selected_real_videos[n_train:n_train+n_val]
    test_real = selected_real_videos[n_train+n_val:]
    
    print(f"  Train: {len(train_real)} videos")
    print(f"  Val:   {len(val_real)} videos")
    print(f"  Test:  {len(test_real)} videos\n")
    
    # Extract frames
    print("Extracting frames from REAL videos...")
    total_real_frames = 0
    
    for split_name, videos in [('train', train_real), ('val', val_real), ('test', test_real)]:
        output_dir = os.path.join(output_root, split_name, 'real')
        print(f"  {split_name.upper()}: ", end='', flush=True)
        
        frames_extracted = 0
        for video_path in tqdm(videos, desc=f"{split_name}", leave=False):
            frames_extracted += extract_frames_from_video(
                str(video_path), output_dir, frames_per_video
            )
        
        print(f"{frames_extracted} frames")
        total_real_frames += frames_extracted
    
    print(f"Total REAL frames extracted: {total_real_frames}\n")
    

    # Process FAKE videos

    fake_video_dir = os.path.join(celeb_df_root, 'Celeb-synthesis')
    all_fake_videos = list(Path(fake_video_dir).glob('*.mp4'))
    
    if len(all_fake_videos) < num_fake_videos:
        print(f"Warning: Found only {len(all_fake_videos)} fake videos, "
                f"requested {num_fake_videos}")
        num_fake_videos = len(all_fake_videos)
    
    # Randomly select subset
    selected_fake_videos = random.sample(all_fake_videos, num_fake_videos)
    
    print(f"Selected {len(selected_fake_videos)} FAKE videos from {len(all_fake_videos)} available")
    
    # Split into train/val/test
    n_train = int(len(selected_fake_videos) * train_split)
    n_val = int(len(selected_fake_videos) * val_split)
    
    train_fake = selected_fake_videos[:n_train]
    val_fake = selected_fake_videos[n_train:n_train+n_val]
    test_fake = selected_fake_videos[n_train+n_val:]
    
    print(f"  Train: {len(train_fake)} videos")
    print(f"  Val:   {len(val_fake)} videos")
    print(f"  Test:  {len(test_fake)} videos\n")
    
    # Extract frames
    print("Extracting frames from FAKE videos...")
    total_fake_frames = 0
    
    for split_name, videos in [('train', train_fake), ('val', val_fake), ('test', test_fake)]:
        output_dir = os.path.join(output_root, split_name, 'fake')
        print(f"  {split_name.upper()}: ", end='', flush=True)
        
        frames_extracted = 0
        for video_path in tqdm(videos, desc=f"{split_name}", leave=False):
            frames_extracted += extract_frames_from_video(
                str(video_path), output_dir, frames_per_video
            )
        
        print(f"{frames_extracted} frames")
        total_fake_frames += frames_extracted
    
    print(f"Total FAKE frames extracted: {total_fake_frames}\n")
    
    # Summary Statistics
    print("="*70)
    print("DATASET CREATION COMPLETE!")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        real_frames = len(list(Path(os.path.join(output_root, split, 'real')).glob('*.jpg')))
        fake_frames = len(list(Path(os.path.join(output_root, split, 'fake')).glob('*.jpg')))
        total = real_frames + fake_frames
        
        print(f"{split.upper():5} - Real: {real_frames:4}, Fake: {fake_frames:4}, Total: {total:4}")
    
    total_frames = total_real_frames + total_fake_frames
    print(f"\nGRAND TOTAL: {total_frames} frames")
    print(f"Dataset saved to: {output_root}")
    print("="*70)
    
    # Save metadata
    metadata = {
        'num_real_videos': num_real_videos,
        'num_fake_videos': num_fake_videos,
        'frames_per_video': frames_per_video,
        'train_split': train_split,
        'val_split': val_split,
        'test_split': 1 - train_split - val_split,
        'seed': seed,
        'total_frames': total_frames
    }
    
    import json
    with open(os.path.join(output_root, 'dataset_info.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\nMetadata saved to: {output_root}/dataset_info.json")

# Quick presets for different experiment sizes

def create_tiny_dataset(celeb_df_root, output_root='data_tiny'):
    """
    Create a TINY dataset for quick testing
    10 videos total (5 real + 5 fake) x 10 frames = 100 frames
    Training time: ~5-10 minutes per epoch
    """
    print("Creating TINY dataset (100 frames total)\n")
    create_small_dataset(
        celeb_df_root=celeb_df_root,
        output_root=output_root,
        num_real_videos=5,
        num_fake_videos=5,
        frames_per_video=10
    )

def create_small_dataset_preset(celeb_df_root, output_root='data_small'):
    """
    Create a SMALL dataset for development
    50 videos total (25 real + 25 fake) x 10 frames = 500 frames
    Training time: ~15-20 minutes per epoch
    """
    print("Creating SMALL dataset (500 frames total)\n")
    create_small_dataset(
        celeb_df_root=celeb_df_root,
        output_root=output_root,
        num_real_videos=25,
        num_fake_videos=25,
        frames_per_video=10
    )

def create_medium_dataset(celeb_df_root, output_root='data_medium'):
    """
    Create a MEDIUM dataset for experiments
    100 videos total (50 real + 50 fake) x 10 frames = 1000 frames
    Training time: ~30-40 minutes per epoch
    """
    print("Creating MEDIUM dataset (1000 frames total)\n")
    create_small_dataset(
        celeb_df_root=celeb_df_root,
        output_root=output_root,
        num_real_videos=50,
        num_fake_videos=50,
        frames_per_video=10
    )


# Main execution

if __name__ == '__main__':
    # Path to downloaded Celeb-DF-v2 folder
    CELEB_DF_PATH = 'path/to/Celeb-DF-v2' 
    
    # Choose one of the presets:
    
    # Option 1: Tiny dataset (100 frames) - fastest for testing code
    # create_tiny_dataset(CELEB_DF_PATH, output_root='data_tiny')
    
    # Option 2: Small dataset (500 frames)
    create_small_dataset_preset(CELEB_DF_PATH, output_root='data_small')
    
    # Option 3: Medium dataset (1000 frames) - more robust experiments
    # create_medium_dataset(CELEB_DF_PATH, output_root='data_medium')
    
    # Option 4: Custom configuration
    # create_small_dataset(
    #     celeb_df_root=CELEB_DF_PATH,
    #     output_root='data_custom',
    #     num_real_videos=30,      # Your choice
    #     num_fake_videos=30,      # Your choice
    #     frames_per_video=10,     # Your choice
    #     train_split=0.7,
    #     val_split=0.15,
    #     seed=42
    # )