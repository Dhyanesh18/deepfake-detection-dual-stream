"""
Prepare DFDC dataset structure to match the expected train/val/test/real/fake format
Copies/symlinks frames from DFDC folders into the required structure
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import random

def prepare_dfdc_dataset(
    dfdc_root,
    output_root,
    metadata_csv='metadata.csv',
    frames_per_video=50,  # How many frames to use per video
    use_symlinks=False,   # Set to True to save disk space (not supported on all systems)
    seed=42
):
    """
    Reorganize DFDC dataset into train/val/test/real/fake structure
    
    Args:
        dfdc_root: Path to dfdc_train_faces_sample folder
        output_root: Where to create the organized dataset
        metadata_csv: Name of metadata file (default: 'metadata.csv')
        frames_per_video: Max frames to use per video (default: 50)
        use_symlinks: Use symbolic links instead of copying (saves space)
        seed: Random seed for frame selection
    """
    
    random.seed(seed)
    
    print("="*70)
    print("PREPARING DFDC DATASET")
    print("="*70)
    print(f"Source: {dfdc_root}")
    print(f"Output: {output_root}")
    print(f"Frames per video: {frames_per_video}")
    print(f"Using symlinks: {use_symlinks}")
    print("="*70 + "\n")
    
    # Read metadata
    metadata_path = os.path.join(dfdc_root, metadata_csv)
    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found!")
        return
    
    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata: {len(df)} videos")
    print(f"Columns: {df.columns.tolist()}\n")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(output_root, split, label), exist_ok=True)
    
    # Process statistics
    stats = {
        'train': {'real': 0, 'fake': 0},
        'val': {'real': 0, 'fake': 0},
        'test': {'real': 0, 'fake': 0}
    }
    
    # Process each video
    print("Processing videos...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Videos"):
        video_name = row['filename'].replace('.mp4', '')
        label = row['label'].lower()  # 'REAL' or 'FAKE' -> 'real' or 'fake'
        split = row['split']  # 'train', 'val', or 'test' (if available)
        
        # Handle split - DFDC might not have explicit val/test splits
        # If 'split' column doesn't exist or is only 'train', create splits manually
        if split not in ['train', 'val', 'test']:
            split = 'train'  # Default to train, we'll split later
        
        # Source folder with frames
        source_folder = os.path.join(dfdc_root, video_name)
        
        if not os.path.exists(source_folder):
            print(f"Warning: Folder {source_folder} not found, skipping...")
            continue
        
        # Get all frame files
        frame_files = sorted([
            f for f in os.listdir(source_folder) 
            if f.endswith(('.png', '.jpg'))
        ], key=lambda x: int(x.split('.')[0]))
        
        if len(frame_files) == 0:
            continue
        
        # Randomly sample frames if too many
        if len(frame_files) > frames_per_video:
            frame_files = random.sample(frame_files, frames_per_video)
        
        # Copy/link frames to output directory
        output_dir = os.path.join(output_root, split, label)
        
        for frame_file in frame_files:
            source_path = os.path.join(source_folder, frame_file)
            # Create unique filename: videoname_framename
            dest_filename = f"{video_name}_{frame_file}"
            dest_path = os.path.join(output_dir, dest_filename)
            
            try:
                if use_symlinks:
                    os.symlink(source_path, dest_path)
                else:
                    shutil.copy2(source_path, dest_path)
                
                stats[split][label] += 1
            except Exception as e:
                print(f"Error copying {source_path}: {e}")
    
    # Print statistics
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE")
    print("="*70)
    
    total_frames = 0
    for split in ['train', 'val', 'test']:
        real_count = stats[split]['real']
        fake_count = stats[split]['fake']
        total = real_count + fake_count
        total_frames += total
        
        if total > 0:
            print(f"{split.upper():5} - Real: {real_count:5}, Fake: {fake_count:5}, Total: {total:5}")
    
    print(f"\nTOTAL FRAMES: {total_frames}")
    print(f"Dataset saved to: {output_root}")
    print("="*70)
    
    # If all data is in 'train', create val/test splits
    if stats['val']['real'] == 0 and stats['val']['fake'] == 0:
        print("\nNo validation/test splits found. Creating splits...")
        create_splits_from_train(output_root, train_split=0.7, val_split=0.15, seed=seed)


def create_splits_from_train(dataset_root, train_split=0.7, val_split=0.15, seed=42):
    """
    If DFDC only has 'train' data, split it into train/val/test
    """
    random.seed(seed)
    
    train_dir = os.path.join(dataset_root, 'train')
    val_dir = os.path.join(dataset_root, 'val')
    test_dir = os.path.join(dataset_root, 'test')
    
    for label in ['real', 'fake']:
        # Get all files in train/label
        train_label_dir = os.path.join(train_dir, label)
        all_files = [f for f in os.listdir(train_label_dir) if f.endswith(('.png', '.jpg'))]
        
        if len(all_files) == 0:
            continue
        
        # Shuffle and split
        random.shuffle(all_files)
        n_train = int(len(all_files) * train_split)
        n_val = int(len(all_files) * val_split)
        
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]
        
        # Move files to val
        val_label_dir = os.path.join(val_dir, label)
        for f in val_files:
            shutil.move(
                os.path.join(train_label_dir, f),
                os.path.join(val_label_dir, f)
            )
        
        # Move files to test
        test_label_dir = os.path.join(test_dir, label)
        for f in test_files:
            shutil.move(
                os.path.join(train_label_dir, f),
                os.path.join(test_label_dir, f)
            )
        
        print(f"{label.upper():5} - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


if __name__ == '__main__':
    # Path to your downloaded DFDC dataset
    DFDC_PATH = 'D:/SEM-5/CV/Project/dfdc_train_faces_sample'
    
    # Where to create the organized dataset
    OUTPUT_PATH = 'data_dfdc'
    
    # Prepare the dataset
    prepare_dfdc_dataset(
        dfdc_root=DFDC_PATH,
        output_root=OUTPUT_PATH,
        frames_per_video=50,  # Use up to 50 frames per video
        use_symlinks=False,   # Set True to save space (if your OS supports it)
        seed=42
    )
    
    print("\nYou can now run train.py with DATASET_ROOT='data_dfdc'")