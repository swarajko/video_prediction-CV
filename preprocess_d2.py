"""
Preprocessing script for d2 dataset
Extracts frames from videos and prepares them for training
"""
import cv2
import os
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def extract_frames_from_video(video_path, output_dir, img_size=128):
    """
    Extract frames from a video file and save them as grayscale images
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        img_size: Target size for resizing frames (default 128 for KTH)
    
    Returns:
        Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    frame_count = 0
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        resized_frame = cv2.resize(gray_frame, (img_size, img_size), interpolation=cv2.INTER_AREA)
        
        # Save frame
        frame_filename = os.path.join(output_dir, f'frame_{str(frame_idx).zfill(5)}.jpg')
        cv2.imwrite(frame_filename, resized_frame)
        
        frame_count += 1
        frame_idx += 1
    
    cap.release()
    return frame_count

def process_d2_dataset(dataset_dir, output_base_dir, img_size=128, train_ratio=0.8):
    """
    Process the entire d2 dataset: extract frames and split into train/val sets
    
    Args:
        dataset_dir: Path to the d2 dataset directory (contains action folders)
        output_base_dir: Base directory for output (will create train/val subdirectories)
        img_size: Target image size (default 128 for KTH)
        train_ratio: Ratio of data to use for training (default 0.8)
    """
    dataset_dir = Path(dataset_dir)
    output_base_dir = Path(output_base_dir)
    
    # Create output directories
    train_dir = output_base_dir / 'train'
    val_dir = output_base_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all video files
    video_files = []
    for action_folder in sorted(dataset_dir.iterdir()):
        if action_folder.is_dir():
            for video_file in sorted(action_folder.glob('*.avi')):
                video_files.append((action_folder.name, video_file))
    
    print(f"Found {len(video_files)} video files")
    
    # Split videos into train and validation sets
    train_videos, val_videos = train_test_split(
        video_files, 
        test_size=1-train_ratio, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Train videos: {len(train_videos)}, Validation videos: {len(val_videos)}")
    
    # Process training videos
    video_idx = 0
    for action_name, video_path in train_videos:
        video_output_dir = train_dir / f'video_{str(video_idx).zfill(5)}'
        frame_count = extract_frames_from_video(video_path, video_output_dir, img_size)
        print(f"Processed train video {video_idx}: {video_path.name} -> {frame_count} frames")
        video_idx += 1
    
    # Process validation videos
    video_idx = 0
    for action_name, video_path in val_videos:
        video_output_dir = val_dir / f'video_{str(video_idx).zfill(5)}'
        frame_count = extract_frames_from_video(video_path, video_output_dir, img_size)
        print(f"Processed val video {video_idx}: {video_path.name} -> {frame_count} frames")
        video_idx += 1
    
    print(f"\nPreprocessing complete!")
    print(f"Training data: {train_dir}")
    print(f"Validation data: {val_dir}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess d2 dataset for training')
    parser.add_argument('--dataset_dir', type=str, 
                        default=r'C:\Users\SWARAJ\Downloads\d2\d2',
                        help='Path to d2 dataset directory')
    parser.add_argument('--output_dir', type=str, 
                        default='./d2_processed',
                        help='Output directory for processed frames')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Target image size (default: 128 for KTH)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data for training (default: 0.8)')
    
    args = parser.parse_args()
    
    process_d2_dataset(
        dataset_dir=args.dataset_dir,
        output_base_dir=args.output_dir,
        img_size=args.img_size,
        train_ratio=args.train_ratio
    )


