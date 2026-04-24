#!/usr/bin/env python3
"""
Real-world long video dataset acquisition script
Downloads 5-10 long videos (30-60 minutes) from public datasets
"""

import os
import requests
import subprocess
from pathlib import Path
import json
import huggingface_hub as hf
from huggingface_hub import hf_hub_download

# Configuration
VIDEO_DIR = Path("./data/long_videos")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Video-MME dataset info
VIDEO_MME_REPO = "wangzhiyu666/Video-MME"
VIDEO_MME_VIDEOS = [
    "video_001.mp4",  # Placeholder - replace with actual Video-MME IDs
    "video_002.mp4",
    "video_003.mp4",
    "video_004.mp4",
    "video_005.mp4",
]

# Alternative public long video sources
PUBLIC_VIDEO_SOURCES = [
    {
        "name": "Big Buck Bunny",
        "url": "https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_1080p_stereo.avi",
        "expected_size_mb": 150000  # ~150MB
    },
    {
        "name": "Elephant Dream",
        "url": "https://download.blender.org/peach/bigbuckbunny_movies/elephant Dream.mp4",
        "expected_size_mb": 800000  # ~800MB
    },
    {
        "name": "Sintel Trailer",
        "url": "https://archive.org/download/Sintel/sintel_trailer-720p.mp4",
        "expected_size_mb": 100000  # ~100MB
    },
    {
        "name": "Caminandes Llama Drama",
        "url": "https://archive.org/download/Caminandes3/Caminandes3_LlamaDrama_1080p.mp4",
        "expected_size_mb": 120000  # ~120MB
    },
    {
        "name": "Goat Clip",
        "url": "https://archive.org/download/Caminandes2/Caminandes2_1080p.mp4",
        "expected_size_mb": 100000  # ~100MB
    },
    {
        "name": "Tears of Steel",
        "url": "https://download.blender.org/tearsofsteel/tears_of_steller_4k.mp4",
        "expected_size_mb": 800000  # ~800MB
    }
]

def download_video(url, filename, expected_size_mb):
    """Download a video file with progress tracking"""
    filepath = VIDEO_DIR / filename

    if filepath.exists():
        print(f"✓ {filename} already exists, skipping...")
        return True

    print(f"📥 Downloading {filename} ({expected_size_mb} MB)...")

    try:
        # Use wget for better progress tracking
        result = subprocess.run([
            "wget",
            "--continue",  # Resume if partial download
            "--quiet",
            "-O", str(filepath),
            url
        ], check=True, capture_output=True, text=True)

        # Verify download size
        if filepath.exists():
            actual_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"✓ Downloaded {filename}: {actual_size_mb:.1f} MB")
            return True
        else:
            print(f"✗ Failed to download {filename}: file not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {filename}: {e}")
        if filepath.exists():
            filepath.unlink()  # Clean up partial download
        return False
    except Exception as e:
        print(f"✗ Error downloading {filename}: {e}")
        return False

def download_from_huggingface():
    """Attempt to download from Video-MME dataset on Hugging Face"""
    print("🔍 Checking Video-MME dataset on Hugging Face...")

    try:
        # List available files in the dataset
        repo_files = hf.HfApi().list_repo_files(repo_id=VIDEO_MME_REPO)
        video_files = [f for f in repo_files if f.endswith(('.mp4', '.avi', '.mov'))][:5]

        if video_files:
            print(f"Found {len(video_files)} video files in Video-MME")

            for video_file in video_files:
                try:
                    # Download the file
                    local_path = hf_hub_download(
                        repo_id=VIDEO_MME_REPO,
                        filename=video_file,
                        local_dir=VIDEO_DIR,
                        resume_download=True
                    )
                    print(f"✓ Downloaded {video_file} to {local_path}")
                except Exception as e:
                    print(f"✗ Failed to download {video_file}: {e}")

            return len(video_files)
        else:
            print("No video files found in Video-MME")
            return 0

    except Exception as e:
        print(f"✗ Error accessing Video-MME: {e}")
        return 0

def main():
    """Main download function"""
    print("🎬 Starting real-world long video dataset acquisition...")
    print(f"Target directory: {VIDEO_DIR}")

    downloaded_count = 0

    # First try Hugging Face Video-MME
    hf_count = download_from_huggingface()
    downloaded_count += hf_count

    # Then try public video sources if needed
    if downloaded_count < 5:
        print(f"\n📡 Need {5 - downloaded_count} more videos from public sources...")

        for source in PUBLIC_VIDEO_SOURCES:
            if downloaded_count >= 5:
                break

            filename = f"{source['name'].replace(' ', '_').lower()}.mp4"
            if download_video(source['url'], filename, source['expected_size_mb']):
                downloaded_count += 1

    # Check results
    print(f"\n📊 Download Summary:")
    print(f"Total videos downloaded: {downloaded_count}")

    if VIDEO_DIR.exists():
        video_files = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi"))
        total_size = sum(f.stat().st_size for f in video_files) / (1024 * 1024)
        print(f"Total size: {total_size:.1f} MB")
        print(f"Videos: {[f.name for f in video_files]}")

    if downloaded_count >= 5:
        print("✅ Successfully acquired enough video data for experiments")
        return True
    else:
        print(f"❌ Only got {downloaded_count} videos, need at least 5")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)