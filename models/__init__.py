"""
train.py — Run this script to train all three models before using the app.

Usage:
    python train.py --modality image
    python train.py --modality video --data_dir data/SDFVD
    python train.py --modality audio --data_dir data/ASVspoof2019
    python train.py --modality all   --data_dir data/
"""

import argparse
import os

os.makedirs("trained_models", exist_ok=True)


def train_image():
    from datasets.loaders import get_image_loaders
    from models.image_model import train_image_model
    print("\n🖼️  Training Image Model (EfficientNet-B0)...")
    train_loader, val_loader = get_image_loaders(batch_size=32, max_samples=10000)
    train_image_model(train_loader, val_loader, epochs=10,
                      save_path="trained_models/image_model.pth")
    print("✅ Image model training complete!")


def train_video(data_dir):
    from datasets.loaders import get_video_loaders
    from models.video_model import train_video_model
    print("\n🎥  Training Video Model (EfficientNet-B0 on frames)...")
    train_loader, val_loader = get_video_loaders(data_dir, batch_size=16)
    train_video_model(train_loader, val_loader, epochs=10,
                      save_path="trained_models/video_model.pth")
    print("✅ Video model training complete!")


def train_audio(data_dir):
    from datasets.loaders import get_audio_loaders
    from models.audio_model import train_audio_model
    print("\n🎵  Training Audio Model (CNN on Mel Spectrogram)...")
    train_loader, val_loader = get_audio_loaders(data_dir, batch_size=32)
    train_audio_model(train_loader, val_loader, epochs=15,
                      save_path="trained_models/audio_model.pth")
    print("✅ Audio model training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Models")
    parser.add_argument("--modality", type=str, default="all",
                        choices=["image", "video", "audio", "all"],
                        help="Which model to train")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Root directory for video/audio datasets")
    args = parser.parse_args()

    if args.modality in ("image", "all"):
        train_image()

    if args.modality in ("video", "all"):
        video_dir = os.path.join(args.data_dir, "SDFVD") if args.modality == "all" else args.data_dir
        train_video(video_dir)

    if args.modality in ("audio", "all"):
        audio_dir = os.path.join(args.data_dir, "ASVspoof2019") if args.modality == "all" else args.data_dir
        train_audio(audio_dir)

    print("\n🎉 All requested models trained and saved to trained_models/")