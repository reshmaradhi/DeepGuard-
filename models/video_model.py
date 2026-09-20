import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os


# ─── Model Definition (with stronger regularization) ─────────────────────────

class VideoDeepfakeDetector(nn.Module):
    def __init__(self):
        super(VideoDeepfakeDetector, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# ─── Transforms ──────────────────────────────────────────────────────────────

def get_transform(train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


# ─── Frame Extraction ────────────────────────────────────────────────────────

def extract_frames(video_path: str, num_frames: int = 16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    return frames


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict_video(video_path: str, model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoDeepfakeDetector().to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transform(train=False)
    frames = extract_frames(video_path, num_frames=16)

    if not frames:
        return "ERROR", 0.0, [], "Could not extract frames from video."

    fake_count = 0
    frame_results = []
    all_confidences = []

    with torch.no_grad():
        for i, frame in enumerate(frames):
            input_tensor = transform(frame).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
            confidence = probs[pred_class].item() * 100
            label = "FAKE" if pred_class == 1 else "REAL"

            if pred_class == 1:
                fake_count += 1

            frame_results.append({
                "frame": i + 1,
                "label": label,
                "confidence": confidence,
                "image": frame
            })
            all_confidences.append(probs[1].item())

    fake_ratio = fake_count / len(frames)
    avg_fake_prob = np.mean(all_confidences) * 100
    final_label = "FAKE" if fake_ratio >= 0.5 else "REAL"
    final_confidence = avg_fake_prob if final_label == "FAKE" else (100 - avg_fake_prob)

    summary = (f"{fake_count}/{len(frames)} frames detected as FAKE "
               f"({fake_ratio*100:.1f}%)")

    return final_label, final_confidence, frame_results, summary


# ─── Training (with all fixes for better video accuracy) ─────────────────────

def train_video_model(train_loader, val_loader, epochs=20,
                      save_path="trained_models/video_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoDeepfakeDetector().to(device)

    # Freeze early layers — only train last 3 blocks + classifier
    for name, param in model.backbone.named_parameters():
        if 'blocks.5' in name or 'blocks.6' in name or 'classifier' in name or 'conv_head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience = 7
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        val_acc = _evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Model saved (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

    print(f"\n✅ Best Val Accuracy: {best_val_acc:.2f}%")
    return model


def _evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total


