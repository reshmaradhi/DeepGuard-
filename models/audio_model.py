import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io


# ─── Model Definition (with stronger regularization) ─────────────────────────

class AudioDeepfakeDetector(nn.Module):
    def __init__(self):
        super(AudioDeepfakeDetector, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),         # added dropout

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),           # increased from 0.4
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),           # increased from 0.3
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─── Audio to Mel Spectrogram ─────────────────────────────────────────────────

def audio_to_melspectrogram(audio_path: str, duration: float = 4.0):
    y, sr = librosa.load(audio_path, sr=16000, duration=duration)

    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000,
        n_fft=1024, hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
        mel_spec_db.max() - mel_spec_db.min() + 1e-8)

    return mel_spec_db.astype(np.float32), y, sr


def melspectrogram_to_tensor(mel_spec: np.ndarray):
    tensor = torch.tensor(mel_spec).unsqueeze(0).unsqueeze(0)
    return tensor


def generate_spectrogram_image(audio_path: str) -> Image.Image:
    y, sr = librosa.load(audio_path, sr=16000, duration=4.0)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor='#0d1117')
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time',
                                   y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0d1117', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict_audio(audio_path: str, model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeDetector().to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mel_spec, y, sr = audio_to_melspectrogram(audio_path)
    input_tensor = melspectrogram_to_tensor(mel_spec).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred_class = output.argmax(dim=1).item()
        confidence = probs[pred_class].item() * 100

    label = "FAKE" if pred_class == 1 else "REAL"
    spectrogram_image = generate_spectrogram_image(audio_path)
    waveform_image = generate_waveform_image(y, sr)

    return label, confidence, spectrogram_image, waveform_image


def generate_waveform_image(y: np.ndarray, sr: int) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 2), facecolor='#0d1117')
    times = np.linspace(0, len(y) / sr, len(y))
    ax.plot(times, y, color='#00d4ff', linewidth=0.5, alpha=0.8)
    ax.set_facecolor('#0d1117')
    ax.set_xlabel('Time (s)', color='white')
    ax.set_ylabel('Amplitude', color='white')
    ax.set_title('Waveform', color='white', fontsize=12)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                facecolor='#0d1117', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()


# ─── Training (with overfitting fixes) ───────────────────────────────────────

def train_audio_model(train_loader, val_loader, epochs=15,
                      save_path="trained_models/audio_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioDeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=1e-2)       # L2 regularization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience = 5
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        train_acc = 100. * correct / total
        val_acc = _evaluate_audio(model, val_loader, device)
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


def _evaluate_audio(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for specs, labels in loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total