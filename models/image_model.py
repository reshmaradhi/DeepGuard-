import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2


# ─── Model Definition (with stronger regularization) ─────────────────────────

class ImageDeepfakeDetector(nn.Module):
    def __init__(self):
        super(ImageDeepfakeDetector, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        in_features = self.backbone.classifier.in_features
        # Higher dropout to reduce overfitting
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),           # increased from 0.3
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),       # added BatchNorm
            nn.Dropout(0.4),           # increased from 0.2
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# ─── Transforms (stronger augmentation to reduce overfitting) ────────────────

def get_transform(train=False):
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
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


# ─── Grad-CAM ────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer = self.model.backbone.blocks[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = gradients.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activations).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        cam -= cam.min()
        if cam.max() != 0:
            cam /= cam.max()

        return cam, class_idx


# ─── Overlay Heatmap ─────────────────────────────────────────────────────────

def overlay_heatmap(original_image: Image.Image, cam: np.ndarray) -> Image.Image:
    img = np.array(original_image.resize((224, 224))).astype(np.uint8)
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlay)


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict_image(image: Image.Image, model_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageDeepfakeDetector().to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = get_transform(train=False)
    input_tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    gradcam = GradCAM(model)
    cam, pred_class = gradcam.generate(input_tensor)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        confidence = probs[pred_class].item() * 100

    label = "FAKE" if pred_class == 1 else "REAL"
    heatmap_image = overlay_heatmap(image, cam)

    return label, confidence, heatmap_image


# ─── Training (with overfitting fixes) ───────────────────────────────────────

def train_image_model(train_loader, val_loader, epochs=15,
                      save_path="trained_models/image_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageDeepfakeDetector().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                  weight_decay=1e-2)       # L2 regularization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    best_val_acc = 0.0
    patience = 5
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
        val_acc = evaluate_model(model, val_loader, device)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  Model saved (Val Acc: {val_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  ⏹ Early stopping at epoch {epoch+1}")
                break

    print(f"\n Best Val Accuracy: {best_val_acc:.2f}%")
    return model


def evaluate_model(model, loader, device):
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
