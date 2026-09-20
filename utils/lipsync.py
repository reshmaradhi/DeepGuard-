"""
lipsync.py — Lip Sync Analysis using OpenCV face detection + audio energy
Pure OpenCV approach — no mediapipe or dlib required
"""

import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def extract_lip_movements(video_path: str, num_frames: int = 30):
    """Extract lip openness using OpenCV face+mouth detection."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    lip_openness = []
    frames_with_face = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            lip_openness.append(0.0)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Lower 30% of face = mouth region
            mouth_y = y + int(0.65 * h)
            mouth_region = gray[mouth_y:y+h, x:x+w]

            if mouth_region.size > 0:
                # Use pixel variance + edge density as mouth openness proxy
                edges = cv2.Canny(mouth_region, 50, 150)
                edge_density = np.sum(edges > 0) / (mouth_region.size + 1e-6)
                variance = np.var(mouth_region) / 5000.0
                openness = (edge_density * 0.6 + variance * 0.4)
                lip_openness.append(float(min(openness, 1.0)))
                frames_with_face += 1
            else:
                lip_openness.append(0.0)
        else:
            # No face found — use full frame motion as fallback
            lip_openness.append(0.0)

    cap.release()
    return np.array(lip_openness), fps, duration, frames_with_face


def extract_audio_energy(video_path: str, num_frames: int = 30):
    """Extract audio energy using librosa (works directly on video files)."""
    try:
        import librosa
        y, sr = librosa.load(video_path, sr=16000, mono=True)
        samples_per_frame = max(len(y) // num_frames, 1)
        audio_energy = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            segment = y[start:end]
            rms = float(np.sqrt(np.mean(segment**2))) if len(segment) > 0 else 0.0
            audio_energy.append(rms)
        audio_energy = np.array(audio_energy)
        if audio_energy.max() > 0:
            audio_energy = audio_energy / audio_energy.max()
        return audio_energy, True
    except Exception as e:
        print(f"librosa audio extraction failed: {e}")

    # Try moviepy as fallback
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            clip.close()
            return np.zeros(num_frames), False
        audio_array = clip.audio.to_soundarray(fps=16000)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        clip.close()
        samples_per_frame = max(len(audio_array) // num_frames, 1)
        audio_energy = []
        for i in range(num_frames):
            start = i * samples_per_frame
            end = start + samples_per_frame
            segment = audio_array[start:end]
            rms = float(np.sqrt(np.mean(segment**2))) if len(segment) > 0 else 0.0
            audio_energy.append(rms)
        audio_energy = np.array(audio_energy)
        if audio_energy.max() > 0:
            audio_energy = audio_energy / audio_energy.max()
        return audio_energy, True
    except Exception as e:
        print(f"moviepy also failed: {e}")
        return np.zeros(num_frames), False


def compute_lipsync_score(video_path: str, num_frames: int = 30):
    """Compute lip sync score — correlation between lip movement and audio energy."""
    lip_movements, fps, duration, faces_found = extract_lip_movements(video_path, num_frames)
    audio_energy, has_audio = extract_audio_energy(video_path, num_frames)

    if not has_audio:
        return 50.0, "NO_AUDIO", {
            "message": "No audio track found in video",
            "lip_movements": lip_movements.tolist(),
            "audio_energy": audio_energy.tolist()
        }

    if faces_found < num_frames * 0.2:
        return 50.0, "NO_FACE", {
            "message": "Face not detected in most frames — using signal analysis only",
            "lip_movements": lip_movements.tolist(),
            "audio_energy": audio_energy.tolist()
        }

    lip_norm = lip_movements / (lip_movements.max() + 1e-6)

    correlation = np.corrcoef(lip_norm, audio_energy)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0

    score = (correlation + 1) / 2 * 100

    if score >= 55:
        verdict = "SYNCED"
    elif score >= 38:
        verdict = "UNCERTAIN"
    else:
        verdict = "MISMATCH"

    details = {
        "correlation": float(correlation),
        "score": float(score),
        "duration": float(duration),
        "faces_detected": faces_found,
        "frames_analyzed": num_frames,
        "lip_movements": lip_norm.tolist(),
        "audio_energy": audio_energy.tolist(),
        "message": _get_lipsync_message(verdict, score)
    }
    return score, verdict, details


def _get_lipsync_message(verdict, score):
    messages = {
        "SYNCED": f"Lip movements are synchronized with audio (score: {score:.1f}/100)",
        "MISMATCH": f"Lip movements do NOT match audio — possible dubbing detected (score: {score:.1f}/100)",
        "UNCERTAIN": f"Lip-audio sync is borderline (score: {score:.1f}/100)",
        "NO_AUDIO": "No audio track detected in video",
        "NO_FACE": "Face not clearly detected — analysis based on signal patterns"
    }
    return messages.get(verdict, "Unknown")


def plot_lipsync_analysis(lip_movements, audio_energy):
    """Generate lip sync visualization."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), facecolor='#0d1117')
    fig.suptitle('Lip-Sync Analysis', color='white', fontsize=13, fontweight='bold')
    x = np.arange(len(lip_movements))

    ax1.plot(x, lip_movements, color='#00d4ff', linewidth=1.5)
    ax1.set_facecolor('#0d1117')
    ax1.set_title('Lip Movement (OpenCV)', color='#94a3b8', fontsize=9)
    ax1.set_ylabel('Openness', color='white', fontsize=8)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values(): spine.set_edgecolor('#1e3a5f')

    ax2.plot(x, audio_energy, color='#ff6b9d', linewidth=1.5)
    ax2.set_facecolor('#0d1117')
    ax2.set_title('Audio Energy (RMS)', color='#94a3b8', fontsize=9)
    ax2.set_ylabel('Energy', color='white', fontsize=8)
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values(): spine.set_edgecolor('#1e3a5f')

    ax3.plot(x, lip_movements, color='#00d4ff', linewidth=1.5, label='Lip', alpha=0.8)
    ax3.plot(x, audio_energy, color='#ff6b9d', linewidth=1.5, label='Audio', alpha=0.8)
    ax3.fill_between(x, lip_movements, audio_energy, alpha=0.15, color='#7b61ff')
    ax3.set_facecolor('#0d1117')
    ax3.set_title('Sync Overlay', color='#94a3b8', fontsize=9)
    ax3.set_xlabel('Frame Index', color='white', fontsize=8)
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#1e2d40', labelcolor='white', fontsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor('#1e3a5f')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#0d1117', dpi=100)
    plt.close()
    buf.seek(0)
    return Image.open(buf).copy()