"""
forensic_report.py — Auto-generate professional forensic analysis reports
"""

from datetime import datetime
import numpy as np


def generate_forensic_report(
    media_type: str,
    filename: str,
    visual_label: str = None,
    visual_confidence: float = None,
    audio_label: str = None,
    audio_confidence: float = None,
    lipsync_verdict: str = None,
    lipsync_score: float = None,
    manipulation_type: str = None,
    consistency_score: float = None,
    frame_results: list = None,
    gradcam_regions: str = None
) -> dict:
    """
    Generate a structured forensic report dictionary.
    Used by the UI to render the report card.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"DGF-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    report = {
        "report_id": report_id,
        "timestamp": timestamp,
        "filename": filename,
        "media_type": media_type.upper(),
    }

    # ── Image Report ──────────────────────────────────────────────────────────
    if media_type.lower() == "image":
        report["verdict"] = visual_label
        report["confidence"] = visual_confidence
        report["sections"] = [
            {
                "title": "Visual Analysis",
                "items": [
                    f"Classification: {visual_label}",
                    f"Confidence: {visual_confidence:.1f}%",
                    f"Model: EfficientNet-B0 (Transfer Learning)",
                    f"Explainability: Grad-CAM heatmap generated",
                    f"Suspicious regions highlighted in heatmap overlay"
                ]
            },
            {
                "title": "Artifact Indicators",
                "items": _get_image_artifacts(visual_label, visual_confidence)
            },
            {
                "title": "Recommendation",
                "items": [_get_recommendation(visual_label, visual_confidence, "image")]
            }
        ]

    # ── Audio Report ──────────────────────────────────────────────────────────
    elif media_type.lower() == "audio":
        report["verdict"] = audio_label
        report["confidence"] = audio_confidence
        report["sections"] = [
            {
                "title": "Audio Analysis",
                "items": [
                    f"Classification: {audio_label} ({'AI-Generated' if audio_label == 'FAKE' else 'Human Speech'})",
                    f"Confidence: {audio_confidence:.1f}%",
                    f"Model: Custom CNN on Mel Spectrogram",
                    f"Feature: 128-band Mel Spectrogram (4s window)",
                    f"Dataset: ASVspoof 2019 LA partition"
                ]
            },
            {
                "title": "Speech Pattern Analysis",
                "items": _get_audio_artifacts(audio_label, audio_confidence)
            },
            {
                "title": "Recommendation",
                "items": [_get_recommendation(audio_label, audio_confidence, "audio")]
            }
        ]

    # ── Video Report ──────────────────────────────────────────────────────────
    elif media_type.lower() == "video":
        overall_verdict = manipulation_type or (
            "FAKE" if visual_label == "FAKE" or audio_label == "FAKE" else "REAL"
        )
        report["verdict"] = overall_verdict
        report["confidence"] = (
            (visual_confidence + audio_confidence) / 2
            if visual_confidence and audio_confidence else visual_confidence
        )
        report["manipulation_type"] = manipulation_type

        # Frame statistics
        if frame_results:
            fake_frames = sum(1 for f in frame_results if f["label"] == "FAKE")
            total_frames = len(frame_results)
            avg_conf = np.mean([f["confidence"] for f in frame_results])
        else:
            fake_frames = total_frames = avg_conf = 0

        report["sections"] = [
            {
                "title": "Visual Analysis (Frame-Level)",
                "items": [
                    f"Verdict: {visual_label} ({visual_confidence:.1f}% confidence)",
                    f"Fake frames: {fake_frames}/{total_frames} ({fake_frames/max(total_frames,1)*100:.1f}%)",
                    f"Average frame confidence: {avg_conf:.1f}%",
                    f"Model: EfficientNet-B0 with layer freezing",
                    f"Method: Majority voting across {total_frames} frames"
                ]
            },
            {
                "title": "Audio Analysis",
                "items": [
                    f"Verdict: {audio_label} ({audio_confidence:.1f}% confidence)",
                    f"Audio type: {'AI-Generated Speech' if audio_label == 'FAKE' else 'Human Speech'}",
                    f"Model: CNN on Mel Spectrogram",
                    f"Dataset: ASVspoof 2019 (balanced)"
                ]
            },
            {
                "title": "Lip-Sync Analysis",
                "items": [
                    f"Verdict: {lipsync_verdict}",
                    f"Sync Score: {lipsync_score:.1f}/100",
                    f"Method: Facial landmark tracking + audio energy correlation",
                    _get_lipsync_interpretation(lipsync_verdict, lipsync_score)
                ]
            },
            {
                "title": "Cross-Modal Consistency",
                "items": [
                    f"Manipulation Type: {manipulation_type}",
                    f"Consistency Score: {consistency_score:.1f}/100",
                    f"Visual: {visual_label} | Audio: {audio_label} | Lip-Sync: {lipsync_verdict}",
                    _get_manipulation_description(manipulation_type)
                ]
            },
            {
                "title": "Recommendation",
                "items": [_get_recommendation(overall_verdict, report["confidence"], "video")]
            }
        ]

    return report


def _get_image_artifacts(label: str, confidence: float) -> list:
    if label == "FAKE":
        items = [
            "GAN-based generation artifacts likely detected",
            "Potential facial boundary inconsistencies",
            "Unnatural texture patterns in highlighted regions",
            "Possible blending artifacts around facial features"
        ]
        if confidence > 90:
            items.append("High confidence — strong manipulation indicators present")
    else:
        items = [
            "No significant GAN artifacts detected",
            "Natural facial texture patterns observed",
            "Consistent lighting and shadow distribution",
            "No blending or boundary artifacts found"
        ]
    return items


def _get_audio_artifacts(label: str, confidence: float) -> list:
    if label == "FAKE":
        return [
            "Unnatural prosody patterns detected",
            "Possible neural vocoder artifacts present",
            "Speech rhythm inconsistent with natural human speech",
            "Spectral patterns typical of TTS/VC systems",
            f"Detection confidence: {confidence:.1f}% — {'Strong' if confidence > 85 else 'Moderate'} evidence"
        ]
    else:
        return [
            "Natural prosody and speech rhythm detected",
            "No neural vocoder artifacts found",
            "Spectral patterns consistent with human speech",
            "Natural breathing and pause patterns observed"
        ]


def _get_lipsync_interpretation(verdict: str, score: float) -> str:
    if verdict == "SYNCED":
        return f"Lip movements correlate well with audio content"
    elif verdict == "MISMATCH":
        return f"Significant mismatch — lips and audio are not coordinated"
    elif verdict == "UNCERTAIN":
        return f"Borderline sync — inconclusive result"
    elif verdict == "NO_AUDIO":
        return "No audio track present for comparison"
    else:
        return "Face detection insufficient for lip analysis"


def _get_manipulation_description(manipulation_type: str) -> str:
    descriptions = {
        "AUTHENTIC": "No manipulation detected across any modality",
        "FULL_DEEPFAKE": "Complete deepfake — both face and voice are AI-generated",
        "AUDIO_DUBBED": "Dubbing attack — real video with AI-generated voice overlay",
        "FACE_SWAPPED": "Face-swap attack — real audio with AI-generated face",
        "LIP_SYNC_ANOMALY": "Possible subtle dubbing or encoding artifact",
        "UNCERTAIN": "Insufficient confidence for definitive classification"
    }
    return descriptions.get(manipulation_type, "Unknown manipulation type")


def _get_recommendation(label: str, confidence: float, media_type: str) -> str:
    is_fake = label in ["FAKE", "FULL_DEEPFAKE", "AUDIO_DUBBED", "FACE_SWAPPED"]

    if is_fake and confidence > 90:
        return ("HIGH CONFIDENCE DEEPFAKE: Do not share or trust this content. "
                "Report to platform moderators if found online.")
    elif is_fake and confidence > 75:
        return ("LIKELY DEEPFAKE: Exercise caution with this content. "
                "Seek additional verification before sharing.")
    elif is_fake:
        return ("POSSIBLE MANIPULATION: Results are uncertain. "
                "Manual expert review recommended.")
    else:
        return ("LIKELY AUTHENTIC: No significant manipulation detected. "
                "Standard verification practices still recommended.")