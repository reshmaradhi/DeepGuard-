"""
crossmodal.py — Cross-Modal Consistency Analysis
Compares visual and audio verdicts to detect partial deepfakes
"""


# ─── Cross-Modal Consistency Score ───────────────────────────────────────────

def compute_crossmodal_score(
    visual_label: str,
    visual_confidence: float,
    audio_label: str,
    audio_confidence: float,
    lipsync_verdict: str,
    lipsync_score: float
):
    """
    Compute cross-modal consistency score and final verdict.

    Args:
        visual_label: 'REAL' or 'FAKE'
        visual_confidence: 0-100
        audio_label: 'REAL' or 'FAKE'  
        audio_confidence: 0-100
        lipsync_verdict: 'SYNCED', 'MISMATCH', 'UNCERTAIN', 'NO_AUDIO', 'NO_FACE'
        lipsync_score: 0-100

    Returns:
        consistency_score: 0-100
        consistency_verdict: string
        manipulation_type: string
        details: dict
    """

    visual_fake = visual_label == "FAKE"
    audio_fake = audio_label == "FAKE"
    lip_mismatch = lipsync_verdict == "MISMATCH"

    # ── Determine manipulation type ──────────────────────────────────────────
    if not visual_fake and not audio_fake and not lip_mismatch:
        manipulation_type = "AUTHENTIC"
        consistency_verdict = "CONSISTENT"
        consistency_score = 95.0
        threat_level = "NONE"
        description = (
            "Both visual and audio modalities appear authentic. "
            "Lip movements are synchronized with the audio. "
            "No signs of deepfake manipulation detected."
        )

    elif visual_fake and audio_fake and lip_mismatch:
        manipulation_type = "FULL_DEEPFAKE"
        consistency_verdict = "CONSISTENT"
        consistency_score = 90.0
        threat_level = "CRITICAL"
        description = (
            "Both visual and audio modalities are detected as fake. "
            "Lip movements are not synchronized with audio. "
            "This is a fully manipulated deepfake across all modalities."
        )

    elif visual_fake and audio_fake and not lip_mismatch:
        manipulation_type = "FULL_DEEPFAKE"
        consistency_verdict = "CONSISTENT"
        consistency_score = 80.0
        threat_level = "HIGH"
        description = (
            "Both visual and audio modalities are detected as fake. "
            "Despite apparent lip sync, both streams show AI manipulation. "
            "Sophisticated deepfake with coordinated audio-visual generation."
        )

    elif not visual_fake and audio_fake:
        manipulation_type = "AUDIO_DUBBED"
        consistency_verdict = "INCONSISTENT"
        consistency_score = 20.0
        threat_level = "HIGH"
        description = (
            "Visual content appears REAL but audio is AI-generated. "
            "This is a classic dubbing attack — real video with fake AI voice. "
            "The person's lips may not match what they appear to be saying."
        )

    elif visual_fake and not audio_fake:
        manipulation_type = "FACE_SWAPPED"
        consistency_verdict = "INCONSISTENT"
        consistency_score = 20.0
        threat_level = "HIGH"
        description = (
            "Audio appears REAL but visual content is AI-manipulated. "
            "This is a face-swap attack — real audio with fake face overlaid. "
            "The voice is authentic but the face has been digitally replaced."
        )

    elif not visual_fake and not audio_fake and lip_mismatch:
        manipulation_type = "LIP_SYNC_ANOMALY"
        consistency_verdict = "UNCERTAIN"
        consistency_score = 50.0
        threat_level = "MEDIUM"
        description = (
            "Visual and audio appear real individually, but lip movements "
            "do not match the audio. This could indicate subtle dubbing "
            "or a recording/compression artifact."
        )

    else:
        manipulation_type = "UNCERTAIN"
        consistency_verdict = "UNCERTAIN"
        consistency_score = 50.0
        threat_level = "LOW"
        description = "Analysis results are inconclusive. Manual review recommended."

    # ── Adjust score based on confidence levels ───────────────────────────────
    avg_confidence = (visual_confidence + audio_confidence) / 2
    if avg_confidence > 90:
        consistency_score = min(100, consistency_score * 1.05)
    elif avg_confidence < 70:
        consistency_score = consistency_score * 0.9

    return consistency_score, consistency_verdict, manipulation_type, {
        "threat_level": threat_level,
        "description": description,
        "visual": {"label": visual_label, "confidence": visual_confidence},
        "audio": {"label": audio_label, "confidence": audio_confidence},
        "lipsync": {"verdict": lipsync_verdict, "score": lipsync_score},
        "manipulation_type": manipulation_type,
        "consistency_score": consistency_score
    }


def get_threat_color(threat_level: str) -> str:
    colors = {
        "NONE": "#00d084",
        "LOW": "#00d4ff",
        "MEDIUM": "#ffb347",
        "HIGH": "#ff6b6b",
        "CRITICAL": "#ff0000"
    }
    return colors.get(threat_level, "#94a3b8")


def get_manipulation_icon(manipulation_type: str) -> str:
    icons = {
        "AUTHENTIC": "✅",
        "FULL_DEEPFAKE": "🚨",
        "AUDIO_DUBBED": "🎙️",
        "FACE_SWAPPED": "👤",
        "LIP_SYNC_ANOMALY": "⚠️",
        "UNCERTAIN": "❓"
    }
    return icons.get(manipulation_type, "❓")