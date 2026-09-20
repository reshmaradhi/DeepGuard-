import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DeepGuard — Multimodal Deepfake Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #080c14; color: #e2e8f0; }
    .main { background-color: #080c14; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1520 0%, #0a1018 100%); border-right: 1px solid #1e2d40; }

    .hero-title { font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff 0%, #7b61ff 50%, #ff6b6b 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.2rem; letter-spacing: -1px; }

    .hero-subtitle { font-size: 1rem; color: #64748b; font-weight: 300;
        letter-spacing: 2px; text-transform: uppercase; margin-bottom: 2rem; }

    .result-card { background: linear-gradient(135deg, #0d1a2a 0%, #111827 100%);
        border: 1px solid #1e3a5f; border-radius: 16px; padding: 2rem;
        margin: 1rem 0; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }

    .result-fake { border-left: 4px solid #ff4444; background: linear-gradient(135deg, #1a0d0d 0%, #111827 100%); }
    .result-real { border-left: 4px solid #00d084; background: linear-gradient(135deg, #0d1a14 0%, #111827 100%); }

    .verdict-fake { font-family: 'Syne', sans-serif; font-size: 2.5rem; font-weight: 800; color: #ff4444; letter-spacing: 4px; }
    .verdict-real { font-family: 'Syne', sans-serif; font-size: 2.5rem; font-weight: 800; color: #00d084; letter-spacing: 4px; }

    .section-header { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700;
        color: #94a3b8; text-transform: uppercase; letter-spacing: 3px;
        margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e2d40; }

    .info-box { background: #0d1a2a; border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1rem 1.5rem; margin: 1rem 0; font-size: 0.9rem; color: #94a3b8; }

    .forensic-report { background: #0a0f1a; border: 1px solid #1e3a5f;
        border-radius: 16px; padding: 1.5rem; margin: 1rem 0; }

    .report-header { font-family: 'Syne', sans-serif; font-size: 1.3rem;
        font-weight: 700; color: #00d4ff; margin-bottom: 1rem; }

    .report-section { margin: 1rem 0; padding: 1rem; background: #0d1520;
        border-radius: 8px; border-left: 3px solid #1e3a5f; }

    .report-section-title { font-weight: 600; color: #7b61ff; font-size: 0.9rem;
        text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem; }

    .report-item { font-size: 0.85rem; color: #94a3b8; padding: 0.2rem 0; }

    .modal-badge { display: inline-block; padding: 0.3rem 0.8rem;
        border-radius: 20px; font-size: 0.75rem; font-weight: 600;
        letter-spacing: 1px; margin: 0.2rem; }

    .badge-fake { background: #ff444422; color: #ff4444; border: 1px solid #ff4444; }
    .badge-real { background: #00d08422; color: #00d084; border: 1px solid #00d084; }
    .badge-warn { background: #ffb34722; color: #ffb347; border: 1px solid #ffb347; }
    .badge-info { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff; }

    .stButton > button { background: linear-gradient(135deg, #0d1a2a, #162033);
        color: #e2e8f0; border: 1px solid #1e3a5f; border-radius: 12px;
        padding: 1rem 2rem; font-family: 'Syne', sans-serif; font-weight: 600;
        font-size: 1rem; transition: all 0.3s ease; width: 100%; }

    .stButton > button:hover { background: linear-gradient(135deg, #00d4ff22, #7b61ff22);
        border-color: #00d4ff; color: #00d4ff; transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,212,255,0.15); }

    [data-testid="stFileUploader"] { background: #0d1520; border: 2px dashed #1e3a5f; border-radius: 16px; padding: 1rem; }
    hr { border-color: #1e2d40; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🛡️</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; font-weight:800;
                    color:#00d4ff; letter-spacing:2px;'>DEEPGUARD</div>
        <div style='font-size:0.7rem; color:#475569; letter-spacing:3px;
                    text-transform:uppercase; margin-top:0.2rem;'>
            Multimodal Detection System
        </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Select Mode</div>", unsafe_allow_html=True)

    modality = st.radio(
        label="",
        options=[
            "🖼️  Image Detection",
            "🎥  Video Detection",
            "🎵  Audio Detection",
            "🔬  Cross-Modal Analysis"
        ],
        index=0
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <b style='color:#e2e8f0'>Detection Modes:</b><br><br>
        <b style='color:#00d4ff'>🖼️ Image</b><br>
        EfficientNet-B0 + Grad-CAM heatmap<br><br>
        <b style='color:#7b61ff'>🎥 Video</b><br>
        Frame-level CNN analysis<br>
        Majority voting across 16 frames<br><br>
        <b style='color:#ff6b9d'>🎵 Audio</b><br>
        Mel Spectrogram CNN<br>
        AI vs Human speech detection<br><br>
        <b style='color:#ffb347'>🔬 Cross-Modal</b><br>
        Visual + Audio + Lip-Sync<br>
        Consistency scoring + Forensic report
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='position:fixed; bottom:1rem; left:1rem; right:1rem;
                font-size:0.7rem; color:#334155; text-align:center;'>
        MTech Deep Learning Project<br>
        Multimodal Deepfake Detection
    </div>
    """, unsafe_allow_html=True)


# ─── Main Header ─────────────────────────────────────────────────────────────

st.markdown("<div class='hero-title'>DeepGuard</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>Multimodal Deepfake Detection System</div>",
            unsafe_allow_html=True)

IMAGE_MODEL_PATH = "trained_models/image_model.pth"
VIDEO_MODEL_PATH = "trained_models/video_model.pth"
AUDIO_MODEL_PATH = "trained_models/audio_model.pth"


def model_exists(path):
    return os.path.exists(path)


def render_verdict(label: str, confidence: float):
    is_fake = label in ["FAKE", "FULL_DEEPFAKE", "AUDIO_DUBBED", "FACE_SWAPPED"]
    css = "result-fake" if is_fake else "result-real"
    verdict_css = "verdict-fake" if is_fake else "verdict-real"
    icon = "⚠️" if is_fake else "✅"
    display = label.replace("_", " ")
    color = "#ff4444" if is_fake else "#00d084"
    st.markdown(f"""
    <div class='result-card {css}'>
        <div style='font-size:0.8rem; color:#64748b; letter-spacing:3px;
                    text-transform:uppercase; margin-bottom:0.5rem;'>Detection Result</div>
        <div class='{verdict_css}'>{icon} {display}</div>
        <div style='margin-top:1rem; font-size:0.9rem; color:#94a3b8;'>
            Confidence: <b style='color:{color}'>{confidence:.1f}%</b>
        </div>
    </div>""", unsafe_allow_html=True)


def render_confidence_bar(confidence: float, label: str):
    color = "#ff4444" if label in ["FAKE","FULL_DEEPFAKE","AUDIO_DUBBED","FACE_SWAPPED"] else "#00d084"
    st.progress(confidence / 100)
    st.markdown(f"<div style='color:{color}; font-size:1.5rem; font-weight:700;'>{confidence:.1f}%</div>",
                unsafe_allow_html=True)


def render_forensic_report(report: dict):
    verdict = report.get("verdict", "UNKNOWN")
    is_fake = verdict in ["FAKE", "FULL_DEEPFAKE", "AUDIO_DUBBED", "FACE_SWAPPED"]
    verdict_color = "#ff4444" if is_fake else "#00d084"

    st.markdown(f"""
    <div class='forensic-report'>
        <div class='report-header'>🔬 Forensic Analysis Report</div>
        <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;'>
            <div style='font-size:0.8rem; color:#64748b;'>Report ID:
                <span style='color:#00d4ff'>{report.get("report_id","N/A")}</span></div>
            <div style='font-size:0.8rem; color:#64748b;'>Time:
                <span style='color:#94a3b8'>{report.get("timestamp","N/A")}</span></div>
            <div style='font-size:0.8rem; color:#64748b;'>File:
                <span style='color:#94a3b8'>{report.get("filename","N/A")}</span></div>
            <div style='font-size:0.8rem; color:#64748b;'>Type:
                <span style='color:#94a3b8'>{report.get("media_type","N/A")}</span></div>
        </div>
        <div style='background:#0d1520; border-radius:12px; padding:1rem;
                    margin-bottom:1rem; border-left:4px solid {verdict_color};'>
            <div style='font-size:0.8rem; color:#64748b; letter-spacing:2px;
                        text-transform:uppercase;'>Overall Verdict</div>
            <div style='font-size:2rem; font-weight:800; color:{verdict_color};
                        font-family:Syne,sans-serif;'>
                {"⚠️" if is_fake else "✅"} {verdict.replace("_"," ")}
            </div>
            <div style='font-size:0.9rem; color:#94a3b8; margin-top:0.5rem;'>
                Confidence: {report.get("confidence", 0):.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    for section in report.get("sections", []):
        items_html = "".join([
            f"<div class='report-item'>• {item}</div>"
            for item in section["items"]
        ])
        st.markdown(f"""
        <div class='report-section'>
            <div class='report-section-title'>{section["title"]}</div>
            {items_html}
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_crossmodal_summary(details: dict):
    visual = details["visual"]
    audio = details["audio"]
    lipsync = details["lipsync"]
    threat = details["threat_level"]
    manip = details["manipulation_type"]
    threat_color = {
        "NONE": "#00d084", "LOW": "#00d4ff",
        "MEDIUM": "#ffb347", "HIGH": "#ff6b6b", "CRITICAL": "#ff0000"
    }.get(threat, "#94a3b8")

    v_badge = "badge-fake" if visual["label"] == "FAKE" else "badge-real"
    a_badge = "badge-fake" if audio["label"] == "FAKE" else "badge-real"
    l_badge = ("badge-warn" if lipsync["verdict"] == "MISMATCH"
               else "badge-real" if lipsync["verdict"] == "SYNCED"
               else "badge-info")

    st.markdown(f"""
    <div class='result-card' style='border-left:4px solid {threat_color};'>
        <div style='font-size:0.8rem; color:#64748b; letter-spacing:3px;
                    text-transform:uppercase;'>Cross-Modal Analysis Result</div>
        <div style='font-size:1.5rem; font-weight:800; color:{threat_color};
                    font-family:Syne,sans-serif; margin:0.5rem 0;'>
            {manip.replace("_", " ")}
        </div>
        <div style='margin:0.8rem 0;'>
            <span class='modal-badge {v_badge}'>
                👁 Visual: {visual["label"]} ({visual["confidence"]:.0f}%)</span>
            <span class='modal-badge {a_badge}'>
                🎵 Audio: {audio["label"]} ({audio["confidence"]:.0f}%)</span>
            <span class='modal-badge {l_badge}'>
                👄 Lip-Sync: {lipsync["verdict"]}</span>
        </div>
        <div style='font-size:0.85rem; color:#94a3b8; margin-top:0.8rem;'>
            {details["description"]}
        </div>
        <div style='margin-top:0.8rem; font-size:0.85rem;'>
            Threat Level: <span style='color:{threat_color}; font-weight:600;'>{threat}</span>
            &nbsp;|&nbsp; Consistency Score:
            <span style='color:{threat_color}; font-weight:600;'>
                {details["consistency_score"]:.1f}/100</span>
        </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  🖼️ IMAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

if "Image" in modality:
    st.markdown("<div class='section-header'>🖼️ Image Deepfake Detection</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Upload a face image. <b>EfficientNet-B0</b> classifies it as
        <b style='color:#ff4444'>Fake</b> or <b style='color:#00d084'>Real</b>,
        and <b>Grad-CAM</b> generates a heatmap highlighting suspicious facial regions.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload an image",
                                type=["jpg", "jpeg", "png", "webp"],
                                key="img_upload")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("<div class='section-header'>Original Image</div>",
                        unsafe_allow_html=True)
            st.image(image, use_column_width=True)

        if st.button("🔍  Analyze Image", key="btn_img"):
            with st.spinner("Running EfficientNet-B0 + Grad-CAM..."):
                try:
                    from models.image_model import predict_image
                    from utils.forensic_report import generate_forensic_report
                    mp = IMAGE_MODEL_PATH if model_exists(IMAGE_MODEL_PATH) else None
                    if not mp:
                        st.warning("⚠️ No trained model found. Train first.")
                    label, conf, heatmap = predict_image(image, mp)

                    with col2:
                        st.markdown("<div class='section-header'>Grad-CAM Heatmap</div>",
                                    unsafe_allow_html=True)
                        st.image(heatmap, use_column_width=True)
                        st.caption("🔴 Red = High suspicion | 🔵 Blue = Low suspicion")

                    with col3:
                        st.markdown("<div class='section-header'>Result</div>",
                                    unsafe_allow_html=True)
                        render_verdict(label, conf)
                        render_confidence_bar(conf, label)
                        if label == "FAKE":
                            st.error("This image appears AI-generated or manipulated.")
                        else:
                            st.success("This image appears authentic.")

                    report = generate_forensic_report(
                        "image", uploaded.name,
                        visual_label=label, visual_confidence=conf)
                    st.markdown("<div class='section-header'>Forensic Report</div>",
                                unsafe_allow_html=True)
                    render_forensic_report(report)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
#  🎥 VIDEO DETECTION (Visual Only)
# ══════════════════════════════════════════════════════════════════════════════

elif "Video" in modality:
    st.markdown("<div class='section-header'>🎥 Video Deepfake Detection</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Upload a video file. The system extracts <b>16 frames</b> and runs
        <b>EfficientNet-B0</b> on each frame independently. The final verdict is
        determined by <b>majority voting</b> — if 50%+ frames are fake, the video
        is classified as FAKE.<br><br>
        <i>For audio analysis and lip-sync detection, use
        <b style='color:#ffb347'>🔬 Cross-Modal Analysis</b> mode.</i>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a video",
                                type=["mp4", "avi", "mov", "mkv"],
                                key="vid_upload")
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(uploaded)

        if st.button("🔍  Analyze Video Frames", key="btn_vid"):
            with st.spinner("Extracting frames and running CNN analysis..."):
                try:
                    from models.video_model import predict_video
                    mp = VIDEO_MODEL_PATH if model_exists(VIDEO_MODEL_PATH) else None
                    if not mp:
                        st.warning("⚠️ No trained model found. Train first.")

                    label, conf, frame_results, summary = predict_video(tmp_path, mp)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        render_verdict(label, conf)
                        render_confidence_bar(conf, label)
                    with col2:
                        st.markdown(f"""
                        <div class='result-card'>
                            <div class='section-header'>Frame Analysis Summary</div>
                            <div style='font-size:1rem; color:#94a3b8;'>{summary}</div>
                            <div style='margin-top:0.8rem; font-size:0.85rem; color:#64748b;'>
                                Model: EfficientNet-B0<br>
                                Method: Majority voting (16 frames)<br>
                                Dataset: SDFVD 2.0 (927 videos)
                            </div>
                        </div>""", unsafe_allow_html=True)

                    # Frame grid
                    st.markdown("<div class='section-header'>Frame-by-Frame Results</div>",
                                unsafe_allow_html=True)
                    cols = st.columns(4)
                    for i, result in enumerate(frame_results[:8]):
                        with cols[i % 4]:
                            st.image(result["image"], use_column_width=True)
                            color = "#ff4444" if result["label"] == "FAKE" else "#00d084"
                            st.markdown(
                                f"<div style='color:{color}; font-size:0.75rem;"
                                f"font-weight:600; text-align:center;'>"
                                f"Frame {result['frame']}: {result['label']}"
                                f" ({result['confidence']:.0f}%)</div>",
                                unsafe_allow_html=True)

                    if label == "FAKE":
                        st.error("⚠️ Video frames show signs of deepfake manipulation. "
                                 "Run Cross-Modal Analysis for deeper investigation.")
                    else:
                        st.success("✅ Video frames appear authentic. "
                                   "Run Cross-Modal Analysis to also check audio.")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass


# ══════════════════════════════════════════════════════════════════════════════
#  🎵 AUDIO DETECTION
# ══════════════════════════════════════════════════════════════════════════════

elif "Audio" in modality:
    st.markdown("<div class='section-header'>🎵 Audio Deepfake Detection</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        Upload an audio file. The system converts it to a <b>128-band Mel Spectrogram</b>
        and classifies it using a <b>custom 4-block CNN</b> trained on ASVspoof 2019.
        Output tells you whether the speech is <b style='color:#00d084'>Human</b>
        or <b style='color:#ff4444'>AI-Generated</b>.
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload audio",
                                type=["wav", "flac", "mp3", "ogg"],
                                key="aud_upload")
    if uploaded:
        st.audio(uploaded)
        ext = uploaded.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        if st.button("🔍  Analyze Audio", key="btn_aud"):
            with st.spinner("Generating Mel Spectrogram and running CNN..."):
                try:
                    from models.audio_model import predict_audio
                    from utils.forensic_report import generate_forensic_report
                    mp = AUDIO_MODEL_PATH if model_exists(AUDIO_MODEL_PATH) else None
                    if not mp:
                        st.warning("⚠️ No trained model found. Train first.")
                    label, conf, spec_img, wave_img = predict_audio(tmp_path, mp)

                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("<div class='section-header'>Waveform</div>",
                                    unsafe_allow_html=True)
                        st.image(wave_img, use_column_width=True)
                        st.markdown("<div class='section-header'>Mel Spectrogram</div>",
                                    unsafe_allow_html=True)
                        st.image(spec_img, use_column_width=True)
                        st.caption("Spectrogram input to CNN — AI speech shows distinct patterns")

                    with col2:
                        audio_type = "AI-Generated Speech" if label == "FAKE" else "Human Speech"
                        render_verdict(label, conf)
                        st.markdown(
                            f"<div style='color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;'>"
                            f"Classification: <b style='color:"
                            f"{'#ff4444' if label=='FAKE' else '#00d084'}'>"
                            f"{audio_type}</b></div>",
                            unsafe_allow_html=True)
                        render_confidence_bar(conf, label)
                        st.markdown("""
                        <div class='info-box' style='margin-top:1rem;'>
                            <b>Model:</b> Custom 4-block CNN<br>
                            <b>Input:</b> 128-band Mel Spectrogram<br>
                            <b>Dataset:</b> ASVspoof 2019 LA (balanced)<br>
                            <b>Accuracy:</b> 98.50%
                        </div>""", unsafe_allow_html=True)

                    report = generate_forensic_report(
                        "audio", uploaded.name,
                        audio_label=label, audio_confidence=conf)
                    st.markdown("<div class='section-header'>Forensic Report</div>",
                                unsafe_allow_html=True)
                    render_forensic_report(report)

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass


# ══════════════════════════════════════════════════════════════════════════════
#  🔬 CROSS-MODAL ANALYSIS (Novel Feature)
# ══════════════════════════════════════════════════════════════════════════════

elif "Cross-Modal" in modality:
    st.markdown("<div class='section-header'>🔬 Cross-Modal Deepfake Analysis</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <b style='color:#ffb347'>Novel Feature</b> — Upload a video with audio.
        The system runs <b>4 simultaneous checks</b> and computes a
        <b>Cross-Modal Consistency Score</b> to detect sophisticated attacks:<br><br>
        <b style='color:#00d4ff'>① Visual Check</b> — Are the video frames fake?<br>
        <b style='color:#ff6b9d'>② Audio Check</b> — Is the speech AI-generated or human?<br>
        <b style='color:#7b61ff'>③ Lip-Sync Check</b> — Do lips match the audio? (dubbed detection)<br>
        <b style='color:#ffb347'>④ Consistency Score</b> — Do visual and audio agree?<br><br>
        Detects: <b>Full Deepfake</b> | <b>Audio Dubbed</b> (real face, fake voice) |
        <b>Face Swapped</b> (fake face, real voice) | <b>Authentic</b>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a video with audio",
                                type=["mp4", "avi", "mov", "mkv"],
                                key="cm_upload")
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.video(uploaded)

        if st.button("🔬  Run Cross-Modal Analysis", key="btn_cm"):
            progress = st.progress(0)
            status = st.empty()

            try:
                from models.video_model import predict_video
                from models.audio_model import predict_audio
                from utils.lipsync import compute_lipsync_score, plot_lipsync_analysis
                from utils.crossmodal import compute_crossmodal_score
                from utils.forensic_report import generate_forensic_report

                # ── ① Visual ─────────────────────────────────────────────────
                status.markdown("**① Running visual frame analysis...**")
                progress.progress(15)
                vmp = VIDEO_MODEL_PATH if model_exists(VIDEO_MODEL_PATH) else None
                visual_label, visual_conf, frame_results, frame_summary = \
                    predict_video(tmp_path, vmp)
                progress.progress(35)

                # ── ② Audio ──────────────────────────────────────────────────
                status.markdown("**② Running audio AI/human detection...**")
                try:
                    amp = AUDIO_MODEL_PATH if model_exists(AUDIO_MODEL_PATH) else None
                    audio_label, audio_conf, spec_img, wave_img = \
                        predict_audio(tmp_path, amp)
                except Exception as ae:
                    st.warning(f"Audio extraction issue: {ae}. Using default.")
                    audio_label, audio_conf = "REAL", 50.0
                    spec_img, wave_img = None, None
                progress.progress(55)

                # ── ③ Lip-Sync ───────────────────────────────────────────────
                status.markdown("**③ Running lip-sync analysis...**")
                lip_score, lip_verdict, lip_details = \
                    compute_lipsync_score(tmp_path)
                lip_plot = plot_lipsync_analysis(
                    np.array(lip_details["lip_movements"]),
                    np.array(lip_details["audio_energy"])
                )
                progress.progress(75)

                # ── ④ Cross-Modal ────────────────────────────────────────────
                status.markdown("**④ Computing cross-modal consistency...**")
                cm_score, cm_verdict, manip_type, cm_details = \
                    compute_crossmodal_score(
                        visual_label, visual_conf,
                        audio_label, audio_conf,
                        lip_verdict, lip_score
                    )
                progress.progress(95)

                # ── Generate Report ──────────────────────────────────────────
                report = generate_forensic_report(
                    "video", uploaded.name,
                    visual_label=visual_label, visual_confidence=visual_conf,
                    audio_label=audio_label, audio_confidence=audio_conf,
                    lipsync_verdict=lip_verdict, lipsync_score=lip_score,
                    manipulation_type=manip_type, consistency_score=cm_score,
                    frame_results=frame_results
                )
                progress.progress(100)
                status.empty()
                st.success("✅ Cross-Modal Analysis Complete!")

                # ── Display Results ───────────────────────────────────────────
                st.markdown("<div class='section-header'>① Visual Analysis</div>",
                            unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                with col1:
                    render_verdict(visual_label, visual_conf)
                with col2:
                    st.markdown(f"""
                    <div class='info-box'>
                        <b>Summary:</b> {frame_summary}<br>
                        <b>Model:</b> EfficientNet-B0<br>
                        <b>Method:</b> Majority voting (16 frames)
                    </div>""", unsafe_allow_html=True)

                # Frame grid
                cols = st.columns(4)
                for i, result in enumerate(frame_results[:8]):
                    with cols[i % 4]:
                        st.image(result["image"], use_column_width=True)
                        color = "#ff4444" if result["label"] == "FAKE" else "#00d084"
                        st.markdown(
                            f"<div style='color:{color}; font-size:0.75rem;"
                            f"font-weight:600; text-align:center;'>"
                            f"Frame {result['frame']}: {result['label']}"
                            f" ({result['confidence']:.0f}%)</div>",
                            unsafe_allow_html=True)

                st.markdown("<div class='section-header'>② Audio Analysis</div>",
                            unsafe_allow_html=True)
                col1, col2 = st.columns([1, 1])
                with col1:
                    audio_type = "AI-Generated" if audio_label == "FAKE" else "Human Speech"
                    render_verdict(audio_label, audio_conf)
                    st.caption(f"Classification: {audio_type}")
                with col2:
                    if wave_img:
                        st.image(wave_img, use_column_width=True)
                    if spec_img:
                        st.image(spec_img, use_column_width=True)

                st.markdown("<div class='section-header'>③ Lip-Sync Analysis</div>",
                            unsafe_allow_html=True)
                lip_color = ("#ff4444" if lip_verdict == "MISMATCH"
                             else "#00d084" if lip_verdict == "SYNCED"
                             else "#ffb347")
                lip_icon = ("❌" if lip_verdict == "MISMATCH"
                            else "✅" if lip_verdict == "SYNCED" else "⚠️")
                st.markdown(f"""
                <div class='result-card' style='border-left:4px solid {lip_color};'>
                    <div style='font-size:0.8rem; color:#64748b; letter-spacing:3px;
                                text-transform:uppercase;'>Lip-Sync Verdict</div>
                    <div style='font-size:1.8rem; font-weight:800; color:{lip_color};
                                font-family:Syne,sans-serif;'>
                        {lip_icon} {lip_verdict}
                    </div>
                    <div style='color:#94a3b8; font-size:0.9rem; margin-top:0.5rem;'>
                        Sync Score: <b style='color:{lip_color}'>{lip_score:.1f}/100</b>
                        &nbsp;|&nbsp; {lip_details["message"]}
                    </div>
                    <div style='color:#64748b; font-size:0.8rem; margin-top:0.5rem;'>
                        Method: OpenCV face detection + RMS audio energy + Pearson correlation
                    </div>
                </div>""", unsafe_allow_html=True)
                st.image(lip_plot, use_column_width=True,
                         caption="Lip Movement vs Audio Energy — aligned peaks = sync")

                st.markdown("<div class='section-header'>④ Cross-Modal Consistency</div>",
                            unsafe_allow_html=True)
                render_crossmodal_summary(cm_details)

                st.markdown("<div class='section-header'>Forensic Report</div>",
                            unsafe_allow_html=True)
                render_forensic_report(report)

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass