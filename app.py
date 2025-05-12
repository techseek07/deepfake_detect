import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from face_extractor import extract_faces_from_video
from predictor import predict_fake_or_real

# ---------------------------
# 1. SETUP DIRECTORIES
# ---------------------------
def setup_directories():
    for directory in ['uploads', 'outputs', 'temp_frames']:
        os.makedirs(directory, exist_ok=True)

setup_directories()

# ---------------------------
# 2. STREAMLIT UI CONFIG
# ---------------------------
st.set_page_config(page_title="DeepGuard", page_icon="🛡️", layout="wide")
st.title("Deepfake Video Detection System")
st.markdown("""
    ### Upload a video file for authenticity analysis
    Supported formats: MP4, MOV, AVI (max 5 minutes)
""")

# ---------------------------
# 3. VIDEO METADATA (using headless OpenCV)
# ---------------------------
def display_video_metadata(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.sidebar.error("❌ Unable to read video metadata")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS) or 0
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration    = frame_count / fps if fps > 0 else 0.0
    cap.release()

    metadata = {
        "Resolution": f"{width}×{height}",
        "Duration": f"{duration:.2f} seconds",
        "Frame Rate": f"{fps:.2f} FPS"
    }
    st.sidebar.subheader("Video Metadata")
    st.sidebar.json(metadata)

# ---------------------------
# 4. VIDEO PROCESSING & ANALYSIS
# ---------------------------
def analyze_video(uploaded_file):
    temp_video_path = os.path.join("uploads", uploaded_file.name)
    try:
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Preview & metadata
        st.subheader("Uploaded Video Preview")
        st.video(temp_video_path)
        display_video_metadata(temp_video_path)

        # Extract faces
        with st.expander("Face Extraction Progress"):
            faces = extract_faces_from_video(temp_video_path)
            st.success(f"Extracted {len(faces)} face frames")

        if not faces:
            st.warning("No faces detected in the video")
            return

        # Predict
        with st.spinner("Analyzing facial features..."):
            predictions, confidence_scores = predict_fake_or_real(faces)

        if not predictions or not confidence_scores:
            st.error("Analysis failed: Could not process facial features")
            return

        # Display Results
        st.subheader("Analysis Results")
        fake_prob = np.mean(confidence_scores)
        final_verdict = (
            "FAKE" if fake_prob > 0.65
            else "REAL" if fake_prob < 0.35
            else "UNCERTAIN"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Fake Probability", f"{fake_prob*100:.1f}%")
        c2.metric("Analyzed Frames", len(predictions))
        c3.metric("Final Verdict", final_verdict, delta_color="off")

        # Detailed Analysis
        with st.expander("Detailed Analysis"):
            tab1, tab2 = st.tabs(["Confidence Timeline", "Frame Samples"])
            with tab1:
                df = pd.DataFrame({
                    "Frame": list(range(len(confidence_scores))),
                    "Fake Confidence": confidence_scores
                })
                fig = px.line(df, x="Frame", y="Fake Confidence",
                              title="Fake Confidence Over Time",
                              labels={"Fake Confidence": "Confidence Score"})
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                cols = st.columns(4)
                for idx, (face, pred) in enumerate(zip(faces[:8], predictions[:8])):
                    with cols[idx % 4]:
                        st.image(face, caption=f"Frame {idx+1}: {'Fake' if pred else 'Real'}")
    except Exception as e:
        st.error(f"Processing error: {e}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# ---------------------------
# 5. MAIN APP FLOW
# ---------------------------
uploaded_file = st.file_uploader(
    "Choose video file",
    type=["mp4", "mov", "avi"],
    help="Maximum file size: 500MB"
)
if uploaded_file:
    analyze_video(uploaded_file)
else:
    st.info("Please upload a video file to begin analysis")

# ---------------------------
# 6. SIDEBAR INFO
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("System Specifications")
st.sidebar.markdown("""
**Detection Engine:**
- Xception Network (Spatial Analysis)
- CNN-LSTM Hybrid (Temporal Patterns)
- Vision Transformer (Global Context)

**Decision Thresholds:**
- 🔴 FAKE: Confidence > 65%
- 🟡 UNCERTAIN: 35–65%
- 🟢 REAL: Confidence < 35%
""")
