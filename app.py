import streamlit as st
import os
import numpy as np
import pandas as pd
import plotly.express as px
import imageio.v3 as iio
import cv2
from face_extractor import extract_faces_from_video
from predictor import predict_fake_or_real

# ---------------------------
# 1. SETUP DIRECTORIES
# ---------------------------
def setup_directories():
    """Create necessary directories if they don't exist"""
    for directory in ['uploads', 'output', 'temp_frames']:
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
# 3. VIDEO PROCESSING & ANALYSIS
# ---------------------------
def display_video_metadata(video_path: str):
    """Display essential video information"""
    try:
        # Get video metadata using imageio
        reader = iio.imopen(video_path, "r")
        metadata = reader.metadata()
        
        video_info = {
            "Resolution": f"{metadata['shape'][1]}x{metadata['shape'][0]}",
            "Duration": f"{reader.properties().duration:.2f} seconds",
            "Frame Rate": f"{metadata['fps']:.2f} FPS"
        }
        reader.close()
    except Exception as e:
        video_info = {"Error": str(e)}
    
    st.sidebar.subheader("Video Metadata")
    st.sidebar.json(video_info)

def analyze_video(uploaded_file):
    """Main video analysis pipeline"""
    temp_video_path = os.path.join("uploads", uploaded_file.name)

    try:
        # Save uploaded file
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display video preview
        st.subheader("Uploaded Video Preview")
        st.video(temp_video_path)
        display_video_metadata(temp_video_path)

        # Extract faces
        with st.expander("Face Extraction Progress"):
            faces = extract_faces_from_video(temp_video_path)
            if faces:
                st.success(f"Extracted {len(faces)} face frames")
            else:
                st.warning("No faces detected in the video")
                return

        # Make predictions
        try:
            with st.spinner("Analyzing facial features..."):
                predictions, confidence_scores = predict_fake_or_real(faces)

            if not predictions or not confidence_scores:
                st.error("Analysis failed: Could not process facial features")
                return

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return

        # Display results
        st.subheader("Analysis Results")
        fake_prob = np.mean(confidence_scores)
        final_verdict = "FAKE" if fake_prob > 0.65 else "REAL" if fake_prob < 0.35 else "UNCERTAIN"

        col1, col2, col3 = st.columns(3)
        col1.metric("Fake Probability", f"{fake_prob * 100:.1f}%")
        col2.metric("Analyzed Frames", len(predictions))
        col3.metric("Final Verdict", final_verdict, delta_color="off")

        # Visualizations
        with st.expander("Detailed Analysis"):
            tab1, tab2 = st.tabs(["Confidence Timeline", "Frame Samples"])

            with tab1:
                df = pd.DataFrame({
                    'Frame': range(len(confidence_scores)),
                    'Fake Confidence': confidence_scores
                })
                fig = px.line(df, x='Frame', y='Fake Confidence',
                              title="Fake Confidence Over Time",
                              labels={'Fake Confidence': 'Confidence Score'})
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                cols = st.columns(4)
                for idx, (face, pred) in enumerate(zip(faces[:8], predictions[:8])):
                    with cols[idx % 4]:
                        st.image(face, caption=f"Frame {idx + 1}: {'Fake' if pred else 'Real'}")

    except Exception as e:
        st.error(f"Processing error: {str(e)}")
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# ---------------------------
# 4. MAIN APP FLOW
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
# 5. SIDEBAR INFORMATION
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
    - 🟡 UNCERTAIN: 35-65%
    - 🟢 REAL: Confidence < 35%
""")
