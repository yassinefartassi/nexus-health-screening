"""Streamlit entry point for the NEXUS_PFE multimodal health-screening test bench.

Three tabs, one per module: Ocular, Dermatology, rPPG (HRV/BPM).

This is the Raspberry Pi 5 deployment of the same desktop test bench: the
src/inference/ and src/preprocessing/ modules are reused unmodified from the
Ubuntu validation project. The only platform-specific piece is the camera
feed for the rPPG tab, which goes through src/camera/create_camera_source()
so the exact same capture loop works against a USB webcam, a phone IP
camera, or the RPi5's rpicam module -- selected via NEXUS_CAMERA_SOURCE
(see src/camera/camera_source.py).
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.camera.camera_source import create_camera_source
from src.inference.dermatology_infer import predict_skin_condition
from src.inference.ocular_infer import predict_eye_condition
from src.inference.rppg_infer import RPPGBuffer
from src.preprocessing.face_roi import FaceROITracker

st.set_page_config(page_title="NEXUS_PFE — Health Screening Test Bench", layout="wide")
st.title("NEXUS_PFE — Multimodal Health Screening Test Bench")
st.caption(
    "Phase 2: Raspberry Pi 5 deployment of the ocular, dermatology, and rPPG "
    "modules validated on desktop -- same src/inference/ code, camera source "
    "swapped via config/env (see docs/deployment.md)."
)
st.sidebar.caption(
    f"Camera source: **{os.environ.get('NEXUS_CAMERA_SOURCE', '(auto-detected)')}**"
)

tab_ocular, tab_derm, tab_rppg = st.tabs(["Ocular", "Dermatology", "rPPG (HRV/BPM)"])


def _render_classifier_tab(title: str, caption: str, predict_fn, key_prefix: str) -> None:
    st.header(title)
    st.caption(caption)

    source = st.radio("Image source", ["Upload", "Camera"], horizontal=True, key=f"{key_prefix}_source")
    image = None
    if source == "Upload":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=f"{key_prefix}_upload")
        if uploaded is not None:
            image = Image.open(uploaded)
    else:
        snapshot = st.camera_input("Take a photo", key=f"{key_prefix}_camera")
        if snapshot is not None:
            image = Image.open(snapshot)

    if image is None:
        st.info("Upload or capture an image to run inference.")
        return

    col_img, col_pred = st.columns(2)
    col_img.image(image, caption="Input", use_container_width=True)

    try:
        predictions = predict_fn(image, top_k=5)
    except FileNotFoundError as exc:
        col_pred.error(f"Model not available: {exc}")
        return

    with col_pred:
        st.subheader("Predictions")
        for pred in predictions:
            st.write(f"**{pred.label}** — {pred.confidence * 100:.1f}%")
            st.progress(min(max(pred.confidence, 0.0), 1.0))


with tab_ocular:
    _render_classifier_tab(
        "Ocular — Eye Condition Classifier",
        "5 classes: Cataract, Conjunctivitis, Eyelid, Normal, Uveitis.",
        predict_eye_condition,
        "ocular",
    )

with tab_derm:
    _render_classifier_tab(
        "Dermatology — Skin Condition Classifier",
        "11 classes (10 curated skin conditions + normal), EfficientNetV2-S, ~61.6% test accuracy — see docs/model_cards.md.",
        predict_skin_condition,
        "derm",
    )

with tab_rppg:
    st.header("rPPG — Heart Rate & HRV (CHROM vs POS)")
    st.caption(
        "Contactless heart-rate estimation from a video feed via chrominance-based rPPG "
        "(CHROM & POS, adapted from rPPG-Toolbox). Buffers ~10s of face-ROI RGB before the "
        "first reading; keep your face steady and evenly lit. Works the same whether the "
        "feed is a USB webcam, a phone IP camera, or the RPi5's rpicam module."
    )
    st.warning(
        "Uses a simple blocking capture loop (no streamlit-webrtc dependency): while "
        "'Start camera' is checked, other tabs won't update until you uncheck it.",
        icon="⚠️",
    )

    if "rppg_buffer" not in st.session_state:
        st.session_state.rppg_buffer = RPPGBuffer()
    if "rppg_tracker" not in st.session_state:
        st.session_state.rppg_tracker = FaceROITracker()

    run = st.checkbox("Start camera", key="rppg_run")

    col_video, col_stats = st.columns([2, 1])
    frame_placeholder = col_video.empty()
    status_placeholder = col_stats.empty()
    bpm_placeholder = col_stats.empty()
    detail_placeholder = col_stats.empty()
    chart_placeholder = st.empty()

    if run:
        cam = create_camera_source()
        tracker: FaceROITracker = st.session_state.rppg_tracker
        buffer: RPPGBuffer = st.session_state.rppg_buffer
        last_result = None
        frame_count = 0

        if not cam.isOpened():
            status_placeholder.error("Could not open camera source.")
        else:
            try:
                while run:
                    ok, frame = cam.read()
                    if not ok:
                        status_placeholder.error("Lost connection to camera.")
                        break

                    roi_result = tracker.process_frame(frame)
                    overlay_rgb = cv2.cvtColor(roi_result.overlay_bgr, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(overlay_rgb, channels="RGB", use_container_width=True)

                    if roi_result.face_found:
                        buffer.add_sample(roi_result.mean_rgb)
                        status_placeholder.success("Face detected — buffering RGB trace")
                    else:
                        status_placeholder.warning("No face detected")

                    frame_count += 1
                    if buffer.is_ready() and frame_count % 15 == 0:
                        try:
                            last_result = buffer.analyze()
                        except ValueError:
                            pass

                    if last_result is not None:
                        bpm_placeholder.metric("Heart Rate (CHROM/POS avg)", f"{last_result.bpm:.0f} BPM")
                        sdnn = last_result.hrv_sdnn_ms
                        rmssd = last_result.hrv_rmssd_ms
                        detail_placeholder.markdown(
                            f"CHROM: **{last_result.bpm_chrom:.0f}** BPM &nbsp;·&nbsp; "
                            f"POS: **{last_result.bpm_pos:.0f}** BPM\n\n"
                            f"HRV — SDNN: **{sdnn:.1f} ms**, RMSSD: **{rmssd:.1f} ms**"
                            if not (np.isnan(sdnn) or np.isnan(rmssd))
                            else f"CHROM: **{last_result.bpm_chrom:.0f}** BPM &nbsp;·&nbsp; "
                            f"POS: **{last_result.bpm_pos:.0f}** BPM\n\nHRV: not enough clean peaks yet"
                        )
                        n = min(len(last_result.bvp_chrom), len(last_result.bvp_pos))
                        chart_df = pd.DataFrame(
                            {"CHROM": last_result.bvp_chrom[-n:], "POS": last_result.bvp_pos[-n:]}
                        )
                        chart_placeholder.line_chart(chart_df)
                    else:
                        bpm_placeholder.progress(
                            buffer.fill_fraction(), text=f"Buffering... {buffer.fill_fraction() * 100:.0f}%"
                        )
            finally:
                cam.release()
    else:
        st.info("Check 'Start camera' to begin heart-rate estimation.")
