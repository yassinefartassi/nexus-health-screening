"""Streamlit entry point for the NEXUS_PFE multimodal health-screening test bench.

Four tabs: Ocular, Dermatology, rPPG (HRV/BPM), and Apprentissage federe (FL).
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from fl_client.status_client import DEFAULT_STATUS_URL, fetch_status
from src.inference.dermatology_infer import predict_skin_condition
from src.inference.ocular_infer import predict_eye_condition
from src.inference.rppg_infer import RPPGBuffer
from src.preprocessing.face_roi import FaceROITracker

st.set_page_config(page_title="NEXUS_PFE — Health Screening Test Bench", layout="wide")
st.title("NEXUS_PFE — Multimodal Health Screening Test Bench")
st.caption("Phase 1: desktop validation of the ocular, dermatology, and rPPG modules before Raspberry Pi 5 / Flutter deployment.")

tab_ocular, tab_derm, tab_rppg, tab_fl = st.tabs(
    ["Ocular", "Dermatology", "rPPG (HRV/BPM)", "Apprentissage fédéré"]
)


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
        "Contactless heart-rate estimation from webcam video via chrominance-based rPPG "
        "(CHROM & POS, adapted from rPPG-Toolbox). Buffers ~10s of face-ROI RGB before the "
        "first reading; keep your face steady and evenly lit."
    )
    st.warning(
        "Uses a simple blocking capture loop (no streamlit-webrtc dependency): while "
        "'Start webcam' is checked, other tabs won't update until you uncheck it.",
        icon="⚠️",
    )

    if "rppg_buffer" not in st.session_state:
        st.session_state.rppg_buffer = RPPGBuffer()
    if "rppg_tracker" not in st.session_state:
        st.session_state.rppg_tracker = FaceROITracker()

    run = st.checkbox("Start webcam", key="rppg_run")

    col_video, col_stats = st.columns([2, 1])
    frame_placeholder = col_video.empty()
    status_placeholder = col_stats.empty()
    bpm_placeholder = col_stats.empty()
    detail_placeholder = col_stats.empty()
    chart_placeholder = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        tracker: FaceROITracker = st.session_state.rppg_tracker
        buffer: RPPGBuffer = st.session_state.rppg_buffer
        last_result = None
        frame_count = 0

        if not cap.isOpened():
            status_placeholder.error("Could not open webcam (device 0).")
        else:
            try:
                while run:
                    ok, frame = cap.read()
                    if not ok:
                        status_placeholder.error("Lost connection to webcam.")
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
                cap.release()
    else:
        st.info("Check 'Start webcam' to begin heart-rate estimation.")

with tab_fl:
    st.header("Apprentissage fédéré (Federated Learning)")
    st.caption(
        "Comment le modèle global de chaque module (dermatologie, oculaire) s'améliore "
        "à partir de plusieurs appareils, sans jamais recevoir leurs images."
    )

    st.success(
        "🔒 **Vos images ne quittent jamais l'appareil.** Seuls les *poids* du modèle "
        "(des tableaux de nombres décrivant ce que le réseau a appris) sont envoyés au "
        "serveur d'agrégation — jamais une photo, jamais une donnée patient.",
        icon="🔒",
    )

    st.subheader("Le cycle FedAvg")
    cycle_cols = st.columns(6)
    cycle_steps = [
        ("📱", "Client", "Entraîne localement sur ses propres données (jamais partagées)"),
        ("➡️", "Poids", "Seuls les poids du modèle local sont envoyés"),
        ("🖥️", "Serveur", "Reçoit les poids de chaque client connecté"),
        ("🧮", "FedAvg", "Moyenne pondérée des poids reçus (par nombre d'échantillons)"),
        ("🌍", "Modèle global", "Nouvelle version versionnée : global_v{round}"),
        ("🔄", "Redistribution", "Le modèle global sert de point de départ au round suivant"),
    ]
    for col, (icon, label, desc) in zip(cycle_cols, cycle_steps):
        with col:
            st.markdown(f"<div style='text-align:center;font-size:2rem'>{icon}</div>", unsafe_allow_html=True)
            st.markdown(f"**{label}**")
            st.caption(desc)

    st.caption(
        "Architecture **modulaire** : dermatologie (EfficientNetV2-S, TensorFlow/Keras) et "
        "oculaire (ResNet-18, PyTorch) ont chacun leur propre modèle global versionné "
        "indépendamment — ce ne sont pas des poids partagés/convertis entre les deux. "
        "rPPG n'a pas de poids entraînables (CHROM/POS déterministe) et n'est donc jamais fédéré."
    )

    st.divider()
    st.subheader("État réel des rounds exécutés")
    st.caption(
        "Ces chiffres proviennent des logs réels du serveur FL "
        "(`nexus_pfe_server_fed/logs/*.csv` et `models/<module>/manifest.json`), lus via "
        "l'API de statut en lecture seule `fl_server/status_api.py` — jamais de valeurs inventées."
    )

    status_url = st.text_input(
        "Adresse de l'API de statut du serveur FL",
        value=DEFAULT_STATUS_URL,
        help="Démarrée séparément sur le serveur avec `scripts\\start_status_api.ps1` "
        "(nexus_pfe_server_fed). Par défaut http://127.0.0.1:8090 si le serveur tourne "
        "sur cette même machine.",
        key="fl_status_url",
    )
    st.button("🔄 Rafraîchir", key="fl_refresh")

    status = fetch_status(status_url) if status_url else None

    if status is None:
        st.warning(
            "⏳ **En attente d'exécution.** Impossible de joindre l'API de statut du serveur FL "
            f"à `{status_url}`. Démarrez le serveur (`fl_server.server`) et l'API de statut "
            "(`fl_server.status_api`) sur le serveur, ou vérifiez l'adresse ci-dessus. "
            "Aucune métrique n'est affichée tant qu'aucun round réel n'a eu lieu.",
            icon="⏳",
        )
    else:
        module_labels = {"dermatology": "Dermatologie", "ocular": "Oculaire"}
        module_cols = st.columns(2)
        for col, module_name in zip(module_cols, ["dermatology", "ocular"]):
            module_status = status.get(module_name)
            with col:
                st.markdown(f"#### {module_labels[module_name]}")
                if not module_status or not module_status.get("rounds"):
                    st.info("⏳ En attente d'exécution — aucun round FL n'a encore été effectué pour ce module.")
                    continue

                st.caption(f"Framework : {module_status.get('framework', '?')}")
                latest = module_status["latest_round"]
                m1, m2, m3 = st.columns(3)
                m1.metric("Round actuel", latest["round"])
                m2.metric("Clients agrégés", latest["num_clients"])
                acc = latest.get("accuracy")
                m3.metric("Précision (held-out)", f"{acc * 100:.1f}%" if acc is not None else "—")

                rounds_df = pd.DataFrame(module_status["rounds"]).set_index("round")
                st.line_chart(rounds_df[["accuracy"]], height=180)
                st.caption("Précision du modèle global sur le jeu de test réservé, par round (0 = baseline pré-entraînée).")
                with st.expander("Détail par round"):
                    st.dataframe(
                        rounds_df[["timestamp", "num_clients", "accuracy", "loss", "comm_bytes"]],
                        use_container_width=True,
                    )

    st.divider()
    st.subheader("Participer en tant que client (ligne de commande)")
    st.caption(
        "L'entraînement local est un processus Python long à s'exécuter — il tourne en ligne "
        "de commande, pas depuis cette interface Streamlit (qui recharge la page à chaque "
        "interaction). Cette appli fournit le client réutilisable `fl_client/` ; lancez-le "
        "séparément :"
    )
    st.code(
        "pip install -r requirements-fl.txt\n"
        "python -m fl_client.client --client-id 0 --server <ip-serveur>:8080 "
        "--model dermatology --num-clients 2",
        language="bash",
    )
    st.caption(
        "`--client-id` sélectionne quelle partition non-IID simulée (seed fixe, "
        "voir `fl_client/data_partition.py`) ce client entraîne localement — il n'y a pas de "
        "vraie base d'utilisateurs multi-appareils derrière cette appli, la partition est "
        "reproductible mais simulée. `--model ocular` fédère le module oculaire (PyTorch) à la "
        "place. Voir le README pour le guide complet de bout en bout."
    )
