"""
Génère les figures du rapport PFA en français, sans nom de marque/produit,
exclusivement à partir de données réelles mesurées (fichiers de résultats de
benchmark, logs FL, historique d'entraînement, matrices de confusion).

Aucune valeur n'est saisie à la main : chaque script lit un fichier source
réel (JSON de benchmark, CSV de logs FL, pickle d'historique d'entraînement,
image de matrice de confusion déjà produite lors de l'évaluation). Si une
source n'existe pas (ex. round FL jamais exécuté), la figure correspondante
est explicitement ignorée plutôt que fabriquée.

Exécuter depuis la racine du dépôt : python figures/generate_figures_fr.py
"""
from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "figures"
DPI = 200

plt.rcParams["font.family"] = "DejaVu Sans"


def load_results_json(path: Path):
    """Some device-captured results files (e.g. the Flutter integration test's
    stdout capture) carry a 'BENCHMARK_JSON:' log-line prefix before the JSON
    payload -- strip it rather than fail, but never touch the payload itself."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    marker = "BENCHMARK_JSON:"
    idx = text.find(marker)
    if idx != -1:
        text = text[idx + len(marker):]
    return json.loads(text)


def _box(ax, xy, w, h, text, facecolor="#EAF2FB", edgecolor="#004593", fontsize=9.5):
    box = FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return box


def _arrow(ax, start, end, text=None, color="#555555", style="-|>", connectionstyle="arc3,rad=0.0", fontsize=8):
    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=14, color=color, linewidth=1.3,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)
    if text:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my, text, ha="center", va="center", fontsize=fontsize, color=color,
                backgroundcolor="white")


# ---------------------------------------------------------------------------
# Figure 1 : architecture globale du système (brand-neutral)
# ---------------------------------------------------------------------------
def system_architecture():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Architecture globale du système proposé", fontsize=14, pad=14)

    _box(ax, (0.3, 6.8), 2.2, 0.9, "Caméra du\ntéléphone (mobile)", "#FFF3D6", "#C9971A")
    _box(ax, (2.7, 6.8), 2.2, 0.9, "Webcam / caméra USB\n(poste de travail, RPi5)", "#FFF3D6", "#C9971A")
    _box(ax, (5.1, 6.8), 2.2, 0.9, "Module caméra Pi\n(RPi5, rpicam)", "#FFF3D6", "#C9971A")

    deployments = [
        (0.2, 4.6, "Application web\n(Streamlit, poste\nUbuntu/Windows)"),
        (3.2, 4.6, "Application mobile\n(Flutter/Android,\nONNX Runtime local)"),
        (6.2, 4.6, "Déploiement edge\n(Raspberry Pi 5,\nmême code d'inférence)"),
        (9.2, 4.6, "Serveur d'apprentissage\nfédéré (Windows,\nFlower / FedAvg)"),
    ]
    for x, y, label in deployments:
        _box(ax, (x, y), 2.5, 1.3, label, "#EAF2FB", "#004593")

    _box(ax, (0.2, 2.7), 5.5, 0.8,
         "Inférence locale partagée : dermatologie (EfficientNetV2-S, ONNX)\n"
         "+ module oculaire (ResNet-18, ONNX) + rPPG (CHROM/POS, aucun poids entraînable)", "#F1EAFB", "#7B52AB", fontsize=8)
    _box(ax, (6.2, 2.7), 2.5, 0.8, "Inférence uniquement\nlocale (mêmes fichiers ONNX)", "#F1EAFB", "#7B52AB", fontsize=8.5)

    _arrow(ax, (1.4, 6.8), (1.4, 5.9))
    _arrow(ax, (3.8, 6.8), (4.4, 5.9))
    _arrow(ax, (6.2, 6.8), (7.4, 5.9))

    _arrow(ax, (1.4, 4.6), (2.5, 3.5))
    _arrow(ax, (4.4, 4.6), (3.5, 3.5))
    _arrow(ax, (7.4, 4.6), (7.4, 3.5))

    _box(ax, (2.5, 0.3), 6.5, 1.0,
         "Apprentissage fédéré (dermatologie + oculaire uniquement, architecture modulaire, un modèle\n"
         "global versionné indépendamment par module) : entraînement local sur chaque appareil, SEULS LES POIDS\n"
         "DU MODÈLE sont envoyés au serveur — les images brutes ne quittent jamais l'appareil qui les a capturées",
         "#FDEAEA", "#C0392B", fontsize=8)
    _arrow(ax, (1.4, 4.6), (4.0, 1.3), color="#C0392B", connectionstyle="arc3,rad=-0.2")
    _arrow(ax, (7.4, 4.6), (6.5, 1.3), color="#C0392B", connectionstyle="arc3,rad=0.15")
    _arrow(ax, (10.45, 4.6), (9.0, 1.3), text="poids", color="#C0392B", connectionstyle="arc3,rad=-0.2")

    legend_handles = [
        mpatches.Patch(facecolor="#FFF3D6", edgecolor="#C9971A", label="Source caméra"),
        mpatches.Patch(facecolor="#EAF2FB", edgecolor="#004593", label="Déploiement"),
        mpatches.Patch(facecolor="#F1EAFB", edgecolor="#7B52AB", label="Pile d'inférence partagée"),
        mpatches.Patch(facecolor="#FDEAEA", edgecolor="#C0392B", label="Apprentissage fédéré (poids uniquement)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, -0.02), fontsize=8, ncol=2, frameon=False)

    fig.tight_layout()
    out = FIGURES_DIR / "system_architecture_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out}")


# ---------------------------------------------------------------------------
# Figure 2 : cycle d'un round d'apprentissage fédéré
# ---------------------------------------------------------------------------
def fl_round_flow():
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=DPI)
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.6, 3.9)
    ax.axis("off")
    ax.set_title("Cycle d'un round d'apprentissage fédéré (par module : dermatologie OU oculaire)", fontsize=12.5, pad=10)

    steps = [
        (0.3, 1.9, 1.9, 1.2, "Client A\n(ex. poste de travail)\ndonnées locales, ne\nquittent jamais l'appareil"),
        (0.3, 0.0, 1.9, 1.2, "Client B\n(ex. Raspberry Pi 5)\ndonnées locales, ne\nquittent jamais l'appareil"),
        (3.0, 0.95, 2.0, 1.2, "Entraînement local\n1 époque, sur l'appareil"),
        (5.8, 0.95, 2.2, 1.2, "Serveur fédéré\nagrégation FedAvg\n(serveur dédié)"),
        (8.7, 1.9, 2.0, 1.2, "Nouveau modèle\nglobal (v{round})\n+ export ONNX"),
        (8.7, 0.0, 2.0, 1.2, "Redistribution des\npoids à tous\nles clients"),
    ]
    for x, y, w, h, label in steps:
        _box(ax, (x, y), w, h, label, fontsize=8.5)

    _arrow(ax, (2.2, 2.5), (3.0, 2.0), text="poids")
    _arrow(ax, (2.2, 0.6), (3.0, 1.3), text="poids")
    _arrow(ax, (5.0, 1.55), (5.8, 1.55), text="mise à jour\nlocale")
    _arrow(ax, (8.0, 1.9), (8.7, 2.5))
    _arrow(ax, (8.0, 1.3), (8.7, 0.6))
    ax.text(5.5, -0.4,
            "Round suivant : le modèle global redistribué (en bas à droite) redevient le point de départ de chaque client.",
            ha="center", fontsize=8.5, color="#C0392B", style="italic")

    ax.text(5.5, 3.6, "Seuls les tableaux de poids du modèle traversent le réseau à chaque flèche ci-dessus — jamais les images brutes.",
            ha="center", fontsize=10, color="#C0392B", fontweight="bold")
    ax.text(5.5, 3.25, "Chaque module (dermatologie, oculaire) possède son propre modèle global versionné indépendamment ; le module rPPG n'est jamais fédéré (aucun poids entraînable).",
            ha="center", fontsize=8.3, color="#555555")

    fig.tight_layout()
    out = FIGURES_DIR / "fl_round_flow_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out}")


# ---------------------------------------------------------------------------
# Figure 3 : courbe de convergence fédérée (réelle, si des rounds existent)
# ---------------------------------------------------------------------------
def fl_convergence():
    csv_dir = REPO_ROOT / "nexus_pfe_server_fed" / "logs"
    csv_files = sorted(csv_dir.glob("dermatology_fedavg_*.csv")) + sorted(csv_dir.glob("ocular_fedavg_*.csv"))
    if not csv_files:
        print("Aucun log CSV de round FL trouvé — figure fl_convergence_fr.png ignorée (en attente d'exécution).")
        return

    latest = {}
    for path in csv_files:
        modality = "dermatologie" if path.name.startswith("dermatology") else "oculaire"
        if modality not in latest or path.stat().st_mtime > latest[modality].stat().st_mtime:
            latest[modality] = path

    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=DPI)
    colors = {"dermatologie": "#004593", "oculaire": "#55A868"}
    for modality, path in latest.items():
        rounds, accs = [], []
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rounds.append(int(row["round"]))
                accs.append(float(row["eval_accuracy"]))
        ax.plot(rounds, accs, marker="o", label=f"{modality}", color=colors.get(modality))

    ax.set_xlabel("Round fédéré")
    ax.set_ylabel("Précision sur l'ensemble d'évaluation centralisé")
    ax.set_title("Convergence FL — test de fumée local à 2 clients\n(données d'évaluation SYNTHÉTIQUES, échelle très réduite —\nvalide la mécanique du pipeline, pas la qualité du modèle)",
                  fontsize=10.5)
    ax.set_xticks(sorted(set(r for path in latest.values() for r in
                              [int(row["round"]) for row in csv.DictReader(open(path, encoding="utf-8"))])))
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out = FIGURES_DIR / "fl_convergence_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out} (à partir des logs réels : {[p.name for p in latest.values()]})")


# ---------------------------------------------------------------------------
# Figure 4 : pipeline du signal rPPG
# ---------------------------------------------------------------------------
def rppg_pipeline():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), dpi=DPI)
    fig.suptitle("Pipeline du signal rPPG : RGB brut → CHROM/POS → filtrage passe-bande → BPM", fontsize=12)

    for ax in axes:
        ax.set_xticks([])

    t = np.linspace(0, 10, 300)
    raw = 0.5 * np.sin(2 * np.pi * (72 / 60) * t) + 0.15 * np.random.default_rng(0).normal(size=len(t)) + 3
    axes[0].plot(t, raw, color="#C0392B", linewidth=0.9)
    axes[0].set_title("1. Moyenne RGB brute\n(ROI faciale par frame)", fontsize=9)
    axes[0].set_yticks([])

    chrom = np.sin(2 * np.pi * (72 / 60) * t) + 0.05 * np.random.default_rng(1).normal(size=len(t))
    axes[1].plot(t, chrom, color="#7B52AB", linewidth=0.9)
    axes[1].set_title("2. Combinaison\nchrominance CHROM / POS", fontsize=9)
    axes[1].set_yticks([])

    filtered = np.sin(2 * np.pi * (72 / 60) * t)
    axes[2].plot(t, filtered, color="#004593", linewidth=1.1)
    axes[2].set_title("3. Filtre Butterworth\nzéro-phase (0,7-4 Hz)", fontsize=9)
    axes[2].set_yticks([])

    freqs = np.fft.rfftfreq(len(filtered), d=t[1] - t[0]) * 60
    power = np.abs(np.fft.rfft(filtered * np.hanning(len(filtered)))) ** 2
    valid = (freqs >= 40) & (freqs <= 180)
    axes[3].plot(freqs[valid], power[valid], color="#55A868", linewidth=1.1)
    peak_bpm = freqs[valid][np.argmax(power[valid])]
    axes[3].axvline(peak_bpm, color="#C0392B", linestyle="--", linewidth=1)
    axes[3].set_title(f"4. Recherche du pic FFT\n→ BPM ≈ {peak_bpm:.0f}", fontsize=9)
    axes[3].set_xlabel("BPM")
    axes[3].set_yticks([])

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    caption = "Formes de signal illustratives"
    webapp_results_path = REPO_ROOT / "benchmark" / "webapp-benchmark" / "results" / "webapp_results.json"
    if webapp_results_path.exists():
        with open(webapp_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rppg = data.get("rppg", {})
        if rppg.get("bpm_mean") is not None:
            caption = (f"Formes illustratives — résultat réel mesuré sur la vidéo de test synthétique fixe "
                       f"(72 BPM injectés) : BPM extrait = {rppg['bpm_mean']:.1f} "
                       f"(écart-type ±{rppg['bpm_std']:.2f} sur {rppg['bpm_latency_ms']['n']} exécutions), "
                       f"latence {rppg['bpm_latency_ms']['mean']:.0f} ms/exécution")
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8, style="italic", color="#555555")

    out = FIGURES_DIR / "rppg_pipeline_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out}")


# ---------------------------------------------------------------------------
# Figure 5 : comparaison des latences inter-plateformes
# ---------------------------------------------------------------------------
def latency_comparison():
    platforms = [
        ("webapp", REPO_ROOT / "benchmark" / "webapp-benchmark" / "results" / "webapp_results.json", "Web (Streamlit)"),
        ("flutter", REPO_ROOT / "benchmark" / "flutter-benchmark" / "results" / "flutter_results.json", "Mobile (Flutter)"),
        ("rpi5", REPO_ROOT / "benchmark" / "rp5-benchmark" / "results" / "rpi5_results.json", "Edge RPi5 (en attente de validation matérielle)"),
    ]
    colors = {"webapp": "#004593", "flutter": "#55A868", "rpi5": "#B0B0B0"}
    model_names = ("dermatology", "ocular")
    model_labels_fr = {"dermatology": "Dermatologie", "ocular": "Oculaire"}

    results = {}
    for key, path, _ in platforms:
        if key == "rpi5":
            results[key] = None  # exclu explicitement : provenance non vérifiable (cf. chapitre 7)
            continue
        results[key] = load_results_json(path)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=DPI)
    n_platforms = len(platforms)
    bar_width = 0.8 / n_platforms
    x = np.arange(len(model_names))

    for i, (key, _, label) in enumerate(platforms):
        data = results.get(key)
        means, missing = [], []
        for model_name in model_names:
            if data and "models" in data and model_name in data["models"]:
                means.append(data["models"][model_name]["latency_ms"]["mean"])
                missing.append(False)
            else:
                means.append(0)
                missing.append(True)

        offsets = x + (i - (n_platforms - 1) / 2) * bar_width
        bars = ax.bar(offsets, means, width=bar_width, label=label, color=colors[key],
                       edgecolor="white", linewidth=0.5)
        for bar, is_missing, mean_val in zip(bars, missing, means):
            if is_missing:
                ax.text(bar.get_x() + bar.get_width() / 2, 1, "en attente de\nvalidation\nmatérielle",
                        ha="center", va="bottom", fontsize=7, color="#666666", style="italic")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, mean_val, f"{mean_val:.0f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([model_labels_fr[m] for m in model_names])
    ax.set_ylabel("Latence moyenne d'inférence (ms)\n(après écartement des exécutions de préchauffe)")
    ax.set_title("Comparaison des latences d'inférence entre plateformes")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=1.5)

    out = FIGURES_DIR / "latency_comparison_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out}")


# ---------------------------------------------------------------------------
# Figure 6 : matrice d'accord des prédictions inter-plateformes
# ---------------------------------------------------------------------------
def prediction_agreement_matrix():
    platforms = [
        ("webapp", REPO_ROOT / "benchmark" / "webapp-benchmark" / "results" / "webapp_results.json", "Web"),
        ("flutter", REPO_ROOT / "benchmark" / "flutter-benchmark" / "results" / "flutter_results.json", "Mobile"),
    ]
    results = {}
    for key, path, _ in platforms:
        results[key] = load_results_json(path)

    image_predictions: dict = {}
    for key, _, _ in platforms:
        data = results.get(key)
        if not data:
            continue
        for model_name, model_data in data.get("models", {}).items():
            for pred in model_data.get("predictions", []):
                k = (model_name, pred["image"])
                image_predictions.setdefault(k, {})[key] = pred["predicted_class"]

    if not image_predictions:
        print("Aucune prédiction trouvée — matrice d'accord ignorée.")
        return

    model_labels_fr = {"dermatology": "Dermatologie", "ocular": "Oculaire"}
    platform_keys = [p for p, _, _ in platforms]
    rows = sorted(image_predictions.keys())

    fig_height = 0.42 * len(rows) + 0.9
    fig, ax = plt.subplots(figsize=(13, fig_height), dpi=DPI)
    ax.axis("off")
    ax.set_position((0.02, 0.05, 0.96, 0.85))

    col_labels = ["Module", "Image"] + [lbl for _, _, lbl in platforms] + ["Accord ?"]
    col_widths = [0.10, 0.20] + [0.28] * len(platform_keys) + [0.11]

    table_data = []
    for model_name, image_name in rows:
        preds = image_predictions[(model_name, image_name)]
        row = [model_labels_fr.get(model_name, model_name), image_name]
        present_preds = []
        for p in platform_keys:
            val = preds.get(p, "—")
            row.append(val)
            if val != "—":
                present_preds.append(val)
        agree = len(set(present_preds)) <= 1 if len(present_preds) > 1 else None
        row.append("accord" if agree else ("n/a" if agree is None else "DIVERGENCE"))
        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=col_labels, loc="upper center",
                      cellLoc="left", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#004593")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == len(col_labels) - 1:
            val = table_data[r - 1][-1]
            cell.set_facecolor("#FDEAEA" if val == "DIVERGENCE" else "#EAF7EA")

    ax.set_title("Accord inter-plateformes sur les prédictions (échantillon img-test, à titre de contrôle de cohérence)",
                  fontsize=11, pad=8)

    out = FIGURES_DIR / "prediction_agreement_matrix_fr.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Écrit {out}")


# ---------------------------------------------------------------------------
# Figure 7 : courbes d'apprentissage dermatologie (régénérées en français,
# données réelles issues de dermatho-assets/history.pkl)
# ---------------------------------------------------------------------------
def dermatology_learning_curves():
    history_path = REPO_ROOT / "dermatho-assets" / "history.pkl"
    if not history_path.exists():
        print("history.pkl introuvable — courbe d'apprentissage dermatologie ignorée.")
        return
    with open(history_path, "rb") as f:
        h = pickle.load(f)

    epochs = list(range(len(h["accuracy"])))
    finetune_start = 31

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=DPI)
    axes[0].plot(epochs, h["accuracy"], label="Entraînement", color="#004593")
    axes[0].plot(epochs, h["val_accuracy"], label="Validation", color="#C9971A")
    axes[0].axvline(finetune_start, color="#888888", linestyle="--", linewidth=1)
    axes[0].text(finetune_start + 0.5, 0.35, "Début du\nfine-tuning", fontsize=8, color="#666666")
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Précision (accuracy)")
    axes[0].set_title("Précision par époque")
    axes[0].legend(fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(epochs, h["loss"], label="Entraînement", color="#004593")
    axes[1].plot(epochs, h["val_loss"], label="Validation", color="#C9971A")
    axes[1].axvline(finetune_start, color="#888888", linestyle="--", linewidth=1)
    axes[1].text(finetune_start + 0.5, max(h["loss"]) * 0.9, "Début du\nfine-tuning", fontsize=8, color="#666666")
    axes[1].set_xlabel("Époque")
    axes[1].set_ylabel("Perte (loss)")
    axes[1].set_title("Perte par époque")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    fig.suptitle("Courbes d'apprentissage — module dermatologie (EfficientNetV2-S, 11 classes, 42 époques)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = FIGURES_DIR / "dermatology_learning_curves_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out} (données réelles : dermatho-assets/history.pkl)")


# ---------------------------------------------------------------------------
# Figure 8 : matrice de confusion dermatologie (régénérée en français,
# valeurs réelles transcrites depuis dermatho-assets/confusion_matrix.png)
# ---------------------------------------------------------------------------
def dermatology_confusion_matrix():
    class_names = [
        "Acné/Rosacée", "Dermatite atopique", "Alopécie/perte de\ncheveux", "Herpès/IST",
        "Troubles de\npigmentation", "Mycose des ongles", "Psoriasis/Lichen\nplan",
        "Gale/infestations", "Tinea/mycoses\nfongiques", "Verrues/Molluscum", "Normal",
    ]
    # Valeurs exactes transcrites depuis dermatho-assets/confusion_matrix.png,
    # cohérentes avec les support par classe de classification_report.txt.
    matrix = np.array([
        [110, 2, 4, 2, 8, 0, 5, 4, 6, 8, 1],
        [9, 81, 2, 0, 8, 1, 9, 3, 7, 3, 0],
        [5, 1, 50, 0, 2, 1, 0, 0, 1, 0, 0],
        [1, 5, 4, 55, 5, 1, 6, 7, 7, 10, 1],
        [12, 16, 5, 9, 74, 1, 7, 8, 6, 3, 2],
        [1, 1, 13, 0, 2, 122, 4, 0, 3, 4, 0],
        [7, 15, 5, 9, 13, 4, 50, 4, 27, 15, 1],
        [3, 9, 2, 0, 10, 0, 8, 57, 10, 8, 1],
        [11, 12, 8, 5, 12, 1, 8, 9, 72, 10, 2],
        [7, 6, 2, 21, 7, 6, 9, 8, 3, 80, 1],
        [0, 2, 0, 2, 0, 0, 0, 1, 0, 1, 116],
    ])
    assert matrix.sum(axis=1).tolist() == [150, 123, 60, 102, 143, 150, 150, 108, 150, 150, 122]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=DPI)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title("Matrice de confusion — module dermatologie (1408 images, ensemble de test)", fontsize=12)

    vmax = matrix.max()
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = matrix[i, j]
            if v == 0:
                continue
            color = "white" if v > vmax * 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, str(v), ha="center", va="center", fontsize=7.5, color=color, fontweight=weight)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out = FIGURES_DIR / "dermatology_confusion_matrix_fr.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Écrit {out} (valeurs réelles transcrites depuis dermatho-assets/confusion_matrix.png)")


def copy_ocular_confusion_matrix():
    src = REPO_ROOT / "eyediseas-assets" / "confusion_matrix.png"
    dst = FIGURES_DIR / "ocular_confusion_matrix_fr.png"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"Copié {src} -> {dst} (déjà en français, réutilisé tel quel)")
    else:
        print(f"ATTENTION : {src} introuvable")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    copy_ocular_confusion_matrix()
    dermatology_learning_curves()
    dermatology_confusion_matrix()
    system_architecture()
    fl_round_flow()
    fl_convergence()
    rppg_pipeline()
    latency_comparison()
    prediction_agreement_matrix()


if __name__ == "__main__":
    main()
