#!/usr/bin/env bash
# Put the NEXUS models where each sub-project expects them. Nothing is stored in git.
#
# - MediaPipe Face Landmarker: public, downloaded from Google.
# - Trained ONNX weights (dermnet_model.onnx, eye_disease_model.onnx): NOT publicly
#   hosted -- available on request from the author (see DATA.md). Once you have them:
#
#     MODELS_DIR=/path/to/received/files scripts/download_models.sh   # local folder
#     MODELS_BASE_URL=https://host/path    scripts/download_models.sh   # private URL
#
# Every ONNX file is checked against its SHA-256 before it is installed.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-}"
MODELS_BASE_URL="${MODELS_BASE_URL:-}"
MEDIAPIPE_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$ROOT/.model_cache"
mkdir -p "$CACHE"

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

get() {  # get <name> [expected sha256] -- from MODELS_DIR, MODELS_BASE_URL, or (mediapipe) Google
  local name="$1" want="${2:-}" out="$CACHE/$1"
  if [[ ! -f "$out" ]]; then
    if [[ "$name" == face_landmarker.task ]]; then
      echo "Downloading $name ..."; curl -fL --retry 3 --progress-bar -o "$out.part" "$MEDIAPIPE_URL"
    elif [[ -n "$MODELS_DIR" ]]; then
      cp "$MODELS_DIR/$name" "$out.part"
    else
      echo "Downloading $name ..."; curl -fL --retry 3 --progress-bar -o "$out.part" "$MODELS_BASE_URL/$name"
    fi
    mv "$out.part" "$out"
  fi
  if [[ -n "$want" && "$(sha256 "$out")" != "$want" ]]; then
    echo "Checksum mismatch for $name -- deleting it." >&2; rm -f "$out"; exit 1
  fi
}
place() { mkdir -p "$(dirname "$2")"; cp "$CACHE/$1" "$2"; echo "  -> ${2#$ROOT/}"; }

get face_landmarker.task
echo "Placing MediaPipe model:"
for p in nexus_pfe nexus_pfe_raspberrypi; do place face_landmarker.task "$ROOT/$p/models/mediapipe/face_landmarker.task"; done

if [[ -z "$MODELS_DIR" && -z "$MODELS_BASE_URL" ]]; then
  cat <<'EOF'

Trained ONNX weights were NOT installed: they are not publicly hosted.
They are available on request (see DATA.md). Once you have the files, re-run with
  MODELS_DIR=/path/to/files scripts/download_models.sh
EOF
  exit 0
fi

get dermnet_model.onnx     1a7adc58e6977898343a31cadcab3c9eb0fab98973f165198d2f03afd305d00a
get eye_disease_model.onnx 7ee0ec22d3212036d80674770fbeec874bad1ae4d4e99aaa657bf62c3781840d
echo "Placing ONNX models:"
for p in nexus_pfe nexus_pfe_raspberrypi; do
  place dermnet_model.onnx     "$ROOT/$p/models/dermnet_model.onnx"
  place eye_disease_model.onnx "$ROOT/$p/models/eye_disease_model.onnx"
done
place dermnet_model.onnx     "$ROOT/nexus_pfe_mobile/assets/models/dermnet_model.onnx"
place eye_disease_model.onnx "$ROOT/nexus_pfe_mobile/assets/models/eye_disease_model_merged.onnx"
echo "Done. (Cached copies in .model_cache/ can be deleted.)"
