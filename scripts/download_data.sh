#!/usr/bin/env bash
# Set up the datasets NEXUS needs. Nothing is re-hosted: each dataset comes from
# its original publisher, under that publisher's terms. See DATA.md.
#
#   scripts/download_data.sh            # check what's present, explain what's missing
#   scripts/download_data.sh --prepare  # ...then build data/*_curated with prepare_datasets.py
#
# Optional: DERMNET_KAGGLE_SLUG=<owner>/<dataset> lets this script fetch DermNet
# with the Kaggle CLI (needs ~/.kaggle/kaggle.json -- never commit that file).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="${NEXUS_RAW_DATA_DIR:-$ROOT/data/raw}"
OCULAR_DIR="Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation"
mkdir -p "$RAW"
missing=0

# 1. DermNet (dermatology) ----------------------------------------------------
if [[ -d "$RAW/dermnet/train" && -d "$RAW/dermnet/test" ]]; then
  echo "[ok] DermNet found at $RAW/dermnet"
elif [[ -n "${DERMNET_KAGGLE_SLUG:-}" ]] && command -v kaggle >/dev/null; then
  echo "Downloading DermNet from Kaggle ($DERMNET_KAGGLE_SLUG) ..."
  kaggle datasets download -d "$DERMNET_KAGGLE_SLUG" -p "$RAW/dermnet" --unzip
else
  missing=1
  cat <<EOF
[missing] DermNet -> expected at $RAW/dermnet/{train,test}/<class>/
          Download it from Kaggle (see DATA.md for the link and citation), or set
          DERMNET_KAGGLE_SLUG and install the Kaggle CLI to let this script do it.
EOF
fi

# 2. Eye diseases (Mendeley Data) --------------------------------------------
if [[ -d "$RAW/$OCULAR_DIR" ]]; then
  echo "[ok] Ocular dataset found"
else
  missing=1
  cat <<EOF
[missing] Ocular dataset -> download the ZIP from
          https://data.mendeley.com/datasets/n9zp473wfw/2
          and extract it so this folder exists:
          $RAW/$OCULAR_DIR/<Cataract|Conjunctivitis|Eyelid|Normal|Uveitis>/
EOF
fi

# 3. UBFC-rPPG (only for nexus_pfe/evaluation/eval_rppg_ubfc.py) ---------------
if [[ -d "$ROOT/data/UBFC-rPPG/DATASET_1" ]]; then
  echo "[ok] UBFC-rPPG found"
else
  cat <<EOF
[optional] UBFC-rPPG -> only needed for the rPPG accuracy evaluation.
          Request it from https://sites.google.com/view/ybenezeth/ubfcrppg
          and extract to $ROOT/data/UBFC-rPPG/DATASET_1/<subject>/
EOF
fi

if [[ "${1:-}" == "--prepare" ]]; then
  if [[ $missing -ne 0 ]]; then echo "Cannot prepare: required datasets are missing." >&2; exit 1; fi
  ( cd "$ROOT/nexus_pfe_server_fed" && NEXUS_RAW_DATA_DIR="$RAW" python scripts/prepare_datasets.py )
  echo "Curated datasets written to nexus_pfe_server_fed/data/. Point DERMNET_TEST_DIR /"
  echo "OCULAR_TEST_DIR (server) and DERMNET_DATA_DIR / OCULAR_DATA_DIR (clients) at them."
fi
exit $missing
