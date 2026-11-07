# Data and models

**No datasets or model weights are stored in this repository.** The images
used to train and evaluate NEXUS are medical photos of real people, owned by
their original publishers. Download them from the sources below, under those
sources' terms. Do not re-upload or commit them. `data/` is git-ignored.

## Quick start

```bash
scripts/download_models.sh          # MediaPipe face landmarker; ONNX weights if MODELS_DIR/MODELS_BASE_URL is set (see below)
scripts/download_data.sh            # checks which datasets are present and explains how to get the rest
scripts/download_data.sh --prepare  # builds the curated train/test folders once the raw data is in place
```

Raw datasets are expected under `data/raw/` (override with `NEXUS_RAW_DATA_DIR`).

## Datasets

### 1. DermNet (dermatology module)

| | |
|---|---|
| Used for | Training and evaluating the 11-class skin-condition classifier and the dermatology federated-learning task |
| Original source | DermNet: <https://dermnetnz.org>. The images are © DermNet, and their [image licence](https://dermnetnz.org/image-licence) applies |
| Copy used | Kaggle mirror "DermNet" by Shubham Goel (23 classes, licence listed as CC BY-NC-ND 4.0): <https://www.kaggle.com/datasets/shubhamgoel27/dermnet> |
| Expected layout | `data/raw/dermnet/train/<class>/` and `data/raw/dermnet/test/<class>/`. Put the normal-skin images from the Lysa Apriani dataset (below) in `data/raw/dermnet/test/normal/` only: `prepare_datasets.py` reads `normal` from that one folder (not `train/normal/`) and splits it 80/20 (seed 42) into the curated train and test sets |
| `normal` class source | Kaggle "Skin Disease and Normal Skin Dataset" by Lysa Apriani: <https://www.kaggle.com/datasets/lysaapriani/skin-disease-and-normal-skin-dataset>. Licence: see the dataset page. Not redistributed here |
| Notes | `nexus_pfe_server_fed/scripts/prepare_datasets.py` keeps 10 DermNet disease classes and adds `normal` from the dataset above |

### 2. Eye diseases (ocular module)

| | |
|---|---|
| Used for | Training and evaluating the 5-class eye-condition classifier and the ocular federated-learning task |
| Source | Ahmed, M. (2024). *Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation*. Mendeley Data, V2. <https://doi.org/10.17632/n9zp473wfw.2> |
| Licence | See the dataset page |
| Expected layout | `data/raw/Image Dataset on Eye Diseases Classification (Uveitis, Conjunctivitis, Cataract, Eyelid) with Symptoms and SMOTE Validation/<Cataract\|Conjunctivitis\|Eyelid\|Normal\|Uveitis>/` |
| Notes | According to its authors, the images were collected from online sources, and the classes were balanced with SMOTE. Reported accuracy should be read with that in mind. |

### 3. UBFC-rPPG (rPPG evaluation only)

| | |
|---|---|
| Used for | `nexus_pfe/evaluation/eval_rppg_ubfc.py`: heart-rate accuracy of CHROM/POS against a pulse-oximeter reference. Not needed to run the apps |
| Source | Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., & Dubois, J. (2019). Unsupervised skin tissue segmentation for remote photoplethysmography. *Pattern Recognition Letters*, 124, 82–90. <https://doi.org/10.1016/j.patrec.2017.10.017> |
| Access | Shared for research purposes. See <https://sites.google.com/view/ybenezeth/ubfcrppg> |
| Expected layout | `data/UBFC-rPPG/DATASET_1/<subject>/vid.avi` + `gtdump.xmp` |

The results in `nexus_pfe/evaluation/results/` contain only derived metrics
(subject IDs, BPM errors). They contain no video frames or faces.

## Where the code reads data

| Component | Variable | Point it at |
|---|---|---|
| FL server evaluation | `DERMNET_TEST_DIR`, `OCULAR_TEST_DIR` (or `--test-dir`) | `nexus_pfe_server_fed/data/{dermatology,ocular}_curated/test` |
| FL clients (desktop / Raspberry Pi) | `DERMNET_DATA_DIR`, `OCULAR_DATA_DIR` | `nexus_pfe_server_fed/data/{dermatology,ocular}_curated/train` |
| Dataset preparation | `NEXUS_RAW_DATA_DIR` | `data/raw` (default) |

If these are unset, the FL code falls back to a small seeded synthetic dataset,
so the pipeline can be demonstrated without any real data.

## Models

| File | What it is | How you get it |
|---|---|---|
| `dermnet_model.onnx` | EfficientNetV2-S, 11 classes, trained on DermNet (see above) | Available on request (see below) |
| `eye_disease_model.onnx` (`eye_disease_model_merged.onnx` on mobile) | ResNet-18, 5 classes, trained on the eye-disease dataset | Available on request (see below) |
| `face_landmarker.task` | Google MediaPipe Face Landmarker (Apache-2.0) | `scripts/download_models.sh`, or downloaded automatically on first use |
| `haarcascade_frontalface_default.xml` | OpenCV Haar cascade (OpenCV licence) | Included in `nexus_pfe_raspberrypi/models/haarcascades/` |

The trained model weights are **not publicly distributed** (the training data's
licences restrict redistribution). They are **available on request** for research
or evaluation: open an issue or contact the author via the GitHub profile.
With the files in hand, run `MODELS_DIR=/path/to/folder scripts/download_models.sh`
(or `MODELS_BASE_URL=<url>`) to verify their SHA-256 and copy them into place.

The models were trained on the datasets above, so their use is also subject to
those datasets' terms. They are research prototypes, **not medical devices**.

## Third-party code

The rPPG CHROM/POS implementation is adapted from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (Responsible AI
License, see its [LICENSE](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/LICENSE)).
It is not vendored here. Clone it separately if you want its reference
implementations or pretrained weights.
