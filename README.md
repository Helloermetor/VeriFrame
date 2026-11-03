# VeriFrame Deepfake Verification Platform

VeriFrame is a web-based deepfake video verification platform integrating four detection models into a unified analysis pipeline.

**GitHub:** https://github.com/Helloermetor/VeriFrame

## 1. Model architecture

| Model | Role | Architecture |
|---|---|---|
| GenD | Primary detector | CLIP ViT-L/14 |
| Custom Xception | Control detector | Custom-trained DeepfakeBench Xception |
| Effort | Challenger 1 | CLIP ViT-L/14 with SVD residual adaptation |
| FakeSTormer | Challenger 2 | C23 ViT-Base with spatio-temporal localization |

GenD and Effort share the same CLIP ViT-L/14 backbone.

The backend runs all four model adapters through `backend/app/services/inference.py`.

**Important:** Raw detector scores are uncalibrated and must not be interpreted as probabilities of authenticity. The analysis pipeline separately applies frozen Platt calibration to GenD and custom Xception for the primary verdict. Effort scores and FakeSTormer outputs remain uncalibrated. The system can return an INCONCLUSIVE verdict.

## 2. Technology stack

- Frontend: React, TypeScript, Vite
- Backend: Python, FastAPI
- Deep learning: PyTorch
- Model libraries: Transformers, MMCV, TIMM
- Video processing: OpenCV
- GPU: NVIDIA CUDA-compatible GPU recommended

## 3. Requirements

Tested setup:

- Windows 10/11
- Node.js 20 or newer
- Python 3.13 for API, Xception, GenD and Effort
- Python 3.10 for FakeSTormer
- NVIDIA GPU with compatible CUDA drivers recommended
- Git and Git LFS

Do not copy virtual environments from another computer. Create them locally.

## 4. Clone the repository

```powershell
git clone https://github.com/Helloermetor/VeriFrame.git
cd VeriFrame
```
## 5. Backend environment setup

VeriFrame uses three separate Python environments because the models have different dependency requirements.

| Environment | Purpose | Python |
|---|---|---|
| `.venv` | FastAPI backend and Xception | 3.13 |
| `.venv-gend` | GenD and Effort | 3.13 |
| `.venv-fakestormer` | FakeSTormer | 3.10 |

Do not copy virtual environments when transferring the project. Create them on the target machine and install dependencies using the corresponding requirements files.

Run backend setup commands from the `backend` directory.

### 5.1 API and Xception

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-xception.txt

Install a compatible CUDA-enabled PyTorch and Torchvision build separately if GPU inference is required.

### 5.2 GenD and Effort

    py -3.13 -m venv .venv-gend
    .\.venv-gend\Scripts\python.exe -m pip install --upgrade pip
    .\.venv-gend\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
    .\.venv-gend\Scripts\python.exe -m pip install -r requirements-gend.txt
    .\.venv-gend\Scripts\python.exe -m pip install loralib==0.1.2 scikit-learn==1.9.1 tensorboard==2.21.0 efficientnet-pytorch==0.7.1

### 5.3 FakeSTormer

FakeSTormer requires Python 3.10.

    py -3.10 -m venv .venv-fakestormer
    .\.venv-fakestormer\Scripts\python.exe -m pip install --upgrade pip
    .\.venv-fakestormer\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
    .\.venv-fakestormer\Scripts\python.exe -m pip install -r requirements-fakestormer.txt

Use compatible CUDA-enabled PyTorch builds for the target machine.
## 6. Model checkpoint setup

Model checkpoints are not all included in the Git repository. Download or provide the required checkpoints before running inference.

### 6.1 GenD and shared CLIP backbone

GenD uses the following Hugging Face models:

- GenD: `yermandy/GenD_CLIP_L_14`
- Backbone: `openai/clip-vit-large-patch14`

The CLIP backbone is shared between GenD and Effort to avoid maintaining duplicate copies.

Run the unified downloader from the `backend` directory:

    .\.venv-gend\Scripts\python.exe .\download_veriframe_models.py

The script downloads or verifies the GenD checkpoint, shared CLIP backbone, and Effort checkpoint using SHA256 checks.

### 6.2 Effort checkpoint

The Effort checkpoint is available from the official Google Drive source:

https://drive.google.com/file/d/1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI/view?usp=drive_link

Expected filename:

`effort_clip_L14_trainOn_FaceForensic.pth`

The unified downloader handles this checkpoint automatically.

### 6.3 FakeSTormer checkpoint

Download the pretrained checkpoint from the official FakeSTormer Dropbox:

https://www.dropbox.com/scl/fo/elk2szqf0du4l6zm5job9/AAdVmNH--6ywHBZGNQJlR5o?rlkey=j8xesf2fu4ahxdw99w5ndrkb2&st=fe6drzpx&dl=0

Place it in:

`backend/model_assets/FakeSTormer/`

Expected filename:

`TopDownDetector_C23_ViTBase224_ST_hm100_tempLOC0.2_4SBI_SAM_mp0.01_temp2_0vidAug_0.35_temp_normlms_normREAL_model_best.pth`

### 6.4 Custom Xception checkpoint

Place the custom Xception checkpoint at:

`model_assets/Xception/xception_custom_best.pth`

Expected SHA256:

`217EF1B394352BAE9F63A34BC88D167A3B78FC36943AF88BBE719503AD3ECFE1`

This is a custom-trained checkpoint. The unified downloader does not download it.
## 7. Environment configuration

The repository includes a `.env.example` file.

From the repository root, create your local environment file:

    Copy-Item .env.example .env

Default configuration:

    VERIFRAME_UPLOAD_DIR=./runtime/uploads
    VERIFRAME_MAX_UPLOAD_BYTES=262144000
    VERIFRAME_CORS_ORIGINS=http://localhost:5173

Update these values according to your local environment when necessary.

## 8. Running VeriFrame

Open two PowerShell terminals.

### Terminal 1: Backend

    cd backend
    .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000 --env-file ..\.env

Backend API:

http://localhost:8000

API documentation:

http://localhost:8000/docs

### Terminal 2: Frontend

    cd frontend
    npm ci
    npm run dev

Frontend:

http://localhost:5173

For a production frontend build, run:

    npm run build
## 9. Repository structure

```text
VeriFrame/
|
+-- backend/
|   +-- app/
|   |   +-- api/
|   |   +-- services/
|   |   +-- ...
|   |
|   +-- model_assets/
|   +-- GenD/
|   +-- Effort/
|   +-- FakeSTormer/
|   +-- model_sources/
|   |   +-- gend/
|   |   +-- effort/
|   |   +-- fakestormer/
|   |
|   +-- requirements.txt
|   +-- requirements-gend.txt
|   +-- requirements-xception.txt
|   +-- requirements-fakestormer.txt
|   +-- download_veriframe_models.py
|   +-- ...
|
+-- frontend/
|   +-- src/
|   +-- package.json
|   +-- package-lock.json
|   +-- ...
|
+-- model_assets/
|   +-- Xception/
|       +-- xception_custom_best.pth
|
+-- integration_tests/
+-- validation/
+-- .env.example
+-- .gitattributes
+-- .gitignore
+-- README.md
```
## 10. Model architecture

VeriFrame uses a four-model deepfake detection committee.

| Model | Role |
|---|---|
| GenD CLIP ViT-L/14 | Primary detector |
| Custom Xception | Control model |
| Effort CLIP ViT-L/14 + SVD | Supporting detector |
| FakeSTormer C23 ViT-Base | Challenger model |

The backend executes the four model adapters through the committee inference service.

FakeSTormer also provides a separate temporal localization output.

Raw detector outputs are uncalibrated and must not be interpreted as definitive probabilities of authenticity. The analysis pipeline applies frozen Platt calibration to GenD and custom Xception for the primary verdict; Effort and FakeSTormer outputs remain uncalibrated.

The final verdict can be `INCONCLUSIVE` when the evidence does not support a reliable decision.
## 11. API implementation status and limitations

- The backend exposes health and model metadata endpoints.
- The models endpoint reports the four integrated model entries.
- The analysis events endpoint currently returns an event snapshot; it is not a live Server-Sent Events stream.
- The `/api/v1/verify/{report_id}` endpoint currently returns HTTP 501 and is not implemented.
- Model inference requires the appropriate checkpoints and Python environments.
- The current integration should not be considered a fully production-hardened deployment.
## 12. Licensing and attribution

Third-party model implementations and checkpoints retain their respective licenses and attribution requirements.

### Effort

The upstream Effort README declares CC BY-NC 4.0 terms. The upstream repository inspected for this project did not contain a standalone LICENSE file, so this declaration should not be treated as confirmation that every file or dependency has the same license. Individual third-party licenses and notices remain applicable. Commercial use requires appropriate separate permission where applicable. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the inspected revision and attribution details.

Paper:

Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection.

https://arxiv.org/abs/2411.15633

### FakeSTormer

FakeSTormer is distributed under the SnT academic license, based on MIT with academic limitations.

The license permits academic research and development purposes. Other uses require a separate license agreement.

Contact:

snt-tto@uni.lu

Preserve the original LICENSE, NOTICE, Third_Party_License_Notice, README, and paper attribution files.

### GenD and Xception

Retain the original model and checkpoint attribution and comply with the applicable upstream licenses and terms.
## 13. Important deployment considerations

1. Install the correct Python version for each model environment.
2. Install compatible CUDA-enabled PyTorch builds if GPU acceleration is required.
3. Download all required model checkpoints before starting inference.
4. Do not copy virtual environments between computers.
5. Keep large checkpoints out of ordinary Git history unless their licensing and distribution permissions have been checked.
6. Configure upload limits, CORS origins, and storage paths before deployment.
7. Validate model outputs and calibration before using the system for consequential decisions.



