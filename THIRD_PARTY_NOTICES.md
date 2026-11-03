# Third-Party Notices

This project incorporates or is based on third-party software and model
components. Their respective licenses and terms remain applicable. The
project's own license does not replace or override those terms.

## 1. DeepfakeBench

The Xception implementation used by this project is based on code from
DeepfakeBench.

- Copyright: 2023, The Chinese University of Hong Kong, Shenzhen.
- License: Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0).
- License text: https://creativecommons.org/licenses/by-nc/4.0/legalcode
- Upstream project: https://github.com/SCLBD/DeepfakeBench

The local DeepfakeBench LICENSE file contains the license text and
third-party dependency notices. The Xception implementation has been
adapted for this project. Changes include integration and inference
adaptations; the original upstream attribution is retained here.

This license is non-commercial. Commercial use requires separate
authorization where applicable.

## 2. Xception implementation and FaceForensics

The Xception network implementation identifies Zhiyuan Yan as its author
and states that it is mainly modified from the FaceForensics Xception
implementation.

- Author identified in source: Zhiyuan Yan.
- Source reference:
  https://github.com/ondyari/FaceForensics/blob/master/classification/network/xception.py
- FaceForensics repository:
  https://github.com/ondyari/FaceForensics
- FaceForensics code license: MIT, as stated by the upstream repository.

The local DeepfakeBench source also contains its own license and
attribution conditions. Both upstream origins should be acknowledged
when redistributing the relevant source files.

## 3. ImageNet-pretrained Xception backbone

The custom Xception checkpoint was initialized using the pretrained
backbone file:

`xception-b5690688.pth`

The file corresponds to the Xception weights distributed through
Cadene's pretrained-models.pytorch project.

- Project: https://github.com/Cadene/pretrained-models.pytorch
- License: BSD 3-Clause.
- Copyright: 2017, Remi Cadene.
- License text:
  https://github.com/Cadene/pretrained-models.pytorch/blob/master/LICENSE.txt

The custom checkpoint was subsequently fine-tuned on FaceForensics++
C23 data using the project's custom training script. The custom
checkpoint is not the released DeepfakeBench `xception_best.pth`
detector checkpoint.

## 4. FaceForensics++ dataset

The custom Xception model was trained using FaceForensics++ data.

The dataset is governed by its separate Terms of Use, not by the
FaceForensics source-code license.

- Terms of Use:
  https://kaldir.vc.in.tum.de/faceforensics_tos.pdf
- Dataset project:
  https://github.com/ondyari/FaceForensics

The terms restrict use to non-commercial research and educational
purposes. They also specify conditions for providing research
colleagues access to the database.

The dataset itself, extracted frames, and dataset-derived media must
not be included in this repository.

The terms do not clearly establish whether a trained checkpoint
derived from the dataset may be publicly redistributed. Therefore,
this repository does not make a redistribution claim for the custom
Xception checkpoint. Obtain clarification from the dataset providers
before publicly distributing that checkpoint.

## 5. Dlib tools and landmark predictor

The Xception preprocessing source includes dlib-related tools.

A BSD 3-Clause license file is present at:

`backend/model_sources/xception/preprocessing/dlib_tools/LICENSE`

The landmark predictor file:

`shape_predictor_81_face_landmarks.dat`

is a separate runtime asset. Its specific redistribution terms have
not yet been independently verified. It is therefore excluded from
public Git distribution pending verification. The inference pipeline
expects this file to be available locally.

## 6. Project-specific Xception checkpoint

The custom checkpoint:

`model_assets/Xception/xception_custom_best.pth`

was produced by fine-tuning an ImageNet-pretrained Xception backbone
on FaceForensics++ C23 data.

Its provenance is documented, but its public redistribution
eligibility has not been established. It is intentionally excluded
from Git distribution until the relevant dataset and model-weight
terms have been clarified.

## 7. Effort

The Effort implementation is adapted from the upstream Effort-AIGI-Detection
repository.

- Project: Effort: Orthogonal Subspace Decomposition for Generalizable
  AI-Generated Image Detection.
- Upstream repository:
  https://github.com/YZY-stack/Effort-AIGI-Detection
- Inspected upstream revision:
  `96f5dea2b534d400cfd7003f053c7e93c8e16461`
- Upstream README license declaration: Creative Commons
  Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- License text: https://creativecommons.org/licenses/by-nc/4.0/legalcode

The upstream README states that its data, codebase, and benchmarking
protocols are mainly based on DeepfakeBench. The local Effort source
contains adaptations and separately attributed third-party components.

The upstream repository inspected for this notice did not contain a
top-level LICENSE file. The CC BY-NC 4.0 declaration is taken from its
README badge; this notice does not independently establish the licensing
status of every file or dependency in the upstream or adapted source tree.

Individual third-party notices and license headers must be retained.
This includes MIT, Modified BSD, and Apache 2.0 components identified
in the local source. The upstream Effort declaration does not replace
those component-specific terms.

The Effort model checkpoint and shared CLIP model assets are separate
from the source-code license. Their redistribution terms have not been
established here; they remain excluded from public Git distribution.

## 8. Disclaimer
This document is an attribution and provenance summary, not legal
advice. Consult the original license texts and dataset terms before
redistributing source code, model weights, datasets, or derived assets.

