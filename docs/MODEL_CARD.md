# Model card

## Model details

Basil CV System is a configurable two-stage research pipeline: binary semantic
leaf segmentation followed by multi-class disease classification. Supported
production backbones include ResNet-18/34 U-Net encoders and EfficientNet V2 S,
ResNet-50, or ConvNeXt Tiny classifiers. Tiny networks exist for software tests.

Version 1.0.0 contains software and configuration only. It does not ship trained
weights, training data, or disease-class definitions.

## Intended use

- controlled research on basil imagery;
- testing whether background removal changes held-out classification performance;
- human-reviewed experiments with documented data provenance and labels.

## Out of scope

- medical, agronomic, pesticide, food-safety, or regulatory decisions;
- autonomous crop removal or treatment;
- disease severity, prognosis, or causal diagnosis;
- individual-leaf counting, sensor fusion, temporal modeling, or other species.

## Performance

No real-data performance is available. `docs/results/smoke/` demonstrates only
that software stages connect and write expected artifacts. Synthetic class names
and confidence values are not estimates of biological validity.

## Inputs and outputs

The input is one RGB image. The pipeline may reject low-quality images, reject
insufficient visible-leaf area, or return a configured class/confidence. Confidence
below threshold becomes `unknown`. Optional artifacts include the probability
mask, binary mask, leaf-only image, and Grad-CAM overlay.

## Risks and limitations

- background, camera, lighting, site, cultivar, season, and disease-stage shift;
- label error and visual overlap between disease, nutrition, pests, and damage;
- occlusion and small visible leaf area;
- probability miscalibration and false confidence;
- segmentation errors propagated to classification;
- spurious visual shortcuts and subgroup performance gaps;
- Grad-CAM that appears plausible without being causal or faithful.

Thresholding cannot guarantee safety. An `unknown` outcome is not proof that all
unsafe cases are rejected.

## Evaluation and oversight

Use plant/session group isolation, untouched test data, per-class and subgroup
metrics, uncertainty intervals, external-site validation, and calibration checks.
Inspect raw image, mask, and explanation together. A qualified person must review
outputs and retain authority over every real-world action.

## Privacy and security

Do not commit private greenhouse imagery or identifiers. Load checkpoints only
from trusted sources and verify provenance. See `SECURITY.md`.
