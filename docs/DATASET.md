# Dataset protocol

No dataset is distributed with this repository. This document defines the
minimum evidence required before training or publishing results.

## Collection and provenance

For every source, record the owner, collection dates, greenhouse/site, camera,
lighting, cultivar, growth stage, disease confirmation method, annotator process,
authorization, and license. Remove personal or operational identifiers that are
unnecessary for the research question.

Do not mix internet images and local greenhouse images without preserving source
domains. A model can otherwise learn source artifacts instead of disease cues.

## Segmentation schema

Images live under `data/basil/segmentation/images/`; binary masks live under
`data/basil/segmentation/masks/`. Matching files normally share a stem. Masks may
use `0/1` or `0/255`; any other palette is rejected.

| Column | Meaning |
| --- | --- |
| `image_filename` | Relative image filename |
| `mask_filename` | Relative binary-mask filename |
| `plant_id` | Stable physical plant/cluster identity |
| `duct_id`, `hole_id` | Optional greenhouse location identifiers |
| `capture_session_id` | Stable session identity |
| `timestamp` | Capture time in a documented timezone |

## Classification schema

Required columns are `image_path`, `label`, and the configured group column
(normally `plant_id`). Leaf-only input also requires `mask_path` or a generated
`leaf_image_path`; mask-aware input always requires `mask_path`.

Label definitions must describe observable inclusion/exclusion criteria. Use a
qualified domain process for disease confirmation. Ambiguous samples should be
excluded or separately adjudicated—not relabeled as the inference outcome
`unknown`.

## Leakage prevention

All images from one configured group remain in exactly one of train, validation,
or test. Choose the group at the strongest plausible dependence level: use
`capture_session_id` or a site/plant composite if `plant_id` could be reused.

Before a run:

1. replace every `UNKNOWN`, blank, and missing group identifier;
2. generate a persisted split with the relevant split script;
3. inspect class/source counts in every partition;
4. freeze the test split and do not tune against it;
5. publish the split hash or sanitized split when licensing permits.

Near-duplicate frames, augmented derivatives, and images derived from the same
raw capture must never cross partitions.

## Quality assurance

- review validator errors and warnings rather than silently dropping rows;
- independently audit a stratified sample of masks and labels;
- report inter-annotator agreement and adjudication where applicable;
- document exclusion counts and reasons;
- evaluate by class and acquisition subgroup, not only aggregate score.

## Candidate public sources—not used

These links are starting points, not endorsed training data:

- [Basil Leaf Disease Detection on Roboflow Universe](https://universe.roboflow.com/paul-justin-e-garde/basil-leaf-disease-detection) lists 305 images, CC BY 4.0, and healthy/fusarium/powdery labels on its page. Verify the files, task, label quality, provenance, current terms, and license yourself before use.
- [Plant leaf segmentation dataset on Zenodo](https://zenodo.org/records/14707857) describes 397 plant-leaf images and masks. It is not basil-specific and has not been imported or evaluated here.

Record an immutable version, retrieval date, and checksum for external data.
Never infer that a web-page label constitutes clinically confirmed disease.
