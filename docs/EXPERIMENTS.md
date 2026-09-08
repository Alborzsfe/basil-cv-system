# Experiment protocol

## Primary question

Does restricting a classifier to segmented basil-leaf regions improve disease
classification compared with the original greenhouse image?

## Controlled comparison

| Run | Input | Augmentation | Purpose |
| --- | --- | --- | --- |
| A | `raw_image` | Off | Background-containing baseline |
| B | `leaf_only` | Off | Isolate the effect of background removal |
| C | `leaf_only` | On | Measure the documented augmentation policy |

Use the same verified labels, exact persisted group split, random seed, backbone,
pretraining state, image size, optimizer, schedule, epoch/early-stopping budget,
and evaluation code. If a factor must change, declare the comparison non-isolated.

## Before training

- pass dataset validation;
- confirm zero group overlap and near-duplicate leakage;
- ensure every class occurs in each partition;
- define the primary metric before seeing test results;
- record class/source/group distributions and label-audit results.

Macro F1 is the default primary classification metric because disease classes may
be imbalanced. Also report accuracy, weighted F1, per-class sensitivity and
specificity, confusion matrix, ROC AUC, and PR AUC where statistically defined.
For segmentation, report Dice and IoU plus precision, recall, and pixel accuracy.

## Execution and reporting

Each training command writes a resolved config, split copy, run manifest, history,
and checkpoints. Evaluate only the selected checkpoint once on the held-out test
partition. Use `scripts/compare_experiments.py` to create the comparison table.

Report sample and independent-group counts, uncertainty intervals, selection rule,
failed/excluded cases, compute, software revision, and limitations. Prefer
group-level bootstrapping when multiple images belong to one plant. A numerical
difference without uncertainty and a predeclared test is not robust evidence.

Do not tune on the test set, report synthetic smoke output as model performance,
or generalize beyond observed cultivars, cameras, sites, lighting, diseases, and
growth stages.
