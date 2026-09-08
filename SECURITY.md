# Security policy

## Supported version

Security fixes are applied to the latest revision of the `main` branch.

## Reporting a vulnerability

Use GitHub's private **Report a vulnerability** flow instead of a public issue.
Include the affected revision, reproduction steps, impact, and suggested
mitigation. Do not include credentials, private greenhouse images, plant
identifiers, or other sensitive data.

## Checkpoint trust boundary

Only load checkpoints created by this project or obtained from a trusted
source. The code uses `torch.load(..., weights_only=True)`, but that is not a
substitute for verifying provenance and file integrity.
