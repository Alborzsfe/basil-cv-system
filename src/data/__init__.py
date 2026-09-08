"""Dataset schema, validation, reporting, and group-aware splitting for the
basil segmentation dataset (Phase 1/2 of the project).

Dataset loader classes for PyTorch (``Dataset``/``DataLoader``) are added in
a later phase once segmentation training begins; this package currently
focuses on making sure the raw annotated data is trustworthy before any
model ever sees it.
"""
