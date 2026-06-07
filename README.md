# PRInter Reproducibility Code

This repository contains the minimal code needed to reproduce the PRInter/iLncPNet-style lncRNA-protein interaction prediction workflow.

It keeps only the essential training and data-processing code. Experimental result tables, reviewer-specific analyses, baseline adaptations, caches, model weights, and datasets are intentionally excluded.

## What Is Included

```text
.
+-- run_chapter5_reproduction.py
+-- scripts/
|   +-- make_degree_subset.py
+-- docs/
|   +-- REPRODUCIBILITY.md
+-- requirements.txt
+-- LICENSE
+-- CITATION.cff
+-- .gitignore
```

## Method Overview

The pipeline has two stages:

1. Learn lncRNA and protein representations from sequence features and network context.
2. Train an RBF-kernel SVM on concatenated lncRNA-protein pair embeddings.

Because the SVM is trained after neural representation learning, the full workflow should be described as a representation-learning-based prediction pipeline rather than a strictly end-to-end neural classifier.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Reproducibility Notes

Hyperparameters for random walks, contrastive learning, SVM classification, negative sampling, and threshold selection are described in `docs/REPRODUCIBILITY.md`.

## License

This code is released under the MIT License. Dataset redistribution may be subject to the licenses or terms of the original database providers.
