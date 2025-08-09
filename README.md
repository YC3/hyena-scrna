![Status](https://img.shields.io/badge/status-under_development-yellow?style=for-the-badge)
🚧 **Work in progress** — Expect breaking changes.


# Hyena-scRNA: 
Foundation model pretraining for scRNA-seq using Hyena architecture.

This repository contains code for pre-training a single-cell RNA-seq (scRNA-seq) foundation model using the Hyena architecture, designed to:

- Learn genome-wide gene expression patterns across entire transcriptomes (20k+ genes)
- Capture long-range gene interactions (e.g., trans-chromosomal, enhancer-promoter) via Hyena’s subquadratic operators
- Enable zero-shot transfer to downstream tasks like cell type annotation, perturbation modeling, rare cell detection, and in silico perturbation for drug target identification

