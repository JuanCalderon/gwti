# GWTI: Gram-Weighted Tracing for Interpretability

This repository contains the full implementation of **GWTI (Gram-Weighted Tracing for Interpretability)**, a lightweight and pre-hoc interpretability framework designed for sparse short-text NLP tasks such as tweet classification and user profiling.

## 🌟 Overview

GWTI combines:
- **q-gram tokenization**
- **TF-IDF vectorization**
- **Linear classifiers**
- A **tracing mechanism** to recover token-level contributions for visualization.

It provides precise, low-cost visual explanations that are particularly effective in high-dimensional, sparse input spaces.

## 📁 Repository Structure

- `experiments/` — Python scripts for reproducing the experiments and figures reported in the PRL manuscript.
- `supplementary/` — Supplementary materials heatmaps, token-level visualizations, and UI snapshots used in the paper 
- `CITATION.cff` — Citation metadata for properly referencing this work.
- `README.md` — This file.

## 📖 Publication

This code accompanies the manuscript:

**“GWTI: A Pre-Hoc Framework for Visual Interpretability in Sparse Short-Text NLP Tasks”**, submitted to *Pattern Recognition Letters (PRL)*.

## 🔗 Citation

To cite this work, please see the [`CITATION.cff`](./CITATION.cff) file or use the following BibTeX (coming soon).

## 📬 Contact

For questions or collaboration inquiries, please contact:

**José J. Calderón**  
Email: jose.calderon@cimav.edu.mx  
ORCID: [0000-0002-XXXX-XXXX](https://orcid.org)

---

© 2025 José J. Calderón, Mario Graff, Eric S. Téllez. Licensed under MIT.
