# AMPDiffusion 8M — Benchmark Results

We release the **16.54M parameter** AMPDiffusion checkpoint (7.40M ESM2-8M layers + 9.14M own parameters) trained on the same dataset of our Cell Biomaterials paper ([S3050-5623(25)00174-6](https://www.cell.com/cell-biomaterials/fulltext/S3050-5623(25)00174-6)) with 19,670 AMP samples. It was trained on ESM2-8M embeddings and reused attention layer weights for initialization.

## Quick Start

```bash
# Install dependencies
pip install torch fair-esm einops ema_pytorch

# Generate sequences
python note/generate.py \
    --checkpoint note/model.pt \
    --num 1000 \
    --length 25 \
    --output generated.csv
```

## Evaluation

All models scored with the same evaluation pipeline for fair comparison. We report two AMPDiffusion variants:
- **AMPDiff-25aa**: 50,000 unique sequences at fixed 25aa (matching other baselines' max length, enables full HydrAMP scoring)
- **AMPDiff-mix**: 50,000 unique sequences at mixed lengths (10-39aa)

Eval CSVs for each model are in `note/eval/`.

### 1. Average MIC — APEX 11-Species (lower = better)

MIC (Minimum Inhibitory Concentration) predicted by [APEX](https://gitlab.com/machine-biology-group-public/apex) ensemble (40 models, 8 architectures x 5 repeats). Averaged across the ensemble. We report the mean across the same 11 pathogen species as the Cell Biomaterials paper:

*A. baumannii ATCC19606, E. coli ATCC11775, E. coli AIG221, E. coli AIG222, K. pneumoniae ATCC13883, P. aeruginosa PAO1, P. aeruginosa PA14, S. aureus ATCC12600, S. aureus (ATCC BAA-1556) MRSA, vancomycin-resistant E. faecalis ATCC700802, vancomycin-resistant E. faecium ATCC700221*

> **Note**: These APEX values use the current public APEX repo and differ in absolute scale from the values reported in the original Cell Biomaterials paper, but the relative ranking of models is consistent.

| Metric | **AMPDiff-25aa** | **AMPDiff-mix** | AmpGAN | Training | HydrAMP | PepCVAE |
|---|---|---|---|---|---|---|
| **Mean** | **84.9** | **87.0** | 96.2 | 100.7 | 105.7 | 115.9 |
| **Median** | **85.2** | **88.7** | 102.0 | 109.7 | 115.1 | 126.5 |
| **% < 50** | 12.8% | 11.7% | 11.8% | 11.3% | 6.1% | 2.9% |
| **% < 100** | **66.0%** | **62.6%** | 48.4% | 41.3% | 36.7% | 22.1% |
| **% < 128** | **94.6%** | **94.4%** | 76.2% | 72.5% | 66.1% | 53.2% |

### 2. HydrAMP Classifier — P(AMP) and P(low MIC) (higher = better)

Binary classifiers from [HydrAMP](https://github.com/szczurek-lab/hydramp). P(AMP) = probability of being an antimicrobial peptide. P(low MIC) = probability of having low MIC. Max sequence length: 25 aa. All percentages computed on N scored (sequences <=25aa), not total generated.

| Metric | **AMPDiff-25aa** | **AMPDiff-mix** | Training | HydrAMP | AmpGAN | PepCVAE |
|---|---|---|---|---|---|---|
| **N scored (<=25aa)** | 50,000 | 26,198 | 13,896 | 50K | 50K | 50K |
| **P(AMP) mean** | **0.905** | 0.882 | 0.888 | 0.782 | 0.631 | 0.516 |
| **P(AMP) > 0.8** | **87.8%** | 84.6% | 84.1% | 76.8% | 54.0% | 41.1% |
| **P(low MIC) mean** | **0.703** | 0.608 | 0.479 | 0.512 | 0.316 | 0.205 |
| **P(low MIC) > 0.5** | **70.7%** | 61.2% | 47.8% | 51.3% | 31.6% | 20.4% |
| **P(low MIC) > 0.8** | **66.5%** | 56.6% | 45.2% | 48.9% | 27.9% | 17.8% |

### 3. ProGen2 Perplexity (lower = better)

Protein language model perplexity from [ProGen2-medium](https://huggingface.co/hugohrban/progen2-medium). Lower PPL indicates more protein-like sequences.

| Metric | **AMPDiff-25aa** | **AMPDiff-mix** | Training | AmpGAN | PepCVAE | HydrAMP |
|---|---|---|---|---|---|---|
| **Mean PPL** | **15.81** | 16.12 | 17.84 | 18.51 | 20.02 | 20.25 |
| **Median PPL** | **16.16** | 16.13 | 17.76 | 18.30 | 19.76 | 20.35 |

### 4. Amino Acid Frequency (AMPDiffusion vs Training)

| AA | AMPDiffusion | Training | Diff |
|---|---|---|---|
| A | 8.34% | 6.76% | +1.58 |
| C | 2.44% | 5.47% | -3.03 |
| D | 1.18% | 2.40% | -1.22 |
| E | 2.52% | 3.04% | -0.52 |
| F | 4.42% | 4.00% | +0.42 |
| G | 7.94% | 7.66% | +0.28 |
| H | 1.33% | 2.23% | -0.90 |
| I | 7.52% | 6.35% | +1.17 |
| **K** | **15.68%** | **11.26%** | **+4.42** |
| **L** | **11.38%** | **8.70%** | **+2.68** |
| M | 4.21% | 1.15% | +3.06 |
| N | 2.47% | 3.73% | -1.26 |
| P | 3.14% | 3.64% | -0.50 |
| Q | 2.57% | 2.88% | -0.31 |
| **R** | **9.84%** | **10.26%** | **-0.42** |
| S | 3.47% | 4.56% | -1.09 |
| T | 2.83% | 3.40% | -0.57 |
| V | 5.34% | 6.20% | -0.86 |
| W | 2.06% | 3.81% | -1.75 |
| Y | 1.31% | 2.49% | -1.18 |

The model enriches cationic residues (K +4.4%, L +2.7%) consistent with strong AMP activity.

Jensen-Shannon divergence from Training: **0.129** (AMPDiffusion), 0.146 (AmpGAN), 0.211 (HydrAMP), 0.084 (PepCVAE).

### 5. Diversity — Jaccard Similarity to Training (k-mers, mixed-length)

| Model | 3-mer unique | 3-mer Jaccard | 3-mer coverage | 6-mer unique | 6-mer Jaccard |
|---|---|---|---|---|---|
| **AMPDiffusion** | 7,983 | 0.989 | **99.8%** | **690,963** | **0.030** |
| AmpGAN | 7,998 | 0.991 | 100% | 392,307 | 0.012 |
| HydrAMP | 7,992 | 0.990 | 99.9% | 458,692 | 0.011 |
| PepCVAE | 7,999 | 0.991 | 100% | 575,985 | 0.009 |

Training has 7,929 unique 3-mers and 142,236 unique 6-mers.

### 6. Internal Diversity

| Metric | AMPDiff-25aa | AMPDiff-mix | AmpGAN | HydrAMP | PepCVAE | Training |
|---|---|---|---|---|---|---|
| Total sequences | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 | 19,670 |
| Unique sequences | 50,000 (100%) | 50,000 (100%) | 49,635 (99.3%) | 50,000 (100%) | 50,000 (100%) | 19,670 (100%) |

## Eval Data

Per-sequence evaluation CSVs are in `note/eval/`:
- `ampdiffusion_eval.csv` — 50,000 generated sequences (mixed length)
- `ampdiffusion_25aa_eval.csv` — 50,000 generated sequences (fixed 25aa)
- `train_eval.csv`
- `ampgan_eval.csv`
- `hydramp_eval.csv`
- `pepcvae_eval.csv`

Each CSV contains: `sequence`, 11 species-specific APEX MIC values, `average_MIC_11sp`, `hydramp_pamp`, `hydramp_pmic`, `progen_ppl`.
