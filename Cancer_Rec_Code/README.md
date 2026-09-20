# Cancer_Rec_Code

This package contains the complete main cancer-recognition workflow from STMap generation through the three training stages and five-fold evaluation.

## Cohort definition

The default five-class cohort is:

- P: 32
- Q: 25
- S: 24
- N: 30
- H: 48
- Total: 159

The default source-data directory is:

```text
./Cancer_Data/Ori-Data/
├── P/
├── Q/
├── S/
├── N/
└── H/
```

## Main workflow

### Stage 1: Train TSB + STB branches to obtain deep feature

The Temporal-Structural Branch (TSB) and Spatiotemporal Texture Branch (STB) are trained for Task1 and Task2. Their fused output is projected to a 64-dimensional deep feature.

### Stage 2: Train Explicit branch to add explicit information

The Stage-1 backbone is frozen. Explicit response descriptors are extracted from four feature families:

- unit dynamics
- reaction kinetics
- normalized temporal shape
- relative-unit responses

The 8,892-dimensional explicit descriptor is encoded to 64 dimensions and concatenated with the 64-dimensional deep feature, producing a 128-dimensional joint representation for Task1 and Task2.

### Stage 3: Task3 decision adjustment

Task1 and Task2 remain fixed. A lightweight residual decision-adjustment network uses their joint features, explicit-response context, and base hierarchical probabilities to produce the final four-class Task3 output.

## Task definitions

- Task1: H+N versus P+Q+S
- Task2: P versus Q versus S
- Task3: H+N versus P versus Q versus S

## STMap generation

The STMap generator reads the processed workbook in each sample directory and builds the Full STMap tensor. The default output path is:

```text
./Cancer_Data/Pro-STMap/full/stmaps_full.npz
```

The default input is the drift-corrected workbook and the `STMap_wide` sheet.

## One-command execution

```bash
bash run_main.sh
```

The script performs:

```text
1. Structural verification
2. Full STMap generation
3. Stage 1: Train TSB + STB branches to obtain deep feature
4. Stage 2: Train Explicit branch to add explicit information
5. Stage 3: Task3 decision adjustment
6. Five-fold result aggregation
```

## Common options

Use another GPU:

```bash
GPU=0 bash run_main.sh
```

Skip STMap generation when the Full STMap already exists:

```bash
SKIP_STMAP=1 bash run_main.sh
```

Run one fold for a quick test:

```bash
FOLD=1 SKIP_STMAP=1 bash run_main.sh
```

Override paths:

```bash
SOURCE_ROOT=/path/to/Ori-Data \
STMAP_ROOT=/path/to/Pro-STMap \
RESULT_ROOT=./results \
bash run_main.sh
```

## Project files

```text
Cancer_Rec_Code/
├── cancer_model.py
├── cancer_pipeline.py
├── explicit_response_features.py
├── stmap_generation.py
├── run_cancer_recognition.py
├── verify_cancer_recognition.py
├── config.yaml
├── requirements.txt
├── run_main.sh
├── README.md
└── VALIDATION_REPORT.txt
```
