# AlphaGenome Finetuning

## Overview

This repository contains two scripts for finetuning the pretrained AlphaGenome model on custom genomic tracks provided as bigwig files. The scripts are adapted from the official AlphaGenome finetuning notebook with several bug fixes and improvements.

- `rebin.py` — Preprocesses bigwig files to 128bp resolution before training
- `finetune.py` — Finetunes the AlphaGenome model on your prepared bigwig files

---

## Requirements

### HuggingFace Authentication
`finetune.py` downloads the pretrained AlphaGenome checkpoint from HuggingFace automatically. Authenticate once before running:
```bash
huggingface-cli login
```
This stores your token securely and is picked up automatically. Do not hardcode your token in any script.

---

## Step 1 — Preprocess Bigwig Files with rebin.py

### Why Rebinning Is Needed

AlphaGenome predicts genomic tracks at different resolutions depending on the biological nature of each assay:

- **CHIP_HISTONE and CHIP_TF** — histone modifications and transcription factor binding are broad signals that span hundreds of base pairs. The model predicts these at **128bp resolution**, producing 8,192 values per 1Mbp window.
- **RNA_SEQ and DNASE** — these are fine-resolution signals where base-pair precision matters. The model predicts these at **1bp resolution**, producing 1,048,576 values per 1Mbp window.

Bigwig files from sources such as GEO are typically stored at 1bp resolution. The AlphaGenome data pipeline extracts bigwig values using `bw.values()`, which always returns data at 1bp resolution regardless of how the bigwig is stored. For `CHIP_HISTONE` training this means the pipeline provides 1,048,576 target values per window, but the model only produces 8,192 predictions. The loss function cannot compare arrays of different sizes and fails with:

```
AssertionError: Arrays have different shapes: [(1, 8192, 4), (1, 1048576, 4)]
```

This bug is not present in the official AlphaGenome notebook because that notebook only demonstrates `RNA_SEQ` and `DNASE` tracks, which happen to work at 1bp resolution. It only surfaces when using `CHIP_HISTONE` or `CHIP_TF` tracks.

`rebin.py` solves this by averaging each bigwig's signal into 128bp windows, producing a `_128bp.bw` file that is passed to `finetune.py`. Only `CHIP_HISTONE` and `CHIP_TF` bigwigs need rebinning — `RNA_SEQ` and `DNASE` files are used at their original resolution.

### Configuring rebin.py

Update the `INPUT_FILES` list in `rebin.py` with the paths to your original bigwig files:

```python
INPUT_FILES = [
    '/path/to/H3K4me3_replicate1.bw',
    '/path/to/H3K4me3_replicate2.bw',
    '/path/to/H3K27ac_replicate1.bw',
]
```

### Running rebin.py

```bash
python rebin.py
```

For each input file, a rebinned version is saved alongside the original with a `_128bp.bw` suffix:
```
H3K4me3_replicate1.bw       <- original, unchanged
H3K4me3_replicate1_128bp.bw <- rebinned, used for training
```

Rebinning only needs to be done once per bigwig file.

---

## Step 2 — Finetune the Model with finetune.py

### Configuring finetune.py

**1. Update TRACK_METADATA with your rebinned bigwig files:**

```python
TRACK_METADATA = pd.DataFrame(
    data=[
        ['CHIP_HISTONE', 'H3K4me3 replicate 1', '.', '/path/to/H3K4me3_replicate1_128bp.bw'],
        ['CHIP_HISTONE', 'H3K27ac replicate 1', '.', '/path/to/H3K27ac_replicate1_128bp.bw'],
    ],
    columns=['output_type', 'name', 'strand', 'file_path'],
)
```

Each row requires:
- `output_type`: One of `CHIP_HISTONE`, `CHIP_TF`, `RNA_SEQ`, `DNASE`, `ATAC`
- `name`: Human-readable track name
- `strand`: `+`, `-`, or `.` for unstranded
- `file_path`: Path to the rebinned `_128bp.bw` file for CHIP tracks, or the original bigwig for RNA_SEQ and DNASE

**2. Edit the configuration block at the top of `finetune.py`:**

| Parameter | Default | Description |
|---|---|---|
| `LEARNING_RATE` | `5e-4` | Adam optimizer learning rate |
| `NUM_TRAIN_STEPS` | `1000` | Total number of training steps |
| `MODEL_VERSION` | `FOLD_0` | AlphaGenome model fold to use |
| `SEQUENCE_LENGTH` | `2**20` (1Mbp) | Genomic context window size |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `ORGANISM` | `HOMO_SAPIENS` | Training organism |
| `NONZERO_MEAN_RESCALING` | `True` | Normalise tracks by nonzero mean |
| `SAVE_CHECKPOINT_DIR` | `/path/to/checkpoints` | Where to save checkpoints |
| `CHECKPOINT_INTERVAL` | `100` | Save a checkpoint every N steps |
| `BIN_SIZE` | `128` | Model output resolution in base pairs |

### Running finetune.py

```bash
python finetune.py
```

### Nonzero Mean Normalisation

If `NONZERO_MEAN_RESCALING` is enabled, the script computes the mean of all non-zero values across each bigwig file before training. This is used by the model to normalise track signal levels and requires scanning all chromosomes, which can take several minutes.

To avoid repeating this on every run, the results are cached to a CSV file and reloaded automatically on subsequent runs.

### Checkpoints

Checkpoints are saved to a timestamped subdirectory under `SAVE_CHECKPOINT_DIR`:

```
checkpoints/20240514_120000/
    checkpoint_00100/
    checkpoint_00200/
    checkpoint_01001/   <- final checkpoint
```

A checkpoint is saved every `CHECKPOINT_INTERVAL` steps throughout training so progress is not lost if the job is interrupted before completion.

---

## Step 3 — Using the Retrained Model

### What Gets Saved

Each checkpoint saved during training is the retrained model. It stores two components:
- `params` — all model weights including the finetuned CHIP_HISTONE output head
- `state` — batch normalisation and other stateful components

The final checkpoint at the end of training is your fully retrained model. Intermediate checkpoints saved every `CHECKPOINT_INTERVAL` steps are also fully usable.

**Important:** The checkpoint saves weights only, not the model architecture or track configuration. When loading for inference you must reconstruct `output_metadata` and `forward_fn` with the same track setup used during training. Keep `finetune.py` alongside your checkpoints as a record of exactly what the model was trained on.

### Loading the Checkpoint

```python
import orbax.checkpoint as ocp

checkpointer = ocp.StandardCheckpointer()
params, state = checkpointer.restore('/path/to/checkpoints/20240514_120000/checkpoint_01001')
```

### Running Inference

Reconstruct the same `output_metadata` and `forward_fn` as used during training, then pass a batch through the model:

```python
from alphagenome_research.finetuning import finetune

# Reconstruct output metadata using the same TRACK_METADATA as training
output_metadata = {
    dna_model.Organism.HOMO_SAPIENS: build_output_metadata(TRACK_METADATA)
}

# Reconstruct forward function with the same jmp_policy as training
forward_fn = finetune.get_forward_fn(
    output_metadata, jmp_policy='params=float32,compute=float32,output=float32'
)

# Get a batch for the genomic region you want to query
batch = next(ds_iter)
batch = bin_batch_targets(batch)

# Run forward pass
with jax.set_mesh(mesh):
    batch = jax.device_put(batch, data_sharding)
    (loss, scalars, predictions), _ = forward_fn.apply(
        params, state, None, batch
    )
```

### Understanding the Output

`predictions` is a dictionary keyed by output type. For a CHIP_HISTONE model:

```python
predictions['chip_histone']  # shape: (batch, 8192, num_tracks)
```

Each of the 8,192 values corresponds to a 128bp bin across the 1Mbp input window. To map bin indices back to genomic coordinates:

```python
interval_start = batch_interval_start  # start position of the queried interval
bin_starts = [interval_start + i * 128 for i in range(8192)]
```

### What the Retrained Model Can Be Used For

- **Predict histone mark signal** at genomic regions not seen during training
- **Score regulatory elements** by comparing predicted signal at candidate regions
- **Variant effect prediction** — compare predictions on reference vs mutant sequences to assess the impact of genetic variants on histone marks

---

## Changes from the Original Notebook

### Bug Fixes

**1. CHIP_HISTONE target resolution mismatch (critical)**

Described in full in the Rebinning section above. Fixed via `rebin.py` (preprocessing bigwigs to 128bp) and `bin_batch_targets()` in `finetune.py` (binning pipeline targets to match model output resolution at runtime).

**2. Device tensor accumulation in loss list**

The original script appended raw JAX device arrays to the loss list on every step, keeping them allocated on the GPU for the duration of training.

**Fix:** Convert to a Python float immediately to free device memory:
```python
loss.append(float(scalars['loss']))
```

### Improvements from the Notebook

**3. Strand-pair nonzero mean averaging**

The notebook includes an averaging step for strand-paired tracks (e.g. RNA_SEQ `+` and `-` strands) after computing nonzero means. This was missing from the original script conversion. It is a no-op for unstranded ChIP-seq data but ensures correct normalisation if stranded tracks are added in future.

**4. Column validation in build_output_metadata**

Added upfront validation of required columns in `TRACK_METADATA`. Produces a clear error message instead of a cryptic `KeyError` if a column is missing.

**5. Nonzero mean caching**

Results are cached to CSV and reloaded on subsequent runs, avoiding the expensive bigwig scan on every job submission.

**6. Intermediate checkpointing**

The original script saved only a single checkpoint at the end of training. Checkpoints are now saved every `CHECKPOINT_INTERVAL` steps so progress is not lost if the job is interrupted.