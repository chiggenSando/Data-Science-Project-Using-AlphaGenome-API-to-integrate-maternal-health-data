import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
tf.config.set_visible_devices([], 'TPU')

import dataclasses
from datetime import datetime
import os
import pprint

from alphagenome.data import fold_intervals
from alphagenome.data import genome
from alphagenome_research.finetuning import dataset as dataset_lib
from alphagenome_research.finetuning import finetune
from alphagenome_research.model import dna_model
from alphagenome_research.model.metadata import metadata as metadata_lib
from etils import epath
import huggingface_hub
import haiku as hk
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import optax
import orbax.checkpoint as ocp
import pandas as pd
import pyBigWig

# ── Configuration ──────────────────────────────────────────────────────────────
LEARNING_RATE = 5e-4
NUM_TRAIN_STEPS = 1000
MODEL_VERSION = dna_model.ModelVersion.FOLD_0
SEQUENCE_LENGTH = int(2**20)
BATCH_SIZE = 1
ORGANISM = dna_model.Organism.HOMO_SAPIENS
NONZERO_MEAN_RESCALING = True
SAVE_CHECKPOINT_DIR = '/data/home/s4076520/alphagenome/checkpoints'
CHECKPOINT_INTERVAL = 100  # Save a checkpoint every N steps

# FIX: CHIP_HISTONE model heads predict at 128bp resolution (8192 bins per 1Mbp).
# The data pipeline returns targets at 1bp resolution (1048576 values).
# BIN_SIZE converts pipeline targets to match model output resolution.
BIN_SIZE = 128

# ── Track Metadata ─────────────────────────────────────────────────────────────
TRACK_METADATA = pd.DataFrame(
    data=[
        [
            'CHIP_HISTONE',
            'H3K4me3 replicate 2',
            '.',
            '/data/home/s4076520/alphagenome/data/GSM6086873_2_TL_H3K4me3_IP_128bp.bw',
        ],
        [
            'CHIP_HISTONE',
            'H3K4me3 replicate 3',
            '.',
            '/data/home/s4076520/alphagenome/data/GSM6086874_3_TL_H3K4me3_IP_128bp.bw',
        ],
        [
            'CHIP_HISTONE',
            'H3K27ac replicate 2',
            '.',
            '/data/home/s4076520/alphagenome/data/GSM6086875_2_TL_H3K27ac_IP_128bp.bw',
        ],
        [
            'CHIP_HISTONE',
            'H3K27ac replicate 3',
            '.',
            '/data/home/s4076520/alphagenome/data/GSM6086876_3_TL_H3K27ac_IP_128bp.bw',
        ],
    ],
    columns=['output_type', 'name', 'strand', 'file_path'],
)

# ── Nonzero Mean ───────────────────────────────────────────────────────────────
_CHROMOSOMES = set(
    ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
)

# FIX: Cache nonzero means to avoid recomputing on every job submission.
NONZERO_MEAN_CACHE = '/data/home/s4076520/alphagenome/data/track_metadata_means.csv'

def nonzero_mean_from_bigwig(bigwig_path, chromosomes=_CHROMOSOMES):
    bw = pyBigWig.open(bigwig_path)
    total_sum = 0.0
    total_nonzero = 0
    for chrom, length in bw.chroms().items():
        if chrom not in chromosomes:
            continue
        values = np.nan_to_num(
            bw.values(chrom, 0, length, numpy=True).astype(np.float32),
            nan=0.0,
            copy=False,
        )
        total_sum += values.sum()
        total_nonzero += np.count_nonzero(values)
    bw.close()
    return total_sum / max(total_nonzero, 1)

if NONZERO_MEAN_RESCALING and ('nonzero_mean' not in TRACK_METADATA.columns):
    if os.path.exists(NONZERO_MEAN_CACHE):
        print('Loading cached nonzero means...')
        cached = pd.read_csv(NONZERO_MEAN_CACHE)
        TRACK_METADATA = TRACK_METADATA.merge(
            cached[['file_path', 'nonzero_mean']], on='file_path', how='left'
        )
        print('Cached means loaded.')
    else:
        print('Computing nonzero means (this may take a few minutes)...')
        TRACK_METADATA['nonzero_mean'] = TRACK_METADATA['file_path'].apply(
            nonzero_mean_from_bigwig
        )
        # FIX (from notebook): Average nonzero_mean across + and - strand pairs
        # for the same track. No-op for unstranded ChIP data (strand='.'),
        # but correct behaviour if stranded tracks are added later.
        for _, group in TRACK_METADATA.groupby(['name', 'output_type']):
            if set(group['strand']) == {'+', '-'}:
                avg = group['nonzero_mean'].mean()
                TRACK_METADATA.loc[group.index, 'nonzero_mean'] = avg
        TRACK_METADATA.to_csv(NONZERO_MEAN_CACHE, index=False)
        print('Nonzero means computed and cached.')

print('Track metadata:')
print(TRACK_METADATA)

# ── Output Metadata ────────────────────────────────────────────────────────────
def build_output_metadata(track_metadata):
    # FIX (from notebook): Validate required columns upfront for a clear error
    # message instead of a cryptic KeyError later.
    required_cols = {'file_path', 'name', 'output_type', 'strand'}
    if not required_cols.issubset(track_metadata.columns):
        raise ValueError(
            f'track_metadata missing columns: '
            f'{required_cols - set(track_metadata.columns)}'
        )
    metadata = {}
    for output_type, df_group in track_metadata.groupby('output_type'):
        try:
            output_type = dna_model.OutputType[str(output_type)]
        except KeyError as e:
            raise ValueError(f'Unknown output_type: {output_type}') from e
        metadata[output_type.name.lower()] = df_group
    return metadata_lib.AlphaGenomeOutputMetadata(**metadata)

output_metadata = {
    dna_model.Organism.HOMO_SAPIENS: build_output_metadata(TRACK_METADATA)
}

# ── Target Binning ─────────────────────────────────────────────────────────────
# FIX: The data pipeline (BigWigExtractor) uses bw.values() which always returns
# 1bp resolution regardless of bigwig bin size. CHIP_HISTONE and CHIP_TF model
# heads predict at 128bp resolution, so targets must be binned to match.
# Note: RNA_SEQ and DNASE heads predict at 1bp resolution so are NOT binned.
_CHIP_FIELDS = ['chip_histone', 'chip_tf']

def bin_batch_targets(batch, bin_size=BIN_SIZE):
    """Bin CHIP track targets from 1bp to 128bp to match model output resolution."""
    updates = {}
    for field in _CHIP_FIELDS:
        val = getattr(batch, field, None)
        if val is not None:
            b, seq_len, n_tracks = val.shape
            n_bins = seq_len // bin_size
            updates[field] = val[:, :n_bins * bin_size, :].reshape(
                b, n_bins, bin_size, n_tracks
            ).mean(axis=2)
    return dataclasses.replace(batch, **updates)

# ── Data Pipeline ──────────────────────────────────────────────────────────────
print('Setting up data pipeline...')
ds_iter = finetune.get_dataset_iterator(
    batch_size=BATCH_SIZE * jax.local_device_count(),
    sequence_length=SEQUENCE_LENGTH,
    output_metadata=output_metadata[ORGANISM],
    organism=ORGANISM,
    model_version=MODEL_VERSION,
    subset=fold_intervals.Subset.TRAIN,
)
batch = next(ds_iter)
batch = bin_batch_targets(batch)  # FIX: bin targets to 128bp resolution
print('Batch shapes (after binning):')
pprint.pprint(jax.tree.map(np.shape, batch))

# ── Model Initialisation ───────────────────────────────────────────────────────
print('Downloading pretrained checkpoint...')
repo = f'google/alphagenome-{MODEL_VERSION.name.lower().replace("_", "-")}'
checkpoint_path = huggingface_hub.snapshot_download(repo_id=repo)
checkpointer = ocp.StandardCheckpointer()
params_base, state_base = checkpointer.restore(checkpoint_path)
print('Checkpoint loaded.')

# ── Device Mesh ────────────────────────────────────────────────────────────────
num_devices = jax.local_device_count()
devices = mesh_utils.create_device_mesh((num_devices,))
mesh = Mesh(devices, axis_names=('data',))
data_sharding = P('data')
replicated_sharding = P()

# ── Initialise New Output Heads ────────────────────────────────────────────────
print('Initialising new output heads...')
forward_fn = finetune.get_forward_fn(
    output_metadata, jmp_policy='params=float32,compute=float32,output=float32'
)
with jax.set_mesh(mesh):
    batch = jax.device_put(batch, data_sharding)
    params_ft, state_ft = jax.jit(
        forward_fn.init,
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )(jax.random.PRNGKey(0), batch)

# ── Weight Surgery ─────────────────────────────────────────────────────────────
params_ft_head = hk.data_structures.filter(
    lambda module_name, *_: 'head' in module_name, params_ft
)
params_base_no_head = hk.data_structures.filter(
    lambda module_name, *_: 'head' not in module_name, params_base
)
params = hk.data_structures.merge(params_base_no_head, params_ft_head)
state = state_base
optimizer = optax.chain(
    optax.clip_by_global_norm(0.5),
    optax.adam(LEARNING_RATE),
)
opt_state = optimizer.init(params)
train_step = jax.jit(
    finetune.get_train_step(forward_fn.apply, optimizer),
    in_shardings=(
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        data_sharding,
    ),
    out_shardings=(
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
        replicated_sharding,
    ),
)

# ── Checkpointing ──────────────────────────────────────────────────────────────
path_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_dir = epath.Path(SAVE_CHECKPOINT_DIR) / path_suffix
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir = str(checkpoint_dir)
print('Saving checkpoints to', checkpoint_dir)

def save(weights, idx):
    ckpt_path = os.path.join(checkpoint_dir, 'checkpoint_{:05d}'.format(idx))
    print(f'Saving checkpoint to {ckpt_path}')
    checkpointer.save(ckpt_path, weights)
    checkpointer.wait_until_finished()
    return ckpt_path

# ── Training Loop ──────────────────────────────────────────────────────────────
print('Starting training...')
loss = []
step = 0
for step in range(NUM_TRAIN_STEPS):
    try:
        batch = next(ds_iter)
    except StopIteration:
        print('Dataset exhausted')
        break
    batch = bin_batch_targets(batch)  # FIX: bin targets to 128bp resolution
    with jax.set_mesh(mesh):
        batch = jax.device_put(batch, data_sharding)
        params, state, opt_state, scalars = train_step(
            params, state, opt_state, batch
        )
    loss.append(float(scalars['loss']))  # FIX: pull scalar to host to avoid device memory accumulation
    if step % 10 == 1:
        print(f'Step {step}, loss: {loss[-1]}')
    # FIX: Save intermediate checkpoints so progress isn't lost if job is killed
    if step > 0 and step % CHECKPOINT_INTERVAL == 0:
        save((params, state), step)

ckpt_path = save((params, state), step + 1)
print(f'Training complete. Checkpoint saved to {ckpt_path}')
