
import os
import time
import torch
import pandas as pd
from collections import defaultdict

from model.model import ARCNet
from utils.dataset_utils import get_dataloaders


device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

crop_size       = 256
num_images      = 3
diffusion_steps = 1000
sample_steps    = 10    
batch_size      = 1     
num_workers     = 4
num_batches     = None  # Set to an integer to limit the number of batches timed per dataset

checkpoint_dir  = './checkpoints'

# Per-dataset checkpoint paths and dataset roots
datasets = {
    'haze': {
        'root':       '/content/extracted_datasets/REVIDE-haze',
        'checkpoint': os.path.join(checkpoint_dir, 'best_model.pth'),
    },
    'snow': {
        'root':       '/content/extracted_datasets/RVSD_train',
        'checkpoint': os.path.join(checkpoint_dir, 'best_model.pth'),
    },
    'rain': {
        'root':       '/content/extracted_datasets/SPAC',
        'checkpoint': os.path.join(checkpoint_dir, 'best_model.pth'),
    },
}

"""## Timing helpers"""

def cuda_sync():
    """Ensure all CUDA kernels have finished before reading the clock."""
    if device.type == 'cuda':
        torch.cuda.synchronize()


def tick():
    cuda_sync()
    return time.perf_counter()


def tock(start):
    cuda_sync()
    return (time.perf_counter() - start) * 1000   # ms


@torch.no_grad()
def time_restoration(model, test_loader, sample_steps, max_batches=None):
    """
    Run inference on *test_loader* and record wall-clock time (ms) for each
    pipeline stage separately.

    Returns
    -------
    dict  stage_name -> list[float]  (one entry per batch)
    """
    model.eval()
    times = defaultdict(list)

    for batch_idx, batch in enumerate(test_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # --- unpack batch (same format as dataset_utils) ---
        degraded = batch['key_frame'].to(device)   # (B, 3, H, W)
        frames   = batch['aux_frames'].to(device)     # (B, K, 3, H, W)
        key_frame  = degraded
        aux_frames = frames

        #  1. Degradation encoder 
        t0 = tick()
        deg_vec = model.degradation_encoder(key_frame)
        times['degradation_encoder'].append(tock(t0))

        #  2. Feature extraction 
        t0 = tick()
        F_0, F_k = model.encoder(key_frame, aux_frames)
        times['feature_extraction'].append(tock(t0))

        #  3. Optical flow + feature alignment 
        t0 = tick()
        aligned, flows = model.alignment(key_frame, aux_frames, F_k)
        times['alignment'].append(tock(t0))

        #  4. Context fusion 
        t0 = tick()
        context = model._build_context(F_0, aligned)
        times['context_fusion'].append(tock(t0))

        #  5. Diffusion sampling 
        t0 = tick()
        _ = model.diffusion.sample(key_frame, context,
                                   T=sample_steps, deg_vec=deg_vec)
        times['diffusion_sampling'].append(tock(t0))

        #  6. Total (sum of all stages) 
        times['total'].append(
            times['degradation_encoder'][-1]
            + times['feature_extraction'][-1]
            + times['alignment'][-1]
            + times['context_fusion'][-1]
            + times['diffusion_sampling'][-1]
        )

        if (batch_idx + 1) % 10 == 0:
            print(f'  batch {batch_idx + 1:>4d}  |  '
                  f'total so far: {times["total"][-1]:.1f} ms')

    return dict(times)


def summarise(times):
    """Return a DataFrame with mean / std / min / max per stage (ms)."""
    import statistics
    rows = []
    stage_order = [
        'degradation_encoder',
        'feature_extraction',
        'alignment',
        'context_fusion',
        'diffusion_sampling',
        'total',
    ]
    for stage in stage_order:
        vals = times[stage]
        rows.append({
            'stage':   stage,
            'mean_ms': round(statistics.mean(vals), 2),
            'std_ms':  round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 2),
            'min_ms':  round(min(vals), 2),
            'max_ms':  round(max(vals), 2),
            'n':       len(vals),
        })
    return pd.DataFrame(rows).set_index('stage')

"""## Run timing — one dataset at a time"""

all_summaries = {}

for degradation_type, cfg in datasets.items():
    print('=' * 60)
    print(f'  Dataset: {degradation_type.upper()}  |  root: {cfg["root"]}')
    print('=' * 60)

    #  Build test loader for this single dataset 
    _, _, test_loader = get_dataloaders(
        [cfg['root']],
        crop_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
        num_images=num_images,
    )
    print(f'  Test batches available: {len(test_loader)}')
    print(f'  Timing on: {min(num_batches, len(test_loader)) if num_batches else len(test_loader)} batches\n')

    #  Load per-dataset checkpoint 
    model = ARCNet(crop_size, num_images, diffusion_steps).to(device)
    if os.path.exists(cfg['checkpoint']):
        model.load_checkpoint(cfg['checkpoint'])
    else:
        print(f'  [WARNING] Checkpoint not found at {cfg["checkpoint"]}.')
        print(f'  Timing with randomly initialised weights.\n')

    #  GPU warm-up (1 dummy forward pass, not counted) 
    dummy = next(iter(test_loader))
    with torch.no_grad():
        model(
            dummy['key_frame'].to(device),
            dummy['aux_frames'].to(device),
            sample_steps=sample_steps,
        )
    print('  Warm-up done. Starting timing...\n')

    #  Time inference 
    times = time_restoration(
        model, test_loader,
        sample_steps=sample_steps,
        max_batches=num_batches,
    )

    summary = summarise(times)
    all_summaries[degradation_type] = summary

    print(f'\n  Results for {degradation_type.upper()} (sample_steps={sample_steps}):')
    print(summary.to_string())
    print()

"""## Combined summary across all datasets"""

# Flatten into one DataFrame with a 'dataset' column for easy comparison
combined_rows = []
for dtype, df in all_summaries.items():
    tmp = df.reset_index()
    tmp.insert(0, 'dataset', dtype)
    combined_rows.append(tmp)

combined = pd.concat(combined_rows, ignore_index=True)

print('Combined timing summary (mean_ms)')
pivot = combined.pivot(index='stage', columns='dataset', values='mean_ms')
# Preserve stage order
stage_order = [
    'degradation_encoder',
    'feature_extraction',
    'alignment',
    'context_fusion',
    'diffusion_sampling',
    'total',
]
pivot = pivot.reindex(stage_order)
print(pivot.to_string())

"""## Bar chart — mean restoration time per stage × dataset"""

import matplotlib.pyplot as plt

# Exclude 'total' from the stacked bars so heights are additive
stages_to_plot = [
    'degradation_encoder',
    'feature_extraction',
    'alignment',
    'context_fusion',
    'diffusion_sampling',
]

plot_data = pivot.loc[stages_to_plot]   # shape: (stages, datasets)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#  Left: stacked bar per dataset 
ax = axes[0]
bottom = {ds: 0.0 for ds in plot_data.columns}
colors = plt.cm.tab10.colors
for i, stage in enumerate(stages_to_plot):
    vals = plot_data.loc[stage]
    ax.bar(vals.index, vals.values, bottom=[bottom[ds] for ds in vals.index],
           label=stage, color=colors[i])
    for ds in vals.index:
        bottom[ds] += vals[ds]

ax.set_title(f'Mean restoration time per stage\n(sample_steps={sample_steps})')
ax.set_ylabel('Time (ms)')
ax.set_xlabel('Dataset')
ax.legend(loc='upper left', fontsize=8)

#  Right: total time per dataset with error bars 
ax2 = axes[1]
total_means = [all_summaries[ds].loc['total', 'mean_ms'] for ds in all_summaries]
total_stds  = [all_summaries[ds].loc['total', 'std_ms']  for ds in all_summaries]
labels      = list(all_summaries.keys())

bars = ax2.bar(labels, total_means, yerr=total_stds,
               capsize=6, color=['steelblue', 'darkorange', 'seagreen'])
for bar, val in zip(bars, total_means):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(total_stds) * 0.1,
             f'{val:.1f} ms', ha='center', va='bottom', fontsize=9)

ax2.set_title(f'Total restoration time (mean ± std)\n(sample_steps={sample_steps})')
ax2.set_ylabel('Time (ms)')
ax2.set_xlabel('Dataset')

plt.tight_layout()
plt.savefig('restoration_timing.png', dpi=150)
plt.show()
print('Chart saved → restoration_timing.png')

