

# Comparing results experiment — All-In-One Video Restoration Evaluation

# Patch deformable-conv, clone repos
deform_conv_code = '''
import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

class DCN_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, deformable_groups=1, bias=True, extra_offset_mask=True):
        super(DCN_layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        self.extra_offset_mask = extra_offset_mask
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels * 2,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size, stride=_pair(self.stride), padding=_pair(self.padding),
            bias=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_offset()
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input_feat, inter):
        feat_degradation = torch.cat([input_feat, inter], dim=1)
        out = self.conv_offset_mask(feat_degradation)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(
            input_feat.contiguous(), offset, self.weight,
            bias=self.bias, stride=self.stride,
            padding=self.padding, dilation=self.dilation, mask=mask)
'''

import os, sys, subprocess
import pandas as pd

from model.model import ARCNet

import torch
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

for repo, url in [
    ('/content/MCDRNet',  'https://github.com/HyaLii/MCDRNet.git'),
    ('/content/BioIR',    'https://github.com/c-yn/BioIR.git'),
    ('/content/AirNet',   'https://github.com/XLearning-SCU/2022-CVPR-AirNet.git'),
    ('/content/PromptIR', 'https://github.com/va1shn9v/PromptIR.git'),
]:
    if not os.path.exists(repo):
        subprocess.run(['git', 'clone', url, repo], check=True)
        print(f'Cloned {repo}.')
    else:
        print(f'{repo} already cloned.')

for path in ['/content/AirNet/net/deform_conv.py',
             '/content/MCDRNet/net/deform_conv.py']:
    with open(path, 'w') as f:
        f.write(deform_conv_code)
print('Patches applied.')

# Import model classes & load dataset
def import_model(repo_path, class_name):
    for key in list(sys.modules.keys()):
        if key.startswith('net') or key.startswith('option'):
            del sys.modules[key]
    sys.path.insert(0, repo_path)
    from net.model import __dict__ as model_dict
    cls = model_dict[class_name]
    sys.path.remove(repo_path)
    return cls

AirNet   = import_model('/content/AirNet',           'AirNet')
PromptIR = import_model('/content/PromptIR',         'PromptIR')
MCDRNet  = import_model('/content/MCDRNet',          'MCDRNet')
BioIR    = import_model('/content/BioIR/All_in_One', 'BioIR')

# Instantiate models & load checkpoints
import argparse

airnet_opt = argparse.Namespace(
    encoder_dim=256, encoder_depth=1, encoder_num_heads=8,
    de_type=['derain', 'dehaze', 'desnow'],
    patch_size=256, cuda=str(device), lr=2e-4, batch_size=6, num_workers=8,
)

mcdr_opt = argparse.Namespace(
    encoder_dim=128, encoder_depth=1, encoder_num_heads=8,
    patch_size=128, de_type=['derain', 'dehaze', 'desnow'],
    epochs=300, epochs_encoder=60, lr=1e-3, batch_size=16, num_workers=10,
    cuda=str(device),
    ckpt_path='/content/drive/MyDrive/MasterDataset/checkpoints/MCDRNet/',
)

checkpoint_path = '/content/drive/MyDrive/MasterDataset/checkpoints'

all_in_one = ARCNet().to(device)
airnet     = AirNet(airnet_opt).to(device)
promptir   = PromptIR(decoder=True).to(device)
mcdrnet    = MCDRNet(mcdr_opt).to(device)
bioir      = BioIR().to(device)

def LoadBestModel(models):
    for model in models:
        name = type(model).__name__.lower()
        if 'air' in name:
            path = os.path.join(checkpoint_path, 'AirNet/AirNet_best_model.pth')
        elif 'bio' in name:
            path = os.path.join(checkpoint_path, 'BioIR/BioIR_best.pth')
        elif 'promp' in name:
            path = os.path.join(checkpoint_path, 'ProptIR/promptir_best.pth')
        elif 'mcdr' in name:
            path = os.path.join(checkpoint_path, 'MCDRNet/best_model.pth')
        elif 'all_in_one' in name:
            path = os.path.join(checkpoint_path, 'ARCNet/best_model.pth')
            model.load_checkpoint(path)
            print(f'Model {name}, LOADED')
            continue
        else:
            print(f'{name} not found')
            continue
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            print(f'Model {name}, LOADED')

LoadBestModel([all_in_one, airnet, promptir, mcdrnet, bioir])

# Metrics & per-degradation test loaders

psnr_metric  = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)

from utils.dataset_utils import get_dataloaders

crop_size = 256
batch_size = 1
num_workers = 2
_, _, test_haze = get_dataloaders(['/content/extracted_datasets/REVIDE-haze'], batch_size=batch_size, num_workers=num_workers, num_images = 6, crop_size = crop_size)
_, _, test_rain = get_dataloaders(['/content/extracted_datasets/SPAC'],        batch_size=batch_size, num_workers=num_workers, num_images = 6, crop_size = crop_size)
_, _, test_snow = get_dataloaders(['/content/extracted_datasets/RVSD_train'],  batch_size=batch_size, num_workers=num_workers, num_images = 6, crop_size = crop_size)

degradation_loaders = {
    'Dehazing':  test_haze,
    'Deraining': test_rain,
    'Desnowing': test_snow,
}

# Forward functions & models dict
def airnet_forward(model, key, auxs):
    return model(x_query=key, x_key=key)[0].unsqueeze(0)

def standard_forward(model, key, auxs):
    restored = model(key)
    return restored.unsqueeze(0) if restored.dim() == 3 else restored

def all_in_one_forward(model, key, auxs):
    return model.forward(key_frame=key, aux_frames=auxs, sample_steps=10)

models = {
    'All_In_One': (all_in_one, all_in_one_forward),
    'AirNet':     (airnet,     airnet_forward),
    'PromptIR':   (promptir,   standard_forward),
    'MCDRNet':    (mcdrnet,    airnet_forward),
    'BioIR':      (bioir,      standard_forward),
}

# Evaluate_on_loader (returns mean + std for error bars)
import numpy as np

@torch.no_grad()
def evaluate_on_loader(model, loader, forward_fn=None):
    model.eval()
    psnr_vals, ssim_vals, lpips_vals = [], [], []
    for batch in loader:
        key  = batch['key_frame'].to(device)
        gt   = batch['gt_key_frame'].to(device)
        auxs = batch['aux_frames'].to(device)
        restored = forward_fn(model, key, auxs) if forward_fn else model(key)
        restored = torch.clamp(restored, 0.0, 1.0)
        psnr_vals.append(psnr_metric(restored, gt).item())
        ssim_vals.append(ssim_metric(restored, gt).item())
        lpips_vals.append(lpips_metric(restored*2-1, gt*2-1).mean().item())
    return {
        'PSNR':      round(float(np.mean(psnr_vals)),  2),
        'SSIM':      round(float(np.mean(ssim_vals)),  4),
        'LPIPS':     round(float(np.mean(lpips_vals)), 4),
        'PSNR_std':  round(float(np.std(psnr_vals)),   4),
        'SSIM_std':  round(float(np.std(ssim_vals)),   6),
        'LPIPS_std': round(float(np.std(lpips_vals)),  6),
    }

# Run evaluation
results = {}

for deg_name, loader in degradation_loaders.items():
    results[deg_name] = {}
    print(f'\nEvaluating on {deg_name}...')
    for model_name, (m, fwd) in models.items():
        print(f'  {model_name}...', end=' ')
        metrics = evaluate_on_loader(m, loader, forward_fn=fwd)
        results[deg_name][model_name] = metrics
        print(f"PSNR: {metrics['PSNR']:.2f}  SSIM: {metrics['SSIM']:.4f}  LPIPS: {metrics['LPIPS']:.4f}")

# Cell 9 — Per-degradation styled tables
def make_table(deg_name):
    rows = []
    for model_name, metrics in results[deg_name].items():
        rows.append({
            'Model':     model_name,
            'PSNR (dB)': metrics['PSNR'],
            'SSIM':      metrics['SSIM'],
            'LPIPS':     metrics['LPIPS'],
        })
    df = pd.DataFrame(rows).set_index('Model')
    styled = df.style \
        .highlight_max(subset=['PSNR (dB)', 'SSIM'], color='lightgreen') \
        .highlight_min(subset=['LPIPS'],              color='lightgreen') \
        .set_caption(f'{deg_name} — Model Comparison') \
        .format({'PSNR (dB)': '{:.2f}', 'SSIM': '{:.4f}', 'LPIPS': '{:.4f}'})
    return df, styled

for deg_name in degradation_loaders:
    df, styled = make_table(deg_name)
    print(f'\n{"="*50}\n{deg_name} Results\n{"="*50}')
    display(styled)

# Combined summary table
print(f'\n{"="*50}\nCombined Summary (avg across all degradations)\n{"="*50}')

summary_rows = []
for model_name in models:
    avg_psnr  = sum(results[d][model_name]['PSNR']  for d in results) / len(results)
    avg_ssim  = sum(results[d][model_name]['SSIM']  for d in results) / len(results)
    avg_lpips = sum(results[d][model_name]['LPIPS'] for d in results) / len(results)
    summary_rows.append({
        'Model':     model_name,
        'Avg PSNR':  round(avg_psnr,  2),
        'Avg SSIM':  round(avg_ssim,  4),
        'Avg LPIPS': round(avg_lpips, 4),
    })

summary_df = pd.DataFrame(summary_rows).set_index('Model')
summary_styled = summary_df.style \
    .highlight_max(subset=['Avg PSNR', 'Avg SSIM'], color='lightgreen') \
    .highlight_min(subset=['Avg LPIPS'],             color='lightgreen') \
    .set_caption('Overall Summary — All Degradations') \
    .format({'Avg PSNR': '{:.2f}', 'Avg SSIM': '{:.4f}', 'Avg LPIPS': '{:.4f}'})
display(summary_styled)


"""---
## Visual Comparison Table

Generates a publication-style figure with `N_SAMPLES_PER_DEG` rows per degradation type.

Columns: **Input | AirNet | PromptIR | MCDRNet | BioIR | All\_In\_One | GT**

Saved to `comparison_table.png` on Drive.
"""


# Imports & configuration
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COLUMNS          = ['Input', 'AirNet', 'PromptIR', 'MCDRNet', 'BioIR', 'All_In_One', 'GT']
N_COLS           = len(COLUMNS)
N_SAMPLES_PER_DEG = 2
SAVE_PATH        = '/content/drive/MyDrive/MasterDataset/comparison_table.png'

DEGRADATION_LOADERS = {
    'Dehazing':  test_haze,
    'Deraining': test_rain,
    'Desnowing': test_snow,
}

MODELS = {
    'All_In_One': (all_in_one, all_in_one_forward),
    'AirNet':     (airnet,     airnet_forward),
    'PromptIR':   (promptir,   standard_forward),
    'MCDRNet':    (mcdrnet,    airnet_forward),
    'BioIR':      (bioir,      standard_forward),
}


def to_np(t):
    """(1,C,H,W) or (C,H,W) tensor in [0,1]  →  H×W×3 uint8."""
    t = t.squeeze(0).clamp(0, 1).cpu().float()
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

@torch.no_grad()
def collect_samples(degradation_loaders, models, n_samples=2):
    """
    Returns samples[deg_name][i] = {'Input': ndarray, 'GT': ndarray,
                                     'AirNet': ndarray, ...}
    """
    samples = {}
    for deg_name, loader in degradation_loaders.items():
        print(f'\n[{deg_name}] collecting {n_samples} samples...')
        samples[deg_name] = []
        total = len(loader)
        if total == 0:
            print(f'  ⚠  loader is empty for {deg_name}')
            continue

        actual = min(n_samples, total)
        step = total / actual
        target_indices = set(int(step * i + step / 2) for i in range(actual))

        for batch_idx, batch in enumerate(loader):
            if batch_idx not in target_indices:
                continue

            key  = batch['key_frame'].to(device)
            gt   = batch['gt_key_frame'].to(device)
            auxs = batch['aux_frames'].to(device)
            entry = {'Input': to_np(key), 'GT': to_np(gt)}
            for model_name, (m, fwd) in models.items():
                m.eval()
                restored = fwd(m, key, auxs)
                restored = torch.clamp(restored, 0.0, 1.0)
                entry[model_name] = to_np(restored)
                print(f'  {model_name} done', end='  ')
            print()
            samples[deg_name].append(entry)
    return samples

def build_comparison_figure(samples, save_path=SAVE_PATH, dpi=150):
    """
    Rows  = degradation × n_samples_per_deg
    Cols  = Input | AirNet | PromptIR | MCDRNet | BioIR | All_In_One | GT
    Uses standard matplotlib tab10 colour cycle for column borders & labels.
    """
    plt.rcdefaults()
    TAB = plt.rcParams['axes.prop_cycle'].by_key()['color']

    BG       = 'white'
    HDR_BG   = '#f5f5f5'
    DEG_BG   = '#fafafa'
    BORDER   = '#cccccc'
    ACCENT   = 'black'
    GT_COLOR = TAB[1]   
    MODEL_COLORS = {
        'Input':     '#888888',
        'AirNet':    TAB[0],   
        'PromptIR':  TAB[2],   
        'MCDRNet':   TAB[3],   
        'BioIR':     TAB[4],   
        'All_In_One':TAB[6],   
        'GT':        TAB[1],
    }

    deg_names  = list(samples.keys())
    n_deg      = len(deg_names)
    n_samples  = max(len(v) for v in samples.values())
    total_rows = n_deg * n_samples

    cell_h, cell_w = 2.8, 2.8
    header_h    = 0.55
    deg_label_w = 0.55

    fig_w = deg_label_w + N_COLS * cell_w + 0.2
    fig_h = header_h + total_rows * cell_h + 0.2
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    outer = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[header_h, total_rows * cell_h],
        hspace=0, left=0.01, right=0.99, top=0.99, bottom=0.01,
    )

    #  Header row 
    hdr_ax = fig.add_subplot(outer[0])
    hdr_ax.set_facecolor(HDR_BG)
    hdr_ax.set_xlim(0, 1); hdr_ax.set_ylim(0, 1); hdr_ax.axis('off')

    sidebar_frac = deg_label_w / fig_w
    col_w_frac   = (1 - sidebar_frac) / N_COLS

    for ci, col in enumerate(COLUMNS):
        x     = sidebar_frac + (ci + 0.5) * col_w_frac
        color = GT_COLOR if col == 'GT' else MODEL_COLORS.get(col, ACCENT)
        hdr_ax.text(x, 0.5, col, ha='center', va='center',
                    fontsize=15, fontweight='bold', color=color,
                    transform=hdr_ax.transAxes)
        bar_x0 = sidebar_frac + ci * col_w_frac + 0.01 * col_w_frac
        bar_x1 = sidebar_frac + (ci+1) * col_w_frac - 0.01 * col_w_frac
        hdr_ax.axhline(y=0.08, xmin=bar_x0, xmax=bar_x1,
                       color=color, linewidth=1.5, alpha=0.6)
    hdr_ax.axhline(y=0.0, color=ACCENT, linewidth=0.8, alpha=0.4)

    #  Data rows 
    data_gs = gridspec.GridSpecFromSubplotSpec(
        total_rows, N_COLS + 1,
        subplot_spec=outer[1],
        hspace=0.02, wspace=0.02,
        width_ratios=[deg_label_w / (N_COLS * cell_w)] + [1] * N_COLS,
    )

    for di, deg_name in enumerate(deg_names):
        deg_samples = samples[deg_name]
        for si, entry in enumerate(deg_samples):
            global_row = di * n_samples + si

            lbl_ax = fig.add_subplot(data_gs[global_row, 0])
            lbl_ax.set_facecolor(DEG_BG); lbl_ax.axis('off')
            if si == 0 and di > 0:
                lbl_ax.axhline(y=1.0, color=BORDER, linewidth=1.2)
            if si == len(deg_samples) // 2 or len(deg_samples) == 1:
                lbl_ax.text(0.5, 0.5, deg_name, ha='center', va='center',
                            rotation=90, fontsize=15, fontweight='bold',
                            color=ACCENT, transform=lbl_ax.transAxes)
            for ci, col in enumerate(COLUMNS):
                ax = fig.add_subplot(data_gs[global_row, ci + 1])
                ax.set_facecolor('white'); ax.axis('off')
                img = entry.get(col)
                if img is not None:
                    ax.imshow(img, aspect='auto', interpolation='lanczos')
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            color='#888888', fontsize=20, transform=ax.transAxes)
                border_color = GT_COLOR if col == 'GT' else MODEL_COLORS.get(col, BORDER)
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(1.2 if col == 'GT' else 0.6)
                    spine.set_alpha(0.9 if col == 'GT' else 0.35)
                    spine.set_visible(True)
                if si == 0 and di > 0:
                    ax.axhline(y=ax.get_ylim()[1] if ax.images else 1,
                               color=BORDER, linewidth=1.0, alpha=0.8)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                facecolor=BG, edgecolor='none')
    print(f'\n✅  Saved → {save_path}')
    plt.show()
    plt.close(fig)
    return save_path

# Run: collect samples & build figure
samples = collect_samples(DEGRADATION_LOADERS, MODELS, n_samples=N_SAMPLES_PER_DEG)
build_comparison_figure(samples, save_path=SAVE_PATH, dpi=180)




# Inference speed benchmark
import time
from torch.utils.data import ConcatDataset, DataLoader

#  Combine all three test sets into one flat loader 
combined_dataset = ConcatDataset([
    test_haze.dataset,
    test_rain.dataset,
    test_snow.dataset,
])
timing_loader = DataLoader(combined_dataset, batch_size=1,
                           shuffle=False, num_workers=2)
print(f'Total images for timing: {len(timing_loader)}')

USE_CUDA = device.type == 'cuda'

@torch.no_grad()
def benchmark_model(model, forward_fn, loader, warmup=5):
    """
    Runs forward_fn on every batch in loader (batch_size=1).
    The first `warmup` images are discarded so GPU JIT / caching
    don't inflate the first few timings.
    Returns a list of per-image latencies in milliseconds.
    """
    model.eval()
    latencies = []

    for i, batch in enumerate(loader):
        key  = batch['key_frame'].to(device)
        auxs = batch['aux_frames'].to(device)

        if USE_CUDA:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt   = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_evt.record()
            _ = forward_fn(model, key, auxs)
            end_evt.record()
            torch.cuda.synchronize()
            elapsed_ms = start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            _ = forward_fn(model, key, auxs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if i >= warmup:
            latencies.append(elapsed_ms)

    return latencies

#  Run benchmark for every model 
timing_results = {}

for model_name, (m, fwd) in models.items():
    print(f'Timing {model_name}...', end=' ', flush=True)
    lats = benchmark_model(m, fwd, timing_loader)
    timing_results[model_name] = {
        'n':    len(lats),
        'mean': float(np.mean(lats)),
        'std':  float(np.std(lats)),
        'min':  float(np.min(lats)),
        'max':  float(np.max(lats)),
    }
    print(f"avg {timing_results[model_name]['mean']:.1f} ms  "
          f"std {timing_results[model_name]['std']:.1f} ms  "
          f"(n={len(lats)})")

#  Build summary lines 
device_tag = f'GPU ({torch.cuda.get_device_name(0)})' if USE_CUDA else 'CPU'
sep = '=' * 66
header = f"{'Model':<18} {'N':>5} {'Mean (ms)':>10} {'Std':>8} {'Min':>8} {'Max':>8}"
rows = [
    f"{name:<18} {t['n']:>5} {t['mean']:>10.2f} "
    f"{t['std']:>8.2f} {t['min']:>8.2f} {t['max']:>8.2f}"
    for name, t in sorted(timing_results.items(), key=lambda x: x[1]['mean'])
]
footer = f"Device: {device_tag}  |  warmup images excluded: 5"

summary_lines = [sep, header, sep] + rows + [sep, footer]

#  Print to notebook 
print('\n'.join(summary_lines))

#  Save to .txt 
txt_path = './inference_speed.txt'
with open(txt_path, 'w') as f:
    f.write('Inference Speed Benchmark\n')
    f.write(f'Run date : {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    f.write(f'Total images timed per model: {len(timing_loader) - 5}  (5 warmup excluded)\n\n')
    f.write('\n'.join(summary_lines))
    f.write('\n')
print(f'\n✅  Saved → {txt_path}')

#  Styled DataFrame 
timing_df = pd.DataFrame(timing_results).T[['n', 'mean', 'std', 'min', 'max']]
timing_df.index.name = 'Model'
timing_df = timing_df.rename(columns={
    'n': 'Images', 'mean': 'Mean ms', 'std': 'Std ms',
    'min': 'Min ms', 'max': 'Max ms',
})
timing_styled = timing_df.style \
    .highlight_min(subset=['Mean ms'], color='lightgreen') \
    .highlight_max(subset=['Mean ms'], color='#ffcccc') \
    .set_caption(f'Inference Speed Benchmark — {device_tag}') \
    .format({'Images': '{:.0f}', 'Mean ms': '{:.2f}', 'Std ms': '{:.2f}',
             'Min ms': '{:.2f}', 'Max ms': '{:.2f}'})
display(timing_styled)

