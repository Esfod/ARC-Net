"""Ablation_Study.ipynb"""

import sys, os, json, copy, math
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import lpips
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


from model.model      import ARCNet
from model.diffusion  import GaussianDiffusion, DiffusionUNet
from model.degradation_encoder import DEG_VEC_DIM
from utils.dataset_utils  import get_dataloaders
from utils.training_utils import EarlyStopping
from utils.train import combined_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)



# Path to your trained checkpoint
CHECKPOINT_PATH = 'checkpoints/best_model.pth'

DATA_ROOTS = [
    '/content/extracted_datasets/REVIDE-haze',
    '/content/extracted_datasets/RVSD_train',
    '/content/extracted_datasets/SPAC',
]

# Where to save results and sample images
OUTPUT_DIR = './ablation_results'

# Hyperparameters for data loading and inference
CROP_SIZE       = 256
NUM_IMAGES      = 6
DIFFUSION_STEPS = 1000
SAMPLE_STEPS    = 10     
NUM_WORKERS     = 2

# Number of distinct scenes to visualise per degradation type (haze / snow / rain)
SAMPLES_PER_TYPE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)
print('Output dir:', OUTPUT_DIR)

_, _, test_loader = get_dataloaders(
    DATA_ROOTS,
    crop_size=CROP_SIZE,
    batch_size=1,
    num_workers=NUM_WORKERS,
    num_images=NUM_IMAGES,
)
print(f'Test samples: {len(test_loader)}')

#  Load model checkpoint 
model = ARCNet(CROP_SIZE, NUM_IMAGES, DIFFUSION_STEPS).to(device)
model.load_checkpoint(CHECKPOINT_PATH)
model.eval()
print('Model ready.')


#  Metrics 
psnr_metric  = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)


#  Context-building helpers 

def _full_context(F_0, aligned):
    """Standard: cat key features with mean of flow-aligned aux features."""
    return {
        scale: torch.cat(
            [F_0[scale], torch.stack([a[scale] for a in aligned]).mean(0)],
            dim=1,
        )
        for scale in F_0
    }

def _zero_context(context):
    """Replace all context tensors with zeros — ablates Swin features."""
    return {k: torch.zeros_like(v) for k, v in context.items()}

def _no_aux_context(F_0):
    """Key frame only: zero out the aux half — ablates temporal context."""
    return {scale: torch.cat([F_0[scale], torch.zeros_like(F_0[scale])], dim=1)
            for scale in F_0}

def _unwarped_context(F_0, F_k):
    """Aux features used but NOT aligned — ablates optical flow warping."""
    mean_aux = {
        scale: torch.stack([F_k[i][scale] for i in range(len(F_k))]).mean(0)
        for scale in F_0
    }
    return {scale: torch.cat([F_0[scale], mean_aux[scale]], dim=1) for scale in F_0}


#  Single-batch inference 

@torch.no_grad()
def run_variant(key_frame, aux_frames, *,
                use_deg_vec=True, use_alignment=True,
                use_aux_frames=True, use_context=True):
    """Run one ablation variant on a single (B,3,H,W) batch."""
    enc = model.encoder
    deg = model.degradation_encoder
    aln = model.alignment
    dif = model.diffusion

    deg_vec = deg(key_frame) if use_deg_vec else None
    F_0, F_k = enc(key_frame, aux_frames)

    if not use_context:
        context = _zero_context(_full_context(F_0, [F_0]))
    elif not use_aux_frames:
        context = _no_aux_context(F_0)
    elif not use_alignment:
        context = _unwarped_context(F_0, F_k)
    else:
        aligned, _ = aln(key_frame, aux_frames, F_k)
        context = _full_context(F_0, aligned)

    return dif.sample(key_frame, context, T=SAMPLE_STEPS, deg_vec=deg_vec)


#  Full test-set evaluation 

@torch.no_grad()
def evaluate_variant(loader, **flags):
    """Evaluate one ablation variant over the full loader."""
    model.encoder.eval()
    model.degradation_encoder.eval()
    model.alignment.eval()
    model.diffusion.eval()

    total_psnr = total_ssim = total_lpips = 0.0
    for batch in loader:
        kf  = batch['key_frame'].to(device)
        aux = batch['aux_frames'].to(device)
        gt  = batch['gt_key_frame'].to(device)
        out = run_variant(kf, aux, **flags)
        total_psnr  += psnr_metric(out, gt).item()
        total_ssim  += ssim_metric(out, gt).item()
        total_lpips += lpips_metric(out * 2 - 1, gt * 2 - 1).mean().item()

    n = len(loader)
    return total_psnr / n, total_ssim / n, total_lpips / n


#  Results table 

def print_table(results):
    W = 36
    print('\n' + '=' * (W + 50))
    print(f"{'Variant':<{W}} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Score':>8}")
    print('=' * (W + 50))
    base = results[0]
    for r in results:
        tag = '' if r is base else f"  Δ={r['score']-base['score']:+.3f}"
        print(f"{r['name']:<{W}} {r['psnr']:>8.3f} {r['ssim']:>8.4f} "
              f"{r['lpips']:>8.4f} {r['score']:>8.4f}{tag}")
    print('=' * (W + 50))
    if len(results) > 1:
        print(f"\nImpact vs Full Model:")
        print(f"  {'Variant':<{W-2}} {'ΔPSNR':>8} {'ΔSSIM':>8} {'ΔLPIPS':>8} {'ΔScore':>8}")
        print('  ' + '-' * (W + 42))
        for r in results[1:]:
            print(f"  {r['name']:<{W-2}} "
                  f"{r['psnr']-base['psnr']:>+8.3f} "
                  f"{r['ssim']-base['ssim']:>+8.4f} "
                  f"{r['lpips']-base['lpips']:>+8.4f} "
                  f"{r['score']-base['score']:>+8.4f}")


print('Helper functions defined.')

#  Save visual samples 
VARIANTS = [
    dict(label='Full Model',         slug='01_Full_Model',
         use_deg_vec=True,  use_alignment=True,  use_aux_frames=True,  use_context=True),
    dict(label='w/o DegradationEnc', slug='02_wo_DegradationEncoder',
         use_deg_vec=False, use_alignment=True,  use_aux_frames=True,  use_context=True),
    dict(label='w/o MultiFrameEnc',  slug='03_wo_MultiScaleContext',
         use_deg_vec=True,  use_alignment=True,  use_aux_frames=True,  use_context=False),
    dict(label='w/o AlignmentMod',   slug='03_wo_AlignmentModule',
         use_deg_vec=True,  use_alignment=False, use_aux_frames=True,  use_context=True),
    dict(label='w/o AuxFrames',      slug='05 _wo_AuxFrames',
         use_deg_vec=True,  use_alignment=True,  use_aux_frames=False, use_context=True),
]

samples_dir     = os.path.join(OUTPUT_DIR, 'samples')
per_variant_dir = os.path.join(samples_dir, 'per_variant')
os.makedirs(samples_dir, exist_ok=True)
for v in VARIANTS:
    os.makedirs(os.path.join(per_variant_dir, v['slug']), exist_ok=True)

model.encoder.eval()
model.degradation_encoder.eval()
model.alignment.eval()
model.diffusion.eval()

def _to_np(t):
    return t.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).numpy()

def _label(ax, title, caption=None):
    """Title above image; metrics overlaid at bottom-centre inside the image."""
    ax.set_title(title, fontsize=15, pad=6)
    if caption:
        ax.text(0.5, 0.03, caption,
                fontsize=13, ha='center', va='bottom',
                transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.75,
                          edgecolor='none', boxstyle='round,pad=0.3'))
    ax.axis('off')

sample_idx = 0
for root in DATA_ROOTS:
    _, _, root_loader = get_dataloaders(
        [root], crop_size=CROP_SIZE, batch_size=1,
        num_workers=NUM_WORKERS, num_images=NUM_IMAGES,
    )

    seen = {}
    for batch in root_loader:
        scene = batch['scene'][0]
        if scene not in seen:
            seen[scene] = batch
        if len(seen) == SAMPLES_PER_TYPE:
            break

    for scene, batch in seen.items():
        sample_idx += 1
        kf    = batch['key_frame'].to(device)
        aux   = batch['aux_frames'].to(device)
        gt    = batch['gt_key_frame'].to(device)
        dtype = batch['degradation_type'][0]
        fname = batch['frame_idx'][0].replace('.png','').replace('.jpg','')
        base  = f'{sample_idx:02d}_{dtype}_{scene}_{fname}'

        outputs = [
            run_variant(kf, aux,
                        use_deg_vec=v['use_deg_vec'],
                        use_alignment=v['use_alignment'],
                        use_aux_frames=v['use_aux_frames'],
                        use_context=v['use_context'])
            for v in VARIANTS
        ]

        for v, out in zip(VARIANTS, outputs):
            save_image(out.cpu(),
                       os.path.join(per_variant_dir, v['slug'], f'{base}.png'))

        n_total   = 2 + len(VARIANTS)
        n_cols_2r = math.ceil(n_total / 2)
        fig, axes = plt.subplots(2, n_cols_2r,
                                 figsize=(4.5 * n_cols_2r, 12.0))
        ax_flat = axes.flatten()

        for ax in ax_flat[n_total:]:
            ax.axis('off')

        ax_flat[0].imshow(_to_np(kf))
        _label(ax_flat[0], 'Input (degraded)')

        ax_flat[1].imshow(_to_np(gt))
        _label(ax_flat[1], 'Ground Truth')

        for col, (v, out) in enumerate(zip(VARIANTS, outputs), start=2):
            p = psnr_metric(out, gt).item()
            s = ssim_metric(out, gt).item()
            l = lpips_metric(out * 2 - 1, gt * 2 - 1).mean().item()
            ax_flat[col].imshow(_to_np(out))
            _label(ax_flat[col],
                   title   = v['label'],
                   caption = f"PSNR {p:.2f}  SSIM {s:.3f}  LPIPS {l:.3f}")

        fig.subplots_adjust(
            left=0.02, right=0.98,
            top=0.96,  bottom=0.02,
            wspace=0.05, hspace=0.25,
        )

        comp_path = os.path.join(samples_dir, f'{base}.png')
        fig.savefig(comp_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'  [{sample_idx:02d}] [{dtype}] {scene}/{fname}  → saved')

print(f'\nAll samples saved to: {samples_dir}')

# Inference-time ablations (full test set) 
# Each variant is evaluated over the entire test loader.

ablation_variants = [
    dict(name='1. Full Model (baseline)',
         use_deg_vec=True,  use_alignment=True,  use_aux_frames=True,  use_context=True),
    dict(name='2. w/o MultiScale Encoder',
         use_deg_vec=True,  use_alignment=True,  use_aux_frames=True,  use_context=False),
    dict(name='3. w/o DegradationEncoder',
         use_deg_vec=False, use_alignment=True,  use_aux_frames=True,  use_context=True),
    dict(name='4. w/o AlignmentModule',
         use_deg_vec=True,  use_alignment=False, use_aux_frames=True,  use_context=True),
    dict(name='5. w/o AuxiliaryFrames',
         use_deg_vec=True,  use_alignment=True,  use_aux_frames=False, use_context=True),
]
results = []
for v in ablation_variants:
    name  = v.pop('name')
    print(f'Evaluating: {name} ...', end='', flush=True)
    p, s, l = evaluate_variant(test_loader, **v)
    sc = combined_score(p, s, l)
    results.append(dict(name=name, psnr=p, ssim=s, lpips=l, score=sc))
    print(f'  PSNR {p:.3f}  SSIM {s:.4f}  LPIPS {l:.4f}  Score {sc:.4f}')

print('Done.')

#  Print results table & save JSON 
print_table(results)

out_json = os.path.join(OUTPUT_DIR, 'ablation_results.json')
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results saved to {out_json}')