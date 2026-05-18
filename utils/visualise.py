import os
import torch
import matplotlib.pyplot as plt
import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from utils.dataset_utils import get_dataloaders


def visualise(
    model,
    dataset_roots,
    save_dir,
    crop_size    = 256,
    num_images   = 3,
    scenes_per_type = 2,
    sample_steps = 10,
    test = False,
):
    """
    Runs the model on sample scenes from each degradation type and saves
    side-by-side (input / restored / ground truth) figures.

    Args:
        model          All_In_One  — loaded and on the correct device
        dataset_roots  dict        — {'haze': path, 'snow': path, 'rain': path}
        save_dir       str         — folder where PNGs are written
        crop_size      int         — must match the crop used during training
        num_images     int         — total frames per sample (key + aux)
        scenes_per_type int        — distinct scenes to visualise per degradation type
        sample_steps   int         — DDIM steps for inference
    """
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)

    psnr_metric  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)

    model.eval()

    def _collect_scenes(root, n):
        _, _, loader = get_dataloaders(
            [root], crop_size=crop_size, batch_size=1,
            num_workers=2, num_images=num_images,
        )
        seen = {}
        for batch in loader:
            scene = batch['scene'][0]
            if scene not in seen:
                seen[scene] = batch
            if len(seen) == n:
                break
        return list(seen.values())

    sample_idx = 0
    for dtype, root in dataset_roots.items():
        for batch in _collect_scenes(root, scenes_per_type):
            if test:
                sample_idx += 1
                key_frame    = batch['key_frame'].to(device)
                aux_frames   = batch['aux_frames'].to(device)
                gt_key_frame = batch['gt_key_frame'].to(device)
                scene_name   = batch['scene'][0]
                if dtype == 'haze':
                    continue
                
                sam = sample_steps / 10
                for f in range(int(sam)):
                    f = (f*10)
                    with torch.no_grad():
                        J = model(key_frame=key_frame, aux_frames=aux_frames,
                                    sample_steps=f)

                    psnr_score  = psnr_metric(J, gt_key_frame).item()
                    ssim_score  = ssim_metric(J, gt_key_frame).item()
                    lpips_score = lpips_metric(J, gt_key_frame).item()

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(key_frame[0].cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[0].set_title('Input')
                    axes[0].axis('off')
                    axes[1].imshow(J[0].cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[1].set_title(
                        f'Restored\nPSNR: {psnr_score:.2f} dB  '
                        f'SSIM: {ssim_score:.4f}  LPIPS: {lpips_score:.4f}, sample_steps: {f}'
                    )
                    axes[1].axis('off')
                    axes[2].imshow(gt_key_frame[0].cpu().permute(1, 2, 0).clamp(0, 1))
                    axes[2].set_title('Ground Truth')
                    axes[2].axis('off')
                    fig.suptitle(f'[{dtype}]  Scene: {scene_name}', fontsize=13)
                    plt.tight_layout()

                    fname = f'{sample_idx:02d}_{dtype}_{scene_name}.png'
                    fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
                    plt.show()

                    print(f'Sample {sample_idx} [{dtype}]  Scene: {scene_name}  '
                        f'PSNR: {psnr_score:.2f} dB  SSIM: {ssim_score:.4f}  '
                        f'LPIPS: {lpips_score:.4f}  → {fname}')
                    
            else:
                sample_idx += 1
                key_frame    = batch['key_frame'].to(device)
                aux_frames   = batch['aux_frames'].to(device)
                gt_key_frame = batch['gt_key_frame'].to(device)
                scene_name   = batch['scene'][0]
                
                with torch.no_grad():
                    J = model(key_frame=key_frame, aux_frames=aux_frames,
                            sample_steps=sample_steps)

                psnr_score  = psnr_metric(J, gt_key_frame).item()
                ssim_score  = ssim_metric(J, gt_key_frame).item()
                lpips_score = lpips_metric(J, gt_key_frame).item()

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(key_frame[0].cpu().permute(1, 2, 0).clamp(0, 1))
                axes[0].set_title('Input')
                axes[0].axis('off')
                axes[1].imshow(J[0].cpu().permute(1, 2, 0).clamp(0, 1))
                axes[1].set_title(
                    f'Restored\nPSNR: {psnr_score:.2f} dB  '
                    f'SSIM: {ssim_score:.4f}  LPIPS: {lpips_score:.4f}'
                )
                axes[1].axis('off')
                axes[2].imshow(gt_key_frame[0].cpu().permute(1, 2, 0).clamp(0, 1))
                axes[2].set_title('Ground Truth')
                axes[2].axis('off')
                fig.suptitle(f'[{dtype}]  Scene: {scene_name}', fontsize=13)
                plt.tight_layout()

                fname = f'{sample_idx:02d}_{dtype}_{scene_name}.png'
                fig.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
                plt.show()

                print(f'Sample {sample_idx} [{dtype}]  Scene: {scene_name}  '
                    f'PSNR: {psnr_score:.2f} dB  SSIM: {ssim_score:.4f}  '
                    f'LPIPS: {lpips_score:.4f}  → {fname}')

    print(f'\nAll figures saved to: {save_dir}')
