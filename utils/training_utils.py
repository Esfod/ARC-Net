# https://pythonguides.com/pytorch-early-stopping/

import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter +=1
            print('No improvement in score')

        if self.counter >= self.patience:
            self.early_stop = True
            print('Score has not improved       Stopping Training')

        return self.early_stop
    
    
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import lpips


DEGRADATION_TO_IDX = {'haze': 0, 'rain': 1, 'snow': 2}


def combined_score(psnr, ssim, lpips_val):
    return (psnr / 40.0) + ssim + (1.0 - lpips_val)


#  Phase 1: degradation encoder pre-training 

def train_Phase1(
    degradation_encoder,
    train_loader,
    val_loader,
    num_epochs     = 100,
    lr             = 1e-3,
    checkpoint_dir = None,
    es_patience    = 5,
    es_min_delta   = 0.5,
):
    """
    Pre-trains the DegradationEncoder with a temporary classification head.
    The head is discarded afterwards; only the encoder weights are kept.
    Encoder is frozen after this returns.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    degradation_encoder.to(device)

    from model.degradation_encoder import DEG_VEC_DIM
    head = nn.Linear(DEG_VEC_DIM, 3).to(device)

    optimizer = torch.optim.AdamW(
        list(degradation_encoder.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6,
    )
    early_stopping = EarlyStopping(patience=es_patience, min_delta=es_min_delta)




    best_val_acc = 0.0
    print('\n Phase 1: DegradationEncoder pre-training ')
    for epoch in range(num_epochs):
        degradation_encoder.train()
        head.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in train_loader:
            key_frame = batch['key_frame'].to(device)
            labels = torch.tensor(
                [DEGRADATION_TO_IDX[d] for d in batch['degradation_type']],
                device=device, dtype=torch.long,
            )
            optimizer.zero_grad()
            logits = head(degradation_encoder(key_frame))
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

        scheduler.step()
        train_acc = correct / total * 100

        degradation_encoder.eval()
        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for vbatch in val_loader:
                vkey    = vbatch['key_frame'].to(device)
                vlabels = torch.tensor(
                    [DEGRADATION_TO_IDX[d] for d in vbatch['degradation_type']],
                    device=device, dtype=torch.long,
                )
                vlogits      = head(degradation_encoder(vkey))
                val_correct += (vlogits.argmax(1) == vlabels).sum().item()
                val_total   += len(vlabels)
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0.0

        avg_loss = total_loss / len(train_loader)
        print(f'  Epoch [{epoch+1:3d}/{num_epochs}]  '
              f'Loss: {avg_loss:.4f}  '
              f'Train: {train_acc:.1f}%  Val: {val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if checkpoint_dir is not None:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(
                    degradation_encoder.state_dict(),
                    os.path.join(checkpoint_dir, 'degradation_encoder_best.pth'),
                )

        if early_stopping(val_acc):
            print(f'  Early stopping at epoch {epoch+1}.')
            break

    if checkpoint_dir is not None:
        best_path = os.path.join(checkpoint_dir, 'degradation_encoder_best.pth')
        if os.path.exists(best_path):
            degradation_encoder.load_state_dict(
                torch.load(best_path, map_location=device)
            )

    print(f'\n  Best val accuracy: {best_val_acc:.1f}%')
    print('  Freezing DegradationEncoder for Phase 2.\n')
    for p in degradation_encoder.parameters():
        p.requires_grad = False
    degradation_encoder.eval()


#  Phase 2: restoration model training 

def train_Phase2(
    encoder,
    degradation_encoder,
    alignment,
    diffusion,
    train_loader,
    val_loader,
    num_epochs     = 300,
    lr             = 1e-5,
    decay          = 1e-4,
    checkpoint_dir = '/checkpoints',
    eval_every     = 5,
    save_every     = 10,
    sample_steps   = 100,
    es_patience    = 3,
    es_min_delta   = 2e-3,
    freeze_encoder = 10,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    early_stopping = EarlyStopping(patience=es_patience, min_delta=es_min_delta)

    # Only encoder + diffusion are updated; alignment/RAFT and classifier are frozen.
    params    = list(encoder.parameters()) + list(diffusion.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=decay)

    warmup_epochs    = max(1, int(num_epochs * 0.05))
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    scaler = GradScaler()

    psnr_metric  = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = lpips.LPIPS(net='alex').to(device)
    lpips_vgg    = lpips.LPIPS(net='vgg').to(device)

    best_score  = 0.0
    best_psnr   = 0.0
    best_ssim   = 0.0
    best_lpips  = 0.0
    psnr_score  = 0.0
    ssim_score  = 0.0
    lpips_score = 0.0
    comb_score  = 0.0
    encoder_frozen = False

    print(' Restoration model training ')
    for epoch in range(num_epochs):
        if freeze_encoder > 0 and epoch < freeze_encoder:
            encoder.train()
            alignment.train()
            for p in encoder.parameters():
                p.requires_grad = True
        else:
            if not encoder_frozen:
                for p in encoder.parameters():
                    p.requires_grad = False
                encoder_frozen = True
                print(f'  [Epoch {epoch+1}] Swin encoder frozen')
                encoder.eval()
                alignment.eval()

        diffusion.train()
        degradation_encoder.eval()

        epoch_loss = 0.0

        for batch in train_loader:
            key_frame    = batch['key_frame'].to(device)
            aux_frames   = batch['aux_frames'].to(device)
            gt_key_frame = batch['gt_key_frame'].to(device)

            optimizer.zero_grad()

            with autocast():
                deg_vec = degradation_encoder(key_frame)  # (B, 736)

                F_0, F_k       = encoder(key_frame, aux_frames)
                aligned, flows = alignment(key_frame, aux_frames, F_k)

                # Simple fusion: concat F_0 with mean of aligned aux features
                context = {
                    scale: torch.cat(
                        [F_0[scale],
                         torch.stack([a[scale] for a in aligned]).mean(0)],
                        dim=1,
                    )
                    for scale in F_0
                }

                loss, delta = diffusion.training_loss(
                    gt_key_frame, key_frame,
                    context=context,
                    lpips_vgg=lpips_vgg,
                    deg_vec=deg_vec,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{num_epochs}]  '
              f'Loss: {avg_loss:.4f}  LR: {current_lr:.6f}')
        

        if (epoch + 1) % eval_every == 0:
            psnr_score, ssim_score, lpips_score = evaluate(
                encoder, degradation_encoder, alignment, diffusion,
                val_loader, psnr_metric, ssim_metric, lpips_metric,
                sample_steps=sample_steps,
            )
            print(f'  PSNR:  {psnr_score:.4f} dB')
            print(f'  SSIM:  {ssim_score:.4f}')
            print(f'  LPIPS: {lpips_score:.4f}')

            comb_score = combined_score(psnr_score, ssim_score, lpips_score)
            
            if comb_score > best_score:
                best_score = comb_score
                best_psnr  = psnr_score
                best_ssim  = ssim_score
                best_lpips = lpips_score
                save_checkpoint(
                    encoder, degradation_encoder, alignment, diffusion, optimizer, epoch,
                    score=comb_score, psnr=psnr_score,
                    ssim=ssim_score,  lpips=lpips_score,
                    path=os.path.join(checkpoint_dir, 'best_model.pth'),
                )
                print(f'  New best: {best_score:.4f}  PSNR: {psnr_score:.4f}  '
                      f'SSIM: {ssim_score:.4f}  LPIPS: {lpips_score:.4f}  — saved')

            if early_stopping(comb_score):
                save_checkpoint(
                    encoder, degradation_encoder, alignment, diffusion, optimizer, epoch,
                    score=comb_score, psnr=psnr_score,
                    ssim=ssim_score,  lpips=lpips_score,
                    path=os.path.join(checkpoint_dir,
                                      f'early_stop_epoch_{epoch+1}.pth'),
                )
                print(f'\n  Early stopping at epoch {epoch+1}.')
                print(f'  Best — PSNR: {best_psnr:.4f}  '
                      f'SSIM: {best_ssim:.4f}  LPIPS: {best_lpips:.4f}')
                break

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                encoder, degradation_encoder, alignment, diffusion, optimizer, epoch,
                score  = comb_score  if (epoch+1) % eval_every == 0 else 0.0,
                psnr   = psnr_score  if (epoch+1) % eval_every == 0 else 0.0,
                ssim   = ssim_score  if (epoch+1) % eval_every == 0 else 0.0,
                lpips  = lpips_score if (epoch+1) % eval_every == 0 else 0.0,
                path   = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'),
            )
            print(f'  Checkpoint saved at epoch {epoch+1}')

    print(f'\nTraining complete.  Best — PSNR: {best_psnr:.4f}  '
          f'SSIM: {best_ssim:.4f}  LPIPS: {best_lpips:.4f}')


#  Evaluation 

@torch.no_grad()
def evaluate(
    encoder, degradation_encoder, alignment, diffusion,
    loader, psnr_metric, ssim_metric, lpips_metric,
    sample_steps=100,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.eval()
    degradation_encoder.eval()
    diffusion.eval()
    alignment.eval()

    total_psnr  = 0.0
    total_ssim  = 0.0
    total_lpips = 0.0

    for batch in loader:
        key_frame    = batch['key_frame'].to(device)
        aux_frames   = batch['aux_frames'].to(device)
        gt_key_frame = batch['gt_key_frame'].to(device)

        deg_vec = degradation_encoder(key_frame)  # (B, 736)

        F_0, F_k       = encoder(key_frame, aux_frames)
        aligned, flows = alignment(key_frame, aux_frames, F_k)

        context = {
            scale: torch.cat(
                [F_0[scale],
                 torch.stack([a[scale] for a in aligned]).mean(0)],
                dim=1,
            )
            for scale in F_0
        }

        J = diffusion.sample(key_frame, context,
                             T=sample_steps, deg_vec=deg_vec)

        total_psnr  += psnr_metric(J, gt_key_frame).item()
        total_ssim  += ssim_metric(J, gt_key_frame).item()
        total_lpips += lpips_metric(
            J * 2 - 1, gt_key_frame * 2 - 1
        ).mean().item()

    n = len(loader)
    return total_psnr / n, total_ssim / n, total_lpips / n


#  Checkpoint helpers 

def save_checkpoint(encoder, degradation_encoder, alignment, diffusion,
                    optimizer, epoch, score, psnr, ssim, lpips, path):
    torch.save({
        'epoch':      epoch,
        'psnr':       psnr,
        'ssim':       ssim,
        'lpips':      lpips,
        'score':      score,
        'encoder':    encoder.state_dict(),
        'degradation_encoder': degradation_encoder.state_dict(),
        'alignment':  alignment.state_dict(),
        'diffusion':  diffusion.state_dict(),
        'optimizer':  optimizer.state_dict(),
    }, path)


def load_checkpoint(path, encoder, degradation_encoder, alignment, diffusion,
                    optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(path, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    if 'degradation_encoder' in ckpt:
        degradation_encoder.load_state_dict(ckpt['degradation_encoder'])
    alignment.load_state_dict(ckpt['alignment'])
    diffusion.load_state_dict(ckpt['diffusion'])
    if optimizer is not None and ckpt.get('optimizer') is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  PSNR:  {ckpt['psnr']:.2f} dB")
    print(f"  SSIM:  {ckpt['ssim']:.4f}")
    print(f"  LPIPS: {ckpt['lpips']:.4f}")
    print(f"  Score: {ckpt['score']:.4f}")
    return ckpt['epoch']


def train(    
    model,
    train_loader,
    val_loader,
    p1_num_epochs     = 100,
    p1_lr             = 1e-3,
    p1_checkpoint_dir = None,
    p1_es_patience    = 5,
    p1_es_min_delta   = 0.5,
    p2_num_epochs     = 300,
    p2_lr             = 1e-5,
    p2_decay          = 1e-4,
    p2_checkpoint_dir = '/checkpoints',
    p2_eval_every     = 5,
    p2_save_every     = 10,
    p2_sample_steps   = 100,
    p2_es_patience    = 3,
    p2_es_min_delta   = 2e-3,
    p2_freeze_encoder = 10,
    ):
    # Phase 1: pre-train degradation encoder
    train_Phase1(
        degradation_encoder = model.degradation_encoder,
        train_loader       = train_loader,
        val_loader         = val_loader,
        num_epochs         = p1_num_epochs,
        lr                 = p1_lr,
        checkpoint_dir     = p1_checkpoint_dir,
        es_patience        = p1_es_patience,
        es_min_delta       = p1_es_min_delta,
    )
    # Phase 2: train full restoration model
    train_Phase2(
        encoder           = model.encoder,
        degradation_encoder= model.degradation_encoder,
        alignment         = model.alignment,
        diffusion         = model.diffusion,
        train_loader      = train_loader,
        val_loader        = val_loader,
        num_epochs        = p2_num_epochs,
        lr                = p2_lr,
        decay             = p2_decay,
        checkpoint_dir    = p2_checkpoint_dir,
        eval_every        = p2_eval_every,
        save_every        = p2_save_every,
        sample_steps      = p2_sample_steps,
        es_patience       = p2_es_patience,
        es_min_delta      = p2_es_min_delta,
        freeze_encoder    = p2_freeze_encoder,
    )