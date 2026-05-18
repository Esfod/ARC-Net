import torch
import os

from utils.dataset_utils import get_dataloaders
from utils.training_utils import train
from model.model import ARCNet


batch_size      = 70
num_workers     = 12
crop_size       = 128   # 256, 384, 512
num_images      = 10

roots = [
    '/dataset/REVIDE-haze',
    '/dataset/RVSD_train',
    '/dataset/SPAC',
]
train_loader, val_loader, test_loader = get_dataloaders(
    roots, crop_size=crop_size, batch_size=batch_size,
    num_workers=num_workers, num_images=num_images,
)

print(f'Train_loader length: {len(train_loader)}')
print(f'Val_loader   length: {len(val_loader)}')
print(f'Test_loader  length: {len(test_loader)}\n')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#  Hyperparameters 
num_epochs      = 300
lr              = 1e-5
decay           = 1e-4
eval_every      = 10
save_every      = 100
sample_steps    = 10
diffusion_steps = 1000

checkpoint_dir      = '/content/drive/MyDrive/MasterDataset/checkpoints'
project_name        = 'ALL-IN-ONE-Class_1_2_test'
run_checkpoint_path = os.path.join(checkpoint_dir, project_name)

#  Initialise model 
model = ARCNet(crop_size, num_images, diffusion_steps).to(device)

train(
    model,
    train_loader,
    val_loader,
    p1_num_epochs     = 100,
    p1_lr             = 1e-3,
    p1_checkpoint_dir = None,
    p1_es_patience    = 5,
    p1_es_min_delta   = 0.5,
    p2_num_epochs     = num_epochs,
    p2_lr             = lr,
    p2_decay          = decay,
    p2_checkpoint_dir = checkpoint_dir,
    p2_eval_every     = eval_every,
    p2_save_every     = save_every,
    p2_sample_steps   = sample_steps,
    p2_es_patience    = 3,
    p2_es_min_delta   = 2e-3,
)

print('Training complete.')