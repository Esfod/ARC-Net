import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import shutil

class Dataset_Combi(Dataset):
  '''
  For each sample returns:
  - key frame:    (3, H, W)     to clean  - frame 000
  - aux_frames:   (2, 3, H, W)  neithbor frames  - frame 001, 002
  - gt_key_frame: (3, H, W)     clean - GT fro key frame only
  '''
  def __init__(self, root, split='Train', crop_size=256, augment=True, num_images = 3, scenes_=None):
    '''
    Args:
      root:       file
      split:      'Train' or 'Test'
      crop_size:  resize image
      augment:    apply random flips (only training) gives more varitiy
      num_images: key + aux frames
    '''
    self.crop_size = crop_size
    self.augment = augment and split == 'Train'
    self.num_images = num_images
    self.samples = []
    self.is_spac = False

    if 'REVIDE' in root:    # Haze
      self.degradation_type = 'haze'
      base = os.path.join(root, 'Train' if split == 'Validation' else split)
      self.restoration_root = os.path.join(base,'hazy')
      self.gt_root = os.path.join(base, 'gt')

      all_scenes = sorted(os.listdir(self.restoration_root))
      scenes = scenes_ if scenes_ is not None else all_scenes
      for scene in scenes:
        frames = sorted(os.listdir(os.path.join(self.restoration_root,scene)))
        frames = [f for f in frames if f.lower().endswith(('.png','.jpg'))]

        for i in range(len(frames) - self.num_images + 1):
          window = frames[i:i+self.num_images]
          if len(window) == self.num_images:
            self.samples.append((scene,*window))
      print(f'[REVIDE | {split}] {len(scenes)} scenes, {len(self.samples)} samples')

    elif 'RVSD' in root: # Snow
      self.degradation_type = 'snow'
      base = os.path.join(root, 'train')
      self.restoration_root = os.path.join(base, 'snow')
      self.gt_root = os.path.join(base, 'clean')

      scenes = scenes_ if scenes_ is not None else sorted(os.listdir(self.restoration_root))
      for scene in scenes:
        frames = sorted(os.listdir(os.path.join(self.restoration_root,scene)))
        frames = [f for f in frames if f.lower().endswith(('.png','.jpg'))]

        for i in range(len(frames) - self.num_images + 1):
          window = frames[i:i+self.num_images]
          if len(window) == self.num_images:
            self.samples.append((scene,*window))

      print(f'[RVSD | {split}] {len(scenes)} scenes, {len(self.samples)} samples')

    elif 'SPAC' in root:  #Rain
      self.degradation_type = 'rain'
      self.is_spac = True
      if (split == 'Train') or (split == 'Validation'):
        base = os.path.join(root, 'Dataset_Training_Synthetic')
      else:
        base = os.path.join(root, 'Dataset_Testing_Synthetic')

      all_gt_folders = sorted([f for f in os.listdir(base) if '_GT' in f])
      gt_folders = [f for f in all_gt_folders if f.replace('_GT','') in scenes_] \
                    if scenes_ is not None else all_gt_folders
      for gt_folder in gt_folders:
        scene_prefix = gt_folder.replace('_GT','')
        gt_dir = os.path.join(base, gt_folder)

        rain_folders = sorted([f for f in os.listdir(base)
                              if '_Rain' in f and scene_prefix in f])
        for rain_folder in rain_folders:
          rain_dir = os.path.join(base, rain_folder)
          frames = sorted([f for f in os.listdir(rain_dir)
                            if f.lower().endswith(('.png','.jpg'))])
          for i in range(len(frames) - num_images + 1):
            window = frames[i:i+self.num_images]
            if len(window) == num_images:
              self.samples.append((rain_dir, gt_dir, *window))
      print(f'[SPAC | {split}] {len(gt_folders)} scenes, {len(self.samples)} samples')


  def __len__(self):
    return len(self.samples)

  def _load_img(self, path):
    return Image.open(path).convert('RGB')

  def _apply_transforms(self, imgs, augment):
    '''
    Apply random flip to images
    Crop images to save memory and detail each scene is croped to a 256x256
    size frame with no downsampling
    '''
    w, h = imgs[0].size

    max_attempts = 20
    for _ in range(max_attempts):
        if augment:
            crop_x = random.randint(0, w - self.crop_size)
            crop_y = random.randint(0, h - self.crop_size)
        else:
            crop_x = (w - self.crop_size) // 2
            crop_y = (h - self.crop_size) // 2

        # Check GT image variance — skip flat crops
        gt_img = imgs[-1]  # GT is always last
        test_crop = gt_img.crop((crop_x, crop_y,
                                  crop_x + self.crop_size,
                                  crop_y + self.crop_size))
        variance = np.array(test_crop).var()

        if variance > 50 or not augment:
            break

    flip = augment and random.random() > 0.5

    result = []
    for img in imgs:
        img = img.crop((crop_x, crop_y,
                        crop_x + self.crop_size,
                        crop_y + self.crop_size))
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = transforms.functional.to_tensor(img)
        result.append(img)

    return result


  def __getitem__(self, idx):
    if self.is_spac:
      rain_dir, gt_dir, *frames = self.samples[idx]
      degraded_imgs = [self._load_img(os.path.join(rain_dir, f)) for f in frames]
      gt_0  = self._load_img(os.path.join(gt_dir, frames[0]))
      scene = os.path.basename(rain_dir)

    else:
      scene, *frames = self.samples[idx]
      degraded_imgs =   [self._load_img(os.path.join(self.restoration_root, scene, f)) for f in frames]
      gt_0 = self._load_img(os.path.join(self.gt_root, scene, frames[0]))


    all_imgs = self._apply_transforms(degraded_imgs + [gt_0], augment=self.augment)
    degraded_tensors = all_imgs[:-1]
    gt_tensor = all_imgs[-1]

    return {
        'key_frame':        degraded_tensors[0],
        'aux_frames':       torch.stack(degraded_tensors[1:]),
        'gt_key_frame':     gt_tensor,
        'scene':            scene,
        'frame_idx':        frames[0],
        'degradation_type': self.degradation_type,
    }

def get_dataloaders(roots, crop_size=256, batch_size=2, num_workers=2, num_images=3, train_split=0.8, val_split=0.1, seed = 42):
  random.seed(seed)

  train_datasets = []
  val_datasets = []
  test_datasets = []

  for root in roots:

    if'RVSD' in root:
      print('RVSD')
      all_scenes = sorted(os.listdir(os.path.join(root,'train','snow')))
    elif 'REVIDE' in root:
      print('REVIDE')
      all_scenes = sorted(os.listdir(os.path.join(root,'Train','hazy')))
    elif 'SPAC' in root:
      print('SPAC')
      base = os.path.join(root, 'Dataset_Training_Synthetic')
      all_scenes = sorted(set(
          f.replace('_GT','') for f in os.listdir(base) if '_GT' in f
      ))

    random.shuffle(all_scenes)
    if 'RVSD' in root:
      train_end = int(len(all_scenes) * train_split)
      val_end = int(len(all_scenes) * (train_split + val_split))
      train_scenes = all_scenes[:train_end]
      val_scenes = all_scenes[train_end:val_end]
      test_scenes = all_scenes[val_end:]


      train_datasets.append(Dataset_Combi(root, split='Train', crop_size=crop_size,
                                          augment=True, num_images=num_images, scenes_=train_scenes))
      val_datasets.append(Dataset_Combi(root, split='Validation',crop_size=crop_size,
                                          augment=False, num_images=num_images, scenes_=val_scenes))
      test_datasets.append(Dataset_Combi(root, split='Test', crop_size=crop_size,
                                          augment=False, num_images=num_images, scenes_=test_scenes))
    else:
      train_end = int(len(all_scenes) * train_split)
      val_end = int(len(all_scenes) * (train_split + val_split))
      train_scenes = all_scenes[:train_end]
      val_scenes = all_scenes[train_end:val_end]


      train_datasets.append(Dataset_Combi(root, split='Train', crop_size=crop_size,
                                          augment=True, num_images=num_images,scenes_=train_scenes))
      val_datasets.append(Dataset_Combi(root, split='Validation', crop_size=crop_size,
                                          augment=False, num_images=num_images,scenes_=val_scenes))
      test_datasets.append(Dataset_Combi(root, split='Test', crop_size=crop_size,
                                          augment=False, num_images=num_images))
  combined_train_dataset = ConcatDataset(train_datasets)
  combined_val_dataset = ConcatDataset(val_datasets)
  combined_test_dataset = ConcatDataset(test_datasets)
  train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle = True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
  val_loader = DataLoader(combined_val_dataset, batch_size=1, shuffle = False,
                         num_workers=num_workers, pin_memory=True)
  test_loader = DataLoader(combined_test_dataset, batch_size=1, shuffle = False,
                             num_workers=num_workers, pin_memory=True)

  return train_loader, val_loader, test_loader


def load_dataset_from_googledrive():
  drive_path = '/content/drive/MyDrive/MasterDataset'
  dataset_name = ['SPAC','REVIDE-haze','RVSD_train']
  extraction_path = '/content/extracted_datasets'

  for dataset in dataset_name:
    path_to_archive = os.path.join(drive_path, dataset + '.tar')
    os.makedirs(extraction_path, exist_ok=True)

    if os.path.exists(path_to_archive):
      if dataset == 'SPAC':
        spac_dir = os.path.join(extraction_path,'SPAC')
        os.makedirs(spac_dir, exist_ok=True)
        print(f'Extracting {dataset}')
        shutil.unpack_archive(path_to_archive, spac_dir)
        print(f'Done Extracting {dataset}')
      else:
        print(f'Extracting {dataset}')
        shutil.unpack_archive(path_to_archive, extraction_path)
        print(f'Done Extracting {dataset}')
    else:
        print(f'Error: File not found at {path_to_archive}')
  print('All done extracting')


def setup_dataloaders():
  roots = [
      '/content/extracted_datasets/REVIDE-haze',
      '/content/extracted_datasets/RVSD_train',
      '/content/extracted_datasets/SPAC',
  ]
  train_loader, val_loader, test_loader = get_dataloaders(roots)
  print(f'Train_loader lenght: {len(train_loader)}')
  print(f'Val_loader lenght: {len(val_loader)}')
  print(f'Test_loader lenght: {len(test_loader)}\n')
  return train_loader, val_loader, test_loader