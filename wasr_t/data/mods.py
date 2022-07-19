import os
from logging import warning
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF


HIST_FRAMES_SUBDIR = 'frames_hist'

class MODSDataset(torch.utils.data.Dataset):
    """MODS dataset wrapper"""

    def __init__(self, seq_mapping, transform=None, normalize_t=None, hist_len=None):
        """
        Args:
            seq_mapping (str): Path to the txt file, containing mods sequence -> subdir mappings
            transform (callable, optional): Tranform to apply to image and masks
            normalize_t (callable, optional): Normalization transform
            hist_len (int, optional): History (past frames) length to include
        """
        seq_mapping = Path(seq_mapping)
        base_dir = seq_mapping.parent

        data = []
        with seq_mapping.open() as file:
            seq_pairs = (tuple(l.split()) for l in file)

            for modd_seq, seq in seq_pairs:
                seq_dir = base_dir / seq
                modd_mapping_fn = seq_dir / 'mapping.txt'
                imu_mapping_fn = seq_dir / 'imu_mapping.txt'

                seq_data = {}
                with (seq_dir / 'imu_mapping.txt').open() as file:
                    imu_pairs = (tuple(l.split()) for l in file)
                    for img_fn, imu_fn in imu_pairs:
                        img_path = seq_dir / img_fn
                        imu_path = seq_dir / imu_fn

                        seq_data[img_path.name] = {
                            'image_path': str(img_path),
                            'imu_path': str(imu_path),
                            'name': img_path.name,
                            'modd_seq': modd_seq,
                            'seq': seq
                        }

                        if hist_len is not None:
                            image_name = Path(img_fn).stem
                            hist_paths = [seq_dir / 'frames_hist' / ('%s_%d.jpg' % (image_name, i)) for i in range(hist_len)]
                            seq_data[img_path.name]['image_hist_paths'] = hist_paths


                with (seq_dir / 'mapping.txt').open() as file:
                    pairs = (tuple(l.split()) for l in file)
                    for img_name, modd_name in pairs:
                        seq_data[img_name]['modd_name'] = modd_name

                data.extend(seq_data.values())

        self.data = data
        self.transform = transform
        self.normalize_t = normalize_t

    def __len__(self):
        return len(self.data)

    def _load_hist_images(self, image_path, hist_paths):
        if not all(Path(p).exists() for p in hist_paths):
            # Fake hist frames: repeat current frame N times
            warning(f'Missing history data for image `{image_path}`. Faking history with copies of the target frame.')
            hist_paths = [image_path for _ in range(len(hist_paths))]

        hist_images = [np.array(Image.open(img_path)) for img_path in hist_paths]

        # Reverse order
        return hist_images[::-1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        entry = self.data[idx]

        img = np.array(Image.open(entry['image_path']))
        imu = np.array(Image.open(entry['imu_path']))

        data = {
            'image': img,
            'imu_mask': imu
        }

        if 'image_hist_paths' in entry:
            hist_images = self._load_hist_images(entry['image_path'], entry['image_hist_paths'])
            data['extra_images'] = hist_images

        extra_images = data['extra_images'] if 'extra_images' in data else []

        # Transform images and masks if transform is provided
        if self.transform is not None:
            transformed = self.transform(data)
            img = transformed['image']
            imu = transformed['imu_mask']
            extra_images = data['extra_images'] if 'extra_images' in data else []

        if self.normalize_t is not None:
            img = self.normalize_t(img)
            extra_images = [self.normalize_t(img) for img in extra_images]
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)
            extra_images = [TF.to_tensor(img) for img in extra_images]

        imu = torch.from_numpy(imu.astype(np.bool))

        metadata_fields = ['seq', 'name', 'modd_seq', 'modd_name']
        metadata = {field: entry[field] for field in metadata_fields}
        metadata['image_path'] = os.path.join(metadata['seq'], metadata['name'])

        features ={
            'image': img,
            'imu_mask': imu,
        }

        if len(extra_images) > 0:
            extra_images = torch.stack(extra_images)
            features['hist_images'] = extra_images

        return features, metadata
