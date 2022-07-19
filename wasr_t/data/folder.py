from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

class FolderDataset(torch.utils.data.Dataset):
    """Dataset wrapper for a general directory of images."""

    def __init__(self, image_dir, normalize_t=None):
        """Creates the dataset.

        Args:
            image_dir (str): path to the image directory. Can contain arbitrary subdirectory structures.
            normalize_t (callable, optional): Transform used to normalize the images. Defaults to None.
        """

        self.image_dir = Path(image_dir)
        self.images = sorted([p.relative_to(image_dir) for p in Path(image_dir).glob('**/*.jpg')])

        self.normalize_t = normalize_t

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rel_path = self.images[idx]
        img_path = self.image_dir / rel_path
        img = np.array(Image.open(str(img_path)))

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)


        features = {
            'image': img
        }


        metadata = {
            'image_name': img_path.name,
            'image_path': str(rel_path)
        }

        return features, metadata
